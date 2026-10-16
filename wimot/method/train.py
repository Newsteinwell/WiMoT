"""Training and stage-wise evaluation of the proposed method and its ablations.

First domain: the dual stem, adapter and projector are trained with SupCon only (100 epochs), then frozen into the
512-dimensional basis phi (stem 256 + first-domain PCA shortcut 256). Later domains: only the affine adapter
(A, b), the projector and the head W used inside L_hist are trained (50 epochs) with 0.5*SupCon + L_hist; class
moments of phi are stored per domain and the OSCL readout is refitted from them after every stage.
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..common import (PROTOCOL, SEMANTIC, AMPLITUDE, empty_cache, file_lock, first_cap, later_cap,
                      metrics, retain, seed_all, sha, state_digest, write_json, read_json)
from . import hist_loss as lm
from .calibration import head_calibration, normalized
from .model import FirstModel, Model, SupConLoss, disable_pca, install_first_pca, model_from_checkpoint
from .readout import StudentDecoder, case_from_key, head_from
from .statistics import load_stats, moments, save_stats, statistic_bytes, stored, transport

T = PROTOCOL['training']
M = PROTOCOL['method']
MAIN = dict(supcon=True, amplitude=True, pca=True, moment=True, aggregate=True, replay=False, adapt=True,
            storage=None, rank=None, semantic='mantis')
VARIANTS = {
    'main': MAIN,
    'no_supcon': dict(MAIN, supcon=False),
    'no_hist': dict(MAIN, moment=False),
    'no_amp': dict(MAIN, amplitude=False),
    'no_pca': dict(MAIN, pca=False),
    'no_adapt': dict(MAIN, adapt=False),
    'no_histstats': dict(MAIN, aggregate=False, moment=False),
    'replay': dict(MAIN, replay=True),
    'triu': dict(MAIN, storage='triu'),
    'rank8': dict(MAIN, storage='rank', rank=8),
    'rank32': dict(MAIN, storage='rank', rank=32),
    'rank128': dict(MAIN, storage='rank', rank=128),
    # Same-input study: amplitude descriptor only (no PCA shortcut), or the amplitude copied into the semantic block.
    'amp_only': dict(MAIN, pca=False, semantic='zero'),
    'amp_full': dict(MAIN, semantic='amp'),
    # Transport-ladder ablation (off by default): the stored class moments of earlier domains stay in the h coordinates of the
    # adapter of their own domain and are not transported through the updated adapter (spec key ``transport``, default True).
    'no_transport': dict(MAIN, transport=False),
    'no_transport_no_hist': dict(MAIN, moment=False, transport=False),
}


SWEEP = re.compile(r'^main_sc(?P<sc>\d+(?:\.\d+)?)_h(?P<h>\d+(?:\.\d+)?)(?:_fsc(?P<fsc>\d+(?:\.\d+)?))?$')


def sweep_name(supcon_weight=None, hist_weight=None, first_supcon_weight=None):
    """Variant name of a loss-weight setting; the paper setting (0.5, 1, first domain 0.5) is ``main`` itself."""
    sc = M['supcon_weight'] if supcon_weight is None else supcon_weight; h = 1. if hist_weight is None else hist_weight
    if (sc, h) == (M['supcon_weight'], 1.) and first_supcon_weight in (None, M['supcon_weight']): return 'main'
    return f'main_sc{sc:g}_h{h:g}' + ('' if first_supcon_weight is None else f'_fsc{first_supcon_weight:g}')


def variant_spec(name):
    """Specification of a variant: an entry of ``VARIANTS`` or a loss-weight setting ``main_sc<w>_h<w>[_fsc<w>]``.

    ``sc``: weight of SupCon in the later domains (paper: 0.5), ``h``: weight of L_hist (paper: 1), ``fsc``: weight of SupCon
    in the first domain (paper: 0.5; omitted, it stays at the paper value). The variants of ``VARIANTS`` carry no weight keys
    and use the defaults, so that the later-domain loss of ``main`` is exactly ``0.5*L_SC + 1*L_hist`` as before.
    """
    if name in VARIANTS: return VARIANTS[name]
    m = SWEEP.match(name)
    if not m: raise KeyError(f'Unknown variant {name}')
    spec = dict(MAIN, supcon_weight=float(m['sc']), hist_weight=float(m['h']))
    if m['fsc'] is not None: spec['first_supcon_weight'] = float(m['fsc'])
    return spec


def load(data, domain, split, spec):
    x, y = data.raw(domain, split)
    return mask(x, spec), y


def mask(x, spec):
    """Input variants: drop the amplitude block, or use only/copy the amplitude descriptor (same-input study)."""
    if not spec['amplitude']:
        x = x.copy(); x[:, SEMANTIC:] = 0
    if spec['semantic'] == 'zero':
        x = x.copy(); x[:, :SEMANTIC] = 0
    elif spec['semantic'] == 'amp':
        x = x.copy(); x[:, :SEMANTIC] = 0; x[:, :AMPLITUDE] = x[:, SEMANTIC:]
    return x


def norm_fit(x):
    """Semantic block: per-coordinate z-score scaled by 1/sqrt(9216); amplitude block: uncentred, one RMS scale / sqrt(512)."""
    mean = x.mean(0, dtype=np.float64); sd = x.std(0, ddof=1, dtype=np.float64).clip(1e-6)
    mean[SEMANTIC:] = 0.; sd[:SEMANTIC] *= np.sqrt(SEMANTIC)
    sd[SEMANTIC:] = max(float(np.sqrt(np.mean(x[:, SEMANTIC:].astype(np.float64)**2))), 1e-6)*np.sqrt(AMPLITUDE)
    return mean, sd


def warm_key(spec):
    key = f"sc{int(spec['supcon'])}_amp{int(spec['amplitude'])}_pca{int(spec['pca'])}"
    if spec.get('first_supcon_weight', M['supcon_weight']) != M['supcon_weight']: key += f"_fsc{spec['first_supcon_weight']:g}"
    return key if spec['semantic'] == 'mantis' else f"{key}_sem{spec['semantic']}"


def train_first(raw, y, classes, spec, seed, device):
    """First-domain model from raw (already masked) features: returns the checkpoint dictionary and a training record."""
    mean, sd = norm_fit(raw); x = ((raw-mean)/sd).astype(np.float32)
    seed_all(seed); model = FirstModel(SEMANTIC+AMPLITUDE, classes).to(device)
    model.stem.calibrate_amplitude(torch.from_numpy(x).to(device))
    opt = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=T['lr'], weight_decay=T['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T['first_epochs'])
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=T['batch_size'],
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    sup = SupConLoss(M['supcon_temperature']); losses = []
    first_weight = spec.get('first_supcon_weight', M['supcon_weight'])
    for epoch in range(T['first_epochs']):
        model.train(); values = []
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device); opt.zero_grad(set_to_none=True)
            z, logits = model(bx); value = logits.sum()*0
            if spec['supcon']: value = value+first_weight*sup(z, by)
            assert torch.isfinite(value); value.backward(); opt.step(); values.append(float(value.detach()))
        scheduler.step(); losses.append(float(np.mean(values)))
    model.eval(); model.stem.requires_grad_(False); model = Model(model).to(device).eval()
    if spec['pca']: basis = install_first_pca(model, x, padded=spec['semantic'] == 'amp')
    else: basis = disable_pca(model)
    model.stem.requires_grad_(False)
    cp = dict(state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
              mean=mean, sd=sd, dimension=SEMANTIC+AMPLITUDE, classes=classes)
    return cp, dict(basis=basis, losses=losses)


def warmup(root, data, seed, spec, device):
    """First-domain model, shared by every variant with the same (supcon, amplitude, pca, input) setting."""
    out = Path(root)/'warmup'/data.dataset/data.domains[0]/f'seed{seed}'/warm_key(spec)
    with file_lock(out.parent/'.lock'):
        if (out/'complete.json').exists():
            proof = read_json(out/'complete.json')
            for name, h in proof['files'].items(): assert sha(out/name) == h, f'Changed warm-up file {out/name}'
            return out
        out.mkdir(parents=True, exist_ok=True)
        raw, y = load(data, data.domains[0], 'train', spec)
        cp, info = train_first(raw, y, data.classes, spec, seed, device)
        torch.save(cp, out/'stage1.pt')
        write_json(out/'complete.json', dict(complete=True, spec=spec, basis=info['basis'], losses=info['losses'],
                                             files={'stage1.pt': sha(out/'stage1.pt')}))
        return out


TRAINABLE = {'adapter.weight', 'adapter.bias', 'projector.weight', 'projector.bias', 'classifier.weight', 'classifier.bias'}


def save_checkpoint(path, model, warm):
    """Stage checkpoint: trainable tensors plus a hash-checked reference to the frozen first-domain checkpoint."""
    trainable = {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}
    assert set(trainable) == TRAINABLE
    torch.save(dict(first_checkpoint=str(warm), first_checkpoint_sha256=sha(warm), trainable_state=trainable), path)


def load_checkpoint(path):
    cp = torch.load(path, map_location='cpu', weights_only=False)
    if 'trainable_state' not in cp: return cp
    source = Path(cp['first_checkpoint'])
    assert sha(source) == cp['first_checkpoint_sha256'], f'Changed first-domain checkpoint {source}'
    first = torch.load(source, map_location='cpu', weights_only=False)
    state = {k: v.clone() for k, v in first['state'].items()}
    for name, value in cp['trainable_state'].items(): state[name] = value.clone()
    return dict(state=state, mean=first['mean'], sd=first['sd'], dimension=first['dimension'], classes=first['classes'])


def infer(model, x, device):
    model.eval()
    with torch.inference_mode():
        return np.concatenate([model.stem(torch.from_numpy(x[i:i+128]).to(device)).cpu().numpy() for i in range(0, len(x), 128)])


def run_dir(root, data, seed, variant):
    return Path(root)/'method'/data.tag/f'seed{seed}'/variant


class Learner:
    """One trajectory held in memory: the frozen first-domain checkpoint ``cp``, the adapter, the stored class moments
    and the OSCL calibration, updated one domain at a time by ``first`` and ``update``."""
    def __init__(self, cp, classes, seed, spec, device='cuda'):
        self.cp, self.classes, self.seed, self.spec, self.device = cp, classes, seed, spec, device
        self.model = model_from_checkpoint(cp).to(device).eval(); self.model.stem.requires_grad_(False)
        self.frozen = state_digest(self.model.stem.state_dict())
        self.stats, self.constraints, self.replay, self.cap, self.scale, self.task = [], [], [], None, 1., 0
        self.calibration = None
        self.transported = spec.get('transport', True)
        if not self.transported and not spec['aggregate']: raise ValueError('transport=False needs the stored moments of every domain')
        self.h_stats = []          # transport=False only: moments of each domain in the h coordinates of its own adapter

    def readout_stats(self):
        """The moments the readout is fitted from: phi coordinates (transported at fit time), or frozen h coordinates."""
        return self.stats if self.transported else self.h_stats

    def norm(self, raw):
        return ((raw-self.cp['mean'])/self.cp['sd']).astype(np.float32)

    def keep(self, s):
        return stored(s, self.spec['storage'], self.spec['rank'])

    def first(self, raw, y):
        """First domain (the model itself was trained by ``train_first``): store its moments; returns the stage record."""
        spec, model, classes = self.spec, self.model, self.classes
        current = infer(model, self.norm(raw), self.device)
        self.stats = [self.keep(moments(current, y, classes))]
        self.constraints = [lm.constraint(model, self.stats[0])] if spec['moment'] else []
        self.scale = lm.first_scale(model, self.stats[0]) if spec['moment'] else 1.
        self.cap = first_cap(len(y), classes)
        self.replay = retain([], current, y, np.arange(len(y)), 0, classes, self.cap//classes, self.seed) if spec['replay'] else []
        self.task = 1
        if not self.transported: self.h_stats = transport(self.stats, model)       # h coordinates of the first-domain adapter
        self.calibration = head_calibration(model, self.stats, current, y, np.zeros(len(y), int), M['readout']['calibration_seed'],
                                            None if self.transported else self.h_stats)
        return dict(stage=1, current_rows=len(y), old_rows=0, calibration=self.calibration)

    def update(self, raw, y):
        """Learn the next domain: adapter training on the arrivals and the stored moments, moment storage, recalibration."""
        spec, model, classes, seed, device, task = self.spec, self.model, self.classes, self.seed, self.device, self.task
        started = time.perf_counter()
        current = infer(model, self.norm(raw), device)
        new = self.keep(moments(current, y, classes))
        self.stats = self.stats+[new] if spec['aggregate'] else [new]
        self.cap = later_cap(self.cap, len(y), classes)
        old = self.replay[0] if spec['replay'] else empty_cache(512)
        if spec['replay']: assert len(old[1]) == self.cap
        pool = np.concatenate([current, old[0]]) if spec['replay'] else current
        targets = np.concatenate([y, old[1]]) if spec['replay'] else y
        tasks = np.concatenate([np.full(len(y), task), old[2]]) if spec['replay'] else np.full(len(y), task)
        loader = DataLoader(TensorDataset(torch.from_numpy(pool), torch.from_numpy(targets), torch.arange(len(pool))),
                            batch_size=T['batch_size'], shuffle=True, generator=torch.Generator().manual_seed(seed+task*100))
        opt = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=T['lr'], weight_decay=T['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T['later_epochs']); sup = SupConLoss(M['supcon_temperature'])
        losses = []
        sc_weight, hist_weight = spec.get('supcon_weight', M['supcon_weight']), spec.get('hist_weight', 1.)
        for epoch in range(T['later_epochs'] if spec['adapt'] else 0):
            model.train(); values = []
            for bx, by, _ in loader:
                bx, by = bx.to(device), by.to(device); opt.zero_grad(set_to_none=True)
                z, logits = model(bx, frozen_features=True); value = logits.sum()*0
                if spec['supcon']: value = value+sc_weight*sup(z, by)
                if spec['moment']: value = value+hist_weight*lm.loss(model, self.constraints, self.scale)
                assert torch.isfinite(value); value.backward(); opt.step(); values.append(float(value.detach()))
            scheduler.step(); losses.append(float(np.mean(values)))
        model.eval(); assert state_digest(model.stem.state_dict()) == self.frozen
        cx, cy, ct = (pool, targets, tasks) if spec['aggregate'] else (current, y, np.zeros(len(y), int))
        if not self.transported:   # new domain in the coordinates of the updated adapter; earlier domains keep theirs
            self.h_stats = self.h_stats+transport([new], model)
        self.calibration = head_calibration(model, self.stats, cx, cy, ct, M['readout']['calibration_seed'],
                                            None if self.transported else self.h_stats)
        row = dict(stage=task+1, current_rows=len(y), old_rows=len(old[1]), losses=losses, calibration=self.calibration,
                   statistic_bytes=statistic_bytes(self.stats))
        if spec['moment']: self.constraints.append(lm.constraint(model, new))
        row['moment_constraint_bytes'] = lm.state_bytes(self.constraints, self.scale) if spec['moment'] else 0
        if spec['replay']:
            self.replay = retain(self.replay, current, y, np.arange(len(y)), task, classes, self.cap//classes, self.seed)
            row.update(retained_rows=len(self.replay[0][1]), replay_bytes=sum(v.nbytes for v in self.replay[0]))
        row['seconds'] = time.perf_counter()-started
        self.task += 1
        return row

    def checkpoint(self):
        """The current stage as a checkpoint dictionary (what ``load_checkpoint`` returns for a saved stage)."""
        state = {k: v.clone() for k, v in self.cp['state'].items()}
        for k, v in self.model.named_parameters():
            if v.requires_grad: state[k] = v.detach().cpu().clone()
        return dict(state=state, mean=self.cp['mean'], sd=self.cp['sd'], dimension=self.cp['dimension'], classes=self.cp['classes'])

    def probabilities(self, raw):
        """Class probabilities of raw feature rows with the current stage; the domain identity is not used."""
        cp = self.checkpoint(); model = model_from_checkpoint(cp).to(self.device).eval().requires_grad_(False)
        phi = infer(model, ((raw-cp['mean'])/cp['sd']).astype(np.float32), self.device)
        return decode(cp, self.readout_stats(), self.calibration['selected']['case'], adapt(cp, phi), self.device,
                      self.transported)


def train(root, data, seed, variant, device='cuda'):
    spec = variant_spec(variant); out = run_dir(root, data, seed, variant)
    if (out/'receipt.json').exists() and read_json(out/'receipt.json')['complete']: return out
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob('stage*'): p.unlink()
    warm = warmup(root, data, seed, spec, device)/'stage1.pt'
    learner = Learner(load_checkpoint(warm), data.classes, seed, spec, device)
    raw, y = load(data, data.domains[0], 'train', spec); first = learner.first(raw, y)
    save_checkpoint(out/'stage1.pt', learner.model, warm); save_stats(out/'stage1_stats.npz', learner.readout_stats())
    receipt = dict(complete=False, dataset=data.dataset, domains=data.domains, seed=seed, variant=variant, spec=spec,
                   fraction=data.fraction, frozen_stem_sha256=learner.frozen, stages=[first])
    write_json(out/'receipt.json', receipt); del raw
    for domain in data.domains[1:]:
        raw, y = load(data, domain, 'train', spec); row = learner.update(raw, y); del raw
        save_checkpoint(out/f'stage{row["stage"]}.pt', learner.model, warm); save_stats(out/f'stage{row["stage"]}_stats.npz', learner.readout_stats())
        receipt['stages'].append(row); write_json(out/'receipt.json', receipt)
    receipt['complete'] = True; write_json(out/'receipt.json', receipt)
    return out


def adapt(cp, phi):
    """h = A phi + b in float64 with the adapter of the checkpoint."""
    return phi.astype(float)@cp['state']['adapter.weight'].numpy().T+cp['state']['adapter.bias'].numpy()


def decode(cp, stats, case_key, h, device='cuda', transported=True):
    """OSCL class probabilities of adapted rows ``h`` from the stored moments ``stats`` (phi coordinates, transported here;
    with ``transported=False`` the moments are already in h coordinates and are used as stored)."""
    model = model_from_checkpoint(cp).to(device).eval().requires_grad_(False)
    base = head_from(transport(stats, model) if transported else stats, cp['classes'])
    return normalized(StudentDecoder(base, *case_from_key(case_key)).predict_proba(h))


def evaluate(out, data, stage, split, device='cuda'):
    """Accuracy of the stage-``stage`` readout on every domain seen so far; the domain identity is never used."""
    out = Path(out); receipt = read_json(out/'receipt.json'); spec = receipt['spec']
    cp = load_checkpoint(out/f'stage{stage}.pt'); model = model_from_checkpoint(cp).to(device).eval().requires_grad_(False)
    hh, yy, tt = [], [], []
    for task, domain in enumerate(data.domains[:stage]):
        raw, y = load(data, domain, split, spec); phi = infer(model, ((raw-cp['mean'])/cp['sd']).astype(np.float32), device)
        hh.append(adapt(cp, phi)); yy.append(y); tt.append(np.full(len(y), task))
    h, y, t = map(np.concatenate, (hh, yy, tt))
    pred = decode(cp, load_stats(out/f'stage{stage}_stats.npz'), receipt['stages'][stage-1]['calibration']['selected']['case'],
                  h, device, spec.get('transport', True)).argmax(1)
    return [float(np.mean(pred[t == i] == y[t == i])) for i in range(stage)]


def test(out, data, device='cuda'):
    """Test matrix (row k: accuracies after stage k) and AFA/AFR; written to test.json next to the receipt."""
    T_ = len(data.domains); matrix = [[None]*T_ for _ in range(T_)]
    for stage in range(1, T_+1): matrix[stage-1][:stage] = evaluate(out, data, stage, 'test', device)
    receipt = read_json(Path(out)/'receipt.json')
    result = dict(family='method', dataset=data.dataset, order=data.order, domains=data.domains, seed=receipt['seed'],
                  method=receipt['variant'], coefficient=None,
                  input='cat9728' if receipt['spec']['semantic'] == 'mantis' else 'amp512', fraction=data.fraction, matrix=matrix,
                  metrics=metrics(matrix),
                  statistic_bytes=receipt['stages'][-1].get('statistic_bytes'),
                  moment_constraint_bytes=receipt['stages'][-1].get('moment_constraint_bytes'))
    write_json(Path(out)/'test.json', result)
    return result
