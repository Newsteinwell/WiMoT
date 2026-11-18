"""Classic continual-learning baselines on frozen features: fine-tuning, ER, EWC, SI, MAS, LwF (each native and +ER),
DER++ and A-GEM. All share one cross-entropy MLP and one first-domain warm-up per (stream, first domain, seed, input);
no checkpoint of the proposed method is ever loaded.
"""
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..common import PROTOCOL, digest, file_lock, first_cap, later_cap, metrics, read_json, retain, seed_all, sha, write_json
from .inputs import Input

T = PROTOCOL['training']


def parse_method(method):
    replay = method.endswith('_er') or method in ('er', 'derpp', 'agem')
    algorithm = method[:-3] if method.endswith('_er') else method
    return algorithm, replay


class Model(nn.Module):
    def __init__(self, dimension, classes):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(dimension, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                      nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.classifier = nn.Linear(128, classes)
        for layer in self.features.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.classifier(self.features(x))


def parameters(model):
    return dict(model.named_parameters())


def theta(model):
    return {k: v.detach().clone() for k, v in parameters(model).items()}


def tensor_bytes(state):
    return sum(v.numel()*v.element_size() for v in state.values())


def optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=T['lr'], weight_decay=T['weight_decay'])


def warmup(root, data, inp, seed, device):
    """First-domain CE training (100 epochs); records the SI path integral of the warm-up."""
    out = Path(root)/'warmup_classic'/data.dataset/data.domains[0]/f'seed{seed}'/inp.kind
    with file_lock(out.parent/f'.{inp.kind}.lock'):
        if (out/'complete.json').exists():
            proof = read_json(out/'complete.json')
            assert proof['complete'] and sha(out/'stage1.pt') == proof['sha256']
            return out
        out.mkdir(parents=True, exist_ok=True)
        x, y = inp.load(data.domains[0], 'train')
        mean, sd = inp.fit(x); x = ((x-mean)/sd).astype(np.float32)
        seed_all(seed); model = Model(inp.dimension, data.classes).to(device)
        opt = optimizer(model); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T['first_epochs'])
        loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                            batch_size=T['batch_size'], shuffle=True, generator=torch.Generator().manual_seed(seed))
        initial = theta(model); path = {k: torch.zeros_like(v) for k, v in initial.items()}
        losses = []
        for epoch in range(T['first_epochs']):
            model.train(); values = []
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device); opt.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(bx), by)
                assert torch.isfinite(loss); loss.backward(); before = theta(model)
                grad = {k: v.grad.detach().clone() for k, v in parameters(model).items()}
                opt.step()
                for k, v in parameters(model).items():
                    path[k].add_(-grad[k]*(v.detach()-before[k]))
                values.append(float(loss.detach()))
            scheduler.step(); losses.append(float(np.mean(values)))
        si = {k: (path[k]/((v.detach()-initial[k]).square()+.1)).clamp_min(0).cpu()
              for k, v in parameters(model).items()}
        torch.save(dict(state={k: v.detach().cpu() for k, v in model.state_dict().items()}, mean=mean, sd=sd,
                        classes=data.classes, dimension=inp.dimension, si_importance=si), out/'stage1.pt')
        write_json(out/'complete.json', dict(complete=True, sha256=sha(out/'stage1.pt'), losses=losses, first_rows=len(y)))
        return out


def load_checkpoint(path):
    cp = torch.load(path, map_location='cpu', weights_only=False)
    if cp.get('first_reference'):
        first = Path(cp['first_reference']); assert sha(first) == cp['first_sha256'], f'Changed warm-up {first}'
        return load_checkpoint(first)
    return cp


def model_from(cp, device):
    model = Model(cp['dimension'], cp['classes']).to(device); model.load_state_dict(cp['state']); return model


def importance(model, x, y, kind, seed, device):
    model.eval(); pars = parameters(model); out = {k: torch.zeros_like(v) for k, v in pars.items()}
    rng = np.random.default_rng(seed); classes = np.unique(y); per = max(1, 256//len(classes))
    ids = np.concatenate([rng.permutation(np.flatnonzero(y == c))[:per] for c in classes])
    for idx in ids:
        logits = model(torch.from_numpy(x[idx:idx+1]).to(device))
        loss = nn.functional.cross_entropy(logits, torch.as_tensor(y[idx:idx+1], device=device)) if kind == 'ewc' else logits.square().sum()
        gradients = torch.autograd.grad(loss, list(pars.values()))
        for (k, _), g in zip(pars.items(), gradients): out[k].add_(g.square() if kind == 'ewc' else g.abs())
    for v in out.values(): v.div_(len(ids))
    return out, dict(rows=len(ids))


def logits_array(model, x, device):
    model.eval()
    with torch.inference_mode():
        return np.concatenate([model(torch.from_numpy(x[i:i+128]).to(device)).cpu().numpy() for i in range(0, len(x), 128)])


def evaluate(out, data, inp, stage, split, device):
    cp = load_checkpoint(Path(out)/f'stage{stage}.pt'); model = model_from(cp, device).eval()
    logits, labels, tasks = [], [], []
    for task, domain in enumerate(data.domains[:stage]):
        x, y = inp.load(domain, split, cp['mean'], cp['sd'])
        logits.append(logits_array(model, x, device)); labels.append(y); tasks.append(np.full(len(y), task))
    z, y, t = map(np.concatenate, (logits, labels, tasks)); z = z.astype(float); z -= z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True); pred = p.argmax(1)
    return [float(np.mean(pred[t == i] == y[t == i])) for i in range(stage)]


def train(out, root, data, inp, seed, method, coefficient=None, device='cuda'):
    """One trajectory. Under reduced labels the replay capacity stays at its first-domain value."""
    algorithm, use_replay = parse_method(method); out = Path(out)
    if (out/'receipt.json').exists() and read_json(out/'receipt.json')['complete']: return out
    out.mkdir(parents=True, exist_ok=True)
    warm = warmup(root, data, inp, seed, device); cp = load_checkpoint(warm/'stage1.pt'); model = model_from(cp, device)
    torch.save(dict(first_reference=str(warm/'stage1.pt'), first_sha256=sha(warm/'stage1.pt')), out/'stage1.pt')
    classes = data.classes; fixed = data.fraction is not None; D = inp.dimension
    current, y = inp.load(data.domains[0], 'train', cp['mean'], cp['sd'])
    cap = first_cap(len(y), classes)
    replay = retain([], current, y, np.arange(len(y)), 0, classes, cap//classes, seed) if use_replay else []
    anchor = theta(model); im = {k: torch.zeros_like(v) for k, v in anchor.items()}
    if algorithm in ('ewc', 'mas'): im, _ = importance(model, current, y, algorithm, seed, device)
    if algorithm == 'si': im = {k: v.to(device) for k, v in cp['si_importance'].items()}
    anchors = [(anchor, im)] if algorithm == 'ewc' else []
    teacher = copy.deepcopy(model).eval().requires_grad_(False) if algorithm == 'lwf' else None
    der = logits_array(model, replay[0][0], device) if algorithm == 'derpp' else None
    T_ = len(data.domains)
    receipt = dict(complete=False, family='classic', method=method, coefficient=coefficient, input=inp.kind, dataset=data.dataset,
                   domains=data.domains, seed=seed, fraction=data.fraction, first_checkpoint_sha256=sha(warm/'stage1.pt'),
                   stages=[dict(stage=1, current_rows=len(y), old_rows=0)], validation_matrix=[[None]*T_ for _ in range(T_)])
    receipt['validation_matrix'][0][:1] = evaluate(out, data, inp, 1, 'validation', device)
    write_json(out/'receipt.json', receipt)
    del current, y
    for task, domain in enumerate(data.domains[1:], 1):
        started = time.perf_counter(); current, y = inp.load(domain, 'train', cp['mean'], cp['sd'])
        cap = later_cap(cap, len(y), classes, fixed)
        old = replay[0] if use_replay else (np.empty((0, D), np.float32), np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64))
        if use_replay: assert len(old[1]) == cap
        mixed = use_replay and algorithm != 'agem'
        pool = np.concatenate([current, old[0]]) if mixed else current
        targets = np.concatenate([y, old[1]]) if mixed else y
        loader = DataLoader(TensorDataset(torch.from_numpy(pool), torch.from_numpy(targets), torch.arange(len(pool))),
                            batch_size=T['batch_size'], shuffle=True, generator=torch.Generator().manual_seed(seed+task*100))
        opt = optimizer(model); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T['later_epochs'])
        rng = np.random.default_rng(seed+task*100); si_start = theta(model)
        si_path = {k: torch.zeros_like(v) for k, v in si_start.items()}; history = []
        acquisition = np.empty((len(y), classes), np.float32); visited = np.zeros(len(y), bool)
        for epoch in range(T['later_epochs']):
            model.train(); curvis = oldvis = 0; losses = []; orders = []
            if algorithm == 'agem':
                ci = np.array_split(rng.permutation(len(y)), int(np.ceil(len(y)/T['batch_size'])))
                oi = np.array_split(rng.permutation(len(old[1])), len(ci))
                batches = [(torch.from_numpy(current[c]), torch.from_numpy(y[c]), torch.from_numpy(c), o) for c, o in zip(ci, oi)]
            else: batches = [(bx, by, ids, None) for bx, by, ids in loader]
            for bx, by, ids, refids in batches:
                bx, by = bx.to(device), by.to(device); idsnp = ids.numpy(); opt.zero_grad(set_to_none=True)
                logits = model(bx); cm = torch.as_tensor(idsnp < len(y), device=device); om = ~cm
                if epoch == 0 and bool(cm.any()):
                    acquisition[idsnp[cm.cpu().numpy()]] = logits[cm].detach().cpu().numpy(); visited[idsnp[cm.cpu().numpy()]] = True
                if algorithm == 'derpp':
                    value = nn.functional.cross_entropy(logits[cm], by[cm]) if bool(cm.any()) else logits.sum()*0
                    if bool(om.any()):
                        cached = torch.from_numpy(der[idsnp[om.cpu().numpy()]-len(y)]).to(device)
                        value = value+.5*nn.functional.mse_loss(logits[om], cached)+.5*nn.functional.cross_entropy(logits[om], by[om])
                else: value = nn.functional.cross_entropy(logits, by)
                if algorithm == 'ewc': value = value+coefficient*sum((fi[k]*(v-an[k]).square()).sum() for an, fi in anchors for k, v in parameters(model).items())
                if algorithm in ('si', 'mas'): value = value+coefficient*sum((im[k]*(v-anchor[k]).square()).sum() for k, v in parameters(model).items())
                if algorithm == 'lwf' and bool(cm.any()):
                    with torch.no_grad(): previous = teacher(bx[cm])
                    value = value+coefficient*4*nn.functional.kl_div(nn.functional.log_softmax(logits[cm]/2, 1), nn.functional.softmax(previous/2, 1), reduction='batchmean')
                assert torch.isfinite(value); value.backward()
                if algorithm == 'agem' and len(refids):
                    pars = list(parameters(model).values()); g = torch.cat([v.grad.reshape(-1).clone() for v in pars])
                    # Keep BN running statistics identical to the current-input update;
                    # the reference pass only supplies a gradient, not extra BN adaptation.
                    buffers = {k: v.clone() for k, v in model.named_buffers()}
                    reference = nn.functional.cross_entropy(model(torch.from_numpy(old[0][refids]).to(device)), torch.from_numpy(old[1][refids]).to(device))
                    rg = torch.autograd.grad(reference, pars); r = torch.cat([v.reshape(-1) for v in rg])
                    for k, v in model.named_buffers(): v.copy_(buffers[k])
                    dot = g@r
                    if dot < 0:
                        g -= dot/(r@r).clamp_min(1e-20)*r; offset = 0
                        for v in pars: v.grad.copy_(g[offset:offset+v.numel()].reshape_as(v)); offset += v.numel()
                        assert float(g@r) > -1e-4
                    oldvis += len(refids)
                before = theta(model) if algorithm == 'si' else None
                grad = {k: v.grad.detach().clone() for k, v in parameters(model).items()} if algorithm == 'si' else None
                opt.step()
                if algorithm == 'si':
                    for k, v in parameters(model).items(): si_path[k].add_(-grad[k]*(v.detach()-before[k]))
                curvis += int(cm.sum()); oldvis += int(om.sum()); losses.append(float(value.detach())); orders.append(idsnp)
            assert curvis == len(y) and oldvis == (cap if use_replay else 0)
            np.testing.assert_array_equal(np.sort(np.concatenate(orders)), np.arange(len(pool)))
            scheduler.step(); history.append(dict(epoch=epoch, loss=float(np.mean(losses)), order_sha256=digest(np.concatenate(orders))))
        assert visited.all(); model.eval()
        if algorithm in ('ewc', 'mas'):
            newim, _ = importance(model, current, y, algorithm, seed+task*100, device)
            if algorithm == 'ewc': anchors.append((theta(model), newim))
            else:
                for k in im: im[k] += newim[k]
        if algorithm == 'si':
            for k, v in parameters(model).items(): im[k] += (si_path[k]/((v.detach()-si_start[k]).square()+.1)).clamp_min(0)
        anchor = theta(model)
        if algorithm == 'lwf': teacher = copy.deepcopy(model).eval().requires_grad_(False)
        stage = task+1
        torch.save(dict(state={k: v.detach().cpu() for k, v in model.state_dict().items()}, mean=cp['mean'], sd=cp['sd'],
                        classes=classes, dimension=D), out/f'stage{stage}.pt')
        row = dict(stage=stage, current_rows=len(y), old_rows=len(old[1]), history=history, seconds=time.perf_counter()-started,
                   model_tensor_bytes=tensor_bytes(model.state_dict()))
        if use_replay:
            replay = retain(replay, current, y, np.arange(len(y)), task, classes, cap//classes, seed)
            if algorithm == 'derpp':
                lookup = {(int(t), int(i)): z for t, i, z in zip(old[2], old[3], der)}
                lookup.update({(task, i): z for i, z in enumerate(acquisition)})
                der = np.stack([lookup[int(t), int(i)] for t, i in zip(replay[0][2], replay[0][3])])
            rr = replay[0]; row.update(retained_rows=len(rr[1]), replay_bytes=sum(v.nbytes for v in rr),
                                       replay_identity_sha256=digest(np.column_stack(rr[2:])))
        receipt['validation_matrix'][task][:stage] = evaluate(out, data, inp, stage, 'validation', device)
        receipt['stages'].append(row); write_json(out/'receipt.json', receipt)
    receipt.update(complete=True, validation_metrics=metrics(receipt['validation_matrix']))
    write_json(out/'receipt.json', receipt); return out


def test(out, data, inp, device='cuda'):
    T_ = len(data.domains); matrix = [[None]*T_ for _ in range(T_)]
    for stage in range(1, T_+1): matrix[stage-1][:stage] = evaluate(out, data, inp, stage, 'test', device)
    return matrix
