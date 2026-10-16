"""Recent domain-incremental methods on frozen features: DUCT (2025), DualCP and VanillaCP (2025), each native and with
the common replay cache, and CN-DPM (2020) whose replay is the common cache (history-aligned)."""
import time
from pathlib import Path

import numpy as np
import torch

from ..common import PROTOCOL, empty_cache, first_cap, metrics, read_json, retain, write_json
from .cn_dpm import CNDPM, CNConfig
from .dualcp import NativeRoutingDualCP
from .duct import DUCT, DUCTConfig, WarmStartedDUCT
from .matched_er import HistoryAlignedCNDPM, ReplayDualCP, ReplayDUCT

R = PROTOCOL['recent']
ALIGNED = ('cn_dpm_aligned', 'duct_er', 'dualcp_er', 'vanillacp_er')
CP = ('dualcp', 'vanillacp', 'dualcp_er', 'vanillacp_er')


def build(method, data, seed, value, device, first, labels, mean, sd, dimension):
    C = data.classes
    if method == 'duct': return WarmStartedDUCT(dimension, seed, device, DUCTConfig(**R['duct']), value, classes=C)
    if method == 'duct_er': return ReplayDUCT(dimension, seed, device, DUCTConfig(**R['duct']), value, C)
    if method == 'cn_dpm_aligned':
        return HistoryAlignedCNDPM(dimension, seed, device, CNConfig(stm_capacity=first_cap(len(labels), C)), value, C)
    m = (ReplayDualCP if method.endswith('_er') else NativeRoutingDualCP)(seed, device, threshold=value, input_dim=dimension,
                                                                          classes=C, **R['dualcp'])
    m.routing_offset = mean/sd
    if method.startswith('dualcp'):
        m.initialize(((first-mean)/sd).astype(np.float32), labels)
        assert 1 < len(m.groups) < C and any(len(g) > 1 for g in m.groups), 'Not a dual hierarchy'
    return m


def prediction(model, x):
    if isinstance(model, CNDPM): return model.predict_proba(x).astype(float).argmax(1)
    if isinstance(model, DUCT): return model.predict_logits(x).astype(float).argmax(1) % model.classes
    return model.predict(x)


def at_stage(model, stage):
    return model.at_stage(stage) if isinstance(model, NativeRoutingDualCP) else model


def evaluate(model, data, inp, mean, sd, stage, split):
    p, y, t = [], [], []
    for task, domain in enumerate(data.domains[:stage]):
        x, labels = inp.load(domain, split, mean, sd)
        p.append(prediction(model, x)); y.append(labels); t.append(np.full(len(labels), task))
    p, y, t = map(np.concatenate, (p, y, t))
    return [float(np.mean(p[t == i] == y[t == i])) for i in range(stage)]


def train(out, data, inp, seed, method, value, device='cuda'):
    """Aligned methods replay the class-priority reservoir of every replay baseline (capacity rule of classic.train)."""
    out = Path(out)
    if (out/'receipt.json').exists() and read_json(out/'receipt.json')['complete']: return out
    out.mkdir(parents=True, exist_ok=True)
    C = data.classes; D = inp.dimension
    first, labels = inp.load(data.domains[0], 'train'); mean, sd = inp.fit(first)
    model = build(method, data, seed, value, device, first, labels, mean, sd, D); cap = first_cap(len(labels), C); del first
    aligned = method in ALIGNED; T_ = len(data.domains)
    receipt = dict(complete=False, family='recent', method=method, coefficient=value, input=inp.kind, dataset=data.dataset,
                   domains=data.domains, seed=seed, fraction=data.fraction, stages=[], validation_matrix=[[None]*T_ for _ in range(T_)])
    memory = []
    for task, domain in enumerate(data.domains):
        x, y = inp.load(domain, 'train', mean, sd); n = len(y); started = time.monotonic(); row = dict(stage=task+1, current_rows=n)
        if aligned:
            if task: cap = min(cap, n//9//C*C); assert 9*cap <= n
            old = memory[0] if task else empty_cache(D)
            assert len(old[1]) == (cap if task else 0)
            model.update_aligned(x, y, old)
            if task: assert len(model.er_epochs) == 50 and all(e['current_visits'] == n and e['old_visits'] == cap for e in model.er_epochs)
            memory = retain(memory, x, y, np.arange(n), task, C, cap//C, seed)
            row.update(old_rows=len(old[1]), retained_rows=len(memory[0][1]))
        else:
            model.update(x, y); row.update(old_rows=0)
        cp = out/('latest.pt' if method in CP else f'stage{task+1}.pt')
        torch.save(dict(model=model, mean=mean, sd=sd, classes=C), cp)
        receipt['validation_matrix'][task][:task+1] = evaluate(model, data, inp, mean, sd, task+1, 'validation')
        row.update(seconds=time.monotonic()-started, memory_bytes=getattr(model, 'memory_bytes', None), checkpoint=cp.name)
        receipt['stages'].append(row); write_json(out/'receipt.json', receipt)
    receipt.update(complete=True, validation_metrics=metrics(receipt['validation_matrix'])); write_json(out/'receipt.json', receipt)
    return out


def test(out, data, inp, method, device='cuda'):
    # Pickled CN-DPM models hold a device generator, so checkpoints are restored on the device that trained them.
    T_ = len(data.domains); matrix = [[None]*T_ for _ in range(T_)]; final = None
    for stage in range(1, T_+1):
        if method in CP:
            final = final or torch.load(Path(out)/'latest.pt', weights_only=False); cp = final; model = at_stage(cp['model'], stage)
        else:
            cp = torch.load(Path(out)/f'stage{stage}.pt', weights_only=False); model = cp['model']
        matrix[stage-1][:stage] = evaluate(model, data, inp, cp['mean'], cp['sd'], stage, 'test')
    return matrix
