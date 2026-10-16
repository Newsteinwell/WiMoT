"""Wi-Fi continual-learning baselines ConSense (AAAI 2025) and EdgeCL (IEEE TWC 2026) on frozen features.

Initialization and mini-batch order are seeded, but both networks draw dropout masks from the global torch generator,
which the original runs never seeded (PyTorch seeds it non-deterministically per process). These trajectories are
therefore reproducible only in distribution; run.py --global-seed fixes the generator for deterministic reruns.
"""
import time
from pathlib import Path

import numpy as np
import torch

from ..common import PROTOCOL, first_cap, metrics, read_json, write_json
from .consense import ConSense
from .edgecl import EdgeCL, EdgeCLWidar
from .inputs import Input

W = PROTOCOL['wifi']


def build(method, data, seed, value, device, input_kind='m9216'):
    c = W['common']
    if method == 'consense':
        return ConSense(seed, device, value, data.classes, c['first_epochs'], c['epochs'], c['batch_size'], c['lr'],
                        **{k: W['consense'][k] for k in ('heads', 'dropout', 'bottleneck')}, input_kind=input_kind)
    cls = EdgeCLWidar if data.dataset.startswith('widar') else EdgeCL
    return cls(seed, device, value, data.classes, c['first_epochs'], c['epochs'], c['batch_size'], c['lr'],
               **{k: W['edgecl'][k] for k in ('beta', 'eta', 'hidden', 'latent', 'heads', 'layers', 'dropout')},
               input_kind=input_kind)


def evaluate(model, data, inp, mean, sd, stage, split):
    z, y, t = [], [], []
    for task, domain in enumerate(data.domains[:stage]):
        x, labels = inp.load(domain, split, mean, sd)
        z.append(model.predict_logits(x)); y.append(labels); t.append(np.full(len(labels), task))
    z, y, t = map(np.concatenate, (z, y, t)); pred = z.argmax(1)
    return [float(np.mean(pred[t == i] == y[t == i])) for i in range(stage)]


def train(out, data, inp, seed, method, value, device='cuda'):
    """Replay capacity: first-domain 10% rule, shrinking so that old/(old+new) <= 1/10, fixed under reduced labels."""
    out = Path(out)
    if (out/'receipt.json').exists() and read_json(out/'receipt.json')['complete']: return out
    out.mkdir(parents=True, exist_ok=True)
    first, y = inp.load(data.domains[0], 'train'); mean, sd = inp.fit(first)
    classes = data.classes; cap = first_cap(len(y), classes); del first
    fixed = data.fraction is not None
    model = build(method, data, seed, value, device, inp.kind); replay = W['replay'][method]
    T_ = len(data.domains)
    receipt = dict(complete=False, family='wifi', method=method, coefficient=value, input=inp.kind, dataset=data.dataset,
                   domains=data.domains, seed=seed, fraction=data.fraction, replay=replay, stages=[],
                   validation_matrix=[[None]*T_ for _ in range(T_)])
    for task, domain in enumerate(data.domains):
        x, labels = inp.load(domain, 'train', mean, sd); n = len(labels)
        if task and not fixed: cap = min(cap, n//9//classes*classes); assert 9*cap <= n
        started = time.monotonic(); model.update(x, labels, cap)
        if replay: model.finish_stage(cap)
        old = model.er_epochs[0]['old_visits']
        assert all(h['current_visits'] == n and h['old_visits'] == old for h in model.er_epochs)
        assert old <= cap if task else old == 0
        if not replay: assert old == 0
        torch.save(dict(model=model, mean=mean, sd=sd), out/f'stage{task+1}.pt')
        receipt['validation_matrix'][task][:task+1] = evaluate(model, data, inp, mean, sd, task+1, 'validation')
        receipt['stages'].append(dict(stage=task+1, current_rows=n, old_rows=old, cap=cap, seconds=time.monotonic()-started))
        write_json(out/'receipt.json', receipt)
    receipt.update(complete=True, validation_metrics=metrics(receipt['validation_matrix']))
    write_json(out/'receipt.json', receipt); return out


def test(out, data, inp, device='cuda'):
    T_ = len(data.domains); matrix = [[None]*T_ for _ in range(T_)]
    for stage in range(1, T_+1):
        cp = torch.load(Path(out)/f'stage{stage}.pt', map_location=device, weights_only=False)
        matrix[stage-1][:stage] = evaluate(cp['model'], data, inp, cp['mean'], cp['sd'], stage, 'test'); del cp
    return matrix
