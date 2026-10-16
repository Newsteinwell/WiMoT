"""Non-continual reference: the classic baseline network trained with cross-entropy on the union of all domains' TRAIN
partitions (union normalization, 100 epochs). It always runs on the CPU, as in the original study."""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..common import PROTOCOL, read_json, seed_all, write_json
from .classic import Model, optimizer

T = PROTOCOL['training']


def evaluate(model, data, inp, mean, sd, split):
    model.eval(); per = []
    for d in data.domains:
        x, y = inp.load(d, split, mean, sd)
        with torch.inference_mode(): per.append(float(np.mean(model(torch.from_numpy(x)).argmax(1).numpy() == y)))
    return per


def train(out, data, inp, seed):
    out = Path(out)
    if (out/'receipt.json').exists() and read_json(out/'receipt.json')['complete']: return out
    out.mkdir(parents=True, exist_ok=True)
    parts = [inp.load(d, 'train') for d in data.domains]
    x = np.concatenate([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
    mean, sd = inp.fit(x); x = ((x-mean)/sd).astype(np.float32)
    seed_all(seed); model = Model(inp.dimension, data.classes)
    opt = optimizer(model); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T['first_epochs'])
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=T['batch_size'],
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    for epoch in range(T['first_epochs']):
        model.train()
        for bx, by in loader:
            opt.zero_grad(set_to_none=True); nn.functional.cross_entropy(model(bx), by).backward(); opt.step()
        sch.step()
    torch.save(dict(state=model.state_dict(), mean=mean, sd=sd, dimension=inp.dimension, classes=data.classes), out/'model.pt')
    write_json(out/'receipt.json', dict(complete=True, family='joint', input=inp.kind, dataset=data.dataset, seed=seed, rows=len(y),
                                        validation_per_domain=evaluate(model, data, inp, mean, sd, 'validation')))
    return out


def test(out, data, inp):
    cp = torch.load(Path(out)/'model.pt', weights_only=False)
    model = Model(cp['dimension'], cp['classes']); model.load_state_dict(cp['state'])
    return evaluate(model, data, inp, cp['mean'], cp['sd'], 'test')
