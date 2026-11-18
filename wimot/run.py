"""Command-line entry point: train or test trajectories, and summarize a study root.

  python -m wimot.run train --features F --root R --dataset D --seed S --family method --method main
  python -m wimot.run train ... --family classic --method si_er [--coefficient 10] [--input cat9728]
  python -m wimot.run train ... --family wifi --method edgecl --seed 42,43,44      (one process, in this order)
  python -m wimot.run test  ...same arguments...                                  (writes test.json)
  python -m wimot.run summarize --root R

Families and methods:
  method   main, no_supcon, no_hist, no_amp, no_pca, no_adapt, no_histstats, replay, triu, rank8, rank32, rank128,
           amp_only, amp_full, no_transport, no_transport_no_hist (stored moments of earlier domains are not transported
           through the updated adapter; transport-ladder ablation)
           main_sc<w>_h<w>[_fsc<w>]  loss-weight setting (sensitivity analysis): SupCon weight <sc> and L_hist weight <h> in the
                                    later domains, optionally SupCon weight <fsc> in the first domain (defaults: 0.5, 1, 0.5)
  classic  ft, er, ewc, ewc_er, si, si_er, mas, mas_er, lwf, lwf_er, derpp, agem
  wifi     consense, edgecl
  recent   duct, dualcp, vanillacp, cn_dpm_aligned, duct_er, dualcp_er, vanillacp_er
  joint    joint (non-continual reference, CPU)
The coefficient defaults to the locked value of configs/locks.json (per input for the same-input study). --order permutes
the domains by a digit string such as 210; --fraction keeps that share of the labels in every later domain. Several
datasets, seeds or methods may be given as comma-separated lists; they are processed in the order given, within one process.
"""
import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import torch

from .common import LOCKS, STREAMS, Data, metrics, read_json, summarize_rows, write_json

FAMILIES = ('method', 'classic', 'wifi', 'recent', 'joint')


def coefficient(family, method, kind, value):
    if value is not None: return value
    if family in ('method', 'joint'): return None
    if kind != 'm9216' and method in LOCKS['same_input'] and kind in LOCKS['same_input'][method]:
        return LOCKS['same_input'][method][kind]
    entry = LOCKS.get(family, {}).get(method)
    return entry['coefficient'] if entry else None


def run_dir(root, data, family, method, kind, seed, value):
    if family == 'method':
        from .method.train import run_dir as method_dir
        return method_dir(root, data, seed, method)
    name = method if value is None else f'{method}_c{value:g}'
    return Path(root)/family/data.tag/f'seed{seed}'/kind/name


RESEED = None


def reseed():
    torch.manual_seed(RESEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(RESEED)


def setup(a):
    global RESEED
    torch.set_num_threads(a.threads); torch.backends.cuda.matmul.allow_tf32 = False
    if a.global_seed is not None: torch.manual_seed(a.global_seed)
    # --reseed-per-job: restore the global generator before every trajectory, so that a trajectory does not depend on the
    # jobs that ran before it in the same process (ConSense/EdgeCL dropout masks).
    RESEED = a.global_seed if getattr(a, 'reseed_per_job', False) else None


def one(a, dataset, seed, method, testing):
    if RESEED is not None: reseed()
    features = Path(a.features)/dataset if (Path(a.features)/dataset).is_dir() else Path(a.features)
    data = Data(features, dataset, a.order, a.fraction, seed)
    family = a.family; kind = 'cat9728' if family == 'method' else a.input
    value = coefficient(family, method, kind, a.coefficient)
    out = run_dir(a.root, data, family, method, kind, seed, value); started = time.monotonic()
    if getattr(a, 'cleanup', False) and (out/'test.json').exists():   # finished trajectory whose checkpoints were deleted
        print('SKIP', family, method, kind, value, data.tag, seed, flush=True); return
    if family == 'method':
        from .method import train as mt
        if testing: result = mt.test(out, data, a.device)
        else: mt.train(a.root, data, seed, method, a.device)
        if not testing: print('TRAINED', family, method, data.tag, seed, round(time.monotonic()-started), flush=True)
        else: print('TEST', family, method, data.tag, seed, round(result['metrics']['AFA'], 4), flush=True)
        if testing and getattr(a, 'cleanup', False):
            for f in list(out.glob('stage*.pt'))+list(out.glob('stage*_stats.npz')): f.unlink()   # keep receipt.json and test.json
        return
    from .baselines.inputs import Input
    inp = Input(kind, data, seed)
    if family == 'classic':
        from .baselines import classic as m
        run = lambda: m.train(out, a.root, data, inp, seed, method, value, a.device); check = lambda: m.test(out, data, inp, a.device)
    elif family == 'wifi':
        from .baselines import wifi as m
        run = lambda: m.train(out, data, inp, seed, method, value, a.device); check = lambda: m.test(out, data, inp, a.device)
    elif family == 'recent':
        from .baselines import recent as m
        run = lambda: m.train(out, data, inp, seed, method, value, a.device); check = lambda: m.test(out, data, inp, method, a.device)
    else:
        from .baselines import joint as m
        run = lambda: m.train(out, data, inp, seed); check = lambda: m.test(out, data, inp)
    if not testing:
        run(); print('TRAINED', family, method, kind, value, data.tag, seed, round(time.monotonic()-started), flush=True); return
    assert read_json(out/'receipt.json')['complete'], f'Incomplete trajectory {out}'
    result = dict(family=family, dataset=dataset, order=data.order, domains=data.domains, seed=seed, method=method,
                  coefficient=value, input=kind, fraction=data.fraction)
    if family == 'joint':
        per = check(); result.update(per_domain=per, metrics=dict(AFA=float(np.mean(per)), AFR=None))
    else:
        matrix = check(); result.update(matrix=matrix, metrics=metrics(matrix))
    write_json(out/'test.json', result)
    print('TEST', family, method, kind, value, data.tag, seed, round(result['metrics']['AFA'], 4), flush=True)
    if getattr(a, 'cleanup', False):
        for f in out.glob('*.pt'): f.unlink()   # keep receipt.json and test.json only (disk budget of long streams)


def summarize(a):
    rows = [read_json(p) for p in sorted(Path(a.root).rglob('test.json'))]
    key = lambda r: '|'.join(str(r.get(k)) for k in ('dataset', 'order', 'fraction', 'input', 'family', 'method', 'coefficient'))
    write_json(Path(a.root)/'summary.json', dict(groups=summarize_rows(rows, key), results=rows))
    print('SUMMARY', a.root, len(rows), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=('train', 'test', 'summarize'))
    p.add_argument('--root', type=Path, required=True, help='Output directory of one study')
    p.add_argument('--features', type=Path, help='Feature directory <domain>_<split>.npz, or a directory holding one such '
                                                 'directory per dataset (see wimot.features.extract)')
    p.add_argument('--dataset', help=f'Comma-separated streams: {", ".join(STREAMS)}')
    p.add_argument('--seed', help='Comma-separated seeds')
    p.add_argument('--family', choices=FAMILIES); p.add_argument('--method', help='Comma-separated methods of the family')
    p.add_argument('--coefficient', type=float); p.add_argument('--input', default='m9216',
                   choices=('m9216', 'cat9728', 'stem9344', 'amp512'))
    p.add_argument('--order'); p.add_argument('--fraction', type=float)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='cuda if available, else cpu'); p.add_argument('--threads', type=int, default=2)
    p.add_argument('--global-seed', type=int, help='Seed the global torch generators once at start-up (ConSense/EdgeCL '
                                                   'dropout); the paper runs left them unseeded')
    p.add_argument('--reseed-per-job', action='store_true', help='With --global-seed: reset the global generators before '
                   'every trajectory instead of once at start-up')
    p.add_argument('--jobs', type=Path, help='File of lines "dataset seed method [coefficient]" run in this order in one '
                                            'process (overrides --dataset/--seed/--method/--coefficient)')
    a = p.parse_args(argv)
    if a.command == 'summarize': return summarize(a)
    setup(a)
    if a.jobs:
        for line in a.jobs.read_text().splitlines():
            if not line.strip() or line.startswith('#'): continue
            dataset, seed, method, *rest = line.split()
            a.coefficient = float(rest[0]) if rest and rest[0] != '-' else None
            one(a, dataset, int(seed), method, a.command == 'test')
        return
    for dataset, seed, method in itertools.product(a.dataset.split(','), map(int, a.seed.split(',')), a.method.split(',')):
        one(a, dataset, seed, method, a.command == 'test')


if __name__ == '__main__':
    main()
