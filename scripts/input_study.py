"""Input-family study: WiMoT and the 21 baselines on one input that does not use a foundation model.

  python scripts/input_study.py select-train --features STORE --out RUNS --tag amp_pca [--shard-name w0]
  python scripts/input_study.py select       --features STORE --out RUNS --tag amp_pca      (writes RUNS/selection.json)
  python scripts/input_study.py final        --features STORE --out RUNS --tag amp_pca [--shard-name w0]

STORE is a feature directory with one sub-directory per dataset (``STORE/<dataset>/<domain>_<split>.npz``): the cached MantisV2
store (the baselines then read only its 512-D amplitude descriptor, input kind ``amp512``; WiMoT runs as ``amp_full``/``amp_only``)
or a store written by ``scripts/make_derived_inputs.py``. The rest follows the protocol of the paper:

  select-train  seed 42, 3DO and WiMANS, every candidate coefficient of every baseline (``configs/protocol.json``), TRAIN and
                VALIDATION only (no TEST access);
  select        per baseline the candidate with the highest mean seed-42 validation AFA over 3DO and WiMANS (first maximum on
                ties); the locked values of ``configs/locks.json`` are never read for these methods;
  final         WiMoT (amp_full, amp_only), the 21 baselines with the selected coefficients and the non-continual joint-training
                reference on 3DO, WiMANS and Widar3.0, seeds 42-44; each trajectory is trained and then tested once.

Several worker processes may run the same phase: a job is claimed with an exclusive file under ``RUNS/claims`` (delete the
file to rerun a job), so the work is balanced dynamically. ``--limit`` stops a worker after that many jobs (smoke tests).
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wimot import run as runner  # noqa: E402
from wimot.common import PROTOCOL, Data, read_json, write_json  # noqa: E402

KIND = 'amp512'
SELECTION_SEED, SELECTION_DATASETS = PROTOCOL['selection']['seed'], PROTOCOL['selection']['datasets']
DATASETS, SEEDS = ('3do', 'wimans', 'widar'), (42, 43, 44)
VARIANTS = ('amp_full', 'amp_only')


def candidates():
    """{(family, method): [values]} of the 21 baselines; methods without a coefficient get [None]."""
    grids = PROTOCOL['classic']['coefficient_grids']; out = {}
    for m in PROTOCOL['classic']['methods']:
        out[('classic', m)] = list(grids[m.removesuffix('_er')]) if m.removesuffix('_er') in grids else [None]
    for family in ('wifi', 'recent'):
        for m in PROTOCOL[family]['methods']:
            out[(family, m)] = list(PROTOCOL[family]['candidates'][m])
    return out


def namespace(a, family, value):
    return SimpleNamespace(root=a.out, features=a.features, family=family, input=KIND, coefficient=value, order=None,
                           fraction=None, device=a.device)


def claim(a, name):
    path = Path(a.out)/'claims'/a.phase_tag/name; path.parent.mkdir(parents=True, exist_ok=True)
    try: os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)); return True
    except FileExistsError: return False


def purge(a, family, dataset, seed, method, value):
    """Deletes the checkpoints and stored moments of a finished trajectory (receipt.json and test.json stay)."""
    out = runner.run_dir(a.out, Data(Path(a.features)/dataset, dataset), family, method, KIND, seed, value)
    for pattern in ('stage*.pt', 'stage*_stats.npz', 'model.pt'):
        for f in Path(out).glob(pattern): f.unlink()


def log(a, row):
    path = Path(a.out)/'logs'/f'{a.phase_tag}_{a.shard_name}.jsonl'; path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as f: f.write(json.dumps(row)+'\n')


def execute(a, jobs):
    """jobs: list of (name, family, dataset, seed, method, value, test). Heavy jobs are listed first."""
    done = 0
    for name, family, dataset, seed, method, value, test in jobs:
        if a.limit is not None and done >= a.limit: break
        if not claim(a, name): continue
        torch.manual_seed(1000+seed)    # ConSense/EdgeCL dropout draws from the global generator: seed it per job
        started = time.monotonic(); ns = namespace(a, family, value)
        try:
            runner.one(ns, dataset, seed, method, False)
            if test: runner.one(ns, dataset, seed, method, True)
            if test and a.cleanup: purge(a, family, dataset, seed, method, value)
        except Exception as error:
            log(a, dict(job=name, error=repr(error), traceback=traceback.format_exc()[-1500:])); print('FAILED', name, repr(error), flush=True); traceback.print_exc(); continue
        log(a, dict(job=name, seconds=round(time.monotonic()-started, 1))); done += 1; print('DONE', name, flush=True)


COST = {'cn_dpm_aligned': 0, 'duct': 1, 'duct_er': 1, 'dualcp': 2, 'dualcp_er': 2, 'vanillacp': 2, 'vanillacp_er': 2,
        'edgecl': 3, 'consense': 3}


def select_jobs():
    jobs = []
    for (family, method), values in candidates().items():
        if len(values) == 1: continue
        for value in values:
            for dataset in SELECTION_DATASETS:
                jobs.append((f'{family}_{method}_{value:g}_{dataset}', family, dataset, SELECTION_SEED, method, value, False))
    return sorted(jobs, key=lambda j: COST.get(j[4], 5))


def validation_afa(a, family, method, value, dataset):
    data = Data(Path(a.features)/dataset, dataset)
    out = runner.run_dir(a.out, data, family, method, KIND, SELECTION_SEED, value)
    receipt = read_json(out/'receipt.json')
    if 'validation_metrics' not in receipt: raise FileNotFoundError(out)    # the trajectory failed after its receipt was opened
    return receipt['validation_metrics']['AFA']


def select(a):
    result = {}
    for (family, method), values in candidates().items():
        if len(values) == 1:
            result[f'{family}/{method}'] = dict(coefficient=values[0], scores=None, note='single candidate'); continue
        try:
            scores = [sum(validation_afa(a, family, method, v, d) for d in SELECTION_DATASETS)/len(SELECTION_DATASETS) for v in values]
        except (FileNotFoundError, KeyError):    # a candidate could not be trained (no complete receipt) on this input (see logs/*.jsonl): no selection possible
            errors = [json.loads(line) for f in (Path(a.out)/'logs').glob('select-train_*.jsonl') for line in open(f)]
            reasons = sorted({e['error'] for e in errors if 'error' in e and e['job'].startswith(f'{family}_{method}_')})
            result[f'{family}/{method}'] = dict(coefficient=None, candidates=values, scores=None, note='not runnable on this input: '+'; '.join(reasons))
            print('UNAVAILABLE', a.tag, family, method, reasons, flush=True); continue
        best = values[max(range(len(values)), key=lambda i: (scores[i], -i))]
        result[f'{family}/{method}'] = dict(coefficient=best, candidates=values, scores=scores)
        print('SELECTED', a.tag, family, method, 'scores', [round(s, 4) for s in scores], 'selected', best, flush=True)
    write_json(Path(a.out)/'selection.json', dict(tag=a.tag, rule=PROTOCOL['selection']['rule'], selected=result))
    if a.cleanup:    # candidates that were not selected are never used again
        for (family, method), values in candidates().items():
            for v in values:
                if len(values) > 1 and result[f'{family}/{method}']['coefficient'] is not None and v != result[f'{family}/{method}']['coefficient']:
                    for d in SELECTION_DATASETS: purge(a, family, d, SELECTION_SEED, method, v)


def final_jobs(a):
    chosen = read_json(Path(a.out)/'selection.json')['selected']; jobs = []
    for seed in SEEDS:
        for dataset in DATASETS:
            for variant in VARIANTS: jobs.append((f'method_{variant}_{dataset}_{seed}', 'method', dataset, seed, variant, None, True))
            jobs.append((f'joint_{dataset}_{seed}', 'joint', dataset, seed, 'joint', None, True))
            for (family, method) in candidates():
                value = chosen[f'{family}/{method}']['coefficient']
                if value is None and chosen[f'{family}/{method}'].get('note', '').startswith('not runnable'): continue
                jobs.append((f'{family}_{method}_{dataset}_{seed}', family, dataset, seed, method, value, True))
    return sorted(jobs, key=lambda j: COST.get(j[4], 5))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('phase', choices=('select-train', 'select', 'final')); p.add_argument('--features', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--tag', required=True)
    p.add_argument('--shard-name', default='w0'); p.add_argument('--threads', type=int, default=2)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu'); p.add_argument('--limit', type=int)
    p.add_argument('--cleanup', action='store_true', help='delete checkpoints of finished trajectories (final) and of '
                   'non-selected candidates (select) to save disk')
    p.add_argument('--datasets', help='restrict the final phase to these comma-separated datasets')
    a = p.parse_args(argv); a.phase_tag = a.phase; a.global_seed = None; runner.setup(a)
    if a.phase == 'select': return select(a)
    jobs = select_jobs() if a.phase == 'select-train' else final_jobs(a)
    if a.datasets: jobs = [j for j in jobs if j[2] in a.datasets.split(',')]
    execute(a, jobs)


if __name__ == '__main__':
    main()
