"""Runs every trajectory of one paper table from configs/jobs/<study>.tsv: all training first, then all tests.

  python -m wimot.study main --features FEATURES --out RUNS [--shard 0/4] [--phase train|test|all]

FEATURES holds one directory per encoder and dataset, FEATURES/<encoder>/<dataset>/<domain>_<split>.npz (encoder 'mantis'
for every table except the backbone study). Results go to RUNS/<study>/<encoder>/...; '--shard i/N' runs every N-th line
starting at line i, so that N processes (or machines sharing RUNS) can split a table. Run '--phase test' after all
shards have trained, then 'python scripts/build_tables.py --runs RUNS'. '--dataset 3do --method si_er,ft --seed 42' keeps
only the matching jobs (a quick subset; the default runs every job of the table).
"""
import argparse
import csv
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from . import run as runner
from .common import REPO


def job_file(study):
    """configs/jobs/<study>.tsv, or configs/jobs_sameall/<name>.tsv for 'sameall_<name>' (baselines with the 9728-D input)."""
    if study.startswith('sameall_'): return REPO/'configs'/'jobs_sameall'/f'{study[8:]}.tsv'
    return REPO/'configs'/'jobs'/f'{study}.tsv'


def jobs(study):
    with open(job_file(study)) as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    value = lambda v, cast: None if v == '-' else cast(v)
    return [dict(family=r['family'], encoder=r['encoder'], input=r['input'], dataset=r['dataset'], order=value(r['order'], str),
                 fraction=value(r['fraction'], float), seed=int(r['seed']), method=r['method'],
                 coefficient=value(r['coefficient'], float)) for r in rows]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('study', choices=sorted([q.stem for q in (REPO/'configs'/'jobs').glob('*.tsv')]
                                          +['sameall_'+q.stem for q in (REPO/'configs'/'jobs_sameall').glob('*.tsv')]))
    p.add_argument('--features', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--shard', default='0/1'); p.add_argument('--phase', choices=('train', 'test', 'all'), default='all')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='cuda if available, else cpu'); p.add_argument('--threads', type=int, default=2)
    p.add_argument('--global-seed', type=int); p.add_argument('--reseed-per-job', action='store_true')
    p.add_argument('--cleanup', action='store_true', help='Train and test every trajectory in turn and delete its checkpoints '
                   '(*.pt) afterwards; receipt.json and test.json stay')
    for name in ('dataset', 'method', 'seed', 'order', 'family'):
        p.add_argument(f'--{name}', help=f'Keep only jobs with these comma-separated {name} values (default: all)')
    a = p.parse_args(argv)
    index, count = map(int, a.shard.split('/'))
    selected = jobs(a.study)
    for name in ('dataset', 'method', 'seed', 'order', 'family'):
        if getattr(a, name):
            wanted = getattr(a, name).split(',')
            selected = [j for j in selected if str(j[name]) in wanted]
    selected = selected[index::count]
    if not selected: raise SystemExit('No job of the study matches the filters')
    runner.setup(a)
    phases = ('train', 'test') if a.phase == 'all' else (a.phase,)
    namespace = lambda job: SimpleNamespace(root=a.out/a.study/job['encoder'], features=a.features/job['encoder'], family=job['family'],
                                            input=job['input'], coefficient=job['coefficient'], order=job['order'],
                                            fraction=job['fraction'], device=a.device, cleanup=a.cleanup)
    if a.cleanup:
        started = time.monotonic()
        for k, job in enumerate(selected):
            for phase in phases: runner.one(namespace(job), job['dataset'], job['seed'], job['method'], phase == 'test')
            print('PROGRESS', a.study, 'all', k+1, len(selected), round(time.monotonic()-started), flush=True)
        return
    for phase in phases:
        started = time.monotonic()
        for k, job in enumerate(selected):
            runner.one(namespace(job), job['dataset'], job['seed'], job['method'], phase == 'test')
            print('PROGRESS', a.study, phase, k+1, len(selected), round(time.monotonic()-started), flush=True)


if __name__ == '__main__':
    main()
