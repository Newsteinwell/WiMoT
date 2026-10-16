"""Summarizes reproduced trajectories per paper table and compares them with results/reference_results.json.

  python scripts/build_tables.py --runs RUNS [--study main ...]     reproduced vs reference
  python scripts/build_tables.py --reference-only [--study main]     the paper numbers from the reference records

For every group (encoder, input, family, method, order, label fraction) and dataset it prints mean +- sample SD of AFA and
AFR in percent over seeds, for the reproduction (R) and the reference (P), and the number of trajectories whose full
accuracy matrix is identical to the reference ('bitwise').
"""
import argparse
import collections
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STUDIES = ('main', 'ablation', 'extra_seeds', 'orders', 'widarext', 'fewlabel', 'sameinput', 'backbone', 'memory')


def key(r, study, encoder):
    method_family = r['family'] == 'method'
    return (study, encoder, r['family'], r['dataset'], r.get('order') or None, r.get('fraction'), int(r['seed']), r['method'],
            None if method_family else r.get('input'), None if method_family or r.get('coefficient') is None else float(r['coefficient']))


def reference():
    out = {}
    for r in json.loads((REPO/'results'/'reference_results.json').read_text())['results']:
        out[key(r, r['study'], r['encoder'])] = r
        if r['study'] == 'ablation' and r['method'] == 'main' and r['seed'] <= 44: out[key(r, 'main', r['encoder'])] = r
    return out


def reproduced(runs):
    out = {}
    for path in Path(runs).glob('*/*/**/test.json'):
        study, encoder = path.relative_to(runs).parts[:2]
        r = json.loads(path.read_text())
        out[key(r, study, encoder)] = dict(r, AFA=r['metrics']['AFA'], AFR=r['metrics'].get('AFR'))
    return out


def fmt(values):
    values = [100*v for v in values if v is not None]
    if not values: return '-'
    return f'{statistics.mean(values):.2f}' + (f'±{statistics.stdev(values):.2f}' if len(values) > 1 else '')


def table(study, ref, rep):
    groups = collections.defaultdict(lambda: collections.defaultdict(dict))
    for k in set(ref) | set(rep):
        if k[0] != study: continue
        _, encoder, family, dataset, order, fraction, seed, method, inp, coefficient = k
        groups[(encoder, inp or '', family, method, order or '', fraction or '')][dataset][seed] = k
    if not groups: return
    datasets = sorted({d for g in groups.values() for d in g})
    print(f'\n== {study} ==')
    head = f"{'encoder':13} {'input':8} {'family':8} {'method':16} {'order':10} {'frac':5}"
    print(head + ''.join(f' | {d:^38}' for d in datasets))
    totals = [0, 0, 0]
    for g in sorted(groups, key=str):
        cells = []
        for d in datasets:
            seeds = groups[g].get(d, {})
            if not seeds: cells.append(' '*38); continue
            P = [ref[k] for k in seeds.values() if k in ref]; R = [rep[k] for k in seeds.values() if k in rep]
            same = sum(1 for k in seeds.values() if k in ref and k in rep and rep[k]['matrix'] == ref[k]['matrix'])
            totals[0] += len(R); totals[1] += len(P); totals[2] += same
            text = f"P {fmt([r['AFA'] for r in P])}/{fmt([r['AFR'] for r in P])}"
            if R: text += f" R {fmt([r['AFA'] for r in R])}/{fmt([r['AFR'] for r in R])} ={same}"
            cells.append(f'{text:38}')
        print(f'{g[0]:13} {g[1]:8} {g[2]:8} {g[3]:16} {g[4]:10} {str(g[5]):5}' + ''.join(f' | {c}' for c in cells))
    print(f'reproduced {totals[0]} of {totals[1]} reference trajectories; bitwise identical matrices: {totals[2]}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', type=Path); p.add_argument('--reference-only', action='store_true')
    p.add_argument('--study', nargs='*', default=list(STUDIES))
    a = p.parse_args()
    if not a.runs and not a.reference_only: p.error('--runs or --reference-only')
    ref = reference(); rep = {} if a.reference_only else reproduced(a.runs)
    print('AFA/AFR in percent, mean±SD over seeds; P = paper (reference), R = this reproduction, =n bitwise-identical seeds')
    for study in a.study: table(study, ref, rep)


if __name__ == '__main__':
    main()
