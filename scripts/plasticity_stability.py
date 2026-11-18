"""Plasticity and stability of every method, computed from the stage-wise test-accuracy matrices of
results/reference_results_same_input.json (no new training; --results selects another records file, for example
the earlier results/reference_results.json whose baselines read the 9216-D Mantis block).

Matrix row k holds the test accuracies on the domains seen after update k.
  new   : mean over updates k >= 2 of the accuracy on the domain just learned (diagonal), a plasticity measure
  old   : mean accuracy on the earlier domains after the last update, a retention measure
  AFR   : average forgetting recorded in the results file (pp)
Main setting: seeds 42-44, original domain order, mean and sample SD over seeds.

    python scripts/plasticity_stability.py            # tables for the main setting and the ablation (seeds 42-46)
"""
import collections
import json
import pathlib
import statistics as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys
_path = sys.argv[sys.argv.index('--results') + 1] if '--results' in sys.argv else ROOT / 'results' / 'reference_results_same_input.json'
REF = json.loads(pathlib.Path(_path).read_text())['results']
DATASETS = ('3do', 'wimans', 'widar')


def metrics(run):
    m = run['matrix']
    t = len(m)
    new = st.mean(m[k][k] for k in range(1, t)) * 100
    old = st.mean(m[-1][k] for k in range(t - 1)) * 100
    return new, old, run['AFR'] * 100


def summarize(runs):
    vals = [metrics(r) for r in runs]
    return {name: (st.mean(v[i] for v in vals), st.stdev(v[i] for v in vals) if len(vals) > 1 else 0.0)
            for i, name in enumerate(('new', 'old', 'afr'))} | {'n': len(vals)}


def cells(rows):
    c = collections.defaultdict(list)
    for r in rows:
        c[(r['dataset'], r['method'])].append(r)
    return {k: summarize(v) for k, v in c.items()}


def main_cells():
    rows = [r for r in REF if (r['study'] == 'main' and r['family'] != 'joint') or (r['study'] == 'ablation' and r['method'] == 'main' and r['seed'] in (42, 43, 44))]
    return cells(rows)


def ablation_cells():
    return cells([r for r in REF if r['study'] == 'ablation'])


def balance_report(c):
    """WiMoT against all baselines: new-domain gap to the best baseline, and AFR/new of the baselines on each side."""
    rep = {}
    for d in DATASETS:
        w = c[(d, 'main')]
        base = {m: v for (dd, m), v in c.items() if dd == d and m != 'main'}
        best_new = max(v['new'][0] for v in base.values())
        rep[d] = dict(
            wimot=w, best_baseline_new=best_new, gap_to_best_new=best_new - w['new'][0],
            best_baseline_old=max((v['old'][0], m) for m, v in base.items()),
            higher_new=sorted((m, v['new'][0], v['afr'][0]) for m, v in base.items() if v['new'][0] >= w['new'][0]),
            lower_afr=sorted((m, v['new'][0], v['afr'][0]) for m, v in base.items() if v['afr'][0] <= w['afr'][0]))
    return rep


def joint_criterion(c, new_gap=2.5, afr_max=1.0):
    """Methods within `new_gap` pp of the best new-domain accuracy and with AFR <= afr_max on every stream."""
    names = {m for (_, m) in c}
    ok = []
    for m in sorted(names):
        good = True
        for d in DATASETS:
            best = max(v['new'][0] for (dd, mm), v in c.items() if dd == d)
            v = c[(d, m)]
            good &= v['new'][0] >= best - new_gap and v['afr'][0] <= afr_max
        if good:
            ok.append(m)
    return ok


if __name__ == '__main__':
    c = main_cells()
    for d in DATASETS:
        print(d)
        for (dd, m), v in sorted(c.items(), key=lambda a: -a[1]['new'][0]):
            if dd == d:
                print(f"  {m:15s} new {v['new'][0]:6.2f}±{v['new'][1]:.2f}  old {v['old'][0]:6.2f}±{v['old'][1]:.2f}  AFR {v['afr'][0]:6.2f}±{v['afr'][1]:.2f}")
    print('joint criterion (all three streams):', joint_criterion(c))
    print('ablation (seeds 42-46)')
    a = ablation_cells()
    for d in DATASETS:
        for m in ('main', 'no_supcon', 'no_hist', 'no_histstats', 'no_adapt'):
            v = a[(d, m)]
            print(f"  {d:7s} {m:13s} new {v['new'][0]:6.2f}  old {v['old'][0]:6.2f}  AFR {v['afr'][0]:6.2f}")
