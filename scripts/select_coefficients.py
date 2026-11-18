"""Recomputes the locked baseline coefficients of configs/locks.json (selection rule of configs/protocol.json).

  python scripts/select_coefficients.py --features FEATURES/mantis --out RUNS/selection [--family classic,wifi,recent]

Every candidate of every method is trained with seed 42 on 3DO and WiMANS (TRAIN/VALIDATION only, no TEST access); the
selected value maximizes the mean validation AFA over the two streams (first maximum on ties). Methods with a single
candidate are reported for completeness. ConSense and EdgeCL use unseeded dropout (see wimot/baselines/wifi.py), so
their validation scores, and possibly the selected value, can vary between reruns; ``--global-seed N --reseed-per-job``
fixes them.

Same-input selection (every baseline reads the 9728-D input): ``--input cat9728``. The runs can be split over processes
(``--shard i/N``, training only) and the selection is then read from the stored receipts (``--aggregate``), optionally
written as a locks file (``--write-locks configs/locks_sameinput_all.json``) that ``WIMOT_LOCKS`` selects at run time.

A DualCP candidate whose class-mean similarity graph does not split the classes into two or more groups, at least one
holding several classes, is not a dual hierarchy (``Not a dual hierarchy``); it is recorded as ineligible (score -inf) on
that stream. ``--extra-candidates dualcp=0.85,dualcp_er=0.85`` appends candidates to the grid of protocol.json (used when
every protocol candidate is ineligible for the 9728-D input).
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wimot import run as runner  # noqa: E402
from wimot.common import LOCKS, PROTOCOL, REPO, Data, read_json  # noqa: E402


EXTRA = {}


def candidates(family):
    if family == 'classic':
        grids = PROTOCOL['classic']['coefficient_grids']
        grids = {m: grids[m.removesuffix('_er')] for m in PROTOCOL['classic']['methods'] if m.removesuffix('_er') in grids}
    else: grids = dict(PROTOCOL[family]['candidates'])
    return {m: grid+[v for v in EXTRA.get(m, []) if v not in grid] for m, grid in grids.items()}


def tasks(families):
    S = PROTOCOL['selection']
    return [(family, method, value, dataset) for family in families for method, grid in candidates(family).items()
            for value in grid for dataset in S['datasets']]


def run_one(a, family, method, value, dataset):
    ns = SimpleNamespace(root=a.out, features=a.features, family=family, input=a.input, coefficient=value,
                         order=None, fraction=None, device=a.device)
    try: runner.one(ns, dataset, PROTOCOL['selection']['seed'], method, False)
    except AssertionError as e:
        if 'dual hierarchy' not in str(e): raise
        out = trajectory_dir(a, family, method, value, dataset); out.mkdir(parents=True, exist_ok=True)
        (out/'ineligible.json').write_text(json.dumps(dict(reason=str(e), family=family, method=method, coefficient=value, dataset=dataset))+'\n')
        print('INELIGIBLE', family, method, value, dataset, str(e), flush=True)


def trajectory_dir(a, family, method, value, dataset):
    data = Data(Path(a.features)/dataset if (Path(a.features)/dataset).is_dir() else a.features, dataset)
    return runner.run_dir(a.out, data, family, method, a.input, PROTOCOL['selection']['seed'], value)


def validation_afa(a, family, method, value, dataset):
    out = trajectory_dir(a, family, method, value, dataset)
    if (out/'ineligible.json').exists(): return None
    return read_json(out/'receipt.json')['validation_metrics']['AFA']


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--features', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--family', default='classic,wifi,recent'); p.add_argument('--device', default='cuda')
    p.add_argument('--threads', type=int, default=2); p.add_argument('--global-seed', type=int)
    p.add_argument('--reseed-per-job', action='store_true')
    p.add_argument('--input', default='m9216', choices=('m9216', 'cat9728'))
    p.add_argument('--shard', help='i/N: train only every N-th candidate run starting at i (no selection printed)')
    p.add_argument('--aggregate', action='store_true', help='Do not train; read the receipts of all candidates')
    p.add_argument('--write-locks', type=Path, help='Write the selected coefficients as a locks file')
    p.add_argument('--extra-candidates', default='', help='method=value[,method=value...] appended to the protocol grids')
    a = p.parse_args(); runner.setup(a)
    for item in filter(None, a.extra_candidates.split(',')):
        m, v = item.split('='); EXTRA.setdefault(m, []).append(float(v))
    families = a.family.split(','); S = PROTOCOL['selection']
    if not a.aggregate:
        todo = tasks(families)
        if a.shard:
            i, n = map(int, a.shard.split('/')); todo = todo[i::n]
        for t in todo: run_one(a, *t)
        if a.shard: return
    selected = {}
    for family in families:
        for method, grid in candidates(family).items():
            per = [[validation_afa(a, family, method, value, d) for d in S['datasets']] for value in grid]
            scores = [float('-inf') if None in v else sum(v)/len(v) for v in per]
            if max(scores) == float('-inf'): raise SystemExit(f'No eligible candidate for {family} {method}')
            best = max(range(len(grid)), key=lambda i: (scores[i], -i))
            selected.setdefault(family, {})[method] = dict(coefficient=grid[best], validation_mean=round(scores[best], 4),
                                                           candidates=grid, scores=[None if s == float('-inf') else round(s, 4) for s in scores])
            locked = LOCKS[family][method]['coefficient']
            print('SELECTED', a.input, family, method, 'scores', selected[family][method]['scores'], 'selected', grid[best],
                  'locked', locked, 'same' if grid[best] == locked else 'DIFFERENT', flush=True)
    if a.write_locks:
        out = dict(description=f"Coefficients selected before any test evaluation by the rule in protocol.json (seed-42 validation AFA "
                               f"averaged over 3DO and WiMANS), for the input '{a.input}' read by every baseline. Selected with "
                               f"scripts/select_coefficients.py --input {a.input}"
                               + (f" --global-seed {a.global_seed} --reseed-per-job" if a.global_seed is not None else "")
                               + ". Use with WIMOT_LOCKS.", input=a.input, global_seed=a.global_seed)
        out.update({f: {m: dict(coefficient=v['coefficient'], validation_mean=v['validation_mean']) for m, v in d.items()}
                    for f, d in selected.items()})
        out['same_input'] = {}
        out['selection_details'] = selected
        a.write_locks.write_text(json.dumps(out, indent=2)+'\n')


if __name__ == '__main__':
    main()
