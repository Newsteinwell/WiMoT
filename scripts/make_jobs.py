"""Writes configs/jobs/<study>.tsv: one line per trajectory reported in the paper, derived from results/reference_results.json.

Columns: family encoder input dataset order fraction seed method coefficient ('-' for none). The main table reports the
proposed method from the ablation runs (seeds 42-44); those lines are repeated in main.tsv so that every table can be
reproduced from its own job list.
"""
import collections
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COLUMNS = ('family', 'encoder', 'input', 'dataset', 'order', 'fraction', 'seed', 'method', 'coefficient')


def cell(value):
    return '-' if value is None else (f'{value:g}' if isinstance(value, float) else str(value))


def main():
    rows = json.loads((REPO/'results'/'reference_results.json').read_text())['results']
    jobs = collections.defaultdict(list)
    for r in rows:
        jobs[r['study']].append(r)
        if r['study'] == 'ablation' and r['method'] == 'main' and r['seed'] <= 44: jobs['main'].append(r)
    out = REPO/'configs'/'jobs'; out.mkdir(parents=True, exist_ok=True)
    for study, items in sorted(jobs.items()):
        lines = sorted({tuple(cell(r.get(k)) for k in COLUMNS) for r in items},
                       key=lambda t: (t[0], t[1], t[2], t[3], t[4], t[5], int(t[6]), t[7], t[8]))
        (out/f'{study}.tsv').write_text('\t'.join(COLUMNS)+'\n'+''.join('\t'.join(t)+'\n' for t in lines))
        print(study, len(lines))


if __name__ == '__main__':
    main()
