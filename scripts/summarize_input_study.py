"""Tables of one input-family study (``scripts/input_study.py``) with the Mantis-input results of the paper as reference.

  python scripts/summarize_input_study.py --raw RUNS/<tag> --out DIR [--title "..."]

Reads every ``test.json`` below RAW (and ``selection.json``), and writes to DIR: ``per_seed.csv`` (one row per trajectory),
``summary.csv`` and ``summary.md`` (mean +- sample SD of AFA and AFR over seeds, percent / percentage points, sorted by the
mean AFA over the three datasets), the paired per-seed gap of the WiMoT variant ``amp_full`` to the strongest baseline of each
dataset, and the selected coefficients. ``results/reference_results.json`` supplies the foundation-model (MantisV2) results.
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASETS = ('3do', 'wimans', 'widar'); LABEL = {'3do': '3DO', 'wimans': 'WiMANS', 'widar': 'Widar3.0'}
NAME = dict(ft='Fine-tuning', er='ER', ewc='EWC', ewc_er='EWC+ER', si='SI', si_er='SI+ER', mas='MAS', mas_er='MAS+ER', lwf='LwF',
            lwf_er='LwF+ER', derpp='DER++', agem='A-GEM', consense='ConSense', edgecl='EdgeCL', duct='DUCT', dualcp='DualCP',
            vanillacp='VanillaCP', cn_dpm_aligned='CN-DPM', duct_er='DUCT+ER', dualcp_er='DualCP+ER', vanillacp_er='VanillaCP+ER')
WIMOT = {'amp_full': 'WiMoT (amp_full, primary)', 'amp_only': 'WiMoT (amp_only)', 'main': 'WiMoT'}
JOINT = 'Joint training (non-continual reference)'


def display(r):
    if r['family'] == 'method': return WIMOT[r['method']]
    if r['family'] == 'joint': return JOINT
    return NAME[r['method']]


def ms(values, scale=100.):
    v = [x*scale for x in values]
    return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else float('nan'))


def fmt(m, s): return f'{m:.2f} ± {s:.2f}'


def collect(raw):
    rows = []
    for p in sorted(Path(raw).rglob('test.json')):
        r = json.loads(p.read_text()); m = r['metrics']
        rows.append(dict(name=display(r), family=r['family'], method=r['method'], dataset=r['dataset'], seed=int(r['seed']),
                         coefficient=r.get('coefficient'), AFA=m['AFA'], AFR=m.get('AFR')))
    return rows


def reference():
    """Mantis-input results of the paper (study ``main``; WiMoT from study ``ablation``, seeds 42-44)."""
    rows = []
    for r in json.loads((REPO/'results'/'reference_results.json').read_text())['results']:
        if r['encoder'] != 'mantis': continue
        if r['family'] == 'method':     # WiMoT 'main' of Table I is stored with the ablation study (seeds 42-46)
            if not (r['study'] == 'ablation' and r['method'] == 'main' and r['seed'] <= 44): continue
        elif r['study'] != 'main': continue
        rows.append(dict(name=display(r) if r['family'] != 'method' else 'WiMoT', family=r['family'], method=r['method'],
                         dataset=r['dataset'], seed=int(r['seed']), AFA=r['AFA'], AFR=r.get('AFR')))
    return rows


def group(rows):
    g = {}
    for r in rows: g.setdefault((r['name'], r['dataset']), {})[r['seed']] = r
    return g


def build(raw, out, title):
    rows = collect(raw); ref = group(reference()); mine = group(rows); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    with open(out/'per_seed.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['method', 'dataset', 'seed', 'coefficient', 'AFA_percent', 'AFR_pp'])
        for r in sorted(rows, key=lambda r: (r['name'], r['dataset'], r['seed'])):
            w.writerow([r['name'], r['dataset'], r['seed'], r['coefficient'], f"{100*r['AFA']:.3f}",
                        '' if r['AFR'] is None else f"{100*r['AFR']:.3f}"])
    # table sorted by mean AFA over the datasets that exist for every method
    def mean_afa(n):
        values = [ms([r['AFA'] for r in mine[(n, d)].values()])[0] for d in DATASETS if (n, d) in mine]
        return statistics.mean(values) if len(values) == 3 else -1
    names = sorted({n for n, _ in mine}, key=lambda n: -mean_afa(n))
    lines = [f'## {title}', '', 'AFA (%) ↑ and AFR (pp) ↓, mean ± sample SD over seeds 42–44; sorted by the mean AFA over the three datasets. '
             'Rows marked `n<3` have fewer than three seeds.', '',
             '| Method | ' + ' | '.join(f'{LABEL[d]} AFA | {LABEL[d]} AFR' for d in DATASETS) + ' | Mean AFA |', '|---|' + '---:|'*7]
    csv_rows = [['method', 'dataset', 'n_seeds', 'AFA_mean', 'AFA_sd', 'AFR_mean', 'AFR_sd']]
    for n in names:
        cells = []; means = []
        for d in DATASETS:
            g = mine.get((n, d))
            if not g: cells += ['–', '–']; continue
            a = ms([r['AFA'] for r in g.values()]); mark = '' if len(g) == 3 else f' (n{len(g)})'
            afr = [r['AFR'] for r in g.values() if r['AFR'] is not None]; fr = ms(afr) if afr else None
            cells += [fmt(*a)+mark, fmt(*fr) if fr else '–']; means.append(a[0])
            csv_rows.append([n, d, len(g), f'{a[0]:.3f}', f'{a[1]:.3f}'] + ([f'{x:.3f}' for x in fr] if fr else ['', '']))
        last = f'{statistics.mean(means):.2f}' if len(means) == 3 else '–'
        lines.append(f'| {n} | ' + ' | '.join(cells) + f' | {last} |')
    with open(out/'summary.csv', 'w', newline='') as f: csv.writer(f).writerows(csv_rows)
    lines += ['', '### Ranking by AFA on each dataset (all methods and the reference)', '', '| Rank | ' + ' | '.join(LABEL[d] for d in DATASETS) + ' |', '|---:|---|---|---|']
    ranked = {d: sorted(((statistics.mean(r['AFA'] for r in mine[(n, d)].values())*100, n) for n in names if (n, d) in mine), reverse=True) for d in DATASETS}
    for k in range(len(names)):
        lines.append(f'| {k+1} | ' + ' | '.join(f'{ranked[d][k][1]} ({ranked[d][k][0]:.2f})' if k < len(ranked[d]) else '–' for d in DATASETS) + ' |')
    # gaps of WiMoT variants to the strongest baseline (highest mean AFA among the 21 baselines on that dataset)
    baselines = {n for (n, d), g in mine.items() if any(r['family'] in ('classic', 'wifi', 'recent') for r in g.values())}
    lines += ['', '### WiMoT versus the strongest baseline on each dataset (paired by seed)', '',
              '| Dataset | Strongest baseline (mean AFA) | Baseline AFA | WiMoT amp_full AFA | Gap, pp (mean ± SD of per-seed differences; seeds with WiMoT ahead) | Mantis-input gap of the paper, pp |', '|---|---|---:|---:|---|---:|']
    for d in DATASETS:
        cand = [(statistics.mean(r['AFA'] for r in mine[(n, d)].values()), n) for n in baselines if (n, d) in mine]
        if not cand or ('WiMoT (amp_full, primary)', d) not in mine: continue
        best_mean, best = max(cand); w = mine[('WiMoT (amp_full, primary)', d)]; b = mine[(best, d)]
        seeds = sorted(set(w) & set(b)); diff = [100*(w[s]['AFA']-b[s]['AFA']) for s in seeds]
        ahead = sum(x > 0 for x in diff)
        rcand = [(statistics.mean(r['AFA'] for r in ref[(n, d)].values()), n) for n in NAME.values() if (n, d) in ref]
        rgap = 100*(statistics.mean(r['AFA'] for r in ref[('WiMoT', d)].values())-max(rcand)[0])
        lines.append(f'| {LABEL[d]} | {best} | {100*best_mean:.2f} | {100*statistics.mean(r["AFA"] for r in w.values()):.2f} | '
                     f'{statistics.mean(diff):+.2f} ± {statistics.stdev(diff) if len(diff) > 1 else float("nan"):.2f} ({ahead}/{len(diff)}) | {rgap:+.2f} |')
    lines += ['', '### Mantis-input reference (WiMoT with 9728-D input, baselines with the 9216-D Mantis block; paper Supplementary Table S9, 9216 column)', '',
              '| Method | ' + ' | '.join(f'{LABEL[d]} AFA' for d in DATASETS) + ' |', '|---|---:|---:|---:|']
    for n in ['WiMoT'] + [NAME[m] for m in NAME]:
        if all((n, d) in ref for d in DATASETS):
            lines.append(f'| {n} | ' + ' | '.join(fmt(*ms([r['AFA'] for r in ref[(n, d)].values()])) for d in DATASETS) + ' |')
    sel = Path(raw)/'selection.json'
    if sel.exists():
        s = json.loads(sel.read_text())['selected']
        lines += ['', '### Coefficients selected on seed-42 validation (3DO and WiMANS)', '', '| Method | Selected | Candidates | Mean validation AFA per candidate |', '|---|---:|---|---|']
        for k, v in s.items():
            lines.append(f"| {NAME[k.split('/')[1]]} | {v['coefficient']} | {v.get('candidates', '–')} | "
                         f"{[round(x, 4) for x in v['scores']] if v.get('scores') else v.get('note', '–')} |")
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    return lines


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raw', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--title', default='Input family')
    a = p.parse_args(); print('\n'.join(build(a.raw, a.out, a.title)))
