"""Combines the input-family studies (``scripts/input_study.py`` + ``scripts/summarize_input_study.py``) into one table set.

  python scripts/combine_input_families.py --root feature_agnostic_20261015 [--families amp512,amp_pca,amp_rp,amp_hand]

Reads ``ROOT/<family>/raw`` (every ``test.json``, ``selection.json``) and writes ``ROOT/summary.md`` and ``ROOT/summary.csv``:
per dataset and family the AFA/AFR of WiMoT (``amp_full``, ``amp_only``), the joint-training reference, and the strongest baseline
(highest mean AFA over the baselines that ran on that dataset), the paired per-seed gap of ``amp_full`` to it, the number of
baselines that WiMoT exceeds, and the trajectory counts per family. The Mantis-input results of the paper are listed as a reference.
"""
import argparse
import csv
import statistics
from pathlib import Path

from summarize_input_study import DATASETS, JOINT, LABEL, NAME, WIMOT, collect, group, ms, reference

TITLE = {'amp512': '512-D amplitude descriptor', 'amp_pca': 'PCA-whitened descriptor (D ≤ 256)',
         'amp_rp': 'Gaussian random projection (D = 128)', 'amp_hand': 'hand-crafted statistics (D = 157)'}
FULL, ONLY = WIMOT['amp_full'], WIMOT['amp_only']
BASELINES = set(NAME.values())


def stats(g, name, d):
    r = g.get((name, d))
    if not r: return None
    afa = ms([x['AFA'] for x in r.values()])
    afr = [x['AFR'] for x in r.values() if x['AFR'] is not None]
    return dict(n=len(r), afa=afa, afr=ms(afr) if afr else None, seeds=r)


def cell(s, key='afa'):
    if s is None or s[key] is None: return '–'
    return f'{s[key][0]:.2f} ± {s[key][1]:.2f}' + ('' if s['n'] == 3 else f' (n{s["n"]})')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--families', default='amp512,amp_pca,amp_rp,amp_hand')
    a = p.parse_args(); fams = a.families.split(',')
    data = {f: group(collect(a.root/f/'raw')) for f in fams}
    ref = group(reference()); csv_rows = [['family', 'dataset', 'quantity', 'name', 'n_seeds', 'AFA_mean', 'AFA_sd', 'AFR_mean', 'AFR_sd']]
    L = ['# Foundation-model-free input study: all input families', '',
         'AFA (%) ↑ / AFR (pp) ↓, mean ± sample SD over seeds 42–44 (`n<3`: fewer seeds). Every family is evaluated with the protocol of the paper; '
         'coefficients of the baselines were selected on seed-42 validation (3DO and WiMANS) separately for each input. '
         'No input family reads a foundation-model output; WiMoT runs as `amp_full` (primary; all stems and the PCA shortcut read the family) and `amp_only` (amplitude stem only).', '']
    for d in DATASETS:
        L += [f'## {LABEL[d]}', '',
              '| Input | WiMoT amp_full AFA | AFR | WiMoT amp_only AFA | Joint AFA | Strongest baseline | Baseline AFA | AFR | Gap, pp (seeds ahead) | Baselines below WiMoT | WiMoT rank (1 = best, among WiMoT and baselines) |',
              '|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|']
        for f in fams:
            g = data[f]; w = stats(g, FULL, d); o = stats(g, ONLY, d); j = stats(g, JOINT, d)
            base = {n: stats(g, n, d) for n in BASELINES if (n, d) in g}
            if not base or w is None: L.append(f'| {f} | incomplete | | | | | | | | | |'); continue
            best = max(base, key=lambda n: base[n]['afa'][0]); b = base[best]
            seeds = sorted(set(w['seeds']) & set(b['seeds'])); diff = [100*(w['seeds'][s]['AFA']-b['seeds'][s]['AFA']) for s in seeds]
            gap = f'{statistics.mean(diff):+.2f} ± {statistics.stdev(diff) if len(diff) > 1 else float("nan"):.2f} ({sum(x > 0 for x in diff)}/{len(diff)})'
            below = sum(w['afa'][0] > v['afa'][0] for v in base.values())
            rank = 1+sum(v['afa'][0] > w['afa'][0] for v in base.values())
            L.append(f'| {f} | {cell(w)} | {cell(w, "afr")} | {cell(o)} | {cell(j)} | {best} | {cell(b)} | {cell(b, "afr")} | {gap} | {below}/{len(base)} | {rank} |')
            csv_rows.append([f, d, 'wimot_amp_full', FULL, w['n'], *[f'{x:.3f}' for x in w['afa']], *([f'{x:.3f}' for x in w['afr']] if w['afr'] else ['', ''])])
            csv_rows.append([f, d, 'strongest_baseline', best, b['n'], *[f'{x:.3f}' for x in b['afa']], *([f'{x:.3f}' for x in b['afr']] if b['afr'] else ['', ''])])
            if j: csv_rows.append([f, d, 'joint', JOINT, j['n'], *[f'{x:.3f}' for x in j['afa']], '', ''])
            if o: csv_rows.append([f, d, 'wimot_amp_only', ONLY, o['n'], *[f'{x:.3f}' for x in o['afa']], *([f'{x:.3f}' for x in o['afr']] if o['afr'] else ['', ''])])
        rb = {n: statistics.mean(r['AFA'] for r in ref[(n, d)].values())*100 for n in BASELINES if (n, d) in ref}
        rw = ms([r['AFA'] for r in ref[('WiMoT', d)].values()]); rbest = max(rb, key=rb.get)
        L.append(f'| Mantis block + amplitude (paper, reference) | {rw[0]:.2f} ± {rw[1]:.2f} | | | | {rbest} | {rb[rbest]:.2f} | | {rw[0]-rb[rbest]:+.2f} (mean) | | |')
        L.append('')
    # AFR leaders
    L += ['## Lowest AFR among the baselines versus WiMoT (amp_full)', '', '| Input | Dataset | WiMoT AFR | Lowest baseline AFR (method, its AFA) |', '|---|---|---:|---|']
    for f in fams:
        for d in DATASETS:
            g = data[f]; w = stats(g, FULL, d); base = {n: stats(g, n, d) for n in BASELINES if (n, d) in g and stats(g, n, d)['afr']}
            if not base or w is None: continue
            n = min(base, key=lambda k: base[k]['afr'][0])
            L.append(f'| {f} | {LABEL[d]} | {cell(w, "afr")} | {n}: {base[n]["afr"][0]:.2f} pp (AFA {base[n]["afa"][0]:.2f}) |')
    # trajectory counts
    L += ['', '## Trajectories per family (finished test.json; seeds 42–44)', '',
          '| Family | WiMoT amp_full | WiMoT amp_only | Joint | Baseline trajectories (expected / finished) | Baselines not runnable on the input |', '|---|---:|---:|---:|---|---|']
    for f in fams:
        g = data[f]; cnt = lambda name: sum(len(g.get((name, d), {})) for d in DATASETS)
        done = sum(len(r) for (n, d), r in g.items() if n in BASELINES)
        import json
        sel = json.loads((a.root/f/'raw'/'selection.json').read_text())['selected']
        bad = sorted(NAME[k.split('/')[1]] for k, v in sel.items() if v.get('coefficient') is None and str(v.get('note', '')).startswith('not runnable'))
        reason = {NAME[k.split('/')[1]]: v['note'].split(': ', 1)[1][:110] for k, v in sel.items() if str(v.get('note', '')).startswith('not runnable')}
        expected = 9*(21-len(bad))
        L.append(f'| {f} | {cnt(FULL)}/9 | {cnt(ONLY)}/9 | {cnt(JOINT)}/9 | {expected} / {done} | ' + ('; '.join(f'{n} ({reason[n]})' for n in bad) or 'none') + ' |')
    (a.root/'summary.md').write_text('\n'.join(L)+'\n')
    with open(a.root/'summary.csv', 'w', newline='') as fh: csv.writer(fh).writerows(csv_rows)
    print('\n'.join(L))


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent)); main()
