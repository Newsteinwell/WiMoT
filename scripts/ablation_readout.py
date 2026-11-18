"""Readout ablation: replace the OSCL readout of trained WiMoT trajectories by other readouts fitted from the same stored moments.

For every trained trajectory (``wimot.run train/test --family method --method main``) and every stage k, the adapter of stage k
and the class moments stored at stage k are loaded; the moments are transported to the adapted space h, and each readout
predicts the TEST rows of the k domains seen so far (the domain identity is not used). Only stored moments are used (no
historical sample), and nothing is trained or tuned here. The OSCL entry is the test matrix of the trajectory itself.

  python scripts/ablation_readout.py eval --features FEATURES/mantis --runs RUNS --out OUT [--dataset 3do,wimans,widar]
                                          [--seed 42,43,44] [--device cuda]
  python scripts/ablation_readout.py summarize --out OUT

RUNS is the ``--root`` of the trajectories (it holds ``method/<dataset>/seed<S>/main``). ``eval`` writes OUT/raw/<dataset>_seed<S>.json
(all accuracy matrices); ``summarize`` writes OUT/summary_*.md and OUT/summary_*.csv (mean and sample SD over seeds, in percent).
"""
import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wimot.common import Data, read_json, write_json                                  # noqa: E402
from wimot.measures import summary_measures                                           # noqa: E402
from wimot.method import train as mt                                                  # noqa: E402
from wimot.method.model import model_from_checkpoint                                  # noqa: E402
from wimot.method.readouts_alt import READOUTS, readout_scores                        # noqa: E402
from wimot.method.statistics import load_stats, transport                             # noqa: E402

ORDER = ('oscl', 'ncm', 'lda', 'qda', 'probe', 'lda_ridge', 'qda_ridge')
LABEL = dict(oscl='OSCL (WiMoT)', ncm='Nearest class mean', lda='LDA', qda='QDA (no domain gate)', probe='Linear probe W',
             lda_ridge='LDA, ridge 1e-3 (supplementary)', qda_ridge='QDA, ridge 1e-3 (supplementary)')
MEASURES = (('AFA', 'AFA (%)'), ('AFR', 'AFR (pp)'), ('new', 'New-domain acc. (%)'), ('old', 'Earlier-domain acc. (%)'))
DATASETS = ('3do', 'wimans', 'widar')
NAMES = dict(**{'3do': '3DO', 'wimans': 'WiMANS', 'widar': 'Widar3.0'})


def evaluate_run(features, root, dataset, seed, device):
    data = Data(Path(features)/dataset, dataset, None, None, seed)
    out = mt.run_dir(root, data, seed, 'main'); receipt = read_json(out/'receipt.json'); spec = receipt['spec']
    assert receipt['complete'], out
    reference = read_json(out/'test.json'); T = len(data.domains)
    raw = {d: mt.load(data, d, 'test', spec) for d in data.domains}
    matrices = {name: [[None]*T for _ in range(T)] for name in ORDER}; matrices['oscl'] = reference['matrix']
    for stage in range(1, T+1):
        cp = mt.load_checkpoint(out/f'stage{stage}.pt')
        model = model_from_checkpoint(cp).to(device).eval().requires_grad_(False)
        moments = transport(load_stats(out/f'stage{stage}_stats.npz'), model)
        weight = model.classifier.weight.detach().cpu().numpy(); bias = model.classifier.bias.detach().cpu().numpy()
        hh, yy, tt = [], [], []
        for task, domain in enumerate(data.domains[:stage]):
            x, y = raw[domain]; phi = mt.infer(model, ((x-cp['mean'])/cp['sd']).astype(np.float32), device)
            hh.append(mt.adapt(cp, phi)); yy.append(y); tt.append(np.full(len(y), task))
        h, y, t = map(np.concatenate, (hh, yy, tt))
        for name in READOUTS:
            pred = readout_scores(name, moments, h, weight, bias).argmax(1)
            matrices[name][stage-1][:stage] = [float(np.mean(pred[t == i] == y[t == i])) for i in range(stage)]
    return dict(dataset=dataset, seed=seed, domains=data.domains, run=str(out), matrices=matrices,
                measures={n: summary_measures(m) for n, m in matrices.items()})


def evaluate(a):
    for dataset in a.dataset.split(','):
        for seed in map(int, a.seed.split(',')):
            started = time.monotonic()
            result = evaluate_run(a.features, a.runs, dataset, seed, a.device)
            write_json(a.out/'raw'/f'{dataset}_seed{seed}.json', result)
            print('READOUTS', dataset, seed, {n: round(100*m['AFA'], 2) for n, m in result['measures'].items()},
                  round(time.monotonic()-started), 's', flush=True)


def cell(values, scale=100.):
    values = [scale*v for v in values]
    return f'{statistics.mean(values):.2f}±{statistics.stdev(values):.2f}' if len(values) > 1 else f'{values[0]:.2f}'


def summarize(a):
    runs = [read_json(p) for p in sorted((a.out/'raw').glob('*.json'))]
    seeds = sorted({r['seed'] for r in runs})
    sets = {'seeds42-44': [s for s in seeds if s <= 44]}
    if any(s > 44 for s in seeds): sets['seeds42-46'] = seeds
    for tag, wanted in sets.items():
        lines = [f'Readout ablation, seeds {wanted[0]}-{wanted[-1]} (mean±sample SD over seeds; AFR in percentage points)', '']
        rows = [['dataset', 'readout', 'n_seeds', *[m[0] for m in MEASURES]]]
        for dataset in DATASETS:
            sel = [r for r in runs if r['dataset'] == dataset and r['seed'] in wanted]
            if not sel: continue
            lines += [f'**{NAMES[dataset]}** (seeds {sorted(r["seed"] for r in sel)})', '',
                      '| Readout | ' + ' | '.join(m[1] for m in MEASURES) + ' |', '|---|' + '---|'*len(MEASURES)]
            for name in ORDER:
                vals = {m: [r['measures'][name][m] for r in sel] for m, _ in MEASURES}
                lines.append(f'| {LABEL[name]} | ' + ' | '.join(cell(vals[m]) for m, _ in MEASURES) + ' |')
                rows.append([dataset, name, len(sel), *[f'{100*statistics.mean(vals[m]):.3f}' for m, _ in MEASURES],
                             *[f'{100*statistics.stdev(vals[m]):.3f}' if len(sel) > 1 else '' for m, _ in MEASURES]])
            lines.append('')
            lines += ['Paired difference OSCL minus readout over the same seeds (mean in pp; seeds with OSCL strictly better / equal / worse):', '',
                      '| Readout | ΔAFA | ΔAFR | ΔNew | ΔEarlier | AFA better/equal/worse |', '|---|---|---|---|---|---|']
            for name in ORDER[1:]:
                d = {m: [100*(r['measures']['oscl'][m]-r['measures'][name][m]) for r in sel] for m, _ in MEASURES}
                w = [sum(v > 1e-9 for v in d['AFA']), sum(abs(v) <= 1e-9 for v in d['AFA']), sum(v < -1e-9 for v in d['AFA'])]
                lines.append(f'| {LABEL[name]} | ' + ' | '.join(f'{statistics.mean(d[m]):+.2f}' for m, _ in MEASURES) + f' | {w[0]}/{w[1]}/{w[2]} |')
            lines.append('')
        rows[0] += [f'{m[0]}_sd' for m in MEASURES]
        (a.out/f'summary_{tag}.md').write_text('\n'.join(lines)+'\n')
        with open(a.out/f'summary_{tag}.csv', 'w', newline='') as f: csv.writer(f).writerows(rows)
        print('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=('eval', 'summarize')); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--features', type=Path); p.add_argument('--runs', type=Path)
    p.add_argument('--dataset', default='3do,wimans,widar'); p.add_argument('--seed', default='42,43,44')
    p.add_argument('--device', default='cuda')
    a = p.parse_args(); torch.backends.cuda.matmul.allow_tf32 = False; torch.set_num_threads(2)
    evaluate(a) if a.command == 'eval' else summarize(a)


if __name__ == '__main__':
    main()
