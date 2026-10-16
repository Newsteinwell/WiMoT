"""AFA of WiMoT against the SupCon weight and the L_hist weight (sensitivity analysis, seeds 42-44).

Reads results/additional_analyses/loss_weight_sweep/summary.csv (written by scripts/ablation_loss_sweep.py summarize) and
results/reference_results_same_input.json; writes a 2 x 3 figure (rows: lambda_sc and lambda_h; columns: datasets) to
assets/loss_weight_sweep.png and results/additional_analyses/loss_weight_sweep/loss_weight_sweep_afa.{png,pdf}.
Shaded band: +-1 sample SD over seeds. Star and dashed vertical line: the weight of the paper. Dashed horizontal line: the
strongest baseline of Table I in the same seeds. The weight 0 is drawn at the left end, separated by an axis break.

    python scripts/plot_loss_sweep.py
"""
import collections
import csv
import json
import math
import pathlib
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUMMARY = ROOT / 'results/additional_analyses/loss_weight_sweep/summary.csv'
REF = ROOT / 'results/reference_results_same_input.json'  # baselines read the 9728-D input
DATASETS = (('3do', '3DO'), ('wimans', 'WiMANS'), ('widar', 'Widar3.0'))
ROWS = (('sc', 'lambda_sc', 0.5, r'SupCon weight $\lambda_{\mathrm{sc}}$ (later domains)', r'$\lambda_{\mathrm{h}}=1$'),
        ('h', 'lambda_h', 1.0, r'$\mathcal{L}_{\mathrm{hist}}$ weight $\lambda_{\mathrm{h}}$', r'$\lambda_{\mathrm{sc}}=0.5$'))
BLUE, GREY, ORANGE = '#0072B2', '#555555', '#B85B16'


def best_baseline():
    c = collections.defaultdict(list)
    for r in json.loads(REF.read_text())['results']:
        if r['study'] == 'main' and r['family'] != 'joint' and r['method'] != 'main' and r['seed'] in (42, 43, 44):
            c[(r['dataset'], r['method'])].append(100 * r['AFA'])
    out = {}
    for d, _ in DATASETS:
        m = {k[1]: st.mean(v) for k, v in c.items() if k[0] == d}
        out[d] = (max(m.values()), max(m, key=m.get))
    return out


def load():
    rows = collections.defaultdict(list)
    for r in csv.DictReader(SUMMARY.open()):
        if r['series'] in ('sc', 'h'):
            rows[(r['dataset'], r['series'])].append((float(r['lambda_sc']), float(r['lambda_h']), float(r['AFA']), float(r['AFA_sd'])))
    return rows


def main():
    base, rows = best_baseline(), load()
    plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6, 'font.family': 'DejaVu Sans'})
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.3), sharey='col')
    for j, (d, dname) in enumerate(DATASETS):
        for i, (series, key, paper, xlabel, fixed) in enumerate(ROWS):
            ax = axes[i][j]
            pts = sorted((r[0] if series == 'sc' else r[1], r[2], r[3]) for r in rows[(d, series)])
            pos = [p for p in pts if p[0] > 0]
            x = [math.log2(p[0]) for p in pos]
            y = [p[1] for p in pos]; s = [p[2] for p in pos]
            x0 = x[0] - 1.6                                     # position of the weight 0
            zero = next(p for p in pts if p[0] == 0)
            ax.axhline(base[d][0], color=GREY, ls='--', lw=0.9, zorder=1)
            ax.fill_between(x, [a - b for a, b in zip(y, s)], [a + b for a, b in zip(y, s)], color=BLUE, alpha=0.18, lw=0)
            ax.plot(x, y, '-o', color=BLUE, ms=3.2, lw=1.3, zorder=3)
            ax.plot([x0, x[0]], [zero[1], y[0]], ':', color=BLUE, lw=0.8, zorder=2)
            ax.errorbar([x0], [zero[1]], yerr=[zero[2]], fmt='o', mfc='white', mec=ORANGE, color=ORANGE, ms=4.2, capsize=2, lw=1, zorder=4)
            px = math.log2(paper)
            ax.axvline(px, color=ORANGE, ls=':', lw=0.9, zorder=1)
            ax.plot([px], [next(p[1] for p in pos if p[0] == paper)], marker='*', color=ORANGE, ms=9, mec='black', mew=0.4, ls='', zorder=5)
            ax.set_xticks([x0] + x)
            ax.set_xticklabels(['0'] + [f'{p[0]:g}' for p in pos])
            ax.set_xlim(x0 - 0.45, x[-1] + 0.45)
            mid = (x0 + x[0]) / 2
            for dx in (-0.09, 0.09):                            # axis break marks
                ax.plot([mid + dx - 0.08, mid + dx + 0.08], [-0.035, 0.035], transform=ax.get_xaxis_transform(), color='black', lw=0.8, clip_on=False)
            ax.set_xlabel(xlabel + ', ' + fixed, fontsize=7)
            ax.grid(axis='y', color='0.9', lw=0.5)
            ax.spines[['top', 'right']].set_visible(False)
            if i == 0: ax.set_title(dname, fontsize=9, fontweight='bold')
            if j == 0: ax.set_ylabel('AFA (%)')
    for j, (d, _) in enumerate(DATASETS):                       # y range: include the baseline line and every band
        lo = min(min(r[2] - r[3] for r in rows[(d, s)]) for s in ('sc', 'h')); lo = min(lo, base[d][0])
        hi = max(max(r[2] + r[3] for r in rows[(d, s)]) for s in ('sc', 'h'))
        pad = 0.08 * (hi - lo)
        axes[0][j].set_ylim(lo - pad, hi + pad)
    h = [plt.Line2D([], [], color=BLUE, marker='o', ms=3.2, lw=1.3), plt.Line2D([], [], color=ORANGE, marker='o', mfc='white', ls='', ms=4.2),
         plt.Line2D([], [], color=ORANGE, marker='*', ls=':', ms=9, mec='black', mew=0.4), plt.Line2D([], [], color=GREY, ls='--', lw=0.9)]
    fig.legend(h, ['WiMoT, mean $\\pm$ 1 SD over seeds 42-44', 'loss removed (weight 0)', 'weight of the paper', 'strongest baseline (Table I)'],
               loc='lower center', ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.045, 1, 1), h_pad=1.2, w_pad=1.0)
    out = ROOT / 'results/additional_analyses/loss_weight_sweep'
    fig.savefig(out / 'loss_weight_sweep_afa.pdf')
    fig.savefig(out / 'loss_weight_sweep_afa.png', dpi=200)
    fig.savefig(ROOT / 'assets/loss_weight_sweep.png', dpi=200)
    print({d: (round(v[0], 2), v[1]) for d, v in base.items()})


if __name__ == '__main__':
    main()
