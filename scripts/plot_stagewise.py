"""Mean test accuracy over the seen domains after each training stage (paper Fig. 3), original domain order, seeds 42-44.

Reads results/reference_results_same_input.json (WiMoT: ablation records of method ``main``; baselines: ``main`` study, 9728-D input)
and writes assets/wimot_stagewise.png.

    python scripts/plot_stagewise.py
"""
import json
import pathlib
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
REF = json.loads((ROOT / 'results' / 'reference_results_same_input.json').read_text())['results']
DATASETS = (('3do', '(a) 3DO, 3 days'), ('wimans', '(b) WiMANS, 3 rooms'), ('widar', '(c) Widar3.0, 9 users'))
# method, label, color, linestyle, marker, filled
CURVES = (('main', 'WiMoT', '#0072B2', '-', 'o', True), ('si_er', 'SI+ER', '#444444', '-', 'v', True),
          ('mas_er', 'MAS+ER', '#8c564b', '--', 's', False), ('lwf_er', 'LwF+ER', '#D55E00', '-.', 'D', False),
          ('derpp', 'DER++', '#555555', '--', 'o', False), ('edgecl', 'EdgeCL', '#007F73', '-', '^', True),
          ('consense', 'ConSense', '#A8457E', '--', 'D', True), ('ft', 'Fine-tuning', '#888888', ':', 'x', False))


def curve(dataset, method):
    runs = [r for r in REF if r['dataset'] == dataset and r['method'] == method and r['seed'] <= 44 and r['order'] is None
            and r['fraction'] is None and r['encoder'] == 'mantis' and r['family'] != 'joint'
            and r['study'] == ('ablation' if method == 'main' else 'main')]
    assert len(runs) == 3, (dataset, method, len(runs))
    T = len(runs[0]['matrix'])
    return [100 * st.mean(st.mean(r['matrix'][s][:s + 1]) for r in runs) for s in range(T)]


def main():
    plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6, 'font.family': 'DejaVu Sans'})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1), gridspec_kw=dict(width_ratios=(1, 1, 1.5)))
    for ax, (d, title) in zip(axes, DATASETS):
        for m, label, color, ls, mk, filled in CURVES:
            y = curve(d, m)
            ax.plot(range(1, len(y) + 1), y, ls=ls, color=color, marker=mk, ms=4.5, lw=1.4 if m == 'main' else 1.0,
                    mfc=color if filled else 'white', label=label, zorder=5 if m == 'main' else 3)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xlabel('Training stage')
        ax.set_xticks(range(1, len(y) + 1))
        ax.grid(axis='y', color='0.9', lw=0.5)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Mean acc. on seen domains (%)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=8, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(ROOT / 'assets' / 'wimot_stagewise.png', dpi=200)


if __name__ == '__main__':
    main()
