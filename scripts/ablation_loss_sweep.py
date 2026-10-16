"""Sensitivity of WiMoT to the loss weights: SupCon (paper 0.5) against L_hist (paper 1).

The later-domain loss is  lambda_sc * L_SC + lambda_h * L_hist  (paper: 0.5 and 1; the first domain uses SupCon alone, weight 0.5).
A setting is a method name of ``wimot.run`` (``main_sc<w>_h<w>[_fsc<w>]``, see ``wimot.method.train.variant_spec``); ``main`` is the
paper setting. This sweep is a sensitivity analysis on the TEST partitions of seeds 42-44: the coefficients locked in
configs/locks.json are not changed by it.

  python scripts/ablation_loss_sweep.py jobs --out DIR --shards 12 [--dataset 3do,wimans,widar] [--seed 42,43,44]
        writes DIR/shard<i>.txt (lines 'dataset seed method'); run each with  python -m wimot.run train|test --jobs DIR/shard<i>.txt
  python scripts/ablation_loss_sweep.py summarize --runs RUNS --out OUT
        tables (markdown, csv) and the plasticity-stability figure (PNG, PDF) from RUNS/method/<dataset>/seed<S>/<setting>/test.json

Grid: lambda_sc in {0, .1, .25, .5, 1, 2} at lambda_h = 1 (later domains only; the first domain keeps the paper setting), lambda_h in
{0, .25, .5, 1, 2, 4} at lambda_sc = .5, lambda_sc in {0, .1, .25, 1, 2} with SupCon re-weighted in the first domain as well
(``fsc``; lambda_sc = 0 and fsc = 0 is the paper row ``no_supcon``), and, on 3DO and WiMANS, the 3 x 3 grid lambda_sc in {.25, .5, 1}
x lambda_h in {.5, 1, 2}.
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from wimot.common import read_json                    # noqa: E402
from wimot.measures import summary_measures           # noqa: E402
from wimot.method.train import sweep_name             # noqa: E402

SC = (0., .1, .25, .5, 1., 2.)
H = (0., .25, .5, 1., 2., 4.)
SC_FIRST = (0., .1, .25, 1., 2.)
GRID = ((.25, .5, 1.), (.5, 1., 2.))
DATASETS = ('3do', 'wimans', 'widar')
NAMES = {'3do': '3DO', 'wimans': 'WiMANS', 'widar': 'Widar3.0'}
MEASURES = (('AFA', 'AFA (%)'), ('AFR', 'AFR (pp)'), ('new', 'New-domain acc. (%)'), ('old', 'Earlier-domain acc. (%)'))


def settings(dataset):
    """(series, lambda_sc, lambda_h, setting name) of every run of ``dataset``; the paper setting belongs to every series."""
    out = [('sc', sc, 1., sweep_name(sc, 1.)) for sc in SC]
    out += [('h', .5, h, sweep_name(.5, h)) for h in H]
    out += [('sc_first', sc, 1., sweep_name(sc, 1., sc)) for sc in SC_FIRST]
    if dataset in ('3do', 'wimans'): out += [('grid', sc, h, sweep_name(sc, h)) for sc in GRID[0] for h in GRID[1]]
    return out


def make_jobs(a):
    """Jobs ordered by priority (paper series first: sc, h; then sc_first; then the 2-D grid), longest streams first within a group."""
    group = {'sc': 0, 'h': 0, 'sc_first': 1, 'grid': 2}; priority = {}
    for dataset in a.dataset.split(','):
        for key, _, _, name in settings(dataset): priority[(dataset, name)] = min(priority.get((dataset, name), 9), group[key])
    jobs = [(d, int(seed), name) for d in a.dataset.split(',') for seed in a.seed.split(',') for name in sorted({s[3] for s in settings(d)})]
    jobs.sort(key=lambda j: (priority[(j[0], j[2])], DATASETS[::-1].index(j[0]), j[1], j[2]))
    a.out.mkdir(parents=True, exist_ok=True)
    for i in range(a.shards):
        (a.out/f'shard{i}.txt').write_text(''.join(f'{d} {s} {m}\n' for d, s, m in jobs[i::a.shards]))
    print(len(jobs), 'trajectories in', a.shards, 'shards')


def collect(runs):
    """{(dataset, setting): {seed: measures}} and the raw test records."""
    out, records = {}, {}
    for path in sorted(Path(runs).glob('method/*/seed*/*/test.json')):
        r = read_json(path); key = (r['dataset'], r['method'])
        out.setdefault(key, {})[r['seed']] = summary_measures(r['matrix']); records[(r['dataset'], r['method'], r['seed'])] = r
    return out, records


def mean_sd(values):
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.)


def fmt(values):
    m, s = mean_sd([100*v for v in values])
    return f'{m:.2f}±{s:.2f}' if len(values) > 1 else f'{m:.2f}'


def reference_rows(seeds):
    """Paper ablation rows (reference results, seeds 42-44) as context: main, no_supcon, no_hist."""
    ref = read_json(REPO/'results'/'reference_results.json')['results']; rows = {}
    for r in ref:
        if r['study'] == 'ablation' and r['method'] in ('main', 'no_supcon', 'no_hist') and r['seed'] in seeds:
            rows.setdefault((r['dataset'], r['method']), {})[r['seed']] = summary_measures(r['matrix'])
    return rows


def table(dataset, rows, data, header):
    lines = [f'| {header} | n | ' + ' | '.join(m[1] for m in MEASURES) + ' |', '|---|---|' + '---|'*len(MEASURES)]
    for label, name in rows:
        seeds = data.get((dataset, name), {})
        if not seeds: continue
        lines.append(f'| {label} | {len(seeds)} | ' + ' | '.join(fmt([v[m] for v in seeds.values()]) for m, _ in MEASURES) + ' |')
    return lines


STYLE = {'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 11, 'axes.labelsize': 12, 'legend.fontsize': 10,
         'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False}
SERIES = (('sc', r'WiMoT, $\lambda_{\mathrm{SC}}$ sweep (later domains)', 'tab:blue', 'o', '-', (4, 6)),
          ('h', r'WiMoT, $\lambda_{\mathrm{hist}}$ sweep', 'tab:red', 's', '-', (4, -11)),
          ('sc_first', r'WiMoT, $\lambda_{\mathrm{SC}}$ sweep (all domains)', 'tab:green', '^', '--', (-14, 6)))


def points(data, dataset, key):
    """[(swept weight, {seed: measures})] of one series with data."""
    out = [(s[2] if key == 'h' else s[1], data.get((dataset, s[3]), {})) for s in settings(dataset) if s[0] == key]
    out = [(lam, d) for lam, d in out if d]
    return sorted(out, key=lambda p: -p[0]) if key == 'h' else out      # ascending lambda_sc / lambda_h ratio


def stat(d, m):
    return mean_sd([100*v[m] for v in d.values()])


def figure(data, out, name='plasticity_stability', keys=('sc', 'h')):
    """Plasticity-stability curves: new-domain accuracy against earlier-domain accuracy (top) and AFR (bottom); mean ± SD over seeds."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4))
    for col, dataset in enumerate(DATASETS):
        for row, (y, ylabel) in enumerate((('old', 'Earlier-domain accuracy (%)'), ('AFR', 'AFR (pp)'))):
            ax = axes[row, col]
            for key, label, color, marker, ls, offset in SERIES:
                pts = points(data, dataset, key) if key in keys else []
                if not pts: continue
                xs = [stat(d, 'new') for _, d in pts]; ys = [stat(d, y) for _, d in pts]
                ax.errorbar([x[0] for x in xs], [v[0] for v in ys], xerr=[x[1] for x in xs], yerr=[v[1] for v in ys], color=color, marker=marker, ms=5.5,
                            lw=1.3, ls=ls, elinewidth=.6, capsize=1.5, alpha=.85, label=label)
                for (lam, _), x, v in zip(pts, xs, ys):
                    ax.annotate(f'{lam:g}', (x[0], v[0]), textcoords='offset points', xytext=offset, fontsize=8, color=color)
            paper = data.get((dataset, 'main'))
            if paper:
                ax.plot(stat(paper, 'new')[0], stat(paper, y)[0], marker='*', ms=15, mfc='none', mec='black', mew=1.3, ls='',
                        label=r'WiMoT, paper setting ($0.5$, $1$)', zorder=5)
            ax.set_xlabel('New-domain accuracy (%)'); ax.set_ylabel(ylabel); ax.grid(alpha=.25, lw=.5)
            if row == 0: ax.set_title(NAMES[dataset])
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(.5, -.01))
    fig.tight_layout(rect=(0, .05, 1, 1))
    for ext in ('png', 'pdf'): fig.savefig(out/f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def ratio_figure(data, out):
    """The three measures against the weight ratio lambda_sc / lambda_h (later domains), lambda_sc and lambda_hist sweeps."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update(STYLE)
    ratios = {}
    for dataset in DATASETS:
        for key in ('sc', 'h'):
            for lam, d in points(data, dataset, key):
                rho = lam if key == 'sc' else (float('inf') if lam == 0 else .5/lam); ratios.setdefault(dataset, {}).setdefault(key, []).append((rho, d))
    order = sorted({r for v in ratios.values() for series in v.values() for r, _ in series}); pos = {r: i for i, r in enumerate(order)}
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex='col')
    for col, dataset in enumerate(DATASETS):
        for row, (m, ylabel) in enumerate((('new', 'New-domain accuracy (%)'), ('old', 'Earlier-domain accuracy (%)'), ('AFR', 'AFR (pp)'))):
            ax = axes[row, col]
            for key, label, color, marker, ls, _ in SERIES[:2]:
                pts = sorted(ratios[dataset][key], key=lambda p: pos[p[0]])
                st = [stat(d, m) for _, d in pts]
                ax.errorbar([pos[r] for r, _ in pts], [v[0] for v in st], yerr=[v[1] for v in st], color=color, marker=marker, ms=5, lw=1.2, ls=ls,
                            elinewidth=.7, capsize=2, alpha=.85, label=label)
            ax.set_ylabel(ylabel); ax.grid(alpha=.25, lw=.5)
            if row == 0: ax.set_title(NAMES[dataset])
            if row == 2:
                ax.set_xticks(range(len(order))); ax.set_xticklabels(['$\\infty$' if r == float('inf') else f'{r:g}' for r in order], fontsize=9)
                ax.set_xlabel(r'$\lambda_{\mathrm{SC}}/\lambda_{\mathrm{hist}}$ (later domains)')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, -.01)); fig.tight_layout(rect=(0, .04, 1, 1))
    for ext in ('png', 'pdf'): fig.savefig(out/f'ratio_curves.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def summarize(a):
    data, records = collect(a.runs); a.out.mkdir(parents=True, exist_ok=True)
    lines = [
        'Loss-weight sensitivity analysis (test partitions, mean±sample SD over seeds; AFR in percentage points).',
        'Paper setting: lambda_sc = 0.5 (later domains and first domain), lambda_h = 1 (`main`). Sensitivity analysis only: no coefficient is changed.', '']
    rows = [['dataset', 'series', 'lambda_sc', 'lambda_h', 'setting', 'n_seeds', *[m[0] for m in MEASURES], *[f'{m[0]}_sd' for m in MEASURES]]]
    for dataset in DATASETS:
        if not any(k[0] == dataset for k in data): continue
        lines += [f'### {NAMES[dataset]}', '']
        for key, title, header, idx in (('sc', 'SupCon weight in the later domains (first domain: paper setting), lambda_h = 1', 'lambda_sc', 1),
                                        ('h', 'L_hist weight, lambda_sc = 0.5', 'lambda_h', 2),
                                        ('sc_first', 'SupCon weight in all domains including the first (lambda_h = 1)', 'lambda_sc (all domains)', 1),
                                        ('grid', '3 x 3 grid', 'lambda_sc / lambda_h', None)):
            sel = [s for s in settings(dataset) if s[0] == key]
            if not sel: continue
            labels = [(f'{s[idx]:g}' if idx else f'{s[1]:g} / {s[2]:g}', s[3]) for s in sel]
            tab = table(dataset, labels, data, header)
            if len(tab) > 2: lines += [f'**{title}**', '', *tab, '']
        ref = reference_rows({42, 43, 44})
        extra = [(f'{n} (paper, reference results)', n) for n in ('no_supcon', 'no_hist') if (dataset, n) in ref]
        merged = dict(data); merged.update({(dataset, f'ref_{n}'): ref[(dataset, n)] for _, n in extra})
        if extra: lines += ['**Paper ablation rows for comparison** (no_supcon: SupCon removed in every domain, first domain untrained; no_hist: L_hist removed)', '',
                            *table(dataset, [(l, f'ref_{n}') for l, n in extra], merged, 'row'), '']
        for key, sc, h, name in settings(dataset):
            seeds_ = data.get((dataset, name), {})
            if not seeds_: continue
            vals = {m: [v[m] for v in seeds_.values()] for m, _ in MEASURES}
            rows.append([dataset, key, f'{sc:g}', f'{h:g}', name, len(seeds_), *[f'{100*statistics.mean(vals[m]):.6f}' for m, _ in MEASURES],
                         *[f'{100*statistics.stdev(vals[m]):.6f}' if len(seeds_) > 1 else '' for m, _ in MEASURES]])
    checks = ['**Consistency checks against the paper ablation rows** (same seeds; identical accuracy matrices expected on the same GPU model)', '']
    ref = read_json(REPO/'results'/'reference_results.json')['results']
    pairs = (('main_sc0.5_h0', 'no_hist', 'L_hist weight 0 vs no_hist'), ('main_sc0_h1_fsc0', 'no_supcon', 'lambda_sc = 0 in all domains vs no_supcon'))
    for mine, theirs, text in pairs:
        for dataset in DATASETS:
            diffs = []
            for r in ref:
                if r['study'] == 'ablation' and r['method'] == theirs and r['dataset'] == dataset and (dataset, mine, r['seed']) in records:
                    new = records[(dataset, mine, r['seed'])]['matrix']
                    diffs.append(max(abs(x-y) for ra, rb in zip(new, r['matrix']) for x, y in zip(ra, rb) if x is not None)*100)
            if diffs: checks.append(f'* {NAMES[dataset]}, {text}: max |difference| of the accuracy matrix per seed (pp) = ' + ', '.join(f'{d:.2f}' for d in diffs))
    checks += ['', '**Joint rescaling of both weights** (AdamW is invariant to the scale of the loss, so only lambda_sc / lambda_h should matter): '
               'max |difference| of the accuracy matrices of settings with the same ratio (later-domain weights only, all seeds), in pp', '']
    for dataset in DATASETS:
        groups = {}
        for key, sc, h, name in settings(dataset):
            if key in ('sc', 'h', 'grid') and sc > 0 and h > 0 and (dataset, name) in data: groups.setdefault(round(sc/h, 9), set()).add((name, sc, h))
        for ratio, members in sorted(groups.items()):
            members = sorted(members, key=lambda m: (m[0] != 'main', m[1], m[2]))
            if len(members) < 2: continue
            worst = max(max(abs(x-y) for ra, rb in zip(records[(dataset, n, sd)]['matrix'], records[(dataset, members[0][0], sd)]['matrix'])
                            for x, y in zip(ra, rb) if x is not None)*100
                        for n, _, _ in members[1:] for sd in data[(dataset, n)] if (dataset, members[0][0], sd) in records)
            checks.append(f'* {NAMES[dataset]}, ratio {ratio:g}: ' + ', '.join(f'({sc:g}, {h:g})' for _, sc, h in members) + f' -> {worst:.2f}')
    (a.out/'summary.md').write_text('\n'.join(lines+checks)+'\n')
    with open(a.out/'summary.csv', 'w', newline='') as f: csv.writer(f).writerows(rows)
    print('\n'.join(lines+checks))
    figure(data, a.out); figure(data, a.out, 'plasticity_stability_with_first_domain', keys=('sc', 'h', 'sc_first')); ratio_figure(data, a.out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=('jobs', 'summarize')); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--runs', type=Path); p.add_argument('--shards', type=int, default=12)
    p.add_argument('--dataset', default='3do,wimans,widar'); p.add_argument('--seed', default='42,43,44')
    a = p.parse_args()
    make_jobs(a) if a.command == 'jobs' else summarize(a)


if __name__ == '__main__':
    main()
