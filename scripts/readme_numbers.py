"""Prints every number quoted in README.md from results/reference_results_same_input.json (percent; mean ± sample SD over seeds 42-44).

  python scripts/readme_numbers.py [--json] [--results results/reference_results_same_input.json]

WiMoT's main-setting trajectories are the ``ablation`` records of method ``main`` with seeds 42-44 (they are also the WiMoT
rows of the ``main`` job list). The baselines of the main comparison, the domain orders, the Widar3.0 orders and rooms, the reduced-label
study and the encoder study receive the 9728-D input of the method (input ``cat9728``; the 9728-D input of the encoder study is built from the
features of the respective encoder); only the baselines of the five-seed study and the Mantis-only comparison receive the 9216-D Mantis
block (input ``m9216``; a lead computed against such a baseline would be flagged with an asterisk). A lead is the mean AFA of WiMoT minus the highest mean AFA among the
baselines that were run in the same setting (all 21 in the main setting; the classic baselines, ConSense and EdgeCL in the other domain
orders; five baselines in the other Widar3.0 orders and rooms; ten with reduced labels). Ties go to the later method of the pool.
The leads with 9216-D baselines are recomputed from results/reference_results.json (``--mantis-only``).
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASETS = ('3do', 'wimans', 'widar')
NAMES = dict(si_er='SI+ER', derpp='DER++', cn_dpm_aligned='CN-DPM', ft='Fine-tuning', mas='MAS', mas_er='MAS+ER', lwf='LwF', lwf_er='LwF+ER',
             er='ER', ewc='EWC', ewc_er='EWC+ER', si='SI', agem='A-GEM', edgecl='EdgeCL', consense='ConSense')
POOL = ['ft', 'ewc', 'si', 'mas', 'lwf', 'er', 'ewc_er', 'si_er', 'mas_er', 'lwf_er', 'derpp', 'agem', 'cn_dpm_aligned', 'duct', 'duct_er',
        'dualcp', 'dualcp_er', 'vanillacp', 'vanillacp_er', 'consense', 'edgecl']


def pick_best(values):
    """Method with the highest mean AFA; ties (to 9 digits) go to the later method of the pool."""
    return max(values, key=lambda m: (round(values[m], 9), POOL.index(m) if m in POOL else -1))


def load(path=None):
    return json.loads(Path(path or REPO/'results'/'reference_results_same_input.json').read_text())['results']


def mean_sd(values):
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else float('nan'))


def cells(records):
    """(dataset, order, fraction, family, method, input) -> {metric: [percent per seed]} for seeds 42-44."""
    out = defaultdict(lambda: defaultdict(list))
    for r in sorted(records, key=lambda r: r['seed']):
        if r['seed'] > 44: continue
        k = (r['dataset'], r['order'], r['fraction'], r['family'], r['method'], r['input'])
        out[k]['AFA'].append(100*r['AFA'])
        if r['AFR'] is not None: out[k]['AFR'].append(100*r['AFR'])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--json', action='store_true'); p.add_argument('--results', default=None)
    a = p.parse_args()
    ref = load(a.results); report = {}
    study = lambda *names: [r for r in ref if r['study'] in names]

    # Table I
    main_cells = cells([r for r in ref if r['family'] != 'joint' and (r['study'] == 'main' or (r['study'] == 'ablation' and r['method'] == 'main'))])
    table = {}
    for d in DATASETS:
        rows = {k[4]: v for k, v in main_cells.items() if k[0] == d}
        baselines = {m: v for m, v in rows.items() if m != 'main'}
        assert len(baselines) == 21, len(baselines)
        best = pick_best({m: statistics.mean(v['AFA']) for m, v in baselines.items()})
        table[d] = dict(WiMoT=rows['main'], strongest=best, **{m: rows[m] for m in baselines})
    report['table1'] = {d: {m: {k: mean_sd(v) for k, v in x.items()} for m, x in t.items() if m != 'strongest'}
                        for d, t in table.items()}
    report['strongest'] = {d: t['strongest'] for d, t in table.items()}
    report['lead_main'] = {d: statistics.mean(t['WiMoT']['AFA'])-statistics.mean(t[t['strongest']]['AFA']) for d, t in table.items()}

    # Mantis-only comparison: the 21 baselines with the 9216-D Mantis block alone (results/reference_results.json)
    old = json.loads((REPO/'results'/'reference_results.json').read_text())['results']
    old_main = cells([r for r in old if r['family'] != 'joint' and (r['study'] == 'main' or (r['study'] == 'ablation' and r['method'] == 'main'))])
    report['mantis_only'] = {}
    for d in DATASETS:
        rows = {k[4]: v for k, v in old_main.items() if k[0] == d}
        base = {m: statistics.mean(v['AFA']) for m, v in rows.items() if m != 'main'}
        b = pick_best(base)
        better = {m: max(base[m], statistics.mean(table[d][m]['AFA'])) for m in base}
        bb = pick_best(better)
        report['mantis_only'][d] = dict(strongest=b, lead=statistics.mean(rows['main']['AFA'])-base[b],
                                        better_variant=bb, lead_better=statistics.mean(rows['main']['AFA'])-better[bb])

    # stage-wise curves of Fig. 3: mean accuracy over the seen domains after each stage (original order)
    curves = {}
    for d in DATASETS:
        for m in ('main', 'si_er', 'derpp', 'cn_dpm_aligned', 'edgecl', 'consense', 'ft'):
            runs = [r for r in ref if r['dataset'] == d and r['method'] == m and r['seed'] <= 44 and r['order'] is None
                    and r['fraction'] is None and r['encoder'] == 'mantis' and r['input'] == ('m9216' if m == 'main' else 'cat9728')
                    and r['study'] == ('ablation' if m == 'main' else 'main')]
            T = len(runs[0]['matrix'])
            curves[f'{d}/{m}'] = [100*statistics.mean(statistics.mean(r['matrix'][s][:s+1]) for r in runs) for s in range(T)]
    report['stage_curves'] = curves

    # joint training (non-continual reference) with the same 9728-dimensional input
    joint = cells([r for r in ref if r['study'] == 'sameinput' and r['family'] == 'joint' and r['input'] == 'cat9728'])
    report['joint'] = {k[0]: mean_sd(v['AFA']) for k, v in joint.items()}

    # leads over the strongest of the 21 baselines in other settings
    def leads(records, key):
        groups = defaultdict(lambda: defaultdict(dict))
        for k, v in cells(records).items():
            groups[key(k)][(k[3], k[4], k[5])] = statistics.mean(v['AFA'])
        out = {}
        for g, v in groups.items():
            ours = {m: x for m, x in v.items() if m[0] == 'method' and m[1] == 'main'}
            base = {m: x for m, x in v.items() if m[0] not in ('method', 'joint')}
            if ours and base:
                best = pick_best({m[1]: x for m, x in base.items()}); star = '*' if [m for m in base if m[1] == best][0][2] == 'm9216' else ''
                out[g] = (list(ours.values())[0]-base[[m for m in base if m[1] == best][0]], best+star)
        return out
    orders = [r for r in study('orders') if r['dataset'] in ('3do', 'wimans')]
    orders += [r for r in ref if r['study'] == 'main' and r['dataset'] in ('3do', 'wimans')]
    orders += [r for r in ref if r['study'] == 'ablation' and r['method'] == 'main' and r['dataset'] in ('3do', 'wimans')]
    report['orders'] = {f'{k[0]}/{k[1] or "012"}': v for k, v in leads(orders, lambda k: (k[0], k[1])).items()}
    wid = [r for r in study('widarext') if r['dataset'] == 'widar']
    report['widar_orders'] = {str(k[1]): v for k, v in leads(wid, lambda k: (k[0], k[1])).items()}
    report['widar_rooms'] = {k[0]: v for k, v in leads([r for r in study('widarext') if r['dataset'] != 'widar'],
                                                       lambda k: (k[0],)).items()}
    few = [r for r in study('fewlabel')]
    report['fewlabel'] = {f'{k[0]}/{k[1]}': v for k, v in leads(few, lambda k: (k[0], k[2])).items()}

    # encoder study: WiMoT minus the strongest of the five baselines (fine-tuning, LwF, SI+ER, DER++, EdgeCL), all reading the 9728-D input of the
    # respective encoder; pretrained MantisV2 from the main comparison (same five baselines)
    FIVE = ('ft', 'lwf', 'si_er', 'derpp', 'edgecl')
    enc = {}
    for d in DATASETS:
        for e in ('mantis', 'mantis_random', 'moment', 'tspulse', 'chronos2', 'timesfm25', 'timesfm30'):
            if e == 'mantis':
                recs = [r for r in ref if r['dataset'] == d and r['seed'] <= 44 and r['order'] is None and r['fraction'] is None
                        and ((r['study'] == 'ablation' and r['method'] == 'main') or (r['study'] == 'main' and r['method'] in FIVE))]
            else:
                recs = [r for r in study('backbone') if r['dataset'] == d and r['seed'] <= 44 and r['encoder'] == e]
            per = defaultdict(list)
            for r in recs: per[r['method']].append(100*r['AFA'])
            base = {m: statistics.mean(per[m]) for m in FIVE}
            best = pick_best(base); ours = statistics.mean(per['main'])
            seeds = [a-b for a, b in zip(per['main'], per[best])]
            enc[f'{d}/{e}'] = dict(wimot=ours, best=best, best_afa=base[best], lead=ours-base[best], seeds_ahead=sum(x > 0 for x in seeds))
    report['encoders'] = enc

    # storage of the class moments (analytic) and the AFA change of the low-rank stores (memory study)
    d, rank = 512, 32
    dense = d*d+d+1; packed = rank*d+rank+1+d+1
    report['rank32_fraction'] = packed/dense
    mem = cells(study('memory')); change = {}
    for ds in DATASETS:
        base = mem[(ds, None, None, 'method', 'main', 'm9216')]['AFA']; low = mem[(ds, None, None, 'method', 'rank32', 'm9216')]['AFA']
        change[ds] = statistics.mean(low)-statistics.mean(base)
    report['rank32_afa_change'] = change
    if a.json: return print(json.dumps(report, indent=1, default=list))

    f = lambda t: f'{t[0]:.2f}±{t[1]:.2f}'
    print('Table I (AFA / AFR, %, mean±SD, seeds 42-44)')
    for m in ('WiMoT', 'si_er', 'derpp', 'cn_dpm_aligned', 'ft'):
        print(f'  {NAMES.get(m, m):10}', '  '.join(f"{d}: {f(report['table1'][d][m]['AFA'])} / "
                                                     f"{f(report['table1'][d][m]['AFR'])}" for d in DATASETS))
    for d in DATASETS:
        s = report['strongest'][d]
        print(f"  strongest baseline {d}: {NAMES.get(s, s)} {f(report['table1'][d][s]['AFA'])}; lead {report['lead_main'][d]:.2f} pp")
    print('Mantis-only comparison, baselines with the 9216-D Mantis block alone:')
    for d, v in report['mantis_only'].items():
        print(f"  {d}: strongest {NAMES.get(v['strongest'], v['strongest'])}, lead {v['lead']:.2f} pp; better of the two input variants "
              f"{NAMES.get(v['better_variant'], v['better_variant'])}, lead {v['lead_better']:.2f} pp")
    print('Stage-wise mean accuracy over seen domains (Fig. 3):')
    for k, v in report['stage_curves'].items(): print(f'  {k:22}', ' '.join(f'{x:.2f}' for x in v))
    print('Joint training, 9728-d input:', {d: f(v) for d, v in report['joint'].items()})
    for name in ('orders', 'widar_orders', 'widar_rooms', 'fewlabel'):
        print(name, {k: f'{v[0]:+.2f} ({NAMES.get(v[1].rstrip("*"), v[1].rstrip("*"))}{"*" if v[1].endswith("*") else ""})' for k, v in sorted(report[name].items())})
    print('Encoder study, WiMoT minus the strongest of five baselines (pp; seeds ahead):')
    for k, v in report['encoders'].items():
        print(f"  {k:22} WiMoT {v['wimot']:.2f}  {NAMES.get(v['best'], v['best']):8} {v['best_afa']:.2f}  lead {v['lead']:+.2f} ({v['seeds_ahead']}/3)")
    print(f"rank-32 storage fraction {100*report['rank32_fraction']:.2f}%; AFA change vs full covariances (pp):",
          {d: round(v, 2) for d, v in report['rank32_afa_change'].items()})


if __name__ == '__main__':
    main()
