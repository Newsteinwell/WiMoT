"""Tables of the same-input study: every baseline reads the 9728-D input (Mantis block + amplitude descriptor) of WiMoT.

  python scripts/same_input_report.py --runs RUNS --out OUT

RUNS holds the study directories written by ``python -m wimot.study sameall_<study> --out RUNS`` (sameall_main, sameall_orders,
sameall_fewlabel, sameall_widarext; one test.json per trajectory). The baseline records of the same-input runs replace the
baseline records (9216-D input) of results/reference_results.json; WiMoT's records are not changed. The script writes, per
study, the new records (same layout as reference_results.json), the tables (csv, markdown), the comparison with the 9216-D
results, and ``old_vs_new.md/csv`` with every summary number of the paper that depends on the baselines.
No test accuracy is used for any choice: coefficients come from configs/locks_sameinput_all.json.
"""
import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
STUDIES = ('main', 'orders', 'fewlabel', 'widarext', 'backbone')
ENCODERS = ('mantis', 'mantis_random', 'moment', 'tspulse', 'chronos2', 'timesfm25', 'timesfm30')
ENAME = dict(mantis='MantisV2 (pretrained)', mantis_random='MantisV2 (random init.)', moment='MOMENT-1-large', tspulse='TSPulse-r1',
             chronos2='Chronos-2', timesfm25='TimesFM 2.5', timesfm30='TimesFM 3.0')
DATASETS = ('3do', 'wimans', 'widar')
DNAME = dict(**{'3do': '3DO', 'wimans': 'WiMANS', 'widar': 'Widar3.0', 'widar_r2': 'Widar3.0 room 2', 'widar_r3': 'Widar3.0 room 3'})
NAME = dict(ft='Fine-tuning', er='ER', ewc='EWC', ewc_er='EWC+ER', si='SI', si_er='SI+ER', mas='MAS', mas_er='MAS+ER', lwf='LwF',
            lwf_er='LwF+ER', derpp='DER++', agem='A-GEM', cn_dpm_aligned='CN-DPM', duct='DUCT', duct_er='DUCT+ER', dualcp='DualCP',
            dualcp_er='DualCP+ER', vanillacp='VanillaCP', vanillacp_er='VanillaCP+ER', consense='ConSense', edgecl='EdgeCL',
            joint='Joint training', main='WiMoT')
FIVE = ('ft', 'lwf', 'si_er', 'derpp', 'edgecl')


def nm(m): return NAME.get(m, m)


# ---------------------------------------------------------------- records
def collect(runs):
    """Records of the same-input runs in the layout of reference_results.json."""
    out = []
    for study in STUDIES:
        for path in sorted((Path(runs)/f'sameall_{study}').rglob('test.json')):
            r = json.loads(path.read_text())
            joint = r['family'] == 'joint'
            encoder = path.relative_to(Path(runs)/f'sameall_{study}').parts[0] if study == 'backbone' else 'mantis'
            out.append(dict(dataset=r['dataset'], seed=r['seed'], order=r['order'], fraction=r['fraction'], encoder=encoder,
                            input=r['input'], study=study, family=r['family'], method=r['method'], coefficient=r['coefficient'],
                            AFA=r['metrics']['AFA'], AFR=r['metrics']['AFR'], matrix=None if joint else r['matrix'],
                            per_domain=r.get('per_domain')))
    return out


def merged(ref, new):
    """Reference records with the baseline records of the four studies replaced by the same-input records, stream by stream
    (streams without same-input runs keep their 9216-D baselines; the backbone study is replaced per encoder and stream)."""
    have = {(r['study'], r['dataset'], r['encoder'] if r['study'] == 'backbone' else None) for r in new}
    keep = [r for r in ref if not (r['study'] in STUDIES and r['family'] != 'method'
                                   and (r['study'], r['dataset'], r['encoder'] if r['study'] == 'backbone' else None) in have)]
    return keep+new


def ms(v):
    v = np.asarray(v, float); return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else float('nan'))


def fmt(t, d=2): return f'{t[0]:.{d}f} ± {t[1]:.{d}f}'


def cells(records, key):
    out = defaultdict(dict)
    for r in sorted(records, key=lambda r: r['seed']):
        if r['seed'] > 44: continue
        out[key(r)][r['seed']] = r
    return out


def is_ours(r): return r['family'] == 'method' and r['method'] == 'main'


def ours_main(ref):
    return [r for r in ref if r['study'] == 'ablation' and is_ours(r) and r['seed'] <= 44]


# ---------------------------------------------------------------- main table
def main_table(recs):
    """{dataset: {method: {'AFA': [..], 'AFR': [..], 'seeds': [..]}}}, percent; 21 baselines + WiMoT."""
    out = {d: {} for d in DATASETS}
    rows = [r for r in recs if (r['study'] == 'main' and r['family'] not in ('joint',)) or (r['study'] == 'ablation' and is_ours(r))]
    for (d, m), g in cells(rows, lambda r: (r['dataset'], r['method'])).items():
        seeds = sorted(g)
        out[d][m] = dict(seeds=seeds, AFA=[100*g[s]['AFA'] for s in seeds], AFR=[100*g[s]['AFR'] for s in seeds],
                         matrix=[g[s]['matrix'] for s in seeds])
    return out


def strongest(cell, exclude=('main',)):
    base = {m: v for m, v in cell.items() if m not in exclude}
    best = max(base, key=lambda m: st.mean(base[m]['AFA']))
    return best, base


def paired(cell, best):
    a, b = cell['main'], cell[best]
    d = [x-y for x, y in zip(a['AFA'], b['AFA'])]
    return st.mean(d), (st.stdev(d) if len(d) > 1 else float('nan')), sum(x > 0 for x in d), len(d)


def backbone_cells(recs, table):
    """{(dataset, encoder): {method: {'AFA': [..], 'AFR': [..]}}} in percent for WiMoT and the five baselines; the pretrained MantisV2 row
    comes from the main table (``table``)."""
    out = {}
    for d in DATASETS:
        out[(d, 'mantis')] = {m: v for m, v in table[d].items() if m == 'main' or m in FIVE}
    rows = [r for r in recs if r['study'] == 'backbone' and r['encoder'] != 'mantis' and r['seed'] <= 44 and (is_ours(r) or r['method'] in FIVE)]
    for (d, e, m), g in cells(rows, lambda r: (r['dataset'], r['encoder'], r['method'])).items():
        if len(g) != 3: continue
        s_ = sorted(g); out.setdefault((d, e), {})[m] = dict(seeds=s_, AFA=[100*g[x]['AFA'] for x in s_], AFR=[100*g[x]['AFR'] for x in s_])
    return {k: v for k, v in out.items() if 'main' in v and all(m in v for m in FIVE)}


# ---------------------------------------------------------------- other settings
def lead_groups(recs, studies, key):
    """{group: {'ours': [AFA per seed], 'base': {(family, method): [AFA per seed]}, 'afr': ...}} in percent."""
    out = defaultdict(lambda: dict(ours={}, base=defaultdict(dict)))
    for r in recs:
        if r['seed'] > 44: continue
        if r['study'] in studies and r['input'] in ('m9216', 'cat9728') and r['family'] != 'joint':
            g = key(r)
            if g is None: continue
            if is_ours(r): out[g]['ours'][r['seed']] = (100*r['AFA'], 100*r['AFR'])
            elif r['family'] != 'method': out[g]['base'][(r['family'], r['method'])][r['seed']] = (100*r['AFA'], 100*r['AFR'])
    res = {}
    for g, v in out.items():
        base = {m: s for m, s in v['base'].items() if len(s) == 3}
        if len(v['ours']) != 3 or not base: continue
        best = max(base, key=lambda m: st.mean(x[0] for x in base[m].values()))
        diffs = [v['ours'][s][0]-base[best][s][0] for s in (42, 43, 44)]
        res[g] = dict(ours=st.mean(x[0] for x in v['ours'].values()), ours_afr=st.mean(x[1] for x in v['ours'].values()),
                      best=best[1], best_afa=st.mean(x[0] for x in base[best].values()),
                      best_afr=st.mean(x[1] for x in base[best].values()), lead=st.mean(diffs),
                      lead_sd=st.stdev(diffs), wins=sum(d > 0 for d in diffs), n_base=len(base),
                      mean_afr=st.mean(st.mean(x[1] for x in b.values()) for b in base.values()),
                      base_afa={m[1]: st.mean(x[0] for x in b.values()) for m, b in base.items()})
    return res


def order_key(r):
    if r['dataset'] not in ('3do', 'wimans'): return None
    if r['study'] == 'orders': return (r['dataset'], r['order'])
    if r['study'] in ('main', 'ablation') and r['order'] is None and r['fraction'] is None and (r['study'] == 'main' or is_ours(r)):
        return (r['dataset'], '012')
    return None


def orders_leads(recs):
    rows = [r for r in recs if r['study'] in ('orders', 'main') or (r['study'] == 'ablation' and is_ours(r))]
    # WiMoT in the other orders is in the study 'orders' (family method)
    return lead_groups(rows, ('orders', 'main', 'ablation'), order_key)


def widar_key(r):
    if r['study'] == 'widarext' and r['dataset'] == 'widar': return ('widar', r['order'])
    if r['study'] == 'widarext': return (r['dataset'], None)
    if r['dataset'] == 'widar' and r['order'] is None and r['fraction'] is None and (r['study'] == 'main' or (r['study'] == 'ablation' and is_ours(r))):
        return ('widar', '012345678')
    return None


def widar_leads(recs):
    return lead_groups(recs, ('widarext', 'main', 'ablation'), widar_key)


def few_key(r):
    if r['study'] == 'fewlabel': return (r['dataset'], r['fraction'])
    return None


def few_leads(recs):
    return lead_groups(recs, ('fewlabel',), few_key)


# ---------------------------------------------------------------- plasticity and stability
def pls(matrix, afr):
    t = len(matrix)
    return 100*st.mean(matrix[k][k] for k in range(1, t)), 100*st.mean(matrix[-1][k] for k in range(t-1)), afr


def plasticity(table):
    out = {}
    for d, cell in table.items():
        for m, v in cell.items():
            vals = [pls(mat, a) for mat, a in zip(v['matrix'], v['AFR'])]
            out[(d, m)] = {k: ms([x[i] for x in vals]) for i, k in enumerate(('new', 'old', 'afr'))}
    return out


# ---------------------------------------------------------------- 42 settings: lead vs baseline forgetting
def lead_points(recs, tag):
    """The 42 settings of the lead-vs-forgetting table; ``recs`` decides whether the baselines read the 9216-D (reference) or
    the 9728-D input (merged same-input records)."""
    P = []
    mt = main_table(recs)
    for d in DATASETS:
        cell = mt[d]; best, base = strongest(cell)
        P.append(dict(ds=d, setting='main', ours=st.mean(cell['main']['AFA']), best=best, best_afr=st.mean(base[best]['AFR']),
                      mean_afr=st.mean(st.mean(v['AFR']) for v in base.values()), lead=st.mean(cell['main']['AFA'])-st.mean(base[best]['AFA']),
                      rerun=True))
    for (d, o), v in sorted(orders_leads(recs).items()):
        if o != '012': P.append(dict(ds=d, setting=f'order {o}', ours=v['ours'], best=v['best'], best_afr=v['best_afr'],
                                     mean_afr=v['mean_afr'], lead=v['lead'], rerun=True))
    for (d, f), v in sorted(few_leads(recs).items()):
        P.append(dict(ds=d, setting=f'labels {int(100*f)}%', ours=v['ours'], best=v['best'], best_afr=v['best_afr'],
                      mean_afr=v['mean_afr'], lead=v['lead'], rerun=True))
    for (d, o), v in sorted(widar_leads(recs).items(), key=lambda t: str(t[0])):
        if o == '012345678': continue
        P.append(dict(ds='widar', setting=f'order {o}' if o else d.replace('widar_', 'room '), ours=v['ours'], best=v['best'],
                      best_afr=v['best_afr'], mean_afr=v['mean_afr'], lead=v['lead'], rerun=True))
    # encoders (baselines read the 9728-D encoder features in the new version)
    enc = lead_groups([r for r in recs if r['study'] == 'backbone' and r['encoder'] != 'mantis'], ('backbone',),
                      lambda r: (r['dataset'], r['encoder']))
    for (d, e), v in sorted(enc.items()):
        P.append(dict(ds=d, setting=f'enc {e}', ours=v['ours'], best=v['best'], best_afr=v['best_afr'], mean_afr=v['mean_afr'],
                      lead=v['lead'], rerun=True))
    return P


def correlations(P):
    from scipy import stats
    out = dict(n=len(P), leads=sum(p['lead'] > 0 for p in P))
    y = np.array([p['lead'] for p in P])
    for key in ('best_afr', 'mean_afr'):
        x = np.array([p[key] for p in P])
        out[key] = dict(pearson=float(stats.pearsonr(x, y)[0]), spearman=float(stats.spearmanr(x, y)[0]))
        xs, ys = [], []
        for d in DATASETS:
            q = [p for p in P if p['ds'] == d]; xq = np.array([p[key] for p in q]); yq = np.array([p['lead'] for p in q])
            xs += list(xq-xq.mean()); ys += list(yq-yq.mean())
            out[key][f'spearman_{d}'] = float(stats.spearmanr(xq, yq)[0])
        out[key]['demeaned_pearson'] = float(stats.pearsonr(xs, ys)[0]); out[key]['demeaned_spearman'] = float(stats.spearmanr(xs, ys)[0])
    return out


# ---------------------------------------------------------------- output helpers
def write_csv(path, header, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def md_table(header, rows):
    return '\n'.join(['| '+' | '.join(header)+' |', '|'+'|'.join('---' for _ in header)+'|']+['| '+' | '.join(str(c) for c in r)+' |' for r in rows])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reference', type=Path, default=REPO/'results'/'reference_results.json')
    a = p.parse_args(); out = a.out
    ref = json.loads(a.reference.read_text())['results']; new = collect(a.runs)
    recs_new = merged(ref, new); recs_old = ref
    report = {}; old_vs_new = []     # rows: (section, quantity, old, new, note)

    # per-study new records
    for s in STUDIES:
        rows = [r for r in new if r['study'] == s]
        (out/s).mkdir(parents=True, exist_ok=True)
        (out/s/'new_results.json').write_text(json.dumps(dict(description=f'Baseline trajectories of the {s} study with the 9728-D input '
                                                              '(same layout as results/reference_results.json); WiMoT unchanged.',
                                                              results=rows)))

    # ------------------------------------------------ Table I
    tn, to = main_table(recs_new), main_table(recs_old)
    lines = []; csv_rows = []; cmp_rows = []; md = ['# Table I with the 9728-D input for every baseline\n',
                                                     'AFA (%) and AFR (pp), mean ± sample SD over seeds 42-44; sorted by AFA within each stream.\n']
    joint = defaultdict(list)
    for r in new:
        if r['family'] == 'joint' and r['study'] == 'main': joint[r['dataset']].append(100*r['AFA'])
    summary = {}
    for d in DATASETS:
        cell = tn[d]; best, base = strongest(cell); bo, bbase = strongest(to[d])
        lead = paired(cell, best)
        old_lead = paired(to[d], bo)
        summary[d] = dict(best=best, best_afa=st.mean(base[best]['AFA']), best_afr=st.mean(base[best]['AFR']), lead=lead,
                          old_best=bo, old_best_afa=st.mean(bbase[bo]['AFA']), old_lead=old_lead, n=len(base),
                          wimot_afa=ms(cell['main']['AFA']), wimot_afr=ms(cell['main']['AFR']))
        md.append(f'\n## {DNAME[d]}\n')
        table = []
        order = sorted(cell, key=lambda m: -st.mean(cell[m]['AFA']))
        for m in order:
            v = cell[m]; o = to[d].get(m)
            dafa = st.mean(v['AFA'])-st.mean(o['AFA']) if o and m != 'main' else None
            dafr = st.mean(v['AFR'])-st.mean(o['AFR']) if o and m != 'main' else None
            table.append((nm(m), fmt(ms(v['AFA'])), fmt(ms(v['AFR'])), '' if o is None or m == 'main' else fmt(ms(o['AFA'])),
                          '' if o is None or m == 'main' else fmt(ms(o['AFR'])), '' if dafa is None else f'{dafa:+.2f}',
                          '' if dafr is None else f'{dafr:+.2f}'))
            csv_rows.append((d, m, ms(v['AFA'])[0], ms(v['AFA'])[1], ms(v['AFR'])[0], ms(v['AFR'])[1],
                             '' if o is None or m == 'main' else ms(o['AFA'])[0], '' if o is None or m == 'main' else ms(o['AFR'])[0],
                             '' if dafa is None else dafa, '' if dafr is None else dafr))
        if d in joint: table.append(('Joint training (reference)', fmt(ms(joint[d])), '–', '', '', '', ''))
        md.append(md_table(['Method', 'AFA new (9728-D)', 'AFR new', 'AFA old (9216-D)', 'AFR old', 'ΔAFA (pp)', 'ΔAFR (pp)'], table))
        md.append(f"\nStrongest of the {len(base)} baselines: **{nm(best)}** (AFA {st.mean(base[best]['AFA']):.2f}, AFR {st.mean(base[best]['AFR']):.2f}); "
                  f"WiMoT lead {lead[0]:+.2f} ± {lead[1]:.2f} pp ({lead[2]}/{lead[3]} seeds). Previous (9216-D): strongest {nm(bo)} "
                  f"{st.mean(bbase[bo]['AFA']):.2f}, lead {old_lead[0]:+.2f} pp. "
                  + ('**The strongest baseline changed.**' if best != bo else 'The strongest baseline is unchanged.'))
    (out/'main').mkdir(parents=True, exist_ok=True)
    (out/'main'/'table1_new.md').write_text('\n'.join(md)+'\n')
    write_csv(out/'main'/'table1_new.csv', ('dataset', 'method', 'AFA_mean', 'AFA_sd', 'AFR_mean', 'AFR_sd', 'AFA_old9216', 'AFR_old9216',
                                             'dAFA_pp', 'dAFR_pp'), csv_rows)
    cols = ('dataset', 'method', 'AFA_mean', 'AFA_sd', 'AFR_mean', 'AFR_sd', 'AFA_old9216', 'AFR_old9216', 'dAFA_pp', 'dAFR_pp')
    (out/'main'/'table1_new.json').write_text(json.dumps(dict(
        description='Table I with the 9728-D input for every baseline; AFA in %, AFR in pp, mean and sample SD over seeds 42-44; old9216 = earlier baselines (9216-D).',
        strongest={d: dict(method=summary[d]['best'], AFA=summary[d]['best_afa'], AFR=summary[d]['best_afr'], lead_mean=summary[d]['lead'][0], lead_sd=summary[d]['lead'][1],
                           seeds_ahead=summary[d]['lead'][2], old_method=summary[d]['old_best'], old_AFA=summary[d]['old_best_afa'], old_lead=summary[d]['old_lead'][0]) for d in DATASETS},
        rows=[dict(zip(cols, r)) for r in csv_rows]), indent=1))
    report['main'] = summary
    for d in DATASETS:
        s = summary[d]
        old_vs_new += [('Table I', f'{DNAME[d]}: strongest baseline', nm(s['old_best']), nm(s['best']), ''),
                       ('Table I', f'{DNAME[d]}: strongest baseline AFA', f"{s['old_best_afa']:.2f}", f"{s['best_afa']:.2f}", ''),
                       ('Table I', f'{DNAME[d]}: WiMoT lead (pp)', f"{s['old_lead'][0]:+.2f}", f"{s['lead'][0]:+.2f}",
                        f"paired ±SD {s['lead'][1]:.2f}, {s['lead'][2]}/{s['lead'][3]} seeds"),
                       ('Table I', f'{DNAME[d]}: AFR of the strongest baseline', f"{st.mean(strongest(to[d])[1][s['old_best']]['AFR']):.2f}",
                        f"{s['best_afr']:.2f}", '')]
        if d in joint:
            old_j = [100*r['AFA'] for r in ref if r['study'] == 'sameinput' and r['family'] == 'joint' and r['input'] == 'cat9728' and r['dataset'] == d]
            old_vs_new.append(('Table I', f'{DNAME[d]}: joint-training reference AFA', f'{st.mean(old_j):.2f}', f'{st.mean(joint[d]):.2f}', 'same 9728-D input in both'))

    # comparison with the earlier same-input records (sameinput study, 9728-D)
    cmp = []
    for r in ref:
        if r['study'] == 'sameinput' and r['input'] == 'cat9728' and r['family'] not in ('method', 'joint'): cmp.append(r)
    prev = cells(cmp, lambda r: (r['dataset'], r['method']))
    cmp_md = ['# New runs against the earlier 9728-D runs (study `sameinput`, results/reference_results.json)\n',
              'Same input and, where re-selected, the same coefficient; differences come from the GPU and, for ConSense/EdgeCL, from the seeded dropout.\n']
    cmp_rows_t = []
    for (d, m), g in sorted(prev.items()):
        v = tn[d].get(m)
        if not v: continue
        oa, orr = ms([100*g[s]['AFA'] for s in sorted(g)]), ms([100*g[s]['AFR'] for s in sorted(g)])
        cmp_rows_t.append((DNAME[d], nm(m), fmt(oa), fmt(ms(v['AFA'])), f'{ms(v["AFA"])[0]-oa[0]:+.2f}', fmt(orr), fmt(ms(v['AFR'])),
                           f'{ms(v["AFR"])[0]-orr[0]:+.2f}',
                           ', '.join(f'{100*(g[s]["AFA"])-x:+.2f}' for s, x in zip(sorted(g), v['AFA'])) ))
    cmp_md.append(md_table(['Stream', 'Method', 'AFA earlier', 'AFA new', 'Δ', 'AFR earlier', 'AFR new', 'Δ', 'per-seed ΔAFA (earlier − new)'], cmp_rows_t))
    (out/'main'/'vs_earlier_9728_runs.md').write_text('\n'.join(cmp_md)+'\n')
    write_csv(out/'main'/'vs_earlier_9728_runs.csv', ('stream', 'method', 'AFA_earlier', 'AFA_new', 'dAFA', 'AFR_earlier', 'AFR_new', 'dAFR', 'per_seed'), cmp_rows_t)

    # ------------------------------------------------ plasticity and stability
    pn_, po_ = plasticity(tn), plasticity(to)
    rows = []; md = ['# Plasticity and stability (9728-D input for every baseline)\n',
                     'New: mean accuracy on the domain just learned (updates 2..T); earlier: mean accuracy on the earlier domains after the last update; AFR in pp. Mean ± SD over seeds 42-44.\n']
    pls_summary = {}
    for d in DATASETS:
        t = []
        order = sorted([k for k in pn_ if k[0] == d], key=lambda k: -pn_[k]['new'][0])
        for (_, m) in order:
            v, o = pn_[(d, m)], po_.get((d, m))
            t.append((nm(m), fmt(v['new']), fmt(v['old']), fmt(v['afr']),
                      '' if o is None or m == 'main' else f"{v['new'][0]-o['new'][0]:+.2f} / {v['old'][0]-o['old'][0]:+.2f} / {v['afr'][0]-o['afr'][0]:+.2f}"))
            rows.append((d, m, v['new'][0], v['new'][1], v['old'][0], v['old'][1], v['afr'][0], v['afr'][1],
                         '' if o is None else o['new'][0], '' if o is None else o['old'][0], '' if o is None else o['afr'][0]))
        md += [f'\n## {DNAME[d]}\n', md_table(['Method', 'New-domain acc.', 'Earlier-domain acc.', 'AFR', 'Δ new / earlier / AFR vs 9216-D'], t)]
        for tag, P in (('new', pn_), ('old', po_)):
            w = P[(d, 'main')]; base = {m: v for (dd, m), v in P.items() if dd == d and m != 'main'}
            bn = max(base, key=lambda m: base[m]['new'][0]); bo_ = max(base, key=lambda m: base[m]['old'][0]); ba = min(base, key=lambda m: base[m]['afr'][0])
            pls_summary[(d, tag)] = dict(wimot=w, best_new=(bn, base[bn]['new'][0], base[bn]['afr'][0]), best_old=(bo_, base[bo_]['old'][0]),
                                         min_afr=(ba, base[ba]['afr'][0], base[ba]['new'][0]),
                                         higher_new=sorted((nm(m), round(v['new'][0], 2), round(v['afr'][0], 2)) for m, v in base.items() if v['new'][0] >= w['new'][0]),
                                         lower_afr=sorted((nm(m), round(v['new'][0], 2), round(v['afr'][0], 2)) for m, v in base.items() if v['afr'][0] <= w['afr'][0]))
    write_csv(out/'main'/'plasticity_stability_new.csv', ('dataset', 'method', 'new_mean', 'new_sd', 'earlier_mean', 'earlier_sd', 'AFR_mean', 'AFR_sd',
                                                          'new_old9216', 'earlier_old9216', 'AFR_old9216'), rows)
    md.append('\n## Column-wise best baseline and criteria\n')
    t = []
    for d in DATASETS:
        for tag in ('old', 'new'):
            s = pls_summary[(d, tag)]
            t.append((DNAME[d], '9216-D' if tag == 'old' else '9728-D', f"{s['wimot']['new'][0]:.2f}", f"{nm(s['best_new'][0])} {s['best_new'][1]:.2f} (AFR {s['best_new'][2]:.2f})",
                      f"{s['best_new'][1]-s['wimot']['new'][0]:+.2f}", f"{nm(s['best_old'][0])} {s['best_old'][1]:.2f}", f"{s['wimot']['old'][0]:.2f}",
                      f"{nm(s['min_afr'][0])} {s['min_afr'][1]:.2f} (new {s['min_afr'][2]:.2f})", f"{s['wimot']['afr'][0]:.2f}",
                      '; '.join(f'{n} {a:.2f}/{r:.2f}' for n, a, r in s['higher_new']) or '-', '; '.join(f'{n} {a:.2f}/{r:.2f}' for n, a, r in s['lower_afr']) or '-'))
    md.append(md_table(['Stream', 'Input', 'WiMoT new', 'Best baseline new-domain acc.', 'Gap to WiMoT (pp)', 'Best baseline earlier-domain acc.', 'WiMoT earlier',
                        'Lowest-AFR baseline', 'WiMoT AFR', 'Baselines with higher new-domain acc. (new/AFR)', 'Baselines with AFR ≤ WiMoT (new/AFR)'], t))
    names = {m for (_, m) in pn_}
    for tag, P in (('9216-D', po_), ('9728-D', pn_)):
        ok = []
        for m in sorted(names):
            good = True
            for d in DATASETS:
                best = max(v['new'][0] for (dd, mm), v in P.items() if dd == d)
                v = P.get((d, m)); good &= v is not None and v['new'][0] >= best-2.5 and v['afr'][0] <= 1.0
            if good: ok.append(nm(m))
        md.append(f'\nMethods within 2.5 pp of the best new-domain accuracy with AFR ≤ 1 pp on all three streams ({tag}): {", ".join(ok) or "none"}.')
        pls_summary[('criterion', tag)] = ok
    (out/'main'/'plasticity_stability_new.md').write_text('\n'.join(md)+'\n')
    report['plasticity'] = {f'{k[0]}|{k[1]}': v for k, v in pls_summary.items()}
    for d in DATASETS:
        for m in ('ft', 'si_er', 'derpp', 'cn_dpm_aligned'):
            o, n_ = po_[(d, m)], pn_[(d, m)]
            old_vs_new.append(('Plasticity/stability', f'{DNAME[d]} {nm(m)}: new / earlier / AFR', f"{o['new'][0]:.2f} / {o['old'][0]:.2f} / {o['afr'][0]:.2f}",
                               f"{n_['new'][0]:.2f} / {n_['old'][0]:.2f} / {n_['afr'][0]:.2f}", ''))
        so, sn = pls_summary[(d, 'old')], pls_summary[(d, 'new')]
        old_vs_new += [('Plasticity/stability', f'{DNAME[d]}: best baseline new-domain accuracy (gap to WiMoT, its AFR)',
                        f"{nm(so['best_new'][0])} {so['best_new'][1]:.2f} ({so['best_new'][1]-so['wimot']['new'][0]:+.2f}; AFR {so['best_new'][2]:.2f})",
                        f"{nm(sn['best_new'][0])} {sn['best_new'][1]:.2f} ({sn['best_new'][1]-sn['wimot']['new'][0]:+.2f}; AFR {sn['best_new'][2]:.2f})", ''),
                       ('Plasticity/stability', f'{DNAME[d]}: best baseline earlier-domain accuracy', f"{nm(so['best_old'][0])} {so['best_old'][1]:.2f}",
                        f"{nm(sn['best_old'][0])} {sn['best_old'][1]:.2f}", f"WiMoT {sn['wimot']['old'][0]:.2f}"),
                       ('Plasticity/stability', f'{DNAME[d]}: lowest-AFR baseline (AFR; its new-domain acc.)', f"{nm(so['min_afr'][0])} {so['min_afr'][1]:.2f}; {so['min_afr'][2]:.2f}",
                        f"{nm(sn['min_afr'][0])} {sn['min_afr'][1]:.2f}; {sn['min_afr'][2]:.2f}", f"WiMoT AFR {sn['wimot']['afr'][0]:.2f}"),
                       ('Plasticity/stability', f'{DNAME[d]}: baselines with higher new-domain acc. than WiMoT ({so["wimot"]["new"][0]:.2f})',
                        '; '.join(f'{n} {a:.2f}/{r:.2f}' for n, a, r in so['higher_new']) or '-', '; '.join(f'{n} {a:.2f}/{r:.2f}' for n, a, r in sn['higher_new']) or '-', 'name acc./AFR'),
                       ('Plasticity/stability', f'{DNAME[d]}: baselines with AFR ≤ WiMoT',
                        '; '.join(f'{n} {a:.2f}/{r:.2f}' for n, a, r in so['lower_afr']) or '-', '; '.join(f'{n} {a:.2f}/{r:.2f}' for n, a, r in sn['lower_afr']) or '-', 'name new-acc./AFR')]
    old_vs_new.append(('Plasticity/stability', 'methods within 2.5 pp of best new-domain acc. with AFR ≤ 1 pp on all streams', ', '.join(pls_summary[('criterion', '9216-D')]) or 'none',
                       ', '.join(pls_summary[('criterion', '9728-D')]) or 'none', ''))

    # ------------------------------------------------ orders
    ol_n, ol_o = orders_leads(recs_new), orders_leads(recs_old)
    rows = []; md = ['# Domain orders (3DO and WiMANS), baselines with the 9728-D input\n',
                     'Original order (012): the 21 baselines of the main table; other orders: the 12 classic baselines, ConSense and EdgeCL. Lead = WiMoT mean AFA minus strongest baseline mean AFA.\n']
    t = []
    for g in sorted(ol_n):
        n_, o_ = ol_n[g], ol_o.get(g)
        t.append((DNAME[g[0]], g[1], f"{n_['ours']:.2f}", nm(n_['best']), f"{n_['best_afa']:.2f}", f"{n_['best_afr']:.2f}", f"{n_['lead']:+.2f} ± {n_['lead_sd']:.2f} ({n_['wins']}/3)",
                  '' if not o_ else f"{nm(o_['best'])} {o_['best_afa']:.2f}", '' if not o_ else f"{o_['lead']:+.2f}"))
        rows.append((g[0], g[1], n_['ours'], n_['best'], n_['best_afa'], n_['best_afr'], n_['lead'], n_['lead_sd'], n_['wins'], n_['n_base'],
                     '' if not o_ else o_['best'], '' if not o_ else o_['best_afa'], '' if not o_ else o_['lead']))
        old_vs_new.append(('Orders', f'{DNAME[g[0]]} order {g[1]}: lead (strongest baseline)', f"{o_['lead']:+.2f} ({nm(o_['best'])})" if o_ else '', f"{n_['lead']:+.2f} ({nm(n_['best'])})", ''))
    md.append(md_table(['Stream', 'Order', 'WiMoT AFA', 'Strongest baseline (new)', 'its AFA', 'its AFR', 'Lead (paired ± SD; seeds ahead)', 'Strongest (9216-D)', 'Lead (9216-D)'], t))
    for d in ('3do', 'wimans'):
        for tag, L in (('9216-D', ol_o), ('9728-D', ol_n)):
            leads = [v['lead'] for (dd, o), v in L.items() if dd == d]
            md.append(f'\n{DNAME[d]} {tag}: leads over the six orders from {min(leads):+.2f} to {max(leads):+.2f} pp; WiMoT ahead in {sum(x > 0 for x in leads)} of {len(leads)} orders.')
        lo, ln = [v['lead'] for (dd, o), v in ol_o.items() if dd == d], [v['lead'] for (dd, o), v in ol_n.items() if dd == d]
        old_vs_new.append(('Orders', f'{DNAME[d]}: range of lead over the six orders (pp); orders ahead',
                          f'{min(lo):+.2f} to {max(lo):+.2f}; {sum(x > 0 for x in lo)}/{len(lo)}', f'{min(ln):+.2f} to {max(ln):+.2f}; {sum(x > 0 for x in ln)}/{len(ln)}', ''))
    (out/'orders').mkdir(parents=True, exist_ok=True)
    (out/'orders'/'orders_new.md').write_text('\n'.join(md)+'\n')
    write_csv(out/'orders'/'orders_new.csv', ('dataset', 'order', 'WiMoT_AFA', 'strongest_new', 'strongest_AFA', 'strongest_AFR', 'lead', 'lead_sd', 'seeds_ahead',
                                              'n_baselines', 'strongest_old9216', 'strongest_AFA_old9216', 'lead_old9216'), rows)
    # per-baseline table of the orders study
    ob = cells([r for r in new if r['study'] == 'orders'], lambda r: (r['dataset'], r['order'], r['method']))
    ob_old = cells([r for r in ref if r['study'] == 'orders' and r['family'] != 'method'], lambda r: (r['dataset'], r['order'], r['method']))
    rows = []
    for k, g in sorted(ob.items()):
        s = sorted(g); o = ob_old.get(k)
        rows.append((*k, ms([100*g[x]['AFA'] for x in s])[0], ms([100*g[x]['AFA'] for x in s])[1], ms([100*g[x]['AFR'] for x in s])[0],
                     '' if not o else ms([100*o[x]['AFA'] for x in sorted(o)])[0], '' if not o else ms([100*o[x]['AFR'] for x in sorted(o)])[0], len(s)))
    write_csv(out/'orders'/'orders_baselines_new.csv', ('dataset', 'order', 'method', 'AFA_mean', 'AFA_sd', 'AFR_mean', 'AFA_old9216', 'AFR_old9216', 'n_seeds'), rows)

    # ------------------------------------------------ few labels
    fl_n, fl_o = few_leads(recs_new), few_leads(recs_old)
    rows = []; md = ['# Reduced labels, baselines with the 9728-D input\n', 'Ten baselines: Fine-tuning, SI, MAS, LwF, ConSense, ER, SI+ER, LwF+ER, DER++, EdgeCL.\n']; t = []
    for g in sorted(fl_n):
        n_, o_ = fl_n[g], fl_o.get(g)
        t.append((DNAME[g[0]], f'{int(100*g[1])}%', f"{n_['ours']:.2f}", nm(n_['best']), f"{n_['best_afa']:.2f}", f"{n_['best_afr']:.2f}", f"{n_['lead']:+.2f} ± {n_['lead_sd']:.2f} ({n_['wins']}/3)",
                  '' if not o_ else f"{nm(o_['best'])} {o_['best_afa']:.2f}", '' if not o_ else f"{o_['lead']:+.2f}"))
        rows.append((g[0], g[1], n_['ours'], n_['best'], n_['best_afa'], n_['best_afr'], n_['lead'], n_['lead_sd'], n_['wins'], '' if not o_ else o_['best'],
                     '' if not o_ else o_['best_afa'], '' if not o_ else o_['lead']))
        old_vs_new.append(('Few labels', f'{DNAME[g[0]]} {int(100*g[1])}% labels: lead (strongest baseline)', f"{o_['lead']:+.2f} ({nm(o_['best'])})" if o_ else '', f"{n_['lead']:+.2f} ({nm(n_['best'])})", ''))
    md.append(md_table(['Stream', 'Labels', 'WiMoT AFA', 'Strongest baseline (new)', 'its AFA', 'its AFR', 'Lead (paired ± SD; seeds ahead)', 'Strongest (9216-D)', 'Lead (9216-D)'], t))
    for d in ('3do', 'widar'):
        lo, ln = [v['lead'] for (dd, f), v in fl_o.items() if dd == d], [v['lead'] for (dd, f), v in fl_n.items() if dd == d]
        old_vs_new.append(('Few labels', f'{DNAME[d]}: range of lead over 25% and 10% labels (pp)', f'{min(lo):+.2f} to {max(lo):+.2f}', f'{min(ln):+.2f} to {max(ln):+.2f}', ''))
    (out/'fewlabel').mkdir(parents=True, exist_ok=True)
    (out/'fewlabel'/'fewlabel_new.md').write_text('\n'.join(md)+'\n')
    write_csv(out/'fewlabel'/'fewlabel_new.csv', ('dataset', 'fraction', 'WiMoT_AFA', 'strongest_new', 'strongest_AFA', 'strongest_AFR', 'lead', 'lead_sd', 'seeds_ahead',
                                                  'strongest_old9216', 'strongest_AFA_old9216', 'lead_old9216'), rows)
    fb = cells([r for r in new if r['study'] == 'fewlabel'], lambda r: (r['dataset'], r['fraction'], r['method']))
    fb_old = cells([r for r in ref if r['study'] == 'fewlabel' and r['family'] != 'method'], lambda r: (r['dataset'], r['fraction'], r['method']))
    rows = []
    for k, g in sorted(fb.items(), key=str):
        s = sorted(g); o = fb_old.get(k)
        rows.append((*k, ms([100*g[x]['AFA'] for x in s])[0], ms([100*g[x]['AFA'] for x in s])[1], ms([100*g[x]['AFR'] for x in s])[0],
                     '' if not o else ms([100*o[x]['AFA'] for x in sorted(o)])[0], '' if not o else ms([100*o[x]['AFR'] for x in sorted(o)])[0], len(s)))
    write_csv(out/'fewlabel'/'fewlabel_baselines_new.csv', ('dataset', 'fraction', 'method', 'AFA_mean', 'AFA_sd', 'AFR_mean', 'AFA_old9216', 'AFR_old9216', 'n_seeds'), rows)

    # ------------------------------------------------ widar orders and rooms
    wl_n, wl_o = widar_leads(recs_new), widar_leads(recs_old)
    rows = []; md = ['# Widar3.0 other user orders and rooms\n', 'Five baselines: Fine-tuning, LwF, SI+ER, DER++, EdgeCL. Rooms 2 and 3: the same five baselines, rerun with the 9728-D input of the room features.\n']; t = []
    for g in sorted(wl_n, key=str):
        n_, o_ = wl_n[g], wl_o.get(g)
        rerun = True
        t.append((g[0] if g[1] is None else f'widar order {g[1]}', f"{n_['ours']:.2f}", nm(n_['best']), f"{n_['best_afa']:.2f}", f"{n_['best_afr']:.2f}", f"{n_['lead']:+.2f} ± {n_['lead_sd']:.2f} ({n_['wins']}/3)",
                  '' if not o_ else f"{nm(o_['best'])} {o_['best_afa']:.2f}", '' if not o_ else f"{o_['lead']:+.2f}", 'main table' if g[1] == '012345678' else 'rerun (9728-D)'))
        rows.append((g[0], g[1], n_['ours'], n_['best'], n_['best_afa'], n_['best_afr'], n_['lead'], n_['lead_sd'], n_['wins'], '' if not o_ else o_['best'], '' if not o_ else o_['lead'], rerun))
        if g[1] not in (None, '012345678') or g[1] is None:
            old_vs_new.append(('Widar3.0 orders/rooms', f'{g[0]} order {g[1]}' if g[1] else f'{DNAME.get(g[0], g[0])}: lead (strongest baseline)',
                               f"{o_['lead']:+.2f} ({nm(o_['best'])})" if o_ else '', f"{n_['lead']:+.2f} ({nm(n_['best'])})", ''))
    md.append(md_table(['Setting', 'WiMoT AFA', 'Strongest baseline (new)', 'its AFA', 'its AFR', 'Lead (paired ± SD; seeds ahead)', 'Strongest (9216-D)', 'Lead (9216-D)', 'Status'], t))
    lo = [v['lead'] for (d, o), v in wl_o.items() if d == 'widar' and o not in (None, '012345678')]; ln = [v['lead'] for (d, o), v in wl_n.items() if d == 'widar' and o not in (None, '012345678')]
    old_vs_new.append(('Widar3.0 orders/rooms', 'range of lead over the three other user orders (pp)', f'{min(lo):+.2f} to {max(lo):+.2f}', f'{min(ln):+.2f} to {max(ln):+.2f}', ''))
    (out/'widarext').mkdir(parents=True, exist_ok=True)
    (out/'widarext'/'widarext_new.md').write_text('\n'.join(md)+'\n')
    write_csv(out/'widarext'/'widarext_new.csv', ('dataset', 'order', 'WiMoT_AFA', 'strongest_new', 'strongest_AFA', 'strongest_AFR', 'lead', 'lead_sd', 'seeds_ahead', 'strongest_old9216', 'lead_old9216', 'rerun'), rows)

    wb = cells([r for r in new if r['study'] == 'widarext'], lambda r: (r['dataset'], r['order'], r['method']))
    wb_old = cells([r for r in ref if r['study'] == 'widarext' and r['family'] != 'method'], lambda r: (r['dataset'], r['order'], r['method']))
    rows = []
    for k, g in sorted(wb.items(), key=str):
        s_ = sorted(g); o = wb_old.get(k)
        rows.append((*k, ms([100*g[x]['AFA'] for x in s_])[0], ms([100*g[x]['AFA'] for x in s_])[1], ms([100*g[x]['AFR'] for x in s_])[0],
                     '' if not o else ms([100*o[x]['AFA'] for x in sorted(o)])[0], '' if not o else ms([100*o[x]['AFR'] for x in sorted(o)])[0], len(s_)))
    write_csv(out/'widarext'/'widarext_baselines_new.csv', ('dataset', 'order', 'method', 'AFA_mean', 'AFA_sd', 'AFR_mean', 'AFA_old9216', 'AFR_old9216', 'n_seeds'), rows)

    # rooms 2 and 3: every baseline, paired against WiMoT
    md = ['# Widar3.0 rooms 2 and 3: baselines with the 9728-D input\n',
          'AFA (%) and AFR (pp), mean ± sample SD over seeds 42-44; paired = WiMoT − baseline per seed (mean ± SD; seeds with WiMoT strictly ahead). Old: baselines read the 9216-D Mantis block.\n']
    for ds in ('widar_r2', 'widar_r3'):
        w = {r['seed']: r for r in recs_new if r['study'] == 'widarext' and r['dataset'] == ds and is_ours(r) and r['seed'] <= 44}
        t = [('WiMoT', fmt(ms([100*w[x]['AFA'] for x in sorted(w)])), f"{st.mean(100*w[x]['AFR'] for x in w):.2f}", '', '', '')]
        for m in FIVE:
            g = {r['seed']: r for r in new if r['study'] == 'widarext' and r['dataset'] == ds and r['method'] == m}
            o = {r['seed']: r for r in recs_old if r['study'] == 'widarext' and r['dataset'] == ds and r['method'] == m and r['family'] != 'method'}
            d = [100*(w[x]['AFA']-g[x]['AFA']) for x in sorted(g)]
            t.append((nm(m), fmt(ms([100*g[x]['AFA'] for x in sorted(g)])), f"{st.mean(100*g[x]['AFR'] for x in g):.2f}",
                      f"{st.mean(100*o[x]['AFA'] for x in o):.2f}", f'{st.mean(d):+.2f} ± {st.stdev(d):.2f} ({sum(x > 0 for x in d)}/3)', f'{[round(x, 2) for x in d]}'))
        md += [f'\n## {DNAME[ds]}\n', md_table(['Method', 'AFA new', 'AFR new', 'AFA old (9216-D)', 'WiMoT − baseline (paired)', 'per seed'], t)]
    (out/'widarext'/'rooms_new.md').write_text('\n'.join(md)+'\n')

    # ------------------------------------------------ encoder study
    bn_cells, bo_cells = backbone_cells(recs_new, tn), backbone_cells(recs_old, to)
    md = ['# Other frozen encoders: baselines with the 9728-D input\n',
          'AFA (%) mean ± sample SD over seeds 42-44; strongest of the five baselines (Fine-tuning, LwF, SI+ER, DER++, EdgeCL) by mean AFA; '
          'lead = paired WiMoT − strongest baseline (pp; seeds ahead). Old: baselines read only the 9216-D encoder block. '
          'The MantisV2 (pretrained) row uses the main-study baselines in both versions.\n']
    rows = []; full_rows = []; stat = {}
    for d in DATASETS:
        md.append(f'\n## {DNAME[d]}\n'); t = []
        for e in ENCODERS:
            cn_, co_ = bn_cells.get((d, e)), bo_cells.get((d, e))
            if not cn_ or not co_: continue
            bst, bs = strongest(cn_); bo_name, _ = strongest(co_)
            ld, lo_ = paired(cn_, bst), paired(co_, bo_name)
            stat[(d, e)] = dict(ours=ms(cn_['main']['AFA']), ours_afr=st.mean(cn_['main']['AFR']), best=bst, best_afa=st.mean(bs[bst]['AFA']),
                                best_afr=st.mean(bs[bst]['AFR']), lead=ld, old_best=bo_name, old_best_afa=st.mean(co_[bo_name]['AFA']), old_lead=lo_,
                                base_afr=[st.mean(v['AFR']) for v in bs.values()])
            t.append((ENAME[e], fmt(ms(cn_['main']['AFA'])), f"{nm(bst)} {st.mean(bs[bst]['AFA']):.2f}", f'{ld[0]:+.2f} ± {ld[1]:.2f} ({ld[2]}/{ld[3]})',
                      f"{nm(bo_name)} {st.mean(co_[bo_name]['AFA']):.2f}", f'{lo_[0]:+.2f} ({lo_[2]}/{lo_[3]})'))
            rows.append((d, e, ms(cn_['main']['AFA'])[0], ms(cn_['main']['AFA'])[1], bst, st.mean(bs[bst]['AFA']), st.mean(bs[bst]['AFR']), ld[0], ld[1], ld[2],
                         bo_name, st.mean(co_[bo_name]['AFA']), lo_[0], lo_[2]))
            for m in ('main',)+FIVE:
                v, o = cn_[m], co_.get(m)
                full_rows.append((d, e, m, ms(v['AFA'])[0], ms(v['AFA'])[1], st.mean(v['AFR']), '' if m == 'main' or not o else ms(o['AFA'])[0],
                                  '' if m == 'main' or not o else st.mean(o['AFR'])))
        md.append(md_table(['Encoder', 'WiMoT AFA', 'Strongest baseline (new)', 'Lead (paired ± SD; seeds ahead)', 'Strongest (9216-D)', 'Lead (9216-D)'], t))
    (out/'backbone').mkdir(parents=True, exist_ok=True)
    (out/'backbone'/'backbone_new.md').write_text('\n'.join(md)+'\n')
    write_csv(out/'backbone'/'backbone_new.csv', ('dataset', 'encoder', 'WiMoT_AFA', 'WiMoT_AFA_sd', 'strongest_new', 'strongest_AFA', 'strongest_AFR', 'lead', 'lead_sd',
                                                 'seeds_ahead', 'strongest_old9216', 'strongest_AFA_old9216', 'lead_old9216', 'seeds_ahead_old9216'), rows)
    write_csv(out/'backbone'/'backbone_all_methods_new.csv', ('dataset', 'encoder', 'method', 'AFA_mean', 'AFA_sd', 'AFR_mean', 'AFA_old9216', 'AFR_old9216'), full_rows)
    fm = ['# All methods per encoder: AFA (%) and AFR (pp), baselines with the 9728-D input (old 9216-D AFA in brackets)\n']
    for d in DATASETS:
        fm += [f'\n## {DNAME[d]}\n']; t = []
        for e in ENCODERS:
            for r in full_rows:
                if r[0] == d and r[1] == e:
                    t.append((ENAME[e], nm(r[2]), f'{r[3]:.2f} ± {r[4]:.2f}', f'{r[5]:.2f}', '' if r[6] == '' else f'{r[6]:.2f}', '' if r[6] == '' else f'{r[7]:.2f}',
                              '' if r[6] == '' else f'{r[3]-r[6]:+.2f}'))
        fm.append(md_table(['Encoder', 'Method', 'AFA new', 'AFR new', 'AFA old (9216-D)', 'AFR old', 'ΔAFA (pp)'], t))
    (out/'backbone'/'backbone_all_methods_new.md').write_text('\n'.join(fm)+'\n')
    # statements of the paper
    others = [e for e in ENCODERS if e != 'mantis']; five = [e for e in others if e != 'mantis_random']
    def rng(v): return f'{min(v):.2f} to {max(v):.2f}'
    def S(d, e, k, tag): return stat[(d, e)][('old_' if tag == 'old' else '')+k]
    def lead_of(d, e, tag): return stat[(d, e)]['old_lead' if tag == 'old' else 'lead']
    def best_afa(d, e, tag): return S(d, e, 'best_afa', tag)
    def best_name(d, e, tag): return S(d, e, 'best', tag) if tag == 'new' else stat[(d, e)]['old_best']
    for d in ('3do', 'widar'):
        for what, f, group in (('lead over the strongest baseline, mean per encoder (pp), all seven encoders', lambda e, t: lead_of(d, e, t)[0], ENCODERS),
                               ('seeds ahead of the strongest baseline (of 3), minimum over all seven encoders', lambda e, t: lead_of(d, e, t)[2], ENCODERS),
                               ('strongest baseline AFA (%), six other encoders', lambda e, t: best_afa(d, e, t), others)):
            vo, vn = [f(e, 'old') for e in group], [f(e, 'new') for e in group]
            old_vs_new.append(('Encoder study', f'{DNAME[d]}: {what}', rng(vo), rng(vn), ''))
        w = [stat[(d, e)]['ours'][0] for e in others]
        old_vs_new.append(('Encoder study', f'{DNAME[d]}: WiMoT AFA range over the six other encoders (unchanged)', rng(w), rng(w), 'WiMoT records are the same'))
        ref_ = stat[(d, 'mantis')]
        old_vs_new.append(('Encoder study', f'{DNAME[d]}: lead larger than with pretrained MantisV2 for every other encoder',
                           f"{sum(lead_of(d, e, 'old')[0] > lead_of(d, 'mantis', 'old')[0] for e in others)} of 6 (pretrained {lead_of(d, 'mantis', 'old')[0]:+.2f})",
                           f"{sum(lead_of(d, e, 'new')[0] > lead_of(d, 'mantis', 'new')[0] for e in others)} of 6 (pretrained {lead_of(d, 'mantis', 'new')[0]:+.2f})", ''))
        old_vs_new.append(('Encoder study', f'{DNAME[d]}: strongest baseline by encoder',
                           '; '.join(f"{e} {nm(best_name(d, e, 'old'))}" for e in ENCODERS), '; '.join(f"{e} {nm(best_name(d, e, 'new'))}" for e in ENCODERS), ''))
    old_vs_new.append(('Encoder study', 'Lead with random-init. MantisV2 (pp)',
                       ' / '.join(f"{lead_of(d, 'mantis_random', 'old')[0]:+.2f}" for d in ('3do', 'wimans', 'widar')),
                       ' / '.join(f"{lead_of(d, 'mantis_random', 'new')[0]:+.2f}" for d in ('3do', 'wimans', 'widar')), '3DO / WiMANS / Widar3.0'))
    old_vs_new.append(('Encoder study', 'WiMANS: lead with pretrained and random-init. MantisV2 (pp)',
                       f"{lead_of('wimans', 'mantis', 'old')[0]:+.2f} / {lead_of('wimans', 'mantis_random', 'old')[0]:+.2f}",
                       f"{lead_of('wimans', 'mantis', 'new')[0]:+.2f} / {lead_of('wimans', 'mantis_random', 'new')[0]:+.2f}", ''))
    def wim_five(tag):
        v = [lead_of('wimans', e, tag) for e in five]
        return (f"lead per encoder {rng([x[0] for x in v])}; WiMoT ahead in {sum(x[0] > 0 for x in v)} of 5 encoders; seeds ahead {[x[2] for x in v]}; "
                f"strongest {', '.join(nm(best_name('wimans', e, tag)) for e in five)}")
    old_vs_new.append(('Encoder study', 'WiMANS: five foundation-model encoders (MOMENT, TSPulse, Chronos-2, TimesFM 2.5/3.0)', wim_five('old'), wim_five('new'), ''))
    def si_gap(tag):
        out_ = []
        for e in five:
            c = (bo_cells if tag == 'old' else bn_cells)[('wimans', e)]
            g = [b-a for a, b in zip(c['main']['AFA'], c['si_er']['AFA'])]
            out_.append(g)
        return out_
    def si_txt(tag):
        g = si_gap(tag); means = [st.mean(x) for x in g]
        return f"SI+ER minus WiMoT per encoder {rng(means)}; SI+ER higher in all seeds for {sum(all(y > 0 for y in x) for x in g)} of 5"
    old_vs_new.append(('Encoder study', 'WiMANS: SI+ER against WiMoT on the five foundation-model encoders', si_txt('old'), si_txt('new'), ''))
    def afr_rng(d, tag):
        c = [(bo_cells if tag == 'old' else bn_cells)[(d, e)] for e in others]
        return (f"strongest-baseline AFR {rng([st.mean(x[strongest(x)[0]]['AFR']) for x in c])}; all five baselines' AFR {rng([st.mean(x[m]['AFR']) for x in c for m in FIVE])}; "
                f"WiMoT AFR {rng([st.mean(x['main']['AFR']) for x in c])}")
    for d in ('wimans', 'widar'):
        old_vs_new.append(('Encoder study', f'{DNAME[d]}: forgetting (AFR, pp) over the six other encoders', afr_rng(d, 'old'), afr_rng(d, 'new'), ''))
    best_wimans = lambda tag: max(st.mean(((bo_cells if tag == 'old' else bn_cells)[('wimans', e)])[m]['AFA']) for e in ENCODERS for m in FIVE)
    old_vs_new.append(('Encoder study', 'WiMANS: highest AFA of any baseline under any encoder (WiMoT highest overall: 89.61, Mantis)', f'{best_wimans("old"):.2f}', f'{best_wimans("new"):.2f}', ''))
    old_vs_new.append(('Encoder study', 'WiMANS: strongest-baseline AFR range over the five foundation-model encoders (pp)',
                       rng([st.mean(strongest(bo_cells[('wimans', e)])[1][strongest(bo_cells[('wimans', e)])[0]]['AFR']) for e in five]),
                       rng([st.mean(strongest(bn_cells[('wimans', e)])[1][strongest(bn_cells[('wimans', e)])[0]]['AFR']) for e in five]), ''))
    rep_ = [(f'{d}|{e}', dict(ours=v['ours'], ours_afr=v['ours_afr'], best=v['best'], best_afa=v['best_afa'], best_afr=v['best_afr'], lead=v['lead'], old_best=v['old_best'],
                              old_best_afa=v['old_best_afa'], old_lead=v['old_lead'])) for (d, e), v in stat.items()]
    report['backbone'] = dict(rep_)

    # ------------------------------------------------ lead vs baseline forgetting (42 settings)
    Pn, Po = lead_points(recs_new, 'new'), lead_points(recs_old, 'old')
    cn, co = correlations(Pn), correlations(Po)
    md = ['# Lead over the strongest baseline against baseline forgetting (42 settings)\n',
          'In the new column every baseline of all 42 settings reads the 9728-D input (encoder features plus amplitude descriptor). '
          'The main setting uses all 21 baselines (the paper table used 18 on 3DO and WiMANS).\n']
    t = []
    Po_by = {(q['ds'], q['setting']): q for q in Po}
    for pn_row in Pn:
        po_row = Po_by[(pn_row['ds'], pn_row['setting'])]
        t.append((DNAME[pn_row['ds']], pn_row['setting'], f"{po_row['lead']:+.2f} ({nm(po_row['best'])}, AFR {po_row['best_afr']:.2f}, mean {po_row['mean_afr']:.2f})",
                  f"{pn_row['lead']:+.2f} ({nm(pn_row['best'])}, AFR {pn_row['best_afr']:.2f}, mean {pn_row['mean_afr']:.2f})", '' if pn_row['rerun'] else 'baselines not rerun (9216-D)'))
    md.append(md_table(['Stream', 'Setting', 'Lead, 9216-D baselines', 'Lead, 9728-D baselines', 'Note'], t))
    md.append(f"\nSettings: {cn['n']} (old {co['n']}); WiMoT ahead in {cn['leads']} (old {co['leads']}).\n")
    md.append(md_table(['Statistic', '9216-D', '9728-D'], [(f'{k} {s}', f"{co[k][s]:.2f}", f"{cn[k][s]:.2f}") for k in ('best_afr', 'mean_afr') for s in
                                                       ('pearson', 'spearman', 'demeaned_pearson', 'demeaned_spearman', 'spearman_3do', 'spearman_wimans', 'spearman_widar')]))
    (out/'lead_vs_forgetting.md').write_text('\n'.join(md)+'\n')
    write_csv(out/'lead_vs_forgetting.csv', ('dataset', 'setting', 'lead_old9216', 'best_old9216', 'best_afr_old9216', 'mean_afr_old9216', 'lead_new', 'best_new', 'best_afr_new', 'mean_afr_new', 'rerun'),
              [(a_['ds'], a_['setting'], Po_by[(a_['ds'], a_['setting'])]['lead'], Po_by[(a_['ds'], a_['setting'])]['best'], Po_by[(a_['ds'], a_['setting'])]['best_afr'], Po_by[(a_['ds'], a_['setting'])]['mean_afr'], a_['lead'], a_['best'], a_['best_afr'], a_['mean_afr'], a_['rerun']) for a_ in Pn])
    old_vs_new += [('Lead vs forgetting', 'settings with WiMoT ahead', f"{co['leads']} of {co['n']}", f"{cn['leads']} of {cn['n']}", ''),
                   ('Lead vs forgetting', 'Pearson / Spearman with AFR of the strongest baseline', f"{co['best_afr']['pearson']:.2f} / {co['best_afr']['spearman']:.2f}", f"{cn['best_afr']['pearson']:.2f} / {cn['best_afr']['spearman']:.2f}", ''),
                   ('Lead vs forgetting', 'Pearson / Spearman with mean AFR of the baselines', f"{co['mean_afr']['pearson']:.2f} / {co['mean_afr']['spearman']:.2f}", f"{cn['mean_afr']['pearson']:.2f} / {cn['mean_afr']['spearman']:.2f}", ''),
                   ('Lead vs forgetting', 'within-dataset (demeaned) Pearson / Spearman, strongest-baseline AFR', f"{co['best_afr']['demeaned_pearson']:.2f} / {co['best_afr']['demeaned_spearman']:.2f}", f"{cn['best_afr']['demeaned_pearson']:.2f} / {cn['best_afr']['demeaned_spearman']:.2f}", ''),
                   ('Lead vs forgetting', 'within-dataset (demeaned) Pearson / Spearman, mean AFR', f"{co['mean_afr']['demeaned_pearson']:.2f} / {co['mean_afr']['demeaned_spearman']:.2f}", f"{cn['mean_afr']['demeaned_pearson']:.2f} / {cn['mean_afr']['demeaned_spearman']:.2f}", ''),
                   ('Lead vs forgetting', 'Spearman within 3DO / WiMANS (mean AFR)', f"{co['mean_afr']['spearman_3do']:.2f} / {co['mean_afr']['spearman_wimans']:.2f}", f"{cn['mean_afr']['spearman_3do']:.2f} / {cn['mean_afr']['spearman_wimans']:.2f}", ''),
                   ('Lead vs forgetting', 'Spearman within 3DO / WiMANS (strongest-baseline AFR)', f"{co['best_afr']['spearman_3do']:.2f} / {co['best_afr']['spearman_wimans']:.2f}", f"{cn['best_afr']['spearman_3do']:.2f} / {cn['best_afr']['spearman_wimans']:.2f}", '')]
    report['lead_vs_forgetting'] = dict(new=cn, old=co)

    # ------------------------------------------------ further baseline-dependent statements of the paper
    REPLAY_FREE = ('ft', 'ewc', 'si', 'mas', 'lwf', 'duct', 'dualcp', 'vanillacp', 'consense')
    RECENT = ('cn_dpm_aligned', 'duct', 'duct_er', 'dualcp', 'dualcp_er', 'vanillacp', 'vanillacp_er', 'consense', 'edgecl')
    for d in DATASETS:
        for label, group in (('strongest replay-free baseline', REPLAY_FREE), ('strongest of the recent methods (CN-DPM, DUCT, DualCP, VanillaCP, ConSense, EdgeCL)', RECENT)):
            res = []
            for T in (to, tn):
                cell = T[d]; b = max((m for m in cell if m in group), key=lambda m: st.mean(cell[m]['AFA']))
                res.append(f"{nm(b)} {st.mean(cell[b]['AFA']):.2f}; lead {st.mean(cell['main']['AFA'])-st.mean(cell[b]['AFA']):+.2f}")
            old_vs_new.append(('Table I (further)', f'{DNAME[d]}: {label}', res[0], res[1], ''))
        bo, bn = strongest(to[d])[0], strongest(tn[d])[0]
        afr_o = st.mean(to[d]['main']['AFR'])-st.mean(to[d][bo]['AFR']); afr_n = st.mean(tn[d]['main']['AFR'])-st.mean(tn[d][bn]['AFR'])
        old_vs_new.append(('Table I (further)', f'{DNAME[d]}: WiMoT AFR minus AFR of the strongest baseline (pp)', f'{afr_o:+.2f}', f'{afr_n:+.2f}', ''))
        better = lambda T: sum(st.mean(T[d]['main']['AFA']) > st.mean(v['AFA']) for m, v in T[d].items() if m != 'main')
        old_vs_new.append(('Table I (further)', f'{DNAME[d]}: baselines with lower mean AFA than WiMoT', f'{better(to)} of 21', f'{better(tn)} of 21', ''))
    # per-user final accuracy on Widar3.0 (WiMoT against the best baseline on each user)
    def per_user(T):
        cell = T['widar']; k = len(cell['main']['matrix'][0])
        fin = {m: [st.mean(100*mat[-1][u] for mat in v['matrix']) for u in range(k)] for m, v in cell.items()}
        best = [max((m for m in fin if m != 'main'), key=lambda m: fin[m][u]) for u in range(k)]
        diff = [fin['main'][u]-fin[best[u]][u] for u in range(k)]
        return diff, best, fin['main']
    do, bo_, _ = per_user(to); dn, bn_, _ = per_user(tn)
    old_vs_new.append(('Table I (further)', 'Widar3.0: WiMoT final accuracy minus best baseline, users 1-8 (range, pp); users where WiMoT is highest',
                       f'{min(do[:8]):+.1f} to {max(do[:8]):+.1f}; {sum(x > 0 for x in do)} of 9', f'{min(dn[:8]):+.1f} to {max(dn[:8]):+.1f}; {sum(x > 0 for x in dn)} of 9', f'last user: {do[8]:+.1f} -> {dn[8]:+.1f}'))
    rows = [(u+1, do[u], nm(bo_[u]), dn[u], nm(bn_[u])) for u in range(len(do))]
    write_csv(out/'main'/'widar_per_user_lead.csv', ('user_index', 'lead_old9216_pp', 'best_old9216', 'lead_new_pp', 'best_new'), rows)
    # domain-order average (3DO, WiMANS): WiMoT against the baseline with the highest mean over the six orders
    for d in ('3do', 'wimans'):
        res = []
        for recs in (recs_old, recs_new):
            rows_ = [r for r in recs if r['dataset'] == d and r['seed'] <= 44 and ((r['study'] == 'orders') or (r['study'] == 'main' and r['order'] is None) or (r['study'] == 'ablation' and is_ours(r)))
                     and r['family'] != 'joint']
            by = defaultdict(lambda: defaultdict(dict))
            for r in rows_:
                o = r['order'] or '012'
                by[(r['family'], r['method'])][o][r['seed']] = 100*r['AFA']
            full = {k: v for k, v in by.items() if len(v) == 6 and all(len(x) == 3 for x in v.values())}
            avg = {k: st.mean(st.mean(x.values()) for x in v.values()) for k, v in full.items()}
            ours = avg[('method', 'main')]; best = max((k for k in avg if k != ('method', 'main')), key=avg.get)
            res.append(f"WiMoT {ours:.2f}; best {nm(best[1])} {avg[best]:.2f}; lead {ours-avg[best]:+.2f} ({len(avg)-1} baselines)")
        old_vs_new.append(('Orders (further)', f'{DNAME[d]}: mean AFA over the six orders, WiMoT and the strongest baseline on the order average', res[0], res[1], ''))

    # ------------------------------------------------ summary files
    write_csv(out/'old_vs_new.csv', ('section', 'quantity', 'old (9216-D baselines)', 'new (9728-D baselines)', 'note'), old_vs_new)
    sections = defaultdict(list)
    for r in old_vs_new: sections[r[0]].append(r[1:])
    md = ['# Old and new numbers that depend on the baselines\n', 'Old: baselines read the 9216-D Mantis block (paper Table I). New: baselines read the 9728-D input of WiMoT. WiMoT is unchanged.\n']
    for s, rows_ in sections.items():
        md += [f'\n## {s}\n', md_table(['Quantity', 'Old', 'New', 'Note'], rows_)]
    (out/'old_vs_new.md').write_text('\n'.join(md)+'\n')
    (out/'report_summary.json').write_text(json.dumps(report, indent=1, default=lambda o: list(o) if isinstance(o, tuple) else str(o)))
    print('written', out, 'new records', len(new))


if __name__ == '__main__':
    main()
