"""Writes results/reference_results_same_input.json: the results of the paper with every baseline reading the 9728-D input.

  python scripts/merge_same_input.py

WiMoT's records are those of results/reference_results.json. The baseline records of the studies main, orders, fewlabel, widarext (including Widar3.0 rooms 2 and 3),
and backbone (per encoder and stream) are replaced by the records of results/same_input/<study>_baselines_9728.json (study results of
``python -m wimot.study sameall_<study>``, summarized by scripts/same_input_report.py). The other studies (memory, five extra seeds,
input study) keep their earlier records, whose baselines read the 9216-D Mantis block.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STUDIES = ('main', 'orders', 'fewlabel', 'widarext', 'backbone')


def main():
    ref = json.loads((REPO/'results'/'reference_results.json').read_text())
    new = []
    for s in STUDIES:
        new += json.loads((REPO/'results'/'same_input'/f'{s}_baselines_9728.json').read_text())['results']
    key = lambda r: (r['study'], r['dataset'], r['encoder'] if r['study'] == 'backbone' else None)
    have = {key(r) for r in new}
    keep = [r for r in ref['results'] if not (r['study'] in STUDIES and r['family'] != 'method' and key(r) in have)]
    out = dict(description='Results of the paper: baselines of the studies main, orders, fewlabel, widarext, and backbone read the 9728-D input of WiMoT '
                           '(results/same_input); other records as in reference_results.json.', results=keep+new)
    (REPO/'results'/'reference_results_same_input.json').write_text(json.dumps(out, separators=(',', ':')))
    print(len(ref['results']), '->', len(out['results']))


if __name__ == '__main__':
    main()
