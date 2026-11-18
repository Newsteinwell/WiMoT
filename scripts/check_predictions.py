"""Checks the stage-wise test predictions against the reference accuracy matrices.

  python scripts/check_predictions.py [--scores dist/stagewise_test_scores.npz]

For every record listed in results/stagewise_test_predictions.json, the per-domain accuracies of the stored predictions
must equal the reference matrix of results/reference_results.json. With --scores, the argmax of the stored class scores
must also equal the stored predictions (modulo the class count for DUCT, whose logits cover every domain/class slot).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def matrix(arrays, n):
    rows, stage = [], 1
    while f'r{n:04d}_stage{stage}_predictions' in arrays:
        p, y, d = (arrays[f'r{n:04d}_stage{stage}_{k}'] for k in ('predictions', 'labels', 'domains'))
        rows.append({int(x): float(np.mean(p[d == x] == y[d == x])) for x in np.unique(d)})
        stage += 1
    return [[row.get(x) for x in range(len(rows))] for row in rows]


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--scores', type=Path)
    args = a.parse_args()
    index = json.loads((REPO/'results'/'stagewise_test_predictions.json').read_text())
    reference = json.loads((REPO/'results'/'reference_results.json').read_text())['results']
    assert hashlib.sha256((REPO/index['file']).read_bytes()).hexdigest() == index['sha256'], 'predictions file changed'
    arrays = np.load(REPO/index['file'])
    bad = [n for n in index['records'] if matrix(arrays, n) != reference[n]['matrix']]
    print('records', len(index['records']), 'matrix mismatches', len(bad), bad[:10])
    if args.scores:
        assert hashlib.sha256(args.scores.read_bytes()).hexdigest() == index['attachment']['sha256'], 'scores file differs'
        scores = np.load(args.scores)
        classes = {k: v['classes'] for k, v in json.loads((REPO/'configs'/'streams.json').read_text()).items()}
        wrong = [k for k in scores.files if not np.array_equal(scores[k].argmax(1) % classes[reference[int(k[1:5])]['dataset']],
                                                                 arrays[k.rsplit('_', 1)[0]+'_predictions'])]
        print('score arrays', len(scores.files), 'argmax mismatches', len(wrong), wrong[:10])
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
