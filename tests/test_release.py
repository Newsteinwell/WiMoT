"""Fast consistency checks (CPU, no data): python -m pytest tests"""
import csv
import json
from pathlib import Path

import numpy as np

from wimot.common import LOCKS, PROTOCOL, STREAMS, label_subset, metrics, retain, stream
from wimot.features.extract import amplitude, pool

REPO = Path(__file__).resolve().parents[1]


def test_locks_are_candidates():
    grids = PROTOCOL['classic']['coefficient_grids']
    for method, entry in LOCKS['classic'].items():
        assert entry['coefficient'] in grids[method.removesuffix('_er')]
    for family in ('wifi', 'recent'):
        for method, entry in LOCKS[family].items():
            assert entry['coefficient'] in PROTOCOL[family]['candidates'][method]


def test_jobs_cover_reference():
    reference = json.loads((REPO/'results'/'reference_results.json').read_text())['results']
    lines = 0
    for path in (REPO/'configs'/'jobs').glob('*.tsv'):
        rows = list(csv.DictReader(path.open(), delimiter='\t'))
        assert rows and all(r['dataset'] in STREAMS for r in rows)
        lines += len(rows)
    main_ours = sum(1 for r in reference if r['study'] == 'ablation' and r['method'] == 'main' and r['seed'] <= 44)
    assert lines == len(reference)+main_ours


def test_metrics():
    m = metrics([[0.9, None, None], [0.8, 0.95, None], [0.7, 0.9, 0.85]])
    assert np.isclose(m['AFA'], (0.7+0.9+0.85)/3) and np.isclose(m['AFR'], ((0.9-0.7)+(0.95-0.9))/2)


def test_orders_and_label_subset():
    assert stream('3do', '210')['domains'] == ['d3', 'd2', 'd1']
    y = np.repeat(np.arange(3), 100)
    a = label_subset(y, '3do', 'd2', 42, 0.1, 3); b = label_subset(y, '3do', 'd2', 42, 0.1, 3)
    assert np.array_equal(a, b) and len(a) == 30


def test_retain_is_deterministic():
    rng = np.random.default_rng(0); x = rng.normal(size=(60, 4)).astype(np.float32); y = np.repeat(np.arange(3), 20)
    r1 = retain([], x, y, np.arange(60), 0, 3, 5, 42); r2 = retain([], x, y, np.arange(60), 0, 3, 5, 42)
    assert all(np.array_equal(u, v) for u, v in zip(r1[0], r2[0])) and len(r1[0][1]) == 15


def test_feature_layout():
    x = np.random.default_rng(1).normal(size=(2, 30, 64))
    assert pool(np.random.default_rng(2).normal(size=(2, 30, 512))).shape == (2, 9216)
    assert amplitude(x).shape == (2, 512)


def test_coefficient_selection_reproduces_locks():
    selection = json.loads((REPO/'results'/'coefficient_selection.json').read_text())
    for family in ('classic', 'wifi', 'recent'):
        for method, entry in LOCKS[family].items():
            runs = selection[family][method]
            assert runs['selected'] == entry['coefficient'] == runs['locked']
            chosen = [c for c in runs['candidates'] if c['coefficient'] == entry['coefficient']][0]
            assert abs(chosen['mean']-entry['validation_mean']) < 5e-5
    for method, per_input in selection['same_input'].items():
        for name, runs in per_input.items():
            assert runs['selected'] == LOCKS['same_input'][method][name]


def test_stagewise_predictions_reproduce_reference():
    import importlib.util
    spec = importlib.util.spec_from_file_location('check_predictions', REPO/'scripts'/'check_predictions.py')
    check = importlib.util.module_from_spec(spec); spec.loader.exec_module(check)
    index = json.loads((REPO/'results'/'stagewise_test_predictions.json').read_text())
    reference = json.loads((REPO/'results'/'reference_results.json').read_text())['results']
    arrays = np.load(REPO/index['file'])
    missing = [n for entry in index['missing'].values() for n in entry['records']]
    assert sorted(index['records']+missing) == list(range(len(reference)))
    assert all(check.matrix(arrays, n) == reference[n]['matrix'] for n in index['records'])
