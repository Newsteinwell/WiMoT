"""The high-level interface and ``wimot.run`` execute the same code: python -m pytest tests/test_api.py (CPU, ~30 s)."""
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from wimot import WiMoT
from wimot.run import main as run_main

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('smoke_test', REPO/'scripts'/'smoke_test.py')
smoke = importlib.util.module_from_spec(spec); spec.loader.exec_module(smoke)
warnings.simplefilter('ignore', RuntimeWarning)   # spurious float64-matmul warnings of NumPy with macOS Accelerate


@pytest.fixture(scope='module')
def features(tmp_path_factory):
    return smoke.make_features(tmp_path_factory.mktemp('features'), seed=1)


def test_wrapper_matches_run_bitwise(features, tmp_path):
    root = tmp_path/'runs'
    common = ['--root', str(root), '--features', str(features), '--dataset', '3do', '--seed', '7', '--device', 'cpu',
              '--family', 'method', '--method', 'main']
    run_main(['train', *common]); run_main(['test', *common])
    reference = json.loads((root/'method'/'3do'/'seed7'/'main'/'test.json').read_text())

    data = {(d, s): np.load(features/f'{d}_{s}.npz') for d in smoke.DOMAINS for s in ('train', 'test')}
    model = WiMoT(num_classes=3, seed=7, device='cpu'); matrix = []; stored = []
    for k, domain in enumerate(smoke.DOMAINS):
        model.fit_domain(data[domain, 'train']['features'], data[domain, 'train']['labels'])
        matrix.append([model.score(data[d, 'test']['features'], data[d, 'test']['labels']) for d in smoke.DOMAINS[:k+1]]
                      + [None]*(len(smoke.DOMAINS)-k-1))
        stored.append(model.storage_bytes)
    assert matrix == reference['matrix']                                    # accuracy matrices are bitwise identical
    assert stored[-1] == reference['statistic_bytes'] and stored == sorted(stored)
    receipt = json.loads((root/'method'/'3do'/'seed7'/'main'/'receipt.json').read_text())
    # the stage records (losses, calibration scores, selected readout cases) are identical as well
    for mine, theirs in zip(model.history, receipt['stages']):
        for key in ('stage', 'current_rows', 'old_rows', 'calibration', 'losses', 'statistic_bytes'):
            if key in theirs: assert json.loads(json.dumps(mine[key])) == theirs[key], key


def test_wrapper_validation(features):
    model = WiMoT(num_classes=3, device='cpu')
    with pytest.raises(RuntimeError): model.predict(np.zeros((2, 9728), np.float32))
    with pytest.raises(ValueError): model.fit_domain(np.zeros((10, 100), np.float32), np.zeros(10, int))
    with pytest.raises(ValueError): WiMoT(num_classes=3, variant='nonexistent')
