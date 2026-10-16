"""The no-transport ablation (``no_transport``, ``no_transport_no_hist``): off by default, and what the flag changes.

  python -m pytest tests/test_transport_ladder.py        (CPU, about 2 min)
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from wimot.method import train as mt
from wimot.method.statistics import load_stats, transport
from wimot.run import main as run_main

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('smoke_test', REPO/'scripts'/'smoke_test.py')
smoke = importlib.util.module_from_spec(spec); spec.loader.exec_module(smoke)


@pytest.fixture(scope='module')
def features(tmp_path_factory):
    return smoke.make_features(tmp_path_factory.mktemp('features'), seed=1)


@pytest.fixture(scope='module')
def root(tmp_path_factory):
    return tmp_path_factory.mktemp('runs')


def trajectory(features, root, method):
    common = ['--root', str(root), '--features', str(features), '--dataset', '3do', '--seed', '7', '--device', 'cpu',
              '--family', 'method', '--method', method]
    run_main(['train', *common]); run_main(['test', *common])
    out = Path(root)/'method'/'3do'/'seed7'/method
    return out, json.loads((out/'receipt.json').read_text()), json.loads((out/'test.json').read_text())


def test_flag_is_off_by_default():
    assert 'transport' not in mt.VARIANTS['main'] and 'transport' not in mt.variant_spec('main_sc0.5_h1')
    assert all(mt.variant_spec(v).get('transport', True) for v in mt.VARIANTS if not v.startswith('no_transport'))
    assert mt.variant_spec('no_transport') == dict(mt.MAIN, transport=False)
    assert mt.variant_spec('no_transport_no_hist') == dict(mt.MAIN, moment=False, transport=False)
    # the first-domain model is shared with the default variants
    assert mt.warm_key(mt.variant_spec('no_transport')) == mt.warm_key(mt.VARIANTS['main'])


def test_default_trajectory_does_not_depend_on_the_flag_code(features, root):
    """With the flag absent the readout statistics are the phi moments and the stage files hold them (as before)."""
    out, receipt, test = trajectory(features, root, 'main')
    assert 'transport' not in receipt['spec']
    stats = load_stats(out/'stage2_stats.npz')
    assert len(stats) == 2


def test_no_transport_changes_only_the_readout(features, root):
    out, rec, test = trajectory(features, root, 'main')
    out2, rec2, test2 = trajectory(features, root, 'no_transport')
    # training (adapter, SupCon, L_hist) is identical: same losses, same adapters
    assert [s['losses'] for s in rec['stages'][1:]] == [s['losses'] for s in rec2['stages'][1:]]
    import torch
    for stage in (1, 2, 3):
        a = mt.load_checkpoint(out/f'stage{stage}.pt'); b = mt.load_checkpoint(out2/f'stage{stage}.pt')
        for k in a['state']: assert torch.equal(a['state'][k], b['state'][k]), (stage, k)
    # first stage: nothing to transport yet, so the readout and the accuracy are identical
    assert test['matrix'][0] == test2['matrix'][0]
    # later stages: the stored moments of earlier domains are those of their own adapters
    phi = load_stats(out/'stage3_stats.npz')                                  # phi moments of all three domains (default run)
    h = load_stats(out2/'stage3_stats.npz')                                   # no-transport run: h moments in own coordinates
    for k in range(3):
        model = mt.model_from_checkpoint(mt.load_checkpoint(out2/f'stage{k+1}.pt'))
        expected = transport([phi[k]], model)[0]
        np.testing.assert_allclose(h[k].means, expected.means, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(h[k].covariances, expected.covariances, rtol=1e-9, atol=1e-12)
    final = mt.model_from_checkpoint(mt.load_checkpoint(out2/'stage3.pt'))
    transported = transport([phi[0]], final)[0]
    assert not np.allclose(h[0].means, transported.means)                     # the adapter moved, the stored moments did not
    # earlier-domain moments do not change from stage 2 to stage 3 in the stored file
    h2 = load_stats(out2/'stage2_stats.npz')
    for k in range(2): np.testing.assert_array_equal(h2[k].means, h[k].means)
    assert test['matrix'][2][:2] != test2['matrix'][2][:2] or test['matrix'][1][:1] != test2['matrix'][1][:1]
    assert rec2['spec']['transport'] is False


def test_no_transport_no_hist_trains_like_no_hist(features, root):
    out, rec, test = trajectory(features, root, 'no_hist')
    out2, rec2, test2 = trajectory(features, root, 'no_transport_no_hist')
    assert [s['losses'] for s in rec['stages'][1:]] == [s['losses'] for s in rec2['stages'][1:]]
    assert test['matrix'][0] == test2['matrix'][0]
    assert rec2['spec']['moment'] is False and rec2['spec']['transport'] is False


def test_api_matches_run_with_flag(features, root):
    """The high-level interface (Learner.probabilities) decodes the stored h moments exactly as ``wimot.run`` test does."""
    from wimot import WiMoT
    out, rec, test = trajectory(features, root, 'no_transport')
    data = {(d, sp): np.load(features/f'{d}_{sp}.npz') for d in smoke.DOMAINS for sp in ('train', 'test')}
    model = WiMoT(num_classes=3, seed=7, device='cpu', variant='no_transport'); matrix = []
    for k, domain in enumerate(smoke.DOMAINS):
        model.fit_domain(data[domain, 'train']['features'], data[domain, 'train']['labels'])
        matrix.append([model.score(data[d, 'test']['features'], data[d, 'test']['labels']) for d in smoke.DOMAINS[:k+1]]
                      + [None]*(len(smoke.DOMAINS)-k-1))
    assert matrix == test['matrix']
