"""Alternative readouts, loss-weight exposure and summary measures; the default behaviour of the method is unchanged.

  python -m pytest tests/test_ablation.py        (CPU, about 1 min)

``tests/golden_default_losses.json`` holds the stage-wise training losses and the test matrix of a synthetic trajectory
(``scripts/smoke_test.py`` features, seed 7, CPU), recorded with the code *before* the loss weights became configurable.
"""
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from wimot.common import metrics
from wimot.measures import summary_measures
from wimot.method import train as mt
from wimot.method.calibration import normalized
from wimot.method.readout import head_from
from wimot.method.readouts_alt import (LDA, QDA, NearestClassMean, merged_moments, probe_scores, readout_scores, regularize)
from wimot.method.statistics import TaskStatistics, moments
from wimot.run import main as run_main

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('smoke_test', REPO/'scripts'/'smoke_test.py')
smoke = importlib.util.module_from_spec(spec); spec.loader.exec_module(smoke)
spec = importlib.util.spec_from_file_location('plasticity_stability', REPO/'scripts'/'plasticity_stability.py')
plasticity = importlib.util.module_from_spec(spec); spec.loader.exec_module(plasticity)
warnings.simplefilter('ignore', RuntimeWarning)


# ---- default behaviour unchanged -------------------------------------------------------------------------------------

@pytest.fixture(scope='module')
def features(tmp_path_factory):
    return smoke.make_features(tmp_path_factory.mktemp('features'), seed=1)


def trajectory(features, root, method):
    common = ['--root', str(root), '--features', str(features), '--dataset', '3do', '--seed', '7', '--device', 'cpu',
              '--family', 'method', '--method', method]
    run_main(['train', *common]); run_main(['test', *common])
    out = Path(root)/'method'/'3do'/'seed7'/method
    return json.loads((out/'receipt.json').read_text()), json.loads((out/'test.json').read_text())


def test_default_losses_match_recorded_values(features, tmp_path):
    golden = json.loads((REPO/'tests'/'golden_default_losses.json').read_text())
    receipt, test = trajectory(features, tmp_path, 'main')
    for mine, theirs in zip([s['losses'] for s in receipt['stages'][1:]], golden['losses']):
        np.testing.assert_allclose(mine, theirs, rtol=1e-3)
    assert [s['calibration']['selected']['case'] for s in receipt['stages']] == golden['cases']
    assert 'supcon_weight' not in receipt['spec'] and 'hist_weight' not in receipt['spec']


def test_explicit_paper_weights_are_bitwise_the_default(features, tmp_path):
    receipt, test = trajectory(features, tmp_path, 'main')
    other, test2 = trajectory(features, tmp_path, 'main_sc0.5_h1')
    assert [s['losses'] for s in receipt['stages'][1:]] == [s['losses'] for s in other['stages'][1:]]
    assert test['matrix'] == test2['matrix']
    assert mt.sweep_name(0.5, 1.) == 'main' and mt.sweep_name() == 'main'


def test_weights_change_training_as_intended(features, tmp_path):
    base, _ = trajectory(features, tmp_path, 'main')
    for name in ('main_sc0_h1', 'main_sc0.5_h0', 'main_sc1_h1', 'main_sc0.5_h2'):
        other, _ = trajectory(features, tmp_path, name)
        assert [s['losses'] for s in other['stages'][1:]] != [s['losses'] for s in base['stages'][1:]], name
    # first-domain moments and the frozen stem are those of the paper setting when only the later domains are re-weighted
    later, _ = trajectory(features, tmp_path, 'main_sc0_h1')
    assert later['frozen_stem_sha256'] == base['frozen_stem_sha256']
    first, _ = trajectory(features, tmp_path, 'main_sc0.5_h1_fsc0')
    assert first['frozen_stem_sha256'] != base['frozen_stem_sha256']


def test_variant_names():
    s = mt.variant_spec('main_sc0.25_h2')
    assert s['supcon_weight'] == .25 and s['hist_weight'] == 2. and 'first_supcon_weight' not in s
    assert mt.variant_spec('main_sc0_h1_fsc0')['first_supcon_weight'] == 0.
    assert mt.variant_spec('no_supcon') is mt.VARIANTS['no_supcon']
    assert mt.warm_key(mt.variant_spec('main_sc0_h1')) == mt.warm_key(mt.VARIANTS['main'])          # shared first-domain model
    assert mt.warm_key(mt.variant_spec('main_sc0_h1_fsc0')) != mt.warm_key(mt.VARIANTS['main'])
    assert mt.warm_key(mt.variant_spec('main_sc1_h1_fsc0.5')) == mt.warm_key(mt.VARIANTS['main'])
    assert mt.sweep_name(0.1, 4.) == 'main_sc0.1_h4' and mt.sweep_name(2., 0.25, 2.) == 'main_sc2_h0.25_fsc2'
    assert mt.sweep_name(0.5, 1., 0.) == 'main_sc0.5_h1_fsc0'
    with pytest.raises(KeyError): mt.variant_spec('main_scx')


def test_oscl_readout_numerics_unchanged():
    rng = np.random.default_rng(123); classes, d = 3, 6; stats = []
    for t in range(2):
        offset = rng.normal(size=d)*.5; x, y = [], []
        for c in range(classes):
            x.append(np.eye(d)[c]*2+offset+rng.normal(size=(40, d))*(.6+.1*t)); y += [c]*40
        stats.append(moments(np.concatenate(x), np.array(y), classes))
    p = normalized(head_from(stats, classes).predict_proba(rng.normal(size=(5, d))))
    golden = [[0.39392972, 0.60607028, 0.0], [0.99995278, 1.036e-05, 3.685e-05], [0.00045327, 0.99954673, 0.0],
              [0.78004776, 0.1253242, 0.09462803], [2.1e-07, 0.99999006, 9.73e-06]]   # recorded with the unmodified readout.py
    np.testing.assert_allclose(p, golden, atol=2e-4)


# ---- alternative readouts ---------------------------------------------------------------------------------------------

def domain_stats(rng, domains=3, classes=4, d=5, n=200, scale=1.):
    """Per-domain class moments (biased covariances) of Gaussian clusters, and the raw samples."""
    stats, samples = [], []
    for t in range(domains):
        offset = rng.normal(size=d)*.4; xs, ys = [], []
        for c in range(classes):
            xs.append(np.eye(d)[c % d]*3+offset+rng.normal(size=(n, d))*scale); ys += [c]*n
        x, y = np.concatenate(xs), np.array(ys); samples.append((x, y))
        cs = [x[y == c] for c in range(classes)]
        stats.append(TaskStatistics(np.full(classes, float(n)), np.stack([r.mean(0) for r in cs]),
                                    np.stack([np.cov(r, rowvar=False, bias=True) for r in cs])))
    return stats, samples


def test_merged_moments_are_the_equal_weight_mixture():
    stats, samples = domain_stats(np.random.default_rng(0))
    counts, mean, cov = merged_moments(stats)
    for c in range(4):
        pooled = np.concatenate([x[y == c] for x, y in samples])      # equal counts per domain: the pooled sample is the mixture
        np.testing.assert_allclose(mean[c], pooled.mean(0), atol=1e-10)
        np.testing.assert_allclose(cov[c], np.cov(pooled, rowvar=False, bias=True), atol=1e-10)
    assert np.all(counts == 600)


def test_ncm_uses_domain_equal_means():
    stats, _ = domain_stats(np.random.default_rng(1))
    mean = np.mean([s.means for s in stats], axis=0)
    q = mean+.01
    assert NearestClassMean(stats).scores(q).argmax(1).tolist() == [0, 1, 2, 3]
    np.testing.assert_allclose(NearestClassMean(stats).scores(q[:1])[0], -((q[0]-mean)**2).sum(1))


def test_lda_matches_closed_form_and_qda_matches_lda_for_equal_covariances():
    stats, _ = domain_stats(np.random.default_rng(2)); _, mean, cov = merged_moments(stats)
    shared = cov.mean(0); q = np.random.default_rng(3).normal(size=(50, 5))*2
    lda = LDA(stats, ridge=0.)
    expected = q@np.linalg.solve(shared, mean.T)-.5*np.einsum('ci,ci->c', mean, np.linalg.solve(shared, mean.T).T)
    np.testing.assert_allclose(lda.scores(q), expected, atol=1e-8)
    equal = [TaskStatistics(stats[0].counts, stats[0].means, np.stack([stats[0].covariances.mean(0)]*4))]   # one domain, one covariance
    assert np.array_equal(QDA(equal, ridge=0.).scores(q).argmax(1), LDA(equal, ridge=0.).scores(q).argmax(1))


def test_qda_beats_lda_for_unequal_covariances():
    rng = np.random.default_rng(4); d = 3
    mu = np.zeros((2, d)); cov = np.stack([np.eye(d)*.1, np.eye(d)*4.])
    stats = [TaskStatistics(np.full(2, 500.), mu, cov)]
    x = np.concatenate([rng.multivariate_normal(mu[c], cov[c], 2000) for c in range(2)]); y = np.repeat([0, 1], 2000)
    accuracy = lambda scores: float(np.mean(scores.argmax(1) == y))
    assert accuracy(QDA(stats, ridge=0.).scores(x)) > accuracy(LDA(stats, ridge=0.).scores(x))+.1


def test_regularize_prior_and_ridge():
    rng = np.random.default_rng(5); a = rng.normal(size=(30, 8)); cov = np.cov(a, rowvar=False)
    r = regularize(cov, 30.)
    assert np.linalg.eigvalsh(r).min() > 0 and np.allclose(r, r.T)
    beta = 0.25*8/(29+0.25*8); np.testing.assert_allclose(r, (1-beta)*cov+beta*np.trace(cov)/8*np.eye(8), atol=1e-12)
    np.testing.assert_allclose(regularize(cov, 30., ridge=0.), cov, atol=1e-12)
    singular = np.cov(rng.normal(size=(4, 8)), rowvar=False)      # rank 3 < 8: regularised scores stay finite
    stats = [TaskStatistics(np.full(2, 4.), rng.normal(size=(2, 8)), np.stack([singular, singular]))]
    assert np.isfinite(QDA(stats).scores(rng.normal(size=(5, 8)))).all() and np.isfinite(LDA(stats).scores(rng.normal(size=(5, 8)))).all()


def test_probe_is_the_linear_head_and_dispatch():
    rng = np.random.default_rng(6); w, b, h = rng.normal(size=(3, 5)), rng.normal(size=3), rng.normal(size=(7, 5))
    np.testing.assert_allclose(probe_scores(h, w, b), h@w.T+b)
    stats, _ = domain_stats(rng, classes=3)
    np.testing.assert_allclose(readout_scores('probe', stats, h, w, b), h@w.T+b)
    for name in ('ncm', 'lda', 'qda', 'lda_ridge', 'qda_ridge'): assert readout_scores(name, stats, h).shape == (7, 3)
    with pytest.raises(KeyError): readout_scores('nope', stats, h)


# ---- summary measures ---------------------------------------------------------------------------------------------------

def test_summary_measures_agree_with_existing_definitions():
    m = [[.9, None, None], [.8, .95, None], [.7, .9, .85]]
    s = summary_measures(m); base = metrics(m)
    assert s['AFA'] == base['AFA'] and s['AFR'] == base['AFR']
    assert np.isclose(s['new'], (.95+.85)/2) and np.isclose(s['old'], (.7+.9)/2)
    new, old, afr = plasticity.metrics(dict(matrix=m, AFR=base['AFR']))
    assert np.isclose(100*s['new'], new) and np.isclose(100*s['old'], old) and np.isclose(100*s['AFR'], afr)
