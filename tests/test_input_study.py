"""Foundation-model-free input stores and the input-study driver (CPU, seconds).

  python -m pytest tests/test_input_study.py
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from wimot.baselines import consense
from wimot.common import AMPLITUDE, SEMANTIC, STREAMS, PROTOCOL

REPO = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, REPO/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


derive, study, smoke = load('make_derived_inputs'), load('input_study'), load('smoke_test')


@pytest.fixture(scope='module')
def features(tmp_path_factory):
    root = tmp_path_factory.mktemp('mantis')
    smoke.make_features(root/'3do', seed=3)
    return root


@pytest.mark.parametrize('kind,dimension', [('amp_pca', 256), ('amp_rp', 128), ('amp_hand', 157)])
def test_derived_store_layout_and_first_domain_fit(features, tmp_path, kind, dimension):
    derive.make(features, tmp_path, '3do', kind)
    domains = STREAMS['3do']['domains']; target = tmp_path/kind/'3do'
    record = json.loads((target/'transform.json').read_text())
    assert record['dimension'] == dimension and record['fitted_on'] == f'{domains[0]}_train'
    with np.load(target/'d1_train.npz') as f: x = f['features']
    assert x.shape[1] == SEMANTIC+AMPLITUDE and not x[:, :SEMANTIC].any() and not x[:, SEMANTIC+dimension:].any()
    assert np.isfinite(x).all()
    # fitting uses the first-domain TRAIN rows only: rewriting every other file leaves the output unchanged
    other = tmp_path/'other'; (other/'3do').mkdir(parents=True)
    for p in (features/'3do').glob('*.npz'):
        with np.load(p) as f: a, y = f['features'].copy(), f['labels']
        if p.name != 'd1_train.npz': a[:, SEMANTIC:] = np.random.default_rng(0).random(a[:, SEMANTIC:].shape)+5
        np.savez(other/'3do'/p.name, features=a, labels=y)
    derive.make(other, tmp_path/'o', '3do', kind)
    with np.load(tmp_path/'o'/kind/'3do'/'d1_train.npz') as f: np.testing.assert_array_equal(f['features'], x)


def test_pca_whitens_first_domain(features, tmp_path):
    derive.make(features, tmp_path, '3do', 'amp_pca')
    with np.load(tmp_path/'amp_pca'/'3do'/'d1_train.npz') as f: g = f['features'][:, SEMANTIC:SEMANTIC+256].astype(np.float64)
    np.testing.assert_allclose(np.cov(g.T), np.eye(256), atol=1e-3)


def test_hand_features_do_not_read_the_semantic_block(features, tmp_path):
    a = np.abs(np.random.default_rng(1).normal(size=(300, AMPLITUDE)))+1
    _, record = derive.build('amp_hand', a)
    assert record['dimension'] == 12*2+5+2*derive.DCT_KEEP


def test_consense_amp_token_and_candidate_listing():
    x = torch.randn(4, AMPLITUDE)
    assert consense.tokens(x, 'amp512').shape == (4, 1, AMPLITUDE) and consense.TOKEN_COUNT['amp512'] == 1
    assert consense.tokens(torch.randn(2, 9216), 'm9216').shape == (2, 18, 512)    # default unchanged
    listing = study.candidates()
    assert len(listing) == 21 and listing[('classic', 'si_er')] == PROTOCOL['classic']['coefficient_grids']['si']
    assert listing[('classic', 'derpp')] == [None]
