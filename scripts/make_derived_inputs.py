"""Builds foundation-model-free input stores from the 512-D amplitude descriptor of the cached 9728-D features.

  python scripts/make_derived_inputs.py --features FEATURES/mantis --out FEATURES_DERIVED --kinds amp_pca,amp_rp,amp_hand

Only the last 512 coordinates of every cached feature row (the per-channel temporal mean and SD of the CSI window, each
resampled to 256 coordinates; ``wimot.features.extract.amplitude``) are read; the 9216 MantisV2 coordinates are never used.
Every transform is fitted on the TRAIN partition of the first domain of the stream only and then applied unchanged to every
domain and partition, so no later-domain or validation/test statistic enters the input.

Output: ``OUT/<kind>/<dataset>/<domain>_<split>.npz`` with ``features`` (N x 9728, float32) and ``labels``, so that the
unchanged runners read it as ``amp512``/``amp_full``: the derived D-dimensional vector (D <= 512) is stored in the first D
coordinates of the amplitude block (coordinates D..511 are zero) and the 9216 semantic coordinates are zero. ``transform.json``
records D, the rank and the fitted constants.

  amp_pca   centred first-domain covariance, PCA whitening onto the k = min(256, #eigenvalues > 1e-6 * largest) leading
            directions (D = k)
  amp_rp    per-coordinate z-score (first-domain TRAIN), then a fixed Gaussian random projection to 128 dimensions
            (numpy default_rng(20261015), entries N(0, 1/128)), D = 128
  amp_hand  hand-crafted statistics of the descriptor (D = 157), then per-coordinate z-score (first-domain TRAIN):
              - 12 order statistics/moments (mean, SD, min, 5/25/50/75/95 % quantiles, max, skewness, kurtosis,
                mean absolute neighbour difference) of the mean block m and 12 of the SD block s;
              - mean, SD and median of the coefficient of variation s/m, the correlation of m and s, and log(sum s^2 / sum m^2);
              - orthonormal DCT-II coefficients 1..64 of log(1 + m/median) and of log(1 + s/median) along the channel axis
                (the spectral shape of the channel profile; the medians come from the first-domain TRAIN).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wimot.common import AMPLITUDE, SEMANTIC, STREAMS, FeatureStore, read_json, sha, write_json  # noqa: E402

RP_SEED, RP_DIM, PCA_MAX, PCA_FLOOR, HALF = 20261015, 128, 256, 1e-6, 256
DCT_KEEP = 64
KINDS = ('amp_pca', 'amp_rp', 'amp_hand')


def zscore_fit(x):
    return x.mean(0), x.std(0, ddof=1).clip(1e-6)


def dct_matrix(n):
    k, i = np.arange(n)[:, None], np.arange(n)[None]
    m = np.sqrt(2./n)*np.cos(np.pi*(i+.5)*k/n); m[0] /= np.sqrt(2.)
    return m


def order_stats(v):
    q = np.quantile(v, [0, .05, .25, .5, .75, .95, 1], axis=1).T
    mean, sd = v.mean(1, keepdims=True), v.std(1, keepdims=True).clip(1e-12)
    z = (v-mean)/sd
    return np.concatenate([mean, sd, q, (z**3).mean(1, keepdims=True), (z**4).mean(1, keepdims=True),
                           np.abs(np.diff(v, axis=1)).mean(1, keepdims=True)], 1)


def fit_pca(a):
    mean = a.mean(0); z = a-mean
    values, vectors = np.linalg.eigh(z.T@z/(len(a)-1))
    order = np.argsort(values)[::-1]; values, vectors = values[order], vectors[:, order]
    k = min(PCA_MAX, int(np.sum(values > values[0]*PCA_FLOOR)))
    return dict(mean=mean, basis=vectors[:, :k]/np.sqrt(values[:k]), k=k, rank_estimate=int(np.sum(values > values[0]*PCA_FLOOR)),
                top_eigenvalues=values[:5].tolist(), kept_eigenvalue_ratio=float(values[k-1]/values[0]))


def hand_raw(a, median_m, median_s, dct):
    m, s = a[:, :HALF], a[:, HALF:]
    lm, ls = np.log1p(np.maximum(m, 0)/median_m), np.log1p(np.maximum(s, 0)/median_s)
    cv = s/np.maximum(m, 1e-6*median_m)
    mc, sc = m-m.mean(1, keepdims=True), s-s.mean(1, keepdims=True)
    corr = (mc*sc).sum(1)/np.sqrt((mc**2).sum(1)*(sc**2).sum(1)).clip(1e-12)
    ratio = np.log((s**2).sum(1).clip(1e-12)/(m**2).sum(1).clip(1e-12))
    extra = np.stack([cv.mean(1), cv.std(1), np.median(cv, 1), corr, ratio], 1)
    return np.concatenate([order_stats(m), order_stats(s), extra, (lm@dct.T)[:, 1:DCT_KEEP+1], (ls@dct.T)[:, 1:DCT_KEEP+1]], 1)


def build(kind, first):
    """Returns (apply: amplitude block (N x 512, float64) -> derived (N x D), record)."""
    if kind == 'amp_pca':
        p = fit_pca(first)
        return (lambda a: (a-p['mean'])@p['basis']), dict(dimension=p['k'], rank_estimate=p['rank_estimate'],
                                                       top_eigenvalues=p['top_eigenvalues'], kept_eigenvalue_ratio=p['kept_eigenvalue_ratio'])
    if kind == 'amp_rp':
        mean, sd = zscore_fit(first); r = np.random.default_rng(RP_SEED).standard_normal((AMPLITUDE, RP_DIM))/np.sqrt(RP_DIM)
        return (lambda a: ((a-mean)/sd)@r), dict(dimension=RP_DIM, seed=RP_SEED, min_coordinate_sd=float(sd.min()))
    if kind == 'amp_hand':
        mm = max(float(np.median(first[:, :HALF])), 1e-6); ms = max(float(np.median(first[:, HALF:])), 1e-6); dct = dct_matrix(HALF)
        raw = hand_raw(first, mm, ms, dct); mean, sd = zscore_fit(raw)
        return (lambda a: (hand_raw(a, mm, ms, dct)-mean)/sd), dict(dimension=raw.shape[1], median_mean_block=mm, median_sd_block=ms)
    raise ValueError(kind)


def make(features, out, dataset, kind):
    store = FeatureStore(features/dataset); domains = STREAMS[dataset]['domains']
    first, _ = store.load(domains[0], 'train'); apply, record = build(kind, first[:, SEMANTIC:].astype(np.float64))
    D = record['dimension']; assert D <= AMPLITUDE
    target = out/kind/dataset; target.mkdir(parents=True, exist_ok=True); sources = {}
    for domain in domains:
        for split in ('train', 'validation', 'test'):
            x, y = store.load(domain, split); g = apply(x[:, SEMANTIC:].astype(np.float64))
            if not np.isfinite(g).all(): raise ValueError(f'Non-finite {kind} features: {dataset}/{domain}_{split}')
            f = np.zeros((len(g), SEMANTIC+AMPLITUDE), np.float32); f[:, SEMANTIC:SEMANTIC+D] = g
            name = f'{domain}_{split}.npz'; np.savez_compressed(target/name, features=f, labels=y)
            sources[name] = dict(source_sha256=sha(features/dataset/name), rows=len(y), max_abs=float(np.abs(g).max()))
    record.update(kind=kind, dataset=dataset, fitted_on=f'{domains[0]}_train', padded_to=AMPLITUDE, sources=sources)
    write_json(target/'transform.json', record)
    print('DERIVED', kind, dataset, 'D =', D, 'max|g| =', max(v['max_abs'] for v in sources.values()), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--features', type=Path, required=True, help='FEATURES/mantis (one directory per dataset)')
    p.add_argument('--out', type=Path, required=True); p.add_argument('--kinds', default=','.join(KINDS))
    p.add_argument('--datasets', default='3do,wimans,widar')
    a = p.parse_args(argv)
    for kind in a.kinds.split(','):
        for dataset in a.datasets.split(','): make(a.features, a.out, dataset, kind)


if __name__ == '__main__':
    main()
