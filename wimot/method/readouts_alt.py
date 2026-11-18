"""Alternative readouts for the readout ablation (not used by WiMoT itself; ``readout.py`` is unchanged).

Every readout is fitted from the stored class moments only (``TaskStatistics`` of each domain, already transported to the
adapted space h = A phi + b, see ``statistics.transport``) and never sees a historical sample. Domains are weighted equally,
classes have equal priors, and the domain identity is not used at prediction time.

  ncm   nearest class mean: mean over domains of the class means, Euclidean distance
  lda   linear discriminant analysis: one covariance shared by all classes, the equal-class-weight average of the merged
        class covariances
  qda   quadratic discriminant analysis: one Gaussian per class (no domain gate, no domain marginalisation), with the
        merged class covariance
  probe the linear head W of the adapter model (``model.classifier``: the map that L_hist regresses onto centred one-hot
        targets during training), applied to h

The merged moments of a class are those of the equal-weight mixture over domains of its per-domain Gaussians:
mean = (1/T) sum_t mu_t, covariance = (1/T) sum_t [Sigma_t + (mu_t - mean)(mu_t - mean)^T].

Covariance regularisation (LDA, QDA): by default the pseudo-count prior of the OSCL readout, (1-b) S + b tr(S)/d I with
b = kappa d / (n - 1 + kappa d), kappa = 0.25 (``protocol.json``) and n the number of training windows behind the covariance;
``ridge=e`` uses the fixed ridge b = e instead. Neither value is tuned on any data.
"""
import numpy as np

from ..common import PROTOCOL
from .readout import prior_coefficients
from .statistics import matrix

KAPPA = PROTOCOL['method']['readout']['kappa']


def merged_moments(stats):
    """Counts, means (C, d) and covariances (C, d, d) of the equal-domain-weight mixture of every class."""
    counts = sum(np.asarray(s.counts, float) for s in stats)
    means = np.stack([s.means for s in stats]).astype(np.float64)
    covs = np.stack([s.covariances for s in stats]).astype(np.float64)
    mean = means.mean(0); delta = means-mean[None]
    return counts, mean, covs.mean(0)+np.einsum('tci,tcj->cij', delta, delta)/len(stats)


def regularize(cov, n, ridge=None, kappa=KAPPA):
    """Shrink a covariance towards a scaled identity (pseudo-count prior with ``n`` windows, or a fixed ridge)."""
    d = len(cov); beta = float(prior_coefficients([n], d, kappa)[0]) if ridge is None else float(ridge)
    scale = max(float(np.trace(cov))/d, 1e-12)
    out = (1-beta)*cov+beta*scale*np.eye(d)
    return (out+out.T)/2


def _inverse(cov):
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-9*max(float(vals.max()), 1e-12))
    return (vecs/vals)@vecs.T, float(np.log(vals).sum())


class NearestClassMean:
    def __init__(self, stats):
        self.d = stats[0].means.shape[1]; self.mean = merged_moments(stats)[1]

    def scores(self, h):
        h = matrix(h, self.d)
        return -((h[:, None, :]-self.mean[None])**2).sum(-1)


class LDA:
    def __init__(self, stats, ridge=None, kappa=KAPPA):
        counts, self.mean, covs = merged_moments(stats); self.d = self.mean.shape[1]
        self.cov = regularize(covs.mean(0), counts.sum(), ridge, kappa)
        self.precision = _inverse(self.cov)[0]
        self.bias = -.5*np.einsum('ci,ij,cj->c', self.mean, self.precision, self.mean)

    def scores(self, h):
        return matrix(h, self.d)@(self.precision@self.mean.T)+self.bias[None]


class QDA:
    def __init__(self, stats, ridge=None, kappa=KAPPA):
        counts, self.mean, covs = merged_moments(stats); self.d = self.mean.shape[1]
        self.covs = [regularize(c, n, ridge, kappa) for c, n in zip(covs, counts)]
        inv = [_inverse(c) for c in self.covs]
        self.precisions = np.stack([i[0] for i in inv]); self.logdets = np.array([i[1] for i in inv])

    def scores(self, h):
        delta = matrix(h, self.d)[:, None, :]-self.mean[None]
        quadratic = np.einsum('nci,cij,ncj->nc', delta, self.precisions, delta, optimize=True)
        return -.5*(quadratic+self.logdets[None])


def probe_scores(h, weight, bias):
    """Scores W h + b of the linear head (``weight``: C x d, ``bias``: C)."""
    return matrix(h)@np.asarray(weight, np.float64).T+np.asarray(bias, np.float64)[None]


def readout_scores(name, stats, h, weight=None, bias=None):
    """Scores (N x C) of the rows ``h`` from the transported per-domain moments ``stats`` with the readout ``name``."""
    if name == 'ncm': return NearestClassMean(stats).scores(h)
    if name == 'lda': return LDA(stats).scores(h)
    if name == 'qda': return QDA(stats).scores(h)
    if name == 'lda_ridge': return LDA(stats, ridge=RIDGE).scores(h)
    if name == 'qda_ridge': return QDA(stats, ridge=RIDGE).scores(h)
    if name == 'probe': return probe_scores(h, weight, bias)
    raise KeyError(name)


RIDGE = 1e-3
READOUTS = ('ncm', 'lda', 'qda', 'lda_ridge', 'qda_ridge', 'probe')
