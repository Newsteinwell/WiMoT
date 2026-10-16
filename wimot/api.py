"""High-level interface to WiMoT: learn domains one at a time from frozen-encoder CSI features and predict.

    from wimot import WiMoT
    model = WiMoT(num_classes=6, seed=42)
    model.fit_domain(X1, y1)        # first domain: trains the feature basis and the adapter, stores class moments
    model.fit_domain(X2, y2)        # next domain: updates the adapter, stores its class moments, refits the readout
    labels = model.predict(X)       # no domain identifier is needed

``X`` holds the 9728 features of ``wimot.features.extract`` (9216 semantic + 512 amplitude coordinates, float32), one row per
window; ``y`` holds integer labels 0..num_classes-1, and every domain must contain every class at least twice. The class is a
thin wrapper around ``wimot.method.train.Learner``, the code that ``wimot.run`` executes, so a trajectory driven through this
interface is bitwise identical to ``python -m wimot.run`` on the same features, seed and device (tests/test_api.py).
"""
import numpy as np
import torch

from .method import train as mt


class WiMoT:
    """Replay-free continual classifier. ``variant`` selects the method (``main``) or one of its ablation/storage variants."""

    def __init__(self, num_classes, seed=42, device=None, variant='main'):
        if variant not in mt.VARIANTS: raise ValueError(f'Unknown variant {variant}; choose from {sorted(mt.VARIANTS)}')
        self.num_classes, self.seed, self.variant, self.spec = int(num_classes), int(seed), variant, mt.VARIANTS[variant]
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cuda.matmul.allow_tf32 = False
        self.learner = None; self.history = []

    @property
    def domains_seen(self):
        return 0 if self.learner is None else self.learner.task

    @property
    def storage_bytes(self):
        """Bytes of the stored class moments (counts, means, covariances of every domain)."""
        return 0 if self.learner is None else mt.statistic_bytes(self.learner.stats)

    def _check(self, X, y=None):
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != mt.SEMANTIC+mt.AMPLITUDE:
            raise ValueError(f'Expected an N x {mt.SEMANTIC+mt.AMPLITUDE} feature matrix, got {X.shape}')
        if y is None: return X, None
        y = np.asarray(y)
        if y.shape != (len(X),) or y.dtype.kind not in 'iu': raise ValueError('Integer labels of shape (N,) required')
        return X, y.astype(np.int64)

    def fit_domain(self, X, y):
        """Learn one more domain. The first call trains the basis; later calls adapt and store moments. Returns ``self``."""
        X, y = self._check(X, y); raw = mt.mask(X, self.spec)
        if self.learner is None:
            cp, info = mt.train_first(raw, y, self.num_classes, self.spec, self.seed, self.device)
            self.learner = mt.Learner(cp, self.num_classes, self.seed, self.spec, self.device)
            self.history.append(self.learner.first(raw, y))
        else:
            self.history.append(self.learner.update(raw, y))
        return self

    def predict_proba(self, X):
        """Class probabilities (N x num_classes) from the readout of the latest update."""
        if self.learner is None: raise RuntimeError('Call fit_domain first')
        X, _ = self._check(X)
        return self.learner.probabilities(mt.mask(X, self.spec))

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))
