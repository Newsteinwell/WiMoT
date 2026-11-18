"""Inputs read by the baselines.

  m9216    the 9216-D Mantis block (main tables); per-coordinate z-score fitted on the first-domain TRAIN, scaled by
           1/sqrt(9216).
  cat9728  Mantis 9216 || amplitude descriptor 512, with the normalization of the proposed method (method.train.norm_fit).
  amp512   amplitude descriptor only: uncentred, one first-domain RMS scale times sqrt(512).
  stem9344 Mantis 9216 || output of the proposed fixed random amplitude stem (same seed and first-domain calibration as the
           proposed model), the stem block scaled by sqrt(128).
"""
import numpy as np
import torch

from ..common import AMPLITUDE, SEMANTIC, seed_all

DIMENSION = dict(m9216=SEMANTIC, cat9728=SEMANTIC+AMPLITUDE, stem9344=SEMANTIC+128, amp512=AMPLITUDE)


def normalize_fit(x):
    """Fit only the first arrived TRAIN partition; never estimate on future data."""
    x = np.asarray(x, np.float64)
    return x.mean(0), x.std(0, ddof=1).clip(1e-6)*np.sqrt(x.shape[1])


def amp_scale(first_amp):
    return max(float(np.sqrt(np.mean(first_amp.astype(np.float64)**2))), 1e-6)*np.sqrt(AMPLITUDE)


class Input:
    """Feature transform and first-domain normalization of one input kind for one (stream, seed)."""
    def __init__(self, kind, data, seed):
        if kind not in DIMENSION: raise ValueError(kind)
        self.kind, self.data, self.dimension = kind, data, DIMENSION[kind]
        if kind == 'stem9344': self._build_stem(seed)

    def _build_stem(self, seed):
        # Same RNG path as the method warm-up: seed_all(seed), then FirstModel(9728, C); calibrated on first-domain TRAIN.
        from ..method.model import FirstModel
        from ..method.train import norm_fit
        f, _ = self.data.raw(self.data.domains[0], 'train')
        mean, sd = norm_fit(f); x = torch.from_numpy(((f-mean)/sd).astype(np.float32))
        seed_all(seed); model = FirstModel(SEMANTIC+AMPLITUDE, self.data.classes).eval()
        model.stem.calibrate_amplitude(x)
        self.stem, self.amp_sd = model.stem, float(sd[SEMANTIC])

    def _stem_out(self, f):
        s = self.stem
        with torch.inference_mode():
            x = torch.from_numpy((f[:, SEMANTIC:]/self.amp_sd).astype(np.float32))
            z = (s.amplitude(x)-s.amplitude_mean)/s.amplitude_sd
        return z.numpy().astype(np.float64)

    def transform(self, f):
        if f.ndim != 2 or f.shape[1] != SEMANTIC+AMPLITUDE: raise ValueError('Expected Mantis+amplitude feature layout')
        if self.kind == 'm9216': return f[:, :SEMANTIC].copy()
        if self.kind == 'cat9728': return f
        if self.kind == 'amp512': return f[:, SEMANTIC:]
        return np.concatenate([f[:, :SEMANTIC], self._stem_out(f)], 1)

    def fit(self, x):
        if self.kind == 'm9216': return normalize_fit(x)
        if self.kind == 'cat9728':
            from ..method.train import norm_fit
            return norm_fit(x)
        if self.kind == 'amp512': return np.zeros(AMPLITUDE), np.full(AMPLITUDE, amp_scale(x))
        mean, sd = normalize_fit(x[:, :SEMANTIC])
        return np.concatenate([mean, np.zeros(128)]), np.concatenate([sd, np.full(128, np.sqrt(128))])

    def load(self, domain, split, mean=None, sd=None):
        f, y = self.data.raw(domain, split); x = self.transform(f)
        if mean is not None: x = ((x-mean)/sd).astype(np.float32)
        return x, y
