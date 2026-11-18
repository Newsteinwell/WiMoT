"""Frozen basis phi = [semantic stem | fixed random amplitude stem | first-domain PCA] and the affine adapter.

The module construction order fixes the random-number consumption of the first-domain model; it must not be
changed if trained checkpoints are to be reproduced.
"""
import copy

import numpy as np
import torch
from torch import nn

from ..common import SEMANTIC, AMPLITUDE, PROTOCOL

INPUT = SEMANTIC+AMPLITUDE


class DualStem(nn.Module):
    """Trainable semantic stem (9216 -> 512 -> 128) and fixed random amplitude stem (512 -> 1024 -> 128)."""
    def __init__(self):
        super().__init__()
        self.semantic = nn.Sequential(nn.Linear(SEMANTIC, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                      nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.amplitude = nn.Sequential(nn.Linear(AMPLITUDE, 1024, bias=False), nn.ReLU(),
                                       nn.Linear(1024, 128, bias=False))
        self.register_buffer('amplitude_mean', torch.zeros(128))
        self.register_buffer('amplitude_sd', torch.ones(128))
        self.register_buffer('amplitude_calibrated', torch.tensor(False))

    @torch.no_grad()
    def calibrate_amplitude(self, x):
        if bool(self.amplitude_calibrated): raise ValueError('First-domain calibration happens only once')
        if x.ndim != 2 or x.shape[1] != INPUT or len(x) < 2: raise ValueError('Invalid input')
        z = self.amplitude(x[:, SEMANTIC:]); sd = z.std(0, unbiased=True)
        floor = torch.maximum(.01*sd.mean(), sd.new_tensor(1e-6))
        self.amplitude_mean.copy_(z.mean(0)); self.amplitude_sd.copy_(sd.clamp_min(floor))
        self.amplitude_calibrated.fill_(True)

    def forward(self, x):
        if not bool(self.amplitude_calibrated): raise ValueError('Calibrate on the first domain before training')
        amp = (self.amplitude(x[:, SEMANTIC:])-self.amplitude_mean)/self.amplitude_sd
        return torch.cat([self.semantic(x[:, :SEMANTIC]), amp], dim=1)


class FirstModel(nn.Module):
    """First-domain network: dual stem, adapter 256 -> 128, unit-norm SupCon projector and a linear head W."""
    def __init__(self, dimension, classes):
        super().__init__()
        if dimension != INPUT: raise ValueError(f'Input dimension must be {INPUT}')
        self.stem = DualStem()
        self.adapter = nn.Linear(256, 128)
        with torch.no_grad():
            self.adapter.weight.copy_(torch.cat([torch.eye(128), torch.eye(128)], dim=1)/np.sqrt(2))
            self.adapter.bias.zero_()
        self.projector = nn.Linear(128, 128)
        for layer in list(self.stem.modules())+[self.projector]:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None: nn.init.zeros_(layer.bias)
        self.stem.amplitude.requires_grad_(False)
        self.classifier = nn.Linear(128, classes)

    def forward(self, x, frozen_features=False):
        h = self.adapter(x if frozen_features else self.stem(x))
        return nn.functional.normalize(self.projector(h), dim=1), self.classifier(h)


class BasisStem(nn.Module):
    """Frozen 512-dimensional basis: the first-domain dual stem (256) and a 256-dimensional semantic shortcut."""
    def __init__(self, dual_stem, projection_seed):
        super().__init__()
        self.dual = copy.deepcopy(dual_stem)
        rng = np.random.default_rng(projection_seed)
        p = (rng.standard_normal((256, SEMANTIC))/np.sqrt(256)).astype(np.float32)
        self.register_buffer('projection', torch.from_numpy(p))
        self.register_buffer('residual_mean', torch.zeros(256))
        self.register_buffer('residual_scale', torch.ones(()))
        self.register_buffer('residual_calibrated', torch.tensor(False))
        self.requires_grad_(False)

    @torch.no_grad()
    def calibrate(self, first_normalized_inputs):
        if bool(self.residual_calibrated): raise ValueError('First-domain calibration is immutable')
        x = first_normalized_inputs
        if x.ndim != 2 or x.shape[1] != INPUT or len(x) < 2 or not torch.isfinite(x).all():
            raise ValueError('Finite first-domain normalized inputs required')
        z = nn.functional.linear(x[:, :SEMANTIC], self.projection)
        mean = z.mean(0)
        scale = (z-mean).square().mean().sqrt().clamp_min(1e-6)
        self.residual_mean.copy_(mean); self.residual_scale.copy_(scale)
        self.residual_calibrated.fill_(True)

    def forward(self, x):
        if not bool(self.residual_calibrated): raise ValueError('Calibrate before inference')
        shortcut = (nn.functional.linear(x[:, :SEMANTIC], self.projection)-self.residual_mean)/self.residual_scale
        return torch.cat((self.dual(x), shortcut), dim=1)


class Model(nn.Module):
    """phi (frozen, 512) -> affine adapter h = A phi + b (128) -> projector (SupCon) and head W (used only in L_hist)."""
    def __init__(self, first_model, projection_seed=None):
        super().__init__()
        if tuple(first_model.adapter.weight.shape) != (128, 256): raise ValueError('Expected a 256 -> 128 first adapter')
        seed = PROTOCOL['method']['residual_projection_seed'] if projection_seed is None else projection_seed
        self.stem = BasisStem(first_model.stem, seed)
        self.adapter = nn.Linear(512, 128)
        with torch.no_grad():
            self.adapter.weight.zero_(); self.adapter.weight[:, :256].copy_(first_model.adapter.weight)
            self.adapter.bias.copy_(first_model.adapter.bias)
        self.projector = copy.deepcopy(first_model.projector)
        self.classifier = copy.deepcopy(first_model.classifier)

    def forward(self, x, frozen_features=False):
        h = self.adapter(x if frozen_features else self.stem(x))
        return nn.functional.normalize(self.projector(h), dim=1), self.classifier(h)


def model_from_checkpoint(cp):
    model = Model(FirstModel(cp['dimension'], cp['classes'])); model.load_state_dict(cp['state'])
    return model


def pca_basis(first_semantic, dimension=256):
    x = np.asarray(first_semantic, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or min(x.shape[0]-1, x.shape[1]) < dimension:
        raise ValueError('Finite first-domain matrix with sufficient PCA rank required (more than 256 first-domain windows)')
    center = x.mean(0); z = x-center
    values, vectors = np.linalg.eigh(z@z.T)
    order = np.argsort(values)[::-1][:dimension]; values = values[order]
    if np.any(values <= max(values[0]*1e-12, 1e-12)): raise ValueError('PCA rank insufficient')
    basis = (vectors[:, order].T@z)/np.sqrt(values)[:, None]
    signs = np.sign(basis[np.arange(dimension), np.argmax(np.abs(basis), axis=1)])
    basis *= signs[:, None]
    np.testing.assert_allclose(basis@basis.T, np.eye(dimension), rtol=1e-7, atol=1e-7)
    residual = z.T@(z@basis.T)-basis.T*values
    if np.linalg.norm(residual) > 1e-7*np.linalg.norm(basis.T*values): raise ValueError('PCA eigen residual too large')
    return basis.astype(np.float32), dict(first_rows=len(x), dimension=dimension,
                                          captured_energy=float(values.sum()/np.sum(z*z)))


@torch.no_grad()
def install_first_pca(model, normalized_first, padded=False):
    """First-domain semantic PCA (256 directions); ``padded`` keeps k = numerical rank (<= 256) and zero-pads."""
    if torch.count_nonzero(model.adapter.weight[:, 256:]): raise ValueError('First adapter extras must be zero')
    if padded:
        x = np.asarray(normalized_first[:, :SEMANTIC], np.float64); z = x-x.mean(0); v = np.linalg.eigvalsh(z@z.T)
        k = min(256, int(np.sum(v > v.max()*1e-6))); basis, proof = pca_basis(x, k)
        full = np.zeros((256, SEMANTIC), np.float32); full[:k] = basis; basis = full; proof.update(padded_to=256)
    else:
        basis, proof = pca_basis(normalized_first[:, :SEMANTIC], 256)
    model.stem.projection.copy_(torch.from_numpy(basis).to(model.stem.projection.device))
    model.stem.residual_calibrated.fill_(False)
    model.stem.calibrate(torch.from_numpy(normalized_first).to(model.stem.projection.device))
    return proof


@torch.no_grad()
def disable_pca(model):
    model.stem.projection.zero_(); model.stem.residual_mean.zero_(); model.stem.residual_scale.fill_(1)
    model.stem.residual_calibrated.fill_(True)
    return dict(disabled=True)


class SupConLoss(nn.Module):
    """Supervised contrastive loss (Khosla et al., 2020) on unit-norm projections."""
    def __init__(self, temperature=.07):
        super().__init__(); self.temperature = temperature

    def forward(self, features, labels):
        device = features.device; batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        similarity = torch.matmul(features, features.T)/self.temperature
        logits_mask = torch.ones_like(mask)-torch.eye(batch_size, device=device)
        mask = mask*logits_mask
        logits_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity-logits_max.detach()
        exp_logits = torch.exp(logits)*logits_mask
        log_prob = logits-torch.log(exp_logits.sum(dim=1, keepdim=True)+1e-12)
        num_positives = mask.sum(dim=1)
        mean_log_prob = (mask*log_prob).sum(dim=1)/(num_positives+1e-12)
        if not bool((num_positives > 0).any()): return features.sum()*0
        return -mean_log_prob[num_positives > 0].mean()
