"""CN-DPM on a independent, frozen Mantis feature representation.

Adapted from Soochan Lee et al., ICLR 2020, arXiv:2001.00689 and the
MIT-licensed official code, commit d592c5d794e4bddc4947f2ed61d6a9e50c6da277.
Copyright (c) 2020 Soochan Lee; adaptations copyright (c) 2026 authors.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This is a *feature-space adaptation*, not a reproduction of image benchmarks.
It retains cumulative DP priors, joint density/classifier responsibilities,
novelty-only short-term memory, sleep-created experts, detached precursor
features, and generative mixture prediction without task identities.

The Gaussian VAE has a linear output because frozen CSI features are continuous
and unbounded. First-update-only feature normalization is frozen thereafter.
The runner sets STM capacity from the same first-domain 10% retention rule;
sleep optimization retains the method-native budget and is separately counted.
There is no sleep triggered by update boundaries and no historical replay.
"""

from dataclasses import asdict, dataclass, replace
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


OFFICIAL_COMMIT = "d592c5d794e4bddc4947f2ed61d6a9e50c6da277"


@dataclass(frozen=True)
class CNConfig:
    # MLP-sharing architecture from configs/mnist-cndpm.yaml.
    base_width: int = 64
    extension_width: int = 16
    latent_dim: int = 16
    z_samples: int = 16
    batch_size: int = 10
    stm_capacity: int = 150
    sleep_batch_size: int = 50
    sleep_steps_g: int = 8000
    sleep_steps_d: int = 2000
    lr_g: float = 0.0004
    lr_d: float = 0.0001
    weight_decay: float = 0.00001
    gradient_clip_value: float = 0.5
    update_min_usage: float = 0.1
    # Domain-incremental configuration mnist_svhn-cndpm.yaml rather than
    # class-split MNIST's log_alpha=-400 and classifier_chill=0.01.
    log_alpha: float = 0.0
    classifier_chill: float = 1.0
    prediction_batch_size: int = 256

    def __post_init__(self):
        ints = (self.base_width, self.extension_width, self.latent_dim,
                self.z_samples, self.batch_size, self.stm_capacity,
                self.sleep_batch_size, self.sleep_steps_g, self.sleep_steps_d,
                self.prediction_batch_size)
        if any(type(v) is not int or v <= 0 for v in ints):
            raise ValueError("Network, memory and optimization sizes must be positive integers")
        floats = (self.lr_g, self.lr_d, self.gradient_clip_value,
                  self.classifier_chill)
        if any(not np.isfinite(v) or v <= 0 for v in floats):
            raise ValueError("Learning rates, clipping and chilling must be finite and positive")
        if (not np.isfinite(self.log_alpha) or not np.isfinite(self.weight_decay)
                or self.weight_decay < 0 or not np.isfinite(self.update_min_usage)
                or self.update_min_usage < 0):
            raise ValueError("Invalid prior, weight decay or minimum usage")


class _SharingExpert(nn.Module):
    """Official detached precursor-sharing MLP VAE and classifier topology."""

    def __init__(self, dim, ordinal, config, classes=3):
        super().__init__()
        self.ordinal, self.config = ordinal, config
        width = config.base_width if ordinal == 0 else config.extension_width
        earlier = 0 if ordinal == 0 else config.base_width + (ordinal - 1) * config.extension_width
        cat_dim = earlier + width
        self.g1 = nn.Linear(dim, width)
        self.g2 = nn.Linear(cat_dim, width)
        self.z_mu = nn.Linear(cat_dim, config.latent_dim)
        self.z_logvar = nn.Linear(cat_dim, config.latent_dim)
        self.decoder = nn.Sequential(nn.Linear(config.latent_dim, config.base_width),
                                     nn.ReLU(), nn.Linear(config.base_width, config.base_width),
                                     nn.ReLU(), nn.Linear(config.base_width, dim))
        self.d1 = nn.Linear(dim, width)
        self.d2 = nn.Linear(cat_dim, width)
        self.d3 = nn.Linear(cat_dim, classes)

    def g_parameters(self):
        modules = [self.g1, self.g2, self.z_mu, self.z_logvar, self.decoder]
        return [p for module in modules for p in module.parameters()]

    def d_parameters(self):
        return [p for module in [self.d1, self.d2, self.d3] for p in module.parameters()]

    def generative_hidden(self, x, previous=None):
        h1 = F.relu(self.g1(x))
        cat1 = h1 if previous is None else torch.cat((previous[0].detach(), h1), 1)
        h2 = F.relu(self.g2(cat1))
        cat2 = h2 if previous is None else torch.cat((previous[1].detach(), h2), 1)
        return cat1, cat2

    def discriminative_hidden(self, x, previous=None):
        h1 = F.relu(self.d1(x))
        cat1 = h1 if previous is None else torch.cat((previous[0].detach(), h1), 1)
        h2 = F.relu(self.d2(cat1))
        cat2 = h2 if previous is None else torch.cat((previous[1].detach(), h2), 1)
        return cat1, cat2

    def density_nll(self, x, hidden, epsilon):
        mu = self.z_mu(hidden[1])
        logvar = self.z_logvar(hidden[1]).clamp(-12, 12)
        z = mu[:, None, :] + torch.exp(.5 * logvar)[:, None, :] * epsilon
        reconstruction = self.decoder(z)
        # Official Gaussian reconstruction NLL, unit variance and min_noise=.001.
        recon = (.5 * ((x[:, None, :] - reconstruction).square() + .001)
                 + .5 * math.log(2 * math.pi)).sum(-1).mean(1)
        kl = .5 * (mu.square() + logvar.exp() - logvar - 1).sum(1)
        return recon + kl

    def log_predictions(self, hidden):
        return F.log_softmax(self.d3(hidden[1]), dim=1)


def _tensor_bytes(tensors):
    return int(sum(x.numel() * x.element_size() for x in tensors))


class CNDPM:
    """Task-free expert expansion with a bounded novelty STM.

    update(X, y) processes each incoming labeled sample once, in one seeded
    permutation. Repeated optimization in a sleep phase only uses the currently
    full novelty STM, which is cleared after that sleep. update call boundaries
    neither create experts nor flush memory. predict accepts only X.
    """

    def __init__(self, input_dim=256, seed=42, device="cuda", config=None, log_alpha=None, classes=3):
        if type(input_dim) is not int or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        self.classes = int(classes)
        if self.classes < 2: raise ValueError("At least two classes required")
        self.input_dim, self.seed, self.device = input_dim, int(seed), torch.device(device)
        self.config = CNConfig() if config is None else config
        if not isinstance(self.config, CNConfig):
            raise TypeError("config must be CNConfig")
        if log_alpha is not None:
            self.config = replace(self.config, log_alpha=float(log_alpha))
        self._torch_rng = torch.Generator(device=self.device).manual_seed(self.seed)
        self.placeholder = self._initialize_expert(0, placeholder=True)
        self.placeholder.requires_grad_(False)
        self.experts, self.optimizers = [], []
        self.log_counts = torch.empty(0, device=self.device)
        self.stm_x, self.stm_y, self.stm_tasks = [], [], []
        self.sleep_access = []
        self.normalizer_mean, self.normalizer_std = None, None
        # Same fixed common random numbers per example, across every prediction
        # call and batch: deterministic MC-ELBO proxy for density likelihood.
        self.eval_epsilon = torch.randn((1, self.config.z_samples, self.config.latent_dim),
                                        generator=self._torch_rng, device=self.device)
        self.update_count, self.update_duration = 0, 0.0
        self.seen_samples, self.sleep_count, self.stm_peak = 0, 0, 0
        self.routing_counts, self.sleep_sizes, self.stage_history = [], [], []
        self.total_sleep_steps_g, self.total_sleep_steps_d = 0, 0

    def _initialize_expert(self, ordinal, placeholder=False):
        # fork_rng prevents constructor initialization from changing a caller's
        # random generator. Per-expert initialization remains independently seeded.
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        initialization_seed = self.seed + 7919 * (1 if placeholder else ordinal + 2)
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(initialization_seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(initialization_seed)
            return _SharingExpert(self.input_dim, ordinal, self.config, self.classes).to(self.device)

    @property
    def expert_count(self):
        return len(self.experts)

    @property
    def num_experts(self):
        return self.expert_count

    @property
    def stm_size(self):
        return len(self.stm_x)

    def _validate_x(self, X, nonempty=False):
        x = np.asarray(X, dtype=np.float32)
        if (x.ndim != 2 or x.shape[1] != self.input_dim or not np.isfinite(x).all()
                or (nonempty and len(x) == 0)):
            raise ValueError("X must be a finite matrix with the configured feature dimension")
        return x

    def _normalize(self, x):
        return ((x - self.normalizer_mean) / self.normalizer_std).astype(np.float32)

    def _epsilon(self, batch, training):
        if not training:
            return self.eval_epsilon.expand(batch, -1, -1)
        return torch.randn((batch, self.config.z_samples, self.config.latent_dim),
                           generator=self._torch_rng, device=self.device)

    def _collect(self, x, training):
        """Joint precursor traversal, retaining separate gradients per expert."""
        epsilon = self._epsilon(len(x), training)
        placeholder_g = self.placeholder.generative_hidden(x)
        placeholder_d = self.placeholder.discriminative_hidden(x)
        densities = [self.placeholder.density_nll(x, placeholder_g, epsilon)]
        predictions = [self.placeholder.log_predictions(placeholder_d)]
        gh, dh = None, None
        for expert in self.experts:
            gh = expert.generative_hidden(x, gh)
            dh = expert.discriminative_hidden(x, dh)
            densities.append(expert.density_nll(x, gh, epsilon))
            predictions.append(expert.log_predictions(dh))
        return torch.stack(densities, 1), torch.stack(predictions, 1)

    def _class_nll(self, predictions, y):
        # Official value-with-chilling/gradient-without-chilling construction.
        ordinary = -predictions.gather(2, y[:, None, None].expand(-1, predictions.shape[1], 1)).squeeze(2)
        chilled = F.log_softmax(predictions / self.config.classifier_chill, dim=2)
        chilled_loss = -chilled.gather(2, y[:, None, None].expand(-1, predictions.shape[1], 1)).squeeze(2)
        return ordinary - ordinary.detach() + chilled_loss.detach()

    def _wake(self, x, y):
        density, pred = self._collect(x, training=True)
        nll = density + self._class_nll(pred, y)
        logprior = torch.cat((torch.tensor([self.config.log_alpha], device=self.device), self.log_counts))
        logjoint = logprior[None, :] - nll
        with torch.no_grad():
            destination = logjoint.argmax(1)
            novel = destination == 0
            if self.experts:
                # Match official normalization, including dummy in max subtraction
                # and the original 1e-7 guard. Novel examples update usage but not
                # expert loss, as in official ndpm.py.
                raw = torch.exp(logjoint - logjoint.max(1, keepdim=True).values)
                raw[:, 0] = 0
                responsibility = raw / (raw.sum(1, keepdim=True) + 1e-7)
                usage = responsibility[:, 1:].sum(0)
                self.log_counts = torch.logaddexp(self.log_counts, usage.log())
            else:
                responsibility = torch.zeros_like(logjoint)
                usage = torch.empty(0, device=self.device)
        if self.experts:
            losses = (nll[:, 1:] * responsibility[:, 1:] * (~novel)[:, None]).sum(0)
            # Official wake prior/MAP regularizer is absent: retain that behavior;
            # weight decay is added explicitly in sleep phases below.
            for go, do in self.optimizers:
                go.zero_grad(set_to_none=True)
                do.zero_grad(set_to_none=True)
            if bool((~novel).any()):
                losses.sum().backward()
                for expert, (go, do), used in zip(self.experts, self.optimizers, usage):
                    if used.item() > self.config.update_min_usage:
                        torch.nn.utils.clip_grad_value_(expert.parameters(), self.config.gradient_clip_value)
                        go.step()
                        do.step()
        novelty_count = int(novel.sum().item())
        self.routing_counts.append({"novel": novelty_count, "assigned": len(x) - novelty_count,
                                    "experts_before_sleep": self.expert_count})
        for row, label in zip(x[novel].detach().cpu(), y[novel].detach().cpu()):
            self.stm_x.append(row.clone())
            self.stm_y.append(int(label))
            self.stm_tasks.append(self.update_count)
            self.stm_peak = max(self.stm_peak, self.stm_size)
            if self.stm_size == self.config.stm_capacity:
                self._sleep()

    def _new_expert_hidden(self, x, new_expert, generative):
        previous = None
        with torch.no_grad():
            for precursor in self.experts[:-1]:
                previous = (precursor.generative_hidden(x, previous) if generative
                            else precursor.discriminative_hidden(x, previous))
        return (new_expert.generative_hidden(x, previous) if generative
                else new_expert.discriminative_hidden(x, previous))

    def _sleep(self):
        if self.stm_size != self.config.stm_capacity:
            raise RuntimeError("Sleep is triggered by a full STM only")
        x = torch.stack(self.stm_x).to(self.device)
        y = torch.tensor(self.stm_y, dtype=torch.long, device=self.device)
        expert = self._initialize_expert(self.expert_count)
        self.experts.append(expert)
        go = torch.optim.Adam(expert.g_parameters(), lr=self.config.lr_g)
        do = torch.optim.Adam(expert.d_parameters(), lr=self.config.lr_d)
        self.optimizers.append((go, do))
        # Official new prior starts at 1 and adds sleep training size; no heldout
        # STM validation records are removed in this adaptation.
        self.log_counts = torch.cat((self.log_counts,
                                     torch.tensor([math.log(len(x) + 1)], device=self.device)))
        old_visit_count = 0
        origins = torch.tensor(self.stm_tasks, device=self.device)
        for generative, steps, optimizer in [(True, self.config.sleep_steps_g, go),
                                             (False, self.config.sleep_steps_d, do)]:
            parameters = expert.g_parameters() if generative else expert.d_parameters()
            for iteration in range(steps):
                ids = torch.randint(len(x), (self.config.sleep_batch_size,),
                                    generator=self._torch_rng, device=self.device)
                old_visit_count += int((origins[ids] < self.update_count).sum())
                if iteration % 1000 == 0: print("CN_SLEEP", self.update_count+1, self.expert_count, "g" if generative else "d", iteration, steps, flush=True)
                xx, yy = x[ids], y[ids]
                hidden = self._new_expert_hidden(xx, expert, generative)
                if generative:
                    loss = expert.density_nll(xx, hidden, self._epsilon(len(xx), True))
                else:
                    pred = expert.log_predictions(hidden)
                    loss = self._class_nll(pred[:, None, :], yy)[:, 0]
                penalty = self.config.weight_decay * sum(p.square().sum() for p in parameters)
                objective = loss.mean() + penalty
                if not torch.isfinite(objective):
                    raise FloatingPointError("Nonfinite CN-DPM sleep objective")
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                torch.nn.utils.clip_grad_value_(parameters, self.config.gradient_clip_value)
                optimizer.step()
        old = sum(t < self.update_count for t in self.stm_tasks)
        self.sleep_access.append(dict(update=self.update_count+1, total_stm_rows=len(x), old_stm_rows=old,
                                      optimizer_sample_visits=(self.config.sleep_steps_g+self.config.sleep_steps_d)*self.config.sleep_batch_size,
                                      old_sample_visits_actual=old_visit_count))
        self.sleep_count += 1
        self.sleep_sizes.append(len(x))
        self.total_sleep_steps_g += self.config.sleep_steps_g
        self.total_sleep_steps_d += self.config.sleep_steps_d
        self.stm_x.clear()
        self.stm_y.clear()
        self.stm_tasks.clear()

    def update(self, X, y):
        started = time.perf_counter()
        x = self._validate_x(X, nonempty=True)
        labels = np.asarray(y)
        if labels.shape != (len(x),) or not np.isin(labels, np.arange(self.classes)).all():
            raise ValueError("y must contain one configured global label per example")
        labels = labels.astype(np.int64)
        if self.normalizer_mean is None:
            self.normalizer_mean = x.mean(0, dtype=np.float64).astype(np.float32)
            std = x.std(0, ddof=0, dtype=np.float64)
            self.normalizer_std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
        x = torch.from_numpy(self._normalize(x)).to(self.device)
        y = torch.from_numpy(labels).to(self.device)
        order = np.random.default_rng(self.seed + self.update_count * 100).permutation(len(x))
        for start in range(0, len(x), self.config.batch_size):
            ids = torch.as_tensor(order[start:start+self.config.batch_size], device=self.device)
            self._wake(x[ids], y[ids])
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.seen_samples += len(x)
        self.update_count += 1
        self.update_duration = time.perf_counter() - started
        self.stage_history.append({"update": self.update_count, "experts": self.expert_count,
                                   "stm_residual": self.stm_size, "sleep_count": self.sleep_count,
                                   "seen_samples": self.seen_samples,
                                   "update_duration": self.update_duration,
                                   "memory_bytes": self.memory_bytes,
                                   "optimizer_bytes": self.optimizer_bytes})
        return self

    def predict_proba(self, X):
        if not self.experts:
            raise RuntimeError("CN-DPM has no trained expert; STM must fill before inference")
        x = self._normalize(self._validate_x(X))
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(x), self.config.prediction_batch_size):
                xx = torch.from_numpy(x[start:start+self.config.prediction_batch_size]).to(self.device)
                densities, pred = self._collect(xx, training=False)
                # At inference y is unknown: gate on density and cumulative usage
                # only; exclude placeholder and marginalize trained experts.
                gate = F.log_softmax(self.log_counts[None, :] - densities[:, 1:], dim=1)
                logprobs = torch.logsumexp(gate[:, :, None] + pred[:, 1:, :], dim=1)
                outputs.append(logprobs.exp().cpu().numpy())
        return np.concatenate(outputs) if outputs else np.empty((0, self.classes), dtype=np.float32)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    @property
    def memory_bytes(self):
        # Deployment weights include placeholder, fixed MC bank, normalizer,
        # expert priors and bounded residual STM. Excludes optimizer workspace.
        size = _tensor_bytes(list(self.placeholder.parameters()) + [self.log_counts, self.eval_epsilon])
        size += sum(_tensor_bytes(expert.parameters()) for expert in self.experts)
        size += _tensor_bytes(self.stm_x) + 8 * len(self.stm_y)
        if self.normalizer_mean is not None:
            size += self.normalizer_mean.nbytes + self.normalizer_std.nbytes
        return int(size)

    @property
    def optimizer_bytes(self):
        return int(sum(_tensor_bytes(v for state in optimizer.state.values()
                                     for v in state.values() if torch.is_tensor(v))
                       for pair in self.optimizers for optimizer in pair))

    def diagnostics(self):
        return {"official_commit": OFFICIAL_COMMIT, "adaptation": "frozen-feature MLP-sharing CN-DPM",
                "config": asdict(self.config), "update_count": self.update_count,
                "seen_samples": self.seen_samples, "expert_count": self.expert_count,
                "stm_size": self.stm_size, "stm_peak": self.stm_peak,
                "sleep_count": self.sleep_count, "sleep_sizes": list(self.sleep_sizes), "sleep_access": list(self.sleep_access),
                "total_sleep_steps_g": self.total_sleep_steps_g,
                "total_sleep_steps_d": self.total_sleep_steps_d,
                "expert_prior_counts": self.log_counts.detach().exp().cpu().tolist(),
                "retained_novelty_sample_bytes": _tensor_bytes(self.stm_x) + 8 * self.stm_size,
                "memory_bytes": self.memory_bytes, "optimizer_bytes": self.optimizer_bytes,
                "history_replay": False, "novelty_stm": True,
                "normalization": "first update training mean/std, then frozen; constant columns std=1",
                "inference_density": f"Gaussian VAE MC-ELBO, fixed common {self.config.z_samples}-sample bank",
                "stage_history": list(self.stage_history)}

    def get_memory_info(self):
        return {"model_and_stm_bytes": self.memory_bytes,
                "optimizer_bytes": self.optimizer_bytes,
                "expert_count": self.expert_count, "stm_size": self.stm_size,
                "stm_peak": self.stm_peak, "stm_capacity": self.config.stm_capacity,
                "retained_novelty_sample_bytes": _tensor_bytes(self.stm_x) + 8 * self.stm_size}
