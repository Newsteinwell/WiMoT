"""DUCT CVPR 2025: independent Mantis-output adaptation of its two consolidations.

Reference: Da-Wei Zhou et al., arXiv:2410.00911, and official repository
Estrella-fugaz/CVPR25-DUCT, commit ecfb579f418b555b029a6ae11e74e5dd47ee6b59.
This is not an official ViT/image experiment reproduction. A trainable MLP
on shared frozen Mantis features replaces the ViT backbone; raw frozen
feature class means supply its immutable semantic anchor coordinates.

MIT License. Copyright (c) 2025 Pluto.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from dataclasses import asdict, dataclass, replace
import math
import time

import numpy as np
from scipy.special import logsumexp
import torch
from torch import nn
from torch.nn import functional as F


OFFICIAL_COMMIT = "ecfb579f418b555b029a6ae11e74e5dd47ee6b59"


@dataclass(frozen=True)
class DUCTConfig:
    hidden_dim: int = 256
    feature_dim: int = 128
    first_epochs: int = 100
    epochs: int = 50
    retrain_epochs: int = 5
    batch_size: int = 128
    learning_rate: float = .1
    backbone_lr_scale: float = .001
    weight_decay: float = .0005
    momentum: float = .9
    merge_scalar: float = .25
    head_merge_ratio: float = .25
    ot_reg: float = .2
    prediction_batch_size: int = 256

    def __post_init__(self):
        ints = (self.hidden_dim, self.feature_dim, self.first_epochs, self.epochs, self.retrain_epochs,
                self.batch_size, self.prediction_batch_size)
        if any(type(v) is not int or v <= 0 for v in ints):
            raise ValueError("Widths, epochs and batch sizes must be positive integers")
        if any(not np.isfinite(v) or v <= 0 for v in
               (self.learning_rate, self.backbone_lr_scale, self.ot_reg)):
            raise ValueError("Learning rates and OT regularization must be positive")
        if any(not np.isfinite(v) or not 0 <= v <= 1 for v in
               (self.merge_scalar, self.head_merge_ratio, self.momentum)):
            raise ValueError("Merge ratios and momentum must lie in [0,1]")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative")


def sinkhorn_uniform(cost, reg=.2, tolerance=1e-9, max_iterations=20000):
    """Uniform balanced OT in log coordinates; no column-mass normalization.

    Rows are current-domain/source classes; columns are historical/target
    classes. The returned plan has row mass 1/n and column mass 1/m.
    Log arithmetic solves the same entropic OT problem as official POT
    sinkhorn. Epsilon scaling supplies dual warm starts when CSI costs/reg
    make a near-diagonal kernel and ordinary Sinkhorn converges very slowly.
    The FINAL objective still uses the caller's original regularization.
    """
    cost = np.asarray(cost, dtype=np.float64)
    if (cost.ndim != 2 or not all(cost.shape) or not np.isfinite(cost).all()
            or (cost < 0).any() or not np.isfinite(reg) or reg <= 0):
        raise ValueError("OT requires a nonempty finite nonnegative cost and positive reg")
    if tolerance <= 0 or max_iterations <= 0:
        raise ValueError("OT tolerance and iteration limit must be positive")
    n, m = cost.shape
    loga, logb = np.full(n, -math.log(n)), np.full(m, -math.log(m))
    logu, logv = np.zeros(n), np.zeros(m)
    epsilon = max(float(reg), float(np.ptp(cost)))
    schedule = []
    while epsilon > reg:
        schedule.append(epsilon)
        epsilon = max(reg, epsilon / 2)
    schedule.append(float(reg))
    previous_epsilon, stages = None, []
    total_iterations = 0
    for stage_epsilon in schedule:
        if previous_epsilon is not None:
            # logu=dual_u/epsilon and logv=dual_v/epsilon. Preserve the
            # physical dual potentials when reducing entropy regularization.
            logu *= previous_epsilon / stage_epsilon
            logv *= previous_epsilon / stage_epsilon
        kernel = -(cost - cost.min()) / stage_epsilon
        final = stage_epsilon == reg
        stage_tolerance = tolerance if final else max(tolerance, 1e-7)
        stage_limit = max_iterations if final else min(max_iterations, 2000)
        for iteration in range(1, stage_limit + 1):
            logu = loga - logsumexp(kernel + logv[None, :], axis=1)
            logv = logb - logsumexp(kernel + logu[:, None], axis=0)
            if iteration == 1 or iteration % 10 == 0 or iteration == stage_limit:
                plan = np.exp(kernel + logu[:, None] + logv[None, :])
                error = max(float(np.max(np.abs(plan.sum(1) - np.exp(loga)))),
                            float(np.max(np.abs(plan.sum(0) - np.exp(logb)))))
                if error < stage_tolerance:
                    break
        total_iterations += iteration
        stages.append({"reg": float(stage_epsilon), "iterations": iteration,
                       "marginal_error": error, "tolerance": stage_tolerance,
                       "converged_to_tolerance": error < stage_tolerance})
        previous_epsilon = stage_epsilon
    if not np.isfinite(plan).all() or error > 1e-6:
        raise RuntimeError(f"Sinkhorn failed to satisfy marginal masses: error={error}")
    return plan, {"iterations": total_iterations, "final_iterations": iteration,
                  "solver": "log_sinkhorn_epsilon_scaling",
                  "epsilon_schedule": stages, "marginal_error": error,
                  "tolerance": tolerance, "converged_to_tolerance": error < tolerance,
                  "transport_cost": float((plan * cost).sum()), "reg": float(reg)}


def transport_head(plan, current_weight):
    """Native DUCT T.T @ W_current, WITHOUT dividing by column masses."""
    plan = np.asarray(plan, dtype=np.float64)
    weight = np.asarray(current_weight, dtype=np.float64)
    if (plan.ndim != 2 or weight.ndim != 2 or plan.shape[0] != weight.shape[0]
            or not np.isfinite(plan).all() or not np.isfinite(weight).all()):
        raise ValueError("Incompatible OT plan and classifier weight shapes")
    return (plan.T @ weight).astype(np.float32)


class CosineHead(nn.Module):
    """Official nb_proxy=1, to_reduce=False, sigma initialized to one."""
    def __init__(self, dim, slots):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(slots, dim))
        self.sigma = nn.Parameter(torch.ones(1))
        nn.init.uniform_(self.weight, -1 / math.sqrt(dim), 1 / math.sqrt(dim))

    def forward(self, x):
        return self.sigma * F.linear(F.normalize(x, p=2, dim=1),
                                     F.normalize(self.weight, p=2, dim=1))


def _bytes(tensors):
    return int(sum(v.numel() * v.element_size() for v in tensors))


class DUCT:
    """Task-vector MLP merge plus semantic OT consolidation of domain heads.

    Training update boundaries are known. Inference receives only X and
    takes the maximum over all domain/class slots, followed by modulo class count.
    Only class means, head weights, initialization and cumulative task-vector
    parameters survive an update. No training example or label is cached.
    """
    def __init__(self, input_dim=256, seed=42, device="cuda", config=None,
                 merge_scalar=None, classes=3):
        if type(input_dim) is not int or input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.classes = int(classes)
        if self.classes < 2: raise ValueError("At least two classes required")
        self.input_dim, self.seed, self.device = input_dim, int(seed), torch.device(device)
        self.config = DUCTConfig() if config is None else config
        if not isinstance(self.config, DUCTConfig):
            raise TypeError("config must be DUCTConfig")
        if merge_scalar is not None:
            self.config = replace(self.config, merge_scalar=float(merge_scalar))
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed)
            self.backbone = nn.Sequential(nn.Linear(input_dim, self.config.hidden_dim),
                                           nn.ReLU(), nn.Linear(self.config.hidden_dim,
                                                                 self.config.feature_dim),
                                           nn.ReLU()).to(self.device)
        self.initial_state = {k: v.detach().cpu().clone()
                              for k, v in self.backbone.state_dict().items()}
        self.cumulative_delta = {k: torch.zeros_like(v) for k, v in self.initial_state.items()}
        self.head = None
        self.class_means = []
        self.num_tasks, self.update_duration = 0, 0.
        self.stage_history, self.ot_history = [], []
        self.last_optimizer_bytes = 0

    def _validate(self, X, y=None, nonempty=False):
        x = np.asarray(X, dtype=np.float32)
        if (x.ndim != 2 or x.shape[1] != self.input_dim or not np.isfinite(x).all()
                or (nonempty and not len(x))):
            raise ValueError("X must be a finite matrix with the configured feature width")
        if y is None:
            return x
        labels = np.asarray(y)
        if labels.shape != (len(x),) or not np.isin(labels, np.arange(self.classes)).all():
            raise ValueError("y requires configured global activity labels")
        if not np.array_equal(np.unique(labels), np.arange(self.classes)):
            raise ValueError("Every DUCT domain update requires all configured classes")
        return x, labels.astype(np.int64)

    def _extend_head(self):
        old_weight = None if self.head is None else self.head.weight.detach().clone()
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed + 100 * self.num_tasks + 17)
            head = CosineHead(self.config.feature_dim, self.classes * (self.num_tasks + 1)).to(self.device)
        if old_weight is not None:
            with torch.no_grad():
                head.weight[:len(old_weight)].copy_(old_weight)
        # Official update_fc does not copy sigma: expanding the head resets it.
        self.head = head

    def _fit(self, x, y, retrain=False):
        cfg = self.config
        groups = [{"params": self.head.parameters(), "lr": cfg.learning_rate}]
        if not retrain:
            groups.insert(0, {"params": self.backbone.parameters(),
                              "lr": cfg.learning_rate * cfg.backbone_lr_scale})
        # Native group lr stays lrate in both phases: official lr_re is shadowed
        # by the explicit fc-group learning rate. Preserve actual official code.
        optimizer = torch.optim.SGD(groups, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        epochs = cfg.retrain_epochs if retrain else cfg.first_epochs if self.num_tasks == 0 else cfg.epochs
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[max(1, int(epochs*.6)), max(1, int(epochs*.8))], gamma=.1)
        rng = np.random.default_rng(self.seed + self.num_tasks * 100 + (23 if retrain else 0))
        losses = []
        for _ in range(epochs):
            order = rng.permutation(len(x))
            total, count = 0., 0
            for start in range(0, len(x), cfg.batch_size):
                ids = torch.as_tensor(order[start:start+cfg.batch_size], device=self.device)
                xx, yy = x[ids], y[ids]
                if retrain:
                    with torch.no_grad():
                        representation = self.backbone(xx)
                else:
                    representation = self.backbone(xx)
                logits = self.head(representation)[:, self.classes * self.num_tasks:]
                loss = F.cross_entropy(logits, yy)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite DUCT loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * len(xx)
                count += len(xx)
            losses.append(total / count)
            scheduler.step()
        self.last_optimizer_bytes = _bytes(value for state in optimizer.state.values()
                                           for value in state.values() if torch.is_tensor(value))
        return losses

    def _merge_backbone(self, trained_state):
        """theta_merged = theta_0 + lambda * sum_s(theta_s - theta_0)."""
        merged = {}
        for key, initial in self.initial_state.items():
            updated = trained_state[key].detach().cpu()
            self.cumulative_delta[key] += self.config.merge_scalar * (updated - initial)
            merged[key] = initial + self.cumulative_delta[key]
        self.backbone.load_state_dict(merged)

    def _transport_historical_heads(self):
        if self.num_tasks == 0:
            return
        current_means = self.class_means[-1]
        current_weights = self.head.weight[-self.classes:].detach().cpu().numpy().copy()
        with torch.no_grad():
            for old_domain in range(self.num_tasks):
                old_means = self.class_means[old_domain]
                cost = np.linalg.norm(current_means[:, None, :] - old_means[None, :, :], axis=2)
                plan, info = sinkhorn_uniform(cost, reg=self.config.ot_reg)
                transported = torch.from_numpy(transport_head(plan, current_weights)).to(self.device)
                old = self.head.weight[self.classes * old_domain:self.classes * (old_domain + 1)]
                old.copy_((1 - self.config.head_merge_ratio) * old
                          + self.config.head_merge_ratio * transported)
                self.ot_history.append({"update": self.num_tasks + 1,
                                        "old_domain_slot": old_domain,
                                        "row_mass": plan.sum(1).tolist(),
                                        "column_mass": plan.sum(0).tolist(), **info})

    def update(self, X, y):
        started = time.perf_counter()
        x, labels = self._validate(X, y, nonempty=True)
        # Means remain in common frozen Mantis coordinates, never updated
        # using future data or current/merged MLP representation coordinates.
        means = np.stack([x[labels == c].mean(0, dtype=np.float64) for c in range(self.classes)])
        self.class_means.append(means)
        self.backbone.load_state_dict(self.initial_state)
        self._extend_head()
        xx, yy = torch.from_numpy(x).to(self.device), torch.from_numpy(labels).to(self.device)
        train_history = self._fit(xx, yy, retrain=False)
        self._merge_backbone(self.backbone.state_dict())
        retrain_history = self._fit(xx, yy, retrain=True)
        self._transport_historical_heads()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.num_tasks += 1
        self.update_duration = time.perf_counter() - started
        self.stage_history.append({"update": self.num_tasks, "current_training_rows": len(x),
                                   "classifier_slots": self.classes * self.num_tasks,
                                   "train_losses": train_history, "retrain_losses": retrain_history,
                                   "update_duration": self.update_duration,
                                   "memory_bytes": self.memory_bytes,
                                   "final_optimizer_workspace_bytes": self.last_optimizer_bytes,
                                   "sigma": float(self.head.sigma.item())})
        return self

    def predict_logits(self, X):
        if not self.num_tasks:
            raise RuntimeError("Update DUCT before prediction")
        x = self._validate(X)
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(x), self.config.prediction_batch_size):
                xx = torch.from_numpy(x[start:start+self.config.prediction_batch_size]).to(self.device)
                outputs.append(self.head(self.backbone(xx)).cpu().numpy())
        return (np.concatenate(outputs) if outputs else
                np.empty((0, self.classes * self.num_tasks), dtype=np.float32))

    def predict(self, X):
        return self.predict_logits(X).argmax(1) % self.classes

    @property
    def memory_bytes(self):
        size = _bytes(self.backbone.parameters())
        size += _bytes(self.initial_state.values()) + _bytes(self.cumulative_delta.values())
        if self.head is not None:
            size += _bytes(self.head.parameters())
        size += sum(v.nbytes for v in self.class_means)
        return int(size)

    def diagnostics(self):
        return {"official_commit": OFFICIAL_COMMIT,
                "adaptation": "DUCT shared-frozen-CSI-feature MLP adaptation",
                "config": asdict(self.config), "num_tasks": self.num_tasks,
                "classifier_slots": self.classes * self.num_tasks, "nb_proxy": 1,
                "predict_rule": "argmax over all domain/class slots, modulo class count; no task ID",
                "semantic_anchor": "raw class means in frozen Mantis256 space, train only",
                "representation": f"Linear{self.input_dim}-{self.config.hidden_dim}-{self.config.feature_dim}, ReLU; no BatchNorm",
                "retained_training_samples": 0, "history_replay": False,
                "memory_bytes": self.memory_bytes,
                "final_optimizer_workspace_bytes": self.last_optimizer_bytes,
                "sinkhorn": "epsilon-scaled log-domain balanced entropic OT; final native reg; T.T @ W without mass division",
                "ot_history": list(self.ot_history), "stage_history": list(self.stage_history),
                "train_boundary_known": True, "inference_task_id_known": False,
                "model_selection": "fixed final epoch; parent runner selects merge_scalar on validation only"}


class WarmStartedDUCT(DUCT):
    """DUCT whose task-vector reference is the 100-epoch first-domain CE-trained adaptation branch."""
    def update(self, X, y):
        if self.num_tasks:
            return super().update(X, y)
        started = time.monotonic(); x, labels = self._validate(X, y, nonempty=True)
        self.class_means.append(np.stack([x[labels == c].mean(0, dtype=np.float64) for c in range(self.classes)]))
        self._extend_head(); xx = torch.from_numpy(x).to(self.device); yy = torch.from_numpy(labels).to(self.device)
        history = self._fit(xx, yy, retrain=False)
        self.initial_state = {k: v.detach().cpu().clone() for k, v in self.backbone.state_dict().items()}
        self.cumulative_delta = {k: torch.zeros_like(v) for k, v in self.initial_state.items()}
        retrain = self._fit(xx, yy, retrain=True)
        self.num_tasks = 1; self.update_duration = time.monotonic()-started
        self.stage_history.append(dict(update=1, current_training_rows=len(x), classifier_slots=self.classes,
                                       train_losses=history, retrain_losses=retrain, update_duration=self.update_duration,
                                       memory_bytes=self.memory_bytes, first_domain_CE_reference=True,
                                       final_optimizer_workspace_bytes=self.last_optimizer_bytes, sigma=float(self.head.sigma.item())))
        return self
