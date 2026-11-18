"""EdgeCL (IEEE TWC 2026), paper-based adaptation to frozen Mantis tokens.

Primary: Hu, Li, Su, Luo, "Cross-Domain Continual Learning for Edge Intelligence
in Wireless ISAC Networks", IEEE Trans. Wireless Commun. 25:1109--1122, 2026,
arXiv:2502.12736v2, Sec. IV, Eqs. (19)--(28), Algorithm 1, Sec. V-A2 settings.
Kept: per-token two-layer MLP encoder (128, 64), two MHSA/FF layers with 8 heads,
first-vector probability predictor, Xavier-uniform initialization, dropout 0.1,
Adam 1e-3, hybrid k-means++/herding core-set (beta=0.9) in the encoder space,
distilled labels with confidence downscaling eta=2 replayed with the same
temperature, and the first-order robustness-enhanced (SAM-type) update.
Adaptation: tokens are the 18 ordered 512-D Mantis group mean/SD vectors with a
learned positional embedding instead of the CSI timestamp encoder; the paper's
fixed E exemplars per class and domain is replaced by the same total row budget as
every replay method here, rebalanced over retained domains by the same hybrid rule.
Same-input study: ``input_kind`` 'cat9728' reads 19 tokens (18 Mantis + the amplitude descriptor) and
'amp512' one amplitude token, each rescaled to unit coordinate (Mantis) or RMS (amplitude) scale.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .consense import TOKENS, WIDTH, tokens as mantis_tokens

TOKEN_COUNT = {'m9216': TOKENS, 'cat9728': TOKENS+1, 'amp512': 1}


def tokens(x, kind='m9216'):
    if kind == 'm9216': return mantis_tokens(x)
    x = torch.as_tensor(x, dtype=torch.float32)
    if kind == 'cat9728': x = torch.cat([x[:, :9216]*np.sqrt(9216), x[:, 9216:]*np.sqrt(512)], 1)
    else: x = x*np.sqrt(512)
    return x.reshape(len(x), -1, WIDTH)


class Discriminator(nn.Module):
    def __init__(self, classes, hidden=128, latent=64, heads=8, layers=2, dropout=.1, count=TOKENS):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(WIDTH, hidden), nn.ReLU(), nn.Linear(hidden, latent))
        self.position = nn.Parameter(torch.zeros(1, count, latent))
        layer = nn.TransformerEncoderLayer(latent, heads, 2*latent, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(latent, classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def features(self, x):
        return self.transformer(self.dropout(self.encoder(x)+self.position))[:, 0]

    def forward(self, x):
        return self.head(self.features(x))


def kmeans_pp(f, k, rng, iterations=100):
    centers = [f[rng.integers(len(f))]]
    for _ in range(1, k):
        d = np.min(((f[:, None]-np.array(centers)[None]) ** 2).sum(-1), 1)
        centers.append(f[rng.choice(len(f), p=d/d.sum())] if d.sum() > 0 else f[rng.integers(len(f))])
    c = np.array(centers)
    for _ in range(iterations):
        assign = ((f[:, None]-c[None]) ** 2).sum(-1).argmin(1)
        new = np.array([f[assign == j].mean(0) if np.any(assign == j) else c[j] for j in range(k)])
        if np.allclose(new, c): break
        c = new
    return c


def hybrid_select(f, count, beta, rng):
    """Eq. (19) nearest-to-centroid exemplars, then Eq. (20) herding for the rest."""
    if count >= len(f): return np.arange(len(f))
    clusters = int(round(beta*count)) if count > 1 else 1
    clusters = min(max(clusters, 1), count-1) if count > 1 else 1
    chosen = []
    for c in kmeans_pp(f, clusters, rng):
        d = ((f-c) ** 2).sum(1); d[chosen] = np.inf; chosen.append(int(d.argmin()))
    mean = f.mean(0); total = f[chosen].sum(0)
    while len(chosen) < count:
        d = (((total[None]+f)/(len(chosen)+1)-mean[None]) ** 2).sum(1); d[chosen] = np.inf
        j = int(d.argmin()); chosen.append(j); total = total+f[j]
    return np.array(chosen)


class EdgeCL:
    def __init__(self, seed, device, radius, classes, first_epochs=100, epochs=50, batch_size=128, lr=1e-3,
                 beta=.9, eta=2., hidden=128, latent=64, heads=8, layers=2, dropout=.1, input_kind='m9216'):
        self.input_kind = input_kind
        self.seed, self.device, self.radius = seed, torch.device(device), float(radius)
        self.classes, self.first_epochs, self.epochs = classes, first_epochs, epochs
        self.batch_size, self.lr, self.beta, self.eta = batch_size, lr, beta, eta
        devices = [self.device.index or 0] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            self.net = Discriminator(classes, hidden, latent, heads, layers, dropout, TOKEN_COUNT[input_kind]).to(self.device)
        # One entry per retained domain: features (n, 9216), hard labels, distilled labels, source indices.
        self.coreset = []; self.sessions = 0; self.metadata = []

    def encode(self, x):
        self.net.eval()
        with torch.inference_mode():
            return np.concatenate([self.net.features(tokens(x[i:i+256], self.input_kind).to(self.device)).double().cpu().numpy()
                                   for i in range(0, len(x), 256)])

    def logits(self, x):
        self.net.eval()
        with torch.inference_mode():
            return np.concatenate([self.net(tokens(x[i:i+256], self.input_kind).to(self.device)).double().cpu().numpy()
                                   for i in range(0, len(x), 256)])

    def rebalance(self, cap):
        domains = len(self.coreset)
        if not domains: return
        per = cap//(self.classes*domains)
        if per < 1: raise ValueError('Replay budget smaller than one row per class and domain')
        rng = np.random.default_rng(self.seed+7919*self.sessions)
        kept = []
        for entry in self.coreset:
            f = self.encode(entry['x']); keep = []
            for c in range(self.classes):
                ids = np.flatnonzero(entry['y'] == c)
                if len(ids) < per: raise ValueError('Insufficient retained class rows')
                keep.append(ids[hybrid_select(f[ids], per, self.beta, rng)])
            keep = np.concatenate(keep)
            kept.append({k: (v if k == 'task' else v[keep]) for k, v in entry.items()})
        self.coreset = kept

    def old_rows(self):
        return sum(len(e['y']) for e in self.coreset)

    def update(self, X, y, cap):
        x = np.asarray(X, np.float32); y = np.asarray(y, np.int64); net = self.net; first = self.sessions == 0
        if not first:
            self.rebalance(cap)
            assert self.old_rows() <= cap
        old_x = np.concatenate([e['x'] for e in self.coreset]) if self.coreset else np.empty((0, x.shape[1]), np.float32)
        old_p = np.concatenate([e['p'] for e in self.coreset]) if self.coreset else np.empty((0, self.classes), np.float32)
        pool = tokens(np.concatenate([x, old_x]), self.input_kind).to(self.device)
        hard = torch.from_numpy(np.concatenate([y, np.zeros(len(old_x), np.int64)])).to(self.device)
        soft = torch.from_numpy(np.concatenate([np.zeros((len(y), self.classes), np.float32), old_p])).to(self.device)
        epochs = self.first_epochs if first else self.epochs
        params = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        rng = np.random.default_rng(self.seed+100*self.sessions); history = []; n = len(y)

        def objective(ids):
            z = net(pool[ids]); cur = torch.as_tensor(ids < n, device=self.device); value = z.new_zeros(())
            if bool(cur.any()): value = value+F.cross_entropy(z[cur], hard[ids][cur], reduction='sum')
            if bool((~cur).any()):
                # Eq. (22): CE between distilled labels and the eta-downscaled prediction.
                value = value-(soft[ids][~cur]*F.log_softmax(z[~cur]/self.eta, 1)).sum()
            return value/len(ids)

        for epoch in range(epochs):
            net.train(); losses = []; cv = ov = 0
            for ids in np.array_split(rng.permutation(len(pool)), int(np.ceil(len(pool)/self.batch_size))):
                optimizer.zero_grad(set_to_none=True); loss = objective(ids)
                if not torch.isfinite(loss): raise FloatingPointError('EdgeCL loss')
                loss.backward()
                if self.radius > 0:
                    # Eqs. (25)--(28): evaluate the gradient at theta + eps*g/||g||, then step from theta.
                    norm = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in params if p.grad is not None)).clamp_min(1e-12)
                    shifts = []
                    with torch.no_grad():
                        for p in params:
                            e = torch.zeros_like(p) if p.grad is None else self.radius*p.grad/norm
                            p.add_(e); shifts.append(e)
                    optimizer.zero_grad(set_to_none=True); objective(ids).backward()
                    with torch.no_grad():
                        for p, e in zip(params, shifts): p.sub_(e)
                optimizer.step(); losses.append(float(loss.detach()))
                cv += int((ids < n).sum()); ov += int((ids >= n).sum())
            scheduler.step(); history.append(dict(epoch=epoch, loss=float(np.mean(losses)), current_visits=cv, old_visits=ov))
        net.eval()
        p = torch.softmax(torch.from_numpy(self.logits(x))/self.eta, 1).float().numpy()
        self.coreset.append(dict(x=x, y=y, p=p, index=np.arange(n), task=self.sessions))
        used = len(old_x); self.sessions += 1
        self.metadata.append(dict(session=self.sessions-1, rows=n, old_rows=used, epochs=epochs))
        self.er_epochs = history
        return self

    def finish_stage(self, cap):
        # Algorithm 1 line 11: keep only the distilled core-set before the next domain arrives.
        self.rebalance(cap)

    def predict_logits(self, X):
        return self.logits(np.asarray(X, np.float32)).astype(np.float32)

    @property
    def memory_bytes(self):
        n = sum(p.numel()*p.element_size() for p in self.net.state_dict().values())
        return int(n+sum(e['x'].nbytes+e['p'].nbytes+e['y'].nbytes for e in self.coreset))

    def diagnostics(self):
        return dict(paper='IEEE TWC 2026 EdgeCL', implementation='paper-based Mantis-token adaptation', radius=self.radius,
                    beta=self.beta, eta=self.eta, replay_rows=self.old_rows(), retained_domains=len(self.coreset),
                    memory_bytes=self.memory_bytes, metadata=self.metadata)


class EdgeCLWidar(EdgeCL):
    """Widar3.0 streams: the per-class budget floor(cap/C) is divided over retained domains and the remainder
    rows go to the most recent domains (identical to EdgeCL whenever the division is exact)."""
    def rebalance(self, cap):
        domains = len(self.coreset)
        if not domains: return
        quota = cap//self.classes; base, extra = divmod(quota, domains)
        rng = np.random.default_rng(self.seed+7919*self.sessions); kept = []
        for k, entry in enumerate(self.coreset):
            per = base+(k >= domains-extra)
            if per == 0: continue
            f = self.encode(entry['x']); keep = []
            for c in range(self.classes):
                ids = np.flatnonzero(entry['y'] == c)
                if len(ids) < per: raise ValueError('Insufficient retained class rows')
                keep.append(ids[hybrid_select(f[ids], per, self.beta, rng)])
            keep = np.concatenate(keep)
            kept.append({key: (v if key == 'task' else v[keep]) for key, v in entry.items()})
        self.coreset = kept
