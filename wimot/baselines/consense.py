"""ConSense (AAAI 2025), paper-based adaptation to frozen Mantis tokens.

Primary: Li et al., "ConSense: Continually Sensing Human Activity with WiFi via
Growing and Picking", AAAI 2025, arXiv:2502.17483, Eqs. 1--6.
Kept: Gaussian range positional encoding, one MHSA stack frozen after the first
session, per-session parallel-adapter prefixes concatenated to keys/values with
all earlier prefixes frozen, and activation-stability masks on every MLP linear
layer. Exemplar-free: only per-neuron mean activations persist between sessions.
Adaptation: the 9216-D Mantis output is read as 18 ordered 512-D group mean/SD
tokens instead of raw CSI time steps; the class set is fixed (domain-incremental),
so the output layer is shared and masked like the other MLP layers. No task ID.
Same-input study: ``input_kind`` 'cat9728' reads 19 tokens (the 18 Mantis tokens and the 512-D amplitude descriptor as one
further token, each rescaled to unit coordinate scale as for EdgeCL); the Gaussian range encoding then spans 19 positions.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TOKENS, WIDTH = 18, 512


TOKEN_COUNT = {'m9216': TOKENS, 'cat9728': TOKENS+1, 'amp512': 1}


def tokens(x, kind='m9216'):
    # First-domain normalization scales each coordinate by sqrt(9216); undo it so tokens have unit variance.
    x = torch.as_tensor(x, dtype=torch.float32)
    if kind == 'amp512':   # one 512-D amplitude token, rescaled to unit RMS (same rule as EdgeCL)
        return (x*np.sqrt(WIDTH)).reshape(len(x), 1, WIDTH)
    if kind == 'cat9728':
        return torch.cat([x[:, :TOKENS*WIDTH]*np.sqrt(TOKENS*WIDTH), x[:, TOKENS*WIDTH:]*np.sqrt(WIDTH)], 1).reshape(len(x), TOKEN_COUNT[kind], WIDTH)
    return (x*np.sqrt(TOKENS*WIDTH)).reshape(len(x), TOKENS, WIDTH)


class GaussianRange(nn.Module):
    """Li et al. 2021 Gaussian range encoding with learnable means and widths."""

    def __init__(self, n, d, k=10, sigma=8.):
        super().__init__()
        self.n = n
        self.mu = nn.Parameter((torch.arange(k, dtype=torch.float32)+.5)*n/k)
        self.sigma = nn.Parameter(torch.full((k,), float(sigma)))
        self.embedding = nn.Parameter(torch.randn(k, d)*.02)

    def forward(self, x):
        t = torch.arange(self.n, device=x.device, dtype=x.dtype)[:, None]
        w = torch.exp(-(t-self.mu).square()/(2*self.sigma.square()))
        return x+(w/w.sum(1, keepdim=True).clamp_min(1e-12))@self.embedding


class PrefixAttention(nn.Module):
    def __init__(self, d, heads, dropout, bottleneck):
        super().__init__()
        self.d, self.heads, self.bottleneck = d, heads, bottleneck
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.adapters = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

    def add_session(self):
        # Eq. 4: P_K, P_V = Tanh(X W_down) W_up.
        self.adapters.append(nn.Sequential(nn.Linear(self.d, self.bottleneck), nn.Tanh(), nn.Linear(self.bottleneck, 2*self.d)))
        return self.adapters[-1]

    def forward(self, x):
        b, n, d = x.shape; h = self.heads
        keys, values = [self.k(x)], [self.v(x)]
        for adapter in self.adapters:
            pk, pv = adapter(x).chunk(2, -1); keys.append(pk); values.append(pv)
        split = lambda z: z.reshape(b, -1, h, d//h).transpose(1, 2)
        q, k, v = split(self.q(x)), split(torch.cat(keys, 1)), split(torch.cat(values, 1))
        a = self.dropout(torch.softmax(q@k.transpose(-1, -2)/np.sqrt(d//h), -1))
        return self.o((a@v).transpose(1, 2).reshape(b, n, d))


class Network(nn.Module):
    def __init__(self, classes, heads=8, dropout=.1, bottleneck=64, hidden=(512, 128), count=TOKENS):
        super().__init__()
        self.position = GaussianRange(count, WIDTH)
        self.attention = PrefixAttention(WIDTH, heads, dropout, bottleneck)
        self.norm = nn.LayerNorm(WIDTH)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.ModuleList([nn.Linear(WIDTH, hidden[0]), nn.Linear(hidden[0], hidden[1]), nn.Linear(hidden[1], classes)])

    def forward(self, x, record=False):
        z = self.dropout(self.position(x))
        z = self.norm(z+self.attention(z)).mean(1)
        acts = []
        for i, layer in enumerate(self.mlp):
            z = layer(z)
            if i < len(self.mlp)-1: z = F.relu(z)
            acts.append(z)
        return (z, acts) if record else z


class ConSense:
    def __init__(self, seed, device, threshold, classes, first_epochs=100, epochs=50, batch_size=128, lr=1e-3,
                 heads=8, dropout=.1, bottleneck=64, input_kind='m9216'):
        self.input_kind = input_kind
        self.seed, self.device, self.threshold = seed, torch.device(device), float(threshold)
        self.classes, self.first_epochs, self.epochs = classes, first_epochs, epochs
        self.batch_size, self.lr, self.sessions = batch_size, lr, 0
        devices = [self.device.index or 0] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            self.net = Network(classes, heads, dropout, bottleneck, count=TOKEN_COUNT[input_kind]).to(self.device)
        self.previous_activation = None; self.metadata = []

    def mean_activations(self, x):
        self.net.eval(); total = None
        with torch.inference_mode():
            for i in range(0, len(x), 256):
                _, acts = self.net(tokens(x[i:i+256], self.input_kind).to(self.device), record=True)
                s = [a.sum(0).double() for a in acts]
                total = s if total is None else [u+v for u, v in zip(total, s)]
        return [v/len(x) for v in total]

    def update(self, X, y, cap=None):
        x = np.asarray(X, np.float32); y = np.asarray(y, np.int64); net = self.net
        first = self.sessions == 0
        if first:
            trainable = list(net.parameters()); masks = None; stable = None
        else:
            for p in net.parameters(): p.requires_grad_(False)
            adapter = net.attention.add_session().to(self.device)
            for p in adapter.parameters(): p.requires_grad_(True)
            current = self.mean_activations(x)
            stable = [(c-p).abs() <= self.threshold for c, p in zip(current, self.previous_activation)]
            masks = []
            for layer, s in zip(net.mlp, stable):
                layer.weight.requires_grad_(True); layer.bias.requires_grad_(True)
                keep = (~s).to(layer.weight.dtype)
                masks.append((layer.weight, keep[:, None])); masks.append((layer.bias, keep))
            trainable = list(adapter.parameters())+[p for layer in net.mlp for p in layer.parameters()]
        epochs = self.first_epochs if first else self.epochs
        optimizer = torch.optim.Adam(trainable, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        rng = np.random.default_rng(self.seed+100*self.sessions); history = []
        xt = tokens(x, self.input_kind).to(self.device); yt = torch.from_numpy(y).to(self.device)
        for epoch in range(epochs):
            net.train(); losses = []; visits = 0
            for ids in np.array_split(rng.permutation(len(y)), int(np.ceil(len(y)/self.batch_size))):
                loss = F.cross_entropy(net(xt[ids]), yt[ids])
                if not torch.isfinite(loss): raise FloatingPointError('ConSense loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                if masks is not None:
                    for p, keep in masks:
                        if p.grad is not None: p.grad.mul_(keep)
                optimizer.step(); losses.append(float(loss.detach())); visits += len(ids)
            scheduler.step(); history.append(dict(epoch=epoch, loss=float(np.mean(losses)), current_visits=visits, old_visits=0))
        net.eval()
        self.previous_activation = self.mean_activations(x)
        self.metadata.append(dict(session=self.sessions, rows=len(y), epochs=epochs,
                                  stable_neurons=None if stable is None else [int(s.sum()) for s in stable],
                                  layer_widths=[l.out_features for l in net.mlp], prefixes=len(net.attention.adapters)))
        self.sessions += 1; self.er_epochs = history
        return self

    def predict_logits(self, X):
        self.net.eval(); x = np.asarray(X, np.float32)
        with torch.inference_mode():
            return np.concatenate([self.net(tokens(x[i:i+256], self.input_kind).to(self.device)).float().cpu().numpy() for i in range(0, len(x), 256)])

    @property
    def memory_bytes(self):
        n = sum(p.numel()*p.element_size() for p in self.net.state_dict().values())
        return int(n+sum(v.numel()*8 for v in (self.previous_activation or [])))

    def diagnostics(self):
        return dict(paper='AAAI2025 ConSense', implementation='paper-based Mantis-token adaptation', threshold=self.threshold,
                    replay_rows=0, prefixes=len(self.net.attention.adapters), memory_bytes=self.memory_bytes, metadata=self.metadata)
