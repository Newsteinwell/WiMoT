"""Configuration, feature loading, metrics, replay reservoir and small I/O helpers shared by every runner."""
import fcntl
import hashlib
import json
import math
import os
import random
import zlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
SEMANTIC, AMPLITUDE = 9216, 512


def config(name):
    return json.loads((REPO/'configs'/f'{name}.json').read_text())


STREAMS = config('streams')
PROTOCOL = config('protocol')
# WIMOT_LOCKS names an alternative locks file (e.g. configs/locks_sameinput_all.json: coefficients selected for the 9728-D input).
LOCKS = json.loads(Path(os.environ.get('WIMOT_LOCKS') or REPO/'configs'/'locks.json').read_text())


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def state_digest(state):
    h = hashlib.sha256()
    for k, v in state.items(): h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False)+'\n')
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@contextmanager
def file_lock(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(f, fcntl.LOCK_UN)


def stream(dataset, order=None):
    """Domains and class count of a stream; ``order`` permutes the domains by a digit string such as '021'."""
    base = STREAMS[dataset]
    domains = list(base['domains'])
    if order:
        if sorted(int(c) for c in order) != list(range(len(domains))): raise ValueError(f'Order {order} is not a permutation')
        domains = [domains[int(c)] for c in order]
    return dict(domains=domains, classes=base['classes'])


class FeatureStore:
    """Cached features ``<root>/<domain>_<split>.npz`` with arrays ``features`` (N x 9728) and ``labels``.

    If ``passport.json`` lists ``files_sha256``, each file is hash-checked on first access.
    """
    def __init__(self, root, verify=True):
        self.root = Path(root); self.verified = set()
        passport = self.root/'passport.json'
        self.hashes = read_json(passport).get('files_sha256', {}) if verify and passport.exists() else {}

    def path(self, domain, split):
        p = self.root/f'{domain}_{split}.npz'
        if p.name in self.hashes and p.name not in self.verified:
            if sha(p) != self.hashes[p.name]: raise ValueError(f'SHA256 mismatch: {p}')
            self.verified.add(p.name)
        return p

    def load(self, domain, split):
        with np.load(self.path(domain, split)) as f:
            return f['features'], f['labels']


def label_subset(y, dataset, domain, seed, fraction, classes):
    """Reduced-label subset of a later domain: per class max(2, ceil(f*n)) rows, keyed by (seed, dataset/domain, f)."""
    rng = np.random.default_rng([seed, zlib.crc32(f'{dataset}/{domain}'.encode()), round(fraction*1000)])
    keep = []
    for c in range(classes):
        ids = np.flatnonzero(y == c)
        keep.append(rng.permutation(ids)[:max(2, math.ceil(fraction*len(ids)))])
    return np.sort(np.concatenate(keep))


class Data:
    """Raw cached features of one stream, with the reduced-label rule applied to later TRAIN partitions."""
    def __init__(self, features, dataset, order=None, fraction=None, seed=None, verify=True):
        self.store = FeatureStore(features, verify); self.dataset = dataset
        s = stream(dataset, order); self.domains, self.classes = s['domains'], s['classes']
        self.order, self.fraction, self.seed = order, fraction, seed

    @property
    def tag(self):
        """Stream identifier used in output paths: dataset, then the order and the label fraction if given."""
        tag = self.dataset
        if self.order: tag += f'_order{self.order}'
        if self.fraction is not None: tag += f'_f{round(self.fraction*100):02d}'
        return tag

    def raw(self, domain, split):
        x, y = self.store.load(domain, split)
        if self.fraction is None or split != 'train' or domain == self.domains[0]: return x, y
        keep = label_subset(y, self.dataset, domain, self.seed, self.fraction, self.classes)
        return x[keep], y[keep]


def metrics(matrix):
    a = np.array([[np.nan if v is None else v for v in row] for row in matrix])
    final = a[-1]
    forgetting = [np.nanmax(a[t:-1, t])-final[t] for t in range(len(a)-1)]
    return dict(AFA=float(final.mean()), AFR=float(np.mean(forgetting)) if forgetting else None,
                BWT=float(np.mean([final[t]-a[t, t] for t in range(len(a)-1)])) if forgetting else None,
                latest_accuracy=float(final[-1]), final_per_domain=final.tolist())


def priority(seed, task, index):
    return int.from_bytes(hashlib.sha256(f'{seed}:{task}:{index}'.encode()).digest()[:8], 'big')


def retain(old, x, y, indices, task, classes, per_class, seed):
    """Class-stratified hash-priority reservoir over retained and arriving rows."""
    entries = old+[(x, y, np.full(len(y), task), indices)]
    arrays = [np.concatenate([row[k] for row in entries]) for k in range(4)]
    xx, yy, tt, ii = arrays
    keys = np.array([priority(seed, t, i) for t, i in zip(tt, ii)], dtype=np.uint64)
    selected = np.concatenate([np.flatnonzero(yy == c)[np.argsort(keys[yy == c], kind='stable')[:per_class]]
                               for c in range(classes)])
    if len(selected) != classes*per_class: raise ValueError('Insufficient current/retained class candidates')
    return [tuple(v[selected].copy() for v in arrays)]


def first_cap(rows, classes):
    return (rows//10)//classes*classes


def later_cap(cap, rows, classes, fixed=False):
    """Replay capacity after a later domain: old/(old+new) <= 1/10, never growing; ``fixed`` keeps the first capacity."""
    if fixed: return cap
    cap = min(cap, (rows//9)//classes*classes); assert 9*cap <= rows
    return cap


def empty_cache(dimension):
    return (np.empty((0, dimension), np.float32), np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64))


def summarize_rows(rows, key):
    """Group test records by ``key(row)`` and report seeds, values, mean and sample SD of AFA/AFR."""
    groups = {}
    for r in rows: groups.setdefault(key(r), []).append(r)
    out = {}
    for k, g in groups.items():
        g = sorted(g, key=lambda r: r['seed'])
        out[k] = {m: dict(seeds=[r['seed'] for r in g], values=[r['metrics'][m] for r in g],
                          mean=float(np.mean([r['metrics'][m] for r in g])),
                          sd=float(np.std([r['metrics'][m] for r in g], ddof=1)) if len(g) > 1 else None)
                  for m in ('AFA', 'AFR')}
    return out
