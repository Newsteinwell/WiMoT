"""Frozen-encoder features of prepared windows: <out>/<domain>_<split>.npz with 'features' (N x 9728) and 'labels'.

  python -m wimot.features.extract --prepared DIR --dataset 3do --out DIR [--model mantis] [--weights DIR]

Per recording, every channel is linearly interpolated to 512 samples and encoded by the frozen model. The channel
embeddings are pooled by ordered nine-group mean and sample SD (9216 dimensions) and concatenated with the 512-D amplitude
descriptor (per-channel mean and SD resampled to 256 coordinates each), giving 9728 dimensions.

  mantis         MantisV2 (paris-noah/MantisV2), layer-2 combined token, two recordings per forward pass, bfloat16 autocast
  mantis_random  MantisV2 architecture with seeded random initialization (torch.manual_seed(20261014))
  moment, tspulse, chronos2, timesfm25, timesfm30
                 other released encoders (Table backbone); an embedding of width D != 512 is mapped to 512 by a PCA fitted
                 on the first-domain TRAIN channel embeddings only. 16 recordings per call, in chunks of channel series.
Model weights are read from --weights (a local snapshot of the revision listed in THIRD_PARTY_NOTICES.md).
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..common import STREAMS, digest, sha, write_json

SPLITS = ('train', 'validation', 'test')


def pool(hidden):
    """Ordered nine-group mean/sample-SD pooling of batch x channels x 512 embeddings."""
    h = np.asarray(hidden, dtype=np.float64)
    if h.ndim != 3 or h.shape[1] < 18 or h.shape[2] != 512: raise ValueError('Expected batch x channels x 512')
    return np.stack([np.stack([g.mean(1), g.std(1, ddof=1)], axis=1)
                     for g in np.array_split(h, 9, axis=1)], axis=1).reshape(len(h), 9216).astype(np.float32)


def amplitude(x):
    """Per-channel mean and sample SD over time, each linearly resampled to 256 coordinates."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or not np.isfinite(x).all(): raise ValueError('Finite batch x channels x time required')
    coords = np.arange(x.shape[1]); target = np.linspace(0, x.shape[1]-1, 256)
    mean = x.mean(-1); sd = x.std(-1, ddof=1)
    return np.array([np.concatenate([np.interp(target, coords, m), np.interp(target, coords, s)])
                     for m, s in zip(mean, sd)], dtype=np.float32)


def state_hash(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def build(name, weights):
    """Returns (model, encode: (N, 512) float32 CUDA series -> (N, D) float32, channel-series chunk size)."""
    if name in ('mantis', 'mantis_random'):
        from mantis.architecture import MantisV2
        if name == 'mantis_random': torch.manual_seed(20261014)
        model = MantisV2(device='cuda', return_transf_layer=2, output_token='combined', pre_training=False)
        if name == 'mantis': model = model.from_pretrained(str(weights))
        model = model.eval().requires_grad_(False)

        def enc(s):
            with torch.autocast('cuda', dtype=torch.bfloat16): return model(s[:, None]).float()
        return model, enc, 2048
    if name == 'moment':
        from momentfm import MOMENTPipeline
        model = MOMENTPipeline.from_pretrained(str(weights), model_kwargs={'task_name': 'embedding'})
        model.init(); model = model.to('cuda').eval().requires_grad_(False)

        def enc(s):
            mask = torch.ones(s.shape, device=s.device, dtype=torch.long)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                return model(x_enc=s[:, None], input_mask=mask).embeddings.float()
        return model, enc, 1024
    if name == 'tspulse':
        from tsfm_public.models.tspulse import TSPulseForReconstruction
        model = TSPulseForReconstruction.from_pretrained(str(weights)).to('cuda').eval().requires_grad_(False)

        def enc(s):
            out = model.backbone(past_values=s[:, :, None], enable_masking=False, return_dict=True)
            return out.last_hidden_flatten_state.float().reshape(len(s), -1)
        return model, enc, 8192
    if name == 'chronos2':
        from chronos import BaseChronosPipeline
        pipe = BaseChronosPipeline.from_pretrained(str(weights), device_map='cuda', torch_dtype=torch.float32)
        model = pipe.model.eval().requires_grad_(False)

        def enc(s):
            out, _, _, n = model.encode(context=s)
            return out[0][:, :n].float().mean(1)
        return model, enc, 1024
    if name == 'timesfm25':
        import timesfm
        from timesfm.torch import util as tutil
        wrapper = timesfm.TimesFM_2p5_200M_torch(torch_compile=False)
        wrapper.load_checkpoint(str(weights))
        model = wrapper.model.eval().requires_grad_(False); p = model.p

        def enc(s):
            b = len(s); x = s.reshape(b, -1, p); m = torch.zeros_like(x, dtype=torch.bool)
            n = torch.zeros(b, device=s.device); mu = torch.zeros(b, device=s.device); sg = torch.zeros(b, device=s.device)
            mus, sgs = [], []
            for i in range(x.shape[1]):
                (n, mu, sg), _ = tutil.update_running_stats(n, mu, sg, x[:, i], m[:, i]); mus.append(mu); sgs.append(sg)
            normed = tutil.revin(x, torch.stack(mus, 1), torch.stack(sgs, 1), reverse=False)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                (_, h, _, _), _ = model(normed, m)
            return h.float().mean(1)
        return model, enc, 4096
    if name == 'timesfm30':
        from timesfm3 import TimesFM3Torch
        model = TimesFM3Torch.from_pretrained(str(weights)).to('cuda').eval().requires_grad_(False)
        p = model.input_patch_len

        def enc(s):
            b = len(s); v = s.reshape(b, 1, -1, p)
            inputs = dict(values=v, masks=torch.zeros_like(v, dtype=torch.bool),
                          patch_is_target=torch.ones(v.shape[:3], dtype=torch.bool, device=s.device))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                h = model(inputs, return_aux_outputs=True)['__call__:transformer_output']
            return h.float().reshape(b, -1, h.shape[-1]).mean(1)
        return model, enc, 4096
    raise ValueError(name)


@torch.inference_mode()
def embed(enc, x, chunk):
    b, c, t = x.shape
    s = torch.from_numpy(np.ascontiguousarray(x, np.float32)).reshape(b*c, 1, t)
    out = []
    for i in range(0, len(s), chunk):
        part = torch.nn.functional.interpolate(s[i:i+chunk].cuda(), size=512, mode='linear', align_corners=False)[:, 0]
        out.append(enc(part).cpu())
    e = torch.cat(out).numpy().astype(np.float32).reshape(b, c, -1)
    if not np.isfinite(e).all(): raise ValueError('Non-finite embedding')
    return e


@torch.inference_mode()
def mantis_semantic(model, x):
    """Main MantisV2 recipe: two recordings per forward pass; repeated on the first two rows as a determinism check."""
    def encode(batch):
        b, c, t = batch.shape
        tensor = torch.from_numpy(np.ascontiguousarray(batch, np.float32)).cuda().reshape(b*c, 1, t)
        tensor = torch.nn.functional.interpolate(tensor, size=512, mode='linear', align_corners=False)
        with torch.autocast('cuda', dtype=torch.bfloat16): hidden = model(tensor)
        return pool(hidden.float().cpu().numpy().reshape(b, c, 512))
    features = np.concatenate([encode(x[i:i+2]) for i in range(0, len(x), 2)])
    np.testing.assert_array_equal(encode(x[:2]), features[:2])
    return features


def load(prepared, domain, split):
    with np.load(Path(prepared)/f'{domain}_{split}.npz') as f: return f['windows'], f['labels']


def run(a):
    torch.backends.cuda.matmul.allow_tf32 = False
    model, enc, chunk = build(a.model, a.weights); before = state_hash(model)
    domains = STREAMS[a.dataset]['domains']; a.out.mkdir(parents=True, exist_ok=True)
    meta = dict(complete=False, dataset=a.dataset, model=a.model, dimension=9728, encoder_state_sha256=before,
                files_sha256={}, arrays_sha256={}, audits={})
    project = None
    if a.model != 'mantis':
        x, _ = load(a.prepared, domains[0], 'train')
        e0 = np.concatenate([embed(enc, x[i:i+a.rows], chunk) for i in range(0, len(x), a.rows)]); width = e0.shape[-1]
        if width != 512:
            flat = e0.reshape(-1, width); mu = flat.mean(0, dtype=np.float64); cov = np.zeros((width, width))
            for i in range(0, len(flat), 65536):
                c = flat[i:i+65536].astype(np.float64)-mu; cov += c.T@c
            cov /= len(flat)-1; vals, vecs = np.linalg.eigh(cov)
            order = np.argsort(vals)[::-1][:512]; vals, vecs = vals[order], vecs[:, order]
            vecs *= np.sign(vecs[np.abs(vecs).argmax(0), np.arange(512)])
            np.savez(a.out/'pca_first_train.npz', mean=mu, components=vecs, eigenvalues=vals)
            project = lambda e: ((e.astype(np.float64)-mu)@vecs).astype(np.float32)
            meta['projection'] = dict(embedding_dimension=width, components=512, explained_variance=float(vals.sum()/np.trace(cov)))
            del flat, cov
    for domain in domains:
        for split in SPLITS:
            file = f'{domain}_{split}.npz'; started = time.perf_counter(); x, y = load(a.prepared, domain, split)
            if a.model == 'mantis': sem = mantis_semantic(model, x)
            else:
                e = e0 if (domain == domains[0] and split == 'train') else \
                    np.concatenate([embed(enc, x[i:i+a.rows], chunk) for i in range(0, len(y), a.rows)])
                again = embed(enc, x[:2], chunk)
                consistency = float(np.abs(again-e[:2]).max()/max(np.abs(e[:2]).mean(), 1e-12))
                if consistency > 0.05: raise ValueError(f'Batch-dependent embedding {consistency}')
                sem = np.concatenate([pool(e[i:i+64] if project is None else project(e[i:i+64])) for i in range(0, len(e), 64)])
                meta['audits'][file] = dict(rebatch_relative_max_diff=consistency)
            features = np.concatenate([sem, amplitude(x)], axis=1)
            assert features.shape == (len(y), 9728) and np.isfinite(features).all()
            np.savez_compressed(a.out/file, features=features, labels=y)
            meta['files_sha256'][file] = sha(a.out/file); meta['arrays_sha256'][file] = dict(features=digest(features), labels=digest(y))
            meta['audits'].setdefault(file, {}).update(samples=len(y), seconds=time.perf_counter()-started)
            write_json(a.out/'passport.json', meta); print('FEATURES', a.model, file, features.shape, flush=True)
    assert state_hash(model) == before, 'Frozen encoder changed'
    meta['complete'] = True; write_json(a.out/'passport.json', meta)
    check(a)


def check(a):
    """Compare the arrays with the digests of the original features (configs/digests.json), when listed."""
    path = Path(__file__).resolve().parents[2]/'configs'/'digests.json'
    if not path.exists(): return
    expected = json.loads(path.read_text())['features'].get(a.model, {}).get(a.dataset, {})
    same = {}
    for file, arrays in expected.items():
        with np.load(a.out/file) as f: same[file] = all(digest(f[k]) == v['sha256'] for k, v in arrays.items())
    print('FEATURES_MATCH_ORIGINAL', a.model, a.dataset, sum(same.values()), '/', len(same), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--prepared', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--dataset', required=True, choices=list(STREAMS))
    p.add_argument('--model', default='mantis', choices=('mantis', 'mantis_random', 'moment', 'tspulse', 'chronos2', 'timesfm25', 'timesfm30'))
    p.add_argument('--weights', type=Path, help='Local model snapshot (not needed for mantis_random)')
    p.add_argument('--rows', type=int, default=16)
    run(p.parse_args(argv))


if __name__ == '__main__':
    main()
