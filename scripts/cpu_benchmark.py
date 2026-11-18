"""CPU cost of WiMoT on one stream (paper Table 10): encoder latency per query, update time per domain, classifier
latency per query, stored-moment size, and peak memory.

    OMP_NUM_THREADS=4 python scripts/cpu_benchmark.py encoder --dataset widar --threads 4 --weights /data/weights/mantis
    OMP_NUM_THREADS=4 python scripts/cpu_benchmark.py stream  --dataset widar --threads 4 --features /data/features/mantis/widar

experiments/run_cpu_benchmark.sh fixes the thread count of every BLAS library before Python starts and writes
results/cpu_benchmark/<part>_<dataset>_t<threads>.json. The encoder input of a query is channels x 512 after
interpolation, so random windows of the prepared shape give the same encoder cost as recorded CSI.
"""
import argparse
import json
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SHAPES = {'3do': (52, 351), 'wimans': (270, 300), 'widar': (540, 300)}
CLASSES = {'3do': 3, 'wimans': 9, 'widar': 6}


def peak_mib():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r/2**20 if sys.platform == 'darwin' else r/2**10


def machine(threads):
    if sys.platform == 'darwin':
        cpu = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True).stdout.strip()
    else:
        cpu = next((l.split(':', 1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines()
                    if l.startswith('model name')), platform.processor())
    return dict(cpu=cpu, platform=platform.platform(), python=platform.python_version(), torch=torch.__version__,
                threads=threads, torch_threads=torch.get_num_threads())


def timed(fn, warmup, repeats):
    for _ in range(warmup): fn()
    out = []
    for _ in range(repeats):
        t = time.perf_counter(); fn(); out.append(time.perf_counter()-t)
    return dict(median_s=statistics.median(out), p90_s=float(np.quantile(out, 0.9)), min_s=min(out), repeats=repeats)


def encoder(a):
    from mantis.architecture import MantisV2
    from wimot.features.extract import amplitude, pool
    model = MantisV2(device='cpu', return_transf_layer=2, output_token='combined', pre_training=False)
    model = model.from_pretrained(str(a.weights)).eval().requires_grad_(False)
    channels, length = SHAPES[a.dataset]; rng = np.random.default_rng(0)

    @torch.inference_mode()
    def query(batch):
        b, c, t = batch.shape
        s = torch.from_numpy(batch).reshape(b*c, 1, t)
        s = torch.nn.functional.interpolate(s, size=512, mode='linear', align_corners=False)
        sem = pool(model(s).numpy().reshape(b, c, 512))
        return np.concatenate([sem, amplitude(batch)], axis=1)

    one = rng.standard_normal((1, channels, length)).astype(np.float32)
    two = rng.standard_normal((2, channels, length)).astype(np.float32)
    assert query(one).shape == (1, 9728)
    single = timed(lambda: query(one), 3, a.repeats)
    pair = timed(lambda: query(two), 2, max(5, a.repeats//2))
    return dict(part='encoder', dataset=a.dataset, input_shape=[channels, 512], machine=machine(a.threads),
                per_query=single, per_recording_in_pairs_s=pair['median_s']/2, peak_rss_mib=peak_mib())


def stream(a):
    from wimot import WiMoT
    from wimot.common import STREAMS

    def load(domain, split):
        with np.load(a.features/f'{domain}_{split}.npz') as f: return f['features'], f['labels']

    domains = STREAMS[a.dataset]['domains']
    model = WiMoT(num_classes=CLASSES[a.dataset], seed=42, device='cpu')
    stages, tests = [], []
    for domain in domains:
        x, y = load(domain, 'train'); tests.append(load(domain, 'test'))
        t = time.perf_counter(); model.fit_domain(x, y); seconds = time.perf_counter()-t
        acc = [model.score(xt, yt) for xt, yt in tests]
        # predict_proba rebuilds the stage model and refits the classifier on every call; the per-query cost of a
        # deployed classifier is the slope between a 1-row and an n-row call, the intercept is the one-off refit.
        xq = np.concatenate([t[0] for t in tests])
        n = min(len(xq), 512); xn = np.resize(xq, (n, xq.shape[1]))
        one = timed(lambda: model.predict_proba(xq[:1]), 2, max(5, a.repeats//4))['median_s']
        many = timed(lambda: model.predict_proba(xn), 2, max(5, a.repeats//4))['median_s']
        per_query = (many-one)/(n-1); refit = one-per_query
        stages.append(dict(domain=domain, train_samples=int(len(y)), update_s=seconds, readout_refit_s=refit,
                           head_per_query_s=per_query, head_call_1_s=one, head_call_n_s=many, head_n=n,
                           mean_seen_accuracy=float(np.mean(acc)), storage_mib=model.storage_bytes/2**20,
                           peak_rss_mib=peak_mib()))
        print(a.dataset, domain, f'update={seconds:.1f}s', f'refit={refit:.3f}s', f'head/query={per_query*1e3:.3f}ms',
              f'acc={np.mean(acc):.4f}', f'store={model.storage_bytes/2**20:.2f}MiB', flush=True)
    return dict(part='stream', dataset=a.dataset, machine=machine(a.threads), stages=stages,
                final_average_accuracy=stages[-1]['mean_seen_accuracy'], peak_rss_mib=peak_mib())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('part', choices=('encoder', 'stream')); p.add_argument('--dataset', required=True, choices=list(SHAPES))
    p.add_argument('--threads', type=int, required=True); p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--weights', type=Path, help='local MantisV2 snapshot (encoder part)')
    p.add_argument('--features', type=Path, help='feature directory of the stream (stream part)')
    p.add_argument('--out', type=Path)
    a = p.parse_args(); torch.set_num_threads(a.threads)
    if a.part == 'encoder' and a.weights is None: p.error('encoder needs --weights')
    if a.part == 'stream' and a.features is None: p.error('stream needs --features')
    out = a.out or ROOT/'results/cpu_benchmark'/f'{a.part}_{a.dataset}_t{a.threads}.json'
    result = encoder(a) if a.part == 'encoder' else stream(a)
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=1))
    print('WROTE', out, flush=True)


if __name__ == '__main__':
    main()
