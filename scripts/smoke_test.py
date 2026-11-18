"""End-to-end smoke test on synthetic features (CPU, a few minutes, no data, no download).

  python scripts/smoke_test.py [--workdir DIR] [--device cpu] [--quick]

It writes small feature files in the format of ``wimot.features.extract`` (``<domain>_<split>.npz`` with ``features``
(N x 9728) and ``labels``, and ``passport.json`` with the file digests) for the three domains of the 3DO stream, then runs the
unchanged command-line entry points on them:

  1. one full WiMoT trajectory (``wimot.run``: three domains, then the test matrix),
  2. one classic baseline, three replay baselines, ConSense, EdgeCL, CN-DPM, DUCT, DualCP+ER, and the joint-training reference,
  3. a small subset of the ``main`` table (``wimot.study``) summarized by ``scripts/build_tables.py``,
  4. the high-level ``wimot.WiMoT`` interface, whose accuracy matrix must equal that of step 1 exactly,
  5. ``scripts/check_predictions.py`` on the shipped stage-wise predictions.

The data are random Gaussian clusters, so the accuracies say nothing about the paper; the test checks that every code path
runs and produces well-formed, finite results. ``--quick`` runs fewer baselines.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DIMENSION, DOMAINS, CLASSES = 9728, ('d1', 'd2', 'd3'), 3
SIZES = dict(train=90, validation=20, test=30)   # windows per class and domain


def make_features(root, seed=0):
    """Synthetic 3DO-like features: class prototypes, a domain offset and noise; written like ``features.extract``."""
    root = Path(root); root.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(seed)
    # classes 0 and 1 are similar and class 2 is opposite, so that DualCP finds a two-level class hierarchy
    a, b = rng.normal(size=(2, DIMENSION)); prototype = .12*np.stack([a, a+.4*b, -2*a])
    hashes = {}
    for k, domain in enumerate(DOMAINS):
        offset = rng.normal(size=DIMENSION)*.25
        for split, n in SIZES.items():
            labels = np.repeat(np.arange(CLASSES), n)
            x = prototype[labels]+offset+rng.normal(size=(len(labels), DIMENSION))
            x[:, 9216:] = np.abs(x[:, 9216:])+1.   # the amplitude descriptor is positive
            path = root/f'{domain}_{split}.npz'
            np.savez(path, features=x.astype(np.float32), labels=labels.astype(np.int64))
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root/'passport.json').write_text(json.dumps(dict(synthetic=True, files_sha256=hashes, complete=True), indent=2))
    return root


def run(*args, log=None):
    started = time.monotonic()
    # NumPy built on macOS Accelerate raises spurious floating-point RuntimeWarnings in float64 matmul; the results are finite
    env = dict(os.environ, PYTHONWARNINGS='ignore::RuntimeWarning')
    done = subprocess.run([sys.executable, *map(str, args)], cwd=REPO, text=True, capture_output=True, env=env)
    if log is not None: Path(log).write_text(done.stdout+done.stderr)
    if done.returncode: raise SystemExit(f'FAILED: {" ".join(map(str, args))}\n{done.stdout[-1500:]}\n{done.stderr[-3000:]}')
    return time.monotonic()-started, done.stdout


def cn_dpm_short(root, common, device):
    """CN-DPM trains each expert with 8000+2000 sleep steps (minutes per expert on a CPU); the smoke test uses 60+20 steps."""
    import functools
    from wimot.baselines import recent
    from wimot.run import main as run_main
    started = time.monotonic(); original = recent.CNConfig
    recent.CNConfig = functools.partial(original, sleep_steps_g=60, sleep_steps_d=20)
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter('ignore', RuntimeWarning)
            run_main(['train', '--root', str(root), *map(str, common), '--family', 'recent', '--method', 'cn_dpm_aligned'])
    finally: recent.CNConfig = original
    return time.monotonic()-started


def check_result(path, joint=False):
    r = json.loads(Path(path).read_text())
    if joint:
        assert len(r['per_domain']) == len(DOMAINS) and all(np.isfinite(r['per_domain']))
        return r
    m = np.array([[np.nan if v is None else v for v in row] for row in r['matrix']])
    assert m.shape == (3, 3) and np.isfinite(m[np.tril_indices(3)]).all() and np.isnan(m[np.triu_indices(3, 1)]).all()
    assert (0 <= m[np.tril_indices(3)]).all() and (m[np.tril_indices(3)] <= 1).all()
    assert np.isfinite(r['metrics']['AFA']) and np.isfinite(r['metrics']['AFR'])
    return r


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workdir', type=Path, help='Keep features and runs here (default: a temporary directory)')
    p.add_argument('--device', default='cpu'); p.add_argument('--quick', action='store_true')
    a = p.parse_args(argv)
    tmp = None if a.workdir else tempfile.TemporaryDirectory(prefix='wimot_smoke_')
    work = Path(a.workdir or tmp.name); work.mkdir(parents=True, exist_ok=True); logs = work/'logs'; logs.mkdir(exist_ok=True)
    features = make_features(work/'features'/'mantis'/'3do'); started = time.monotonic()
    print(f'synthetic features: {features} ({sum(f.stat().st_size for f in features.iterdir())/1e6:.1f} MB)')
    common = ['--features', features, '--dataset', '3do', '--seed', 42, '--device', a.device]

    # 1. one full WiMoT trajectory
    root = work/'runs'/'direct'
    seconds, _ = run('-m', 'wimot.run', 'train', '--root', root, *common, '--family', 'method', '--method', 'main', log=logs/'method.log')
    run('-m', 'wimot.run', 'test', '--root', root, *common, '--family', 'method', '--method', 'main')
    result = check_result(root/'method'/'3do'/'seed42'/'main'/'test.json')
    receipt = json.loads((root/'method'/'3do'/'seed42'/'main'/'receipt.json').read_text())
    assert receipt['complete'] and len(receipt['stages']) == 3 and receipt['stages'][-1]['statistic_bytes'] > 0
    print(f'[1] WiMoT 3DO trajectory ({seconds:.0f} s): AFA {100*result["metrics"]["AFA"]:.1f}%, AFR {100*result["metrics"]["AFR"]:.1f}%')

    # 2. baselines of every family
    baselines = [('classic', 'ft'), ('classic', 'si_er'), ('classic', 'er'), ('classic', 'derpp'), ('wifi', 'edgecl'),
                 ('wifi', 'consense'), ('recent', 'cn_dpm_aligned'), ('recent', 'duct'), ('recent', 'dualcp_er'), ('joint', 'joint')]
    if a.quick: baselines = [('classic', 'ft'), ('classic', 'si_er'), ('joint', 'joint')]
    for family, method in baselines:
        extra = ['--input', 'cat9728'] if family == 'joint' else []
        if method == 'cn_dpm_aligned': seconds = cn_dpm_short(root, common, a.device)
        else:
            seconds, _ = run('-m', 'wimot.run', 'train', '--root', root, *common, '--family', family, '--method', method, *extra,
                             log=logs/f'{family}_{method}.log')
        run('-m', 'wimot.run', 'test', '--root', root, *common, '--family', family, '--method', method, *extra)
        tests = sorted((root/family/'3do'/'seed42').rglob('test.json'))
        r = check_result([t for t in tests if json.loads(t.read_text())['method'] == method][0], joint=family == 'joint')
        score = np.mean(r['per_domain']) if family == 'joint' else r['metrics']['AFA']
        print(f'[2] {family}/{method:15} ({seconds:3.0f} s): mean accuracy {100*score:.1f}%')

    # 3. a small subset of the main table, then the table summary
    runs = work/'runs'/'study'
    seconds, out = run('-m', 'wimot.study', 'main', '--features', work/'features', '--out', runs, '--device', a.device,
                       '--dataset', '3do', '--seed', 42, '--method', 'main,ft,si_er', log=logs/'study.log')
    _, table = run('scripts/build_tables.py', '--runs', runs, '--study', 'main')
    assert 'reproduced 3 of' in table, table
    count = len(list(runs.rglob('test.json'))); assert count == 3, count
    print(f'[3] study.py main (3 jobs, {seconds:.0f} s) and build_tables.py: {table.strip().splitlines()[-1]}')
    _, table = run('scripts/build_tables.py', '--reference-only', '--study', 'main')
    assert 'ours' not in table and table.count('\n') > 20

    # 4. the high-level interface equals the command-line trajectory
    from wimot import WiMoT
    warnings.simplefilter('ignore', RuntimeWarning)
    arrays = {(d, s): np.load(features/f'{d}_{s}.npz') for d in DOMAINS for s in SIZES}
    model = WiMoT(num_classes=CLASSES, seed=42, device=a.device); matrix = []
    for k, domain in enumerate(DOMAINS):
        model.fit_domain(arrays[domain, 'train']['features'], arrays[domain, 'train']['labels'])
        scores = [model.score(arrays[d, 'test']['features'], arrays[d, 'test']['labels']) for d in DOMAINS[:k+1]]
        matrix.append(scores+[None]*(len(DOMAINS)-k-1))
    assert model.domains_seen == 3 and model.predict(arrays['d1', 'test']['features']).shape == (len(DOMAINS[:1])*CLASSES*SIZES['test'],)
    assert matrix == result['matrix'], f'{matrix} != {result["matrix"]}'
    print('[4] wimot.WiMoT accuracy matrix identical to the wimot.run matrix')

    # 5. shipped stage-wise predictions
    _, out = run('scripts/check_predictions.py')
    assert 'matrix mismatches 0' in out, out
    print(f'[5] check_predictions.py: {out.strip()}')
    print(f'smoke test passed in {time.monotonic()-started:.0f} s')
    if tmp: tmp.cleanup()


if __name__ == '__main__':
    main()
