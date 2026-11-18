"""Prepare fixed-length amplitude windows from the official raw data, reproducing the partitions of data/splits.

  python -m wimot.data.prepare 3do          --raw DIR --out DIR   DIR/<day>/<activity><session>/csiposreg.csv
  python -m wimot.data.prepare wimans-split --annotation annotation.csv --out wimans_split.json
  python -m wimot.data.prepare wimans       --raw DIR --out DIR   DIR/<recording>.npy (official wifi_csi/amp)
  python -m wimot.data.prepare widar        --raw DIR --out DIR [--stream widar|widar_r2|widar_r3]   *.dat anywhere below DIR

Every output file <domain>_<split>.npz holds 'windows' (N x channels x time, float32) and 'labels'; the recording or
window identities are checked against data/splits/<stream>.json, and, if configs/digests.json is present, the SHA-256
of the windows and labels arrays against the original preparation.
"""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from ..common import REPO, digest, sha, write_json

SPLITS = REPO/'data'/'splits'


def reference(stream):
    return json.loads((SPLITS/f'{stream}.json').read_text())


def check_digests(stream, out):
    path = REPO/'configs'/'digests.json'
    if not path.exists(): return
    expected = json.loads(path.read_text())['prepared'].get(stream, {})
    for file, arrays in expected.items():
        with np.load(Path(out)/file) as f:
            for key, value in arrays.items():
                if digest(f[key]) != value['sha256']: raise ValueError(f'{file}:{key} differs from the original preparation')
    print('PREPARED_ARRAYS_MATCH', stream, len(expected), flush=True)


# ---------------------------------------------------------------- 3DO
ACTIVITIES = {'w': 0, 's': 1, 'l': 2}


def read_3do(csv_path):
    import pandas as pd
    d = pd.read_csv(csv_path, usecols=['class', 'data'])
    y = d['class'].to_numpy()
    real = np.asarray(list(range(6, 32))+list(range(33, 59)))*2
    imag = real-1
    a = np.empty((len(d), 52), dtype=np.complex64)
    for row, txt in enumerate(d['data']):
        values = np.fromstring(txt.strip('[]'), sep=',', dtype=np.float32)
        if len(values) <= real[-1]: raise ValueError(f'Short CSI row: {csv_path}, {row}')
        a[row] = values[real]+1j*values[imag]
    return a, y


def prepare_3do(raw, out):
    ref = reference('3do'); out.mkdir(parents=True, exist_ok=True); window = ref['window']; centre = window//2
    for day in ('d1', 'd2', 'd3'):
        for split in ('train', 'validation', 'test'):
            count = ref['counts_per_activity'][split]
            windows, labels, ids, starts = [], [], [], []
            for activity, label in ACTIVITIES.items():
                sessions = ref['sessions_day3'][split][activity] if day == 'd3' else ref['sessions_day1_day2'][split]
                per_session = [count//len(sessions)]*len(sessions)
                for i in range(count % len(sessions)): per_session[i] += 1
                for session, n in zip(sessions, per_session):
                    name = f'{day}/{activity}{session}'; path = Path(raw)/name/'csiposreg.csv'
                    if sha(path) != ref['sources'][name]['csv_sha256']: raise ValueError(f'{path} differs from the official file')
                    a, y = read_3do(path)
                    total = len(a)-window+1
                    candidates = np.flatnonzero(y[centre:centre+total] == label+1)
                    if n > len(candidates): raise ValueError(f'Insufficient labelled windows: {name}')
                    seed = ref['seed']+int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
                    for start in np.sort(np.random.default_rng(seed).choice(candidates, n, replace=False)):
                        windows.append(np.abs(a[start:start+window]).T.astype(np.float32))
                        labels.append(label); ids.append(name); starts.append(int(start))
            file = f'{day}_{split}.npz'; expected = ref['files'][file]
            assert ids == expected['sessions'] and starts == expected['starts'] and labels == expected['labels'], file
            np.savez_compressed(out/file, windows=np.stack(windows), labels=np.asarray(labels),
                                sessions=np.asarray(ids), starts=np.asarray(starts))
            print('PREPARED', file, len(labels), flush=True)
    check_digests('3do', out)


# ---------------------------------------------------------------- WiMANS
WIMANS_ACTIONS = ['nothing', 'walk', 'rotation', 'jump', 'wave', 'lie_down', 'pick_up', 'sit_down', 'stand_up']
WIMANS_ROOMS = ['classroom', 'meeting_room', 'empty_room']


def wimans_split(annotation, out):
    """Single-user 5 GHz recordings; per room, subject and activity the 11 recordings are split 7/2/2 at random."""
    rows = list(csv.DictReader(Path(annotation).open(encoding='utf-8-sig')))
    selected = [r for r in rows if r['number_of_users'] == '1' and r['wifi_band'] == '5']
    for r in selected:
        people = [i for i in range(1, 7) if r[f'user_{i}_activity']]
        if len(people) != 1: raise ValueError('Single-user annotation is ambiguous')
        r['subject'] = people[0]
        r['action'] = r[f'user_{people[0]}_activity'].strip().lower().replace(' ', '_')
    if set(r['action'] for r in selected) != set(WIMANS_ACTIONS): raise ValueError('Unexpected activity vocabulary')
    splits = {room: {s: [] for s in ('train', 'validation', 'test')} for room in WIMANS_ROOMS}
    rng = np.random.default_rng(20261010)
    for room in WIMANS_ROOMS:
        for subject in range(1, 7):
            for action in WIMANS_ACTIONS:
                trials = sorted((r for r in selected if r['environment'] == room and r['subject'] == subject
                                 and r['action'] == action), key=lambda r: r['label'])
                if len(trials) != 11: raise ValueError('Expected 11 recordings per subject/activity/room')
                order = rng.permutation(11)
                for split, idx in zip(('train', 'validation', 'test'), (order[:7], order[7:9], order[9:])):
                    splits[room][split].extend(dict(recording=trials[i]['label'], subject=subject, action=action,
                                                    label=WIMANS_ACTIONS.index(action)) for i in idx)
    ref = reference('wimans')['files']
    for room in WIMANS_ROOMS:
        for split, rows in splits[room].items():
            assert [r['recording'] for r in rows] == ref[f'{room}_{split}.npz']['recordings'], (room, split)
    write_json(out, dict(seed=20261010, domains=WIMANS_ROOMS, actions=WIMANS_ACTIONS, splits=splits))
    print('WIMANS_SPLIT_MATCHES', sum(len(v) for r in splits.values() for v in r.values()), flush=True)


def wimans_recording(values):
    values = np.asarray(values)
    if (values.ndim != 4 or values.shape[1:] != (3, 3, 30) or len(values) < 300 or
            not np.isfinite(values).all() or (values < 0).any()): raise ValueError('Malformed CSI recording')
    x = np.stack([part.mean(0, dtype=np.float64) for part in np.array_split(values, 300, axis=0)])
    x = x.transpose(1, 2, 3, 0).reshape(270, 300).astype(np.float32)
    return ((x-x.mean(dtype=np.float64))/max(float(x.std(dtype=np.float64, ddof=1)), 1e-6)).astype(np.float32)


def prepare_wimans(raw, out):
    ref = reference('wimans'); out.mkdir(parents=True, exist_ok=True)
    for file, entry in ref['files'].items():
        windows = [wimans_recording(np.load(Path(raw)/f'{rid}.npy', allow_pickle=False)) for rid in entry['recordings']]
        np.savez_compressed(out/file, windows=np.stack(windows), labels=np.array(entry['labels'], dtype=np.int64),
                            recordings=np.array(entry['recordings']))
        print('PREPARED', file, len(windows), flush=True)
    check_digests('wimans', out)


# ---------------------------------------------------------------- Widar3.0
WIDAR_NAME = re.compile(r'user(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d)\.dat$')


def read_receiver(path):
    import csiread
    reader = csiread.Intel(str(path), nrxnum=3, ntxnum=1, pl_size=0, if_report=False)
    reader.read()
    csi = np.abs(np.asarray(reader.csi)[:, :, :, 0])  # packets x 30 subcarriers x 3 antennas
    keep = np.isfinite(csi).all((1, 2)) & (csi.reshape(len(csi), -1).max(1) > 0)
    return csi[keep]


def widar_recording(paths, bins=300):
    links = []
    for path in paths:
        csi = read_receiver(path)
        if len(csi) < bins: raise ValueError(f'{Path(path).name}: {len(csi)} packets')
        binned = np.stack([part.mean(0, dtype=np.float64) for part in np.array_split(csi, bins, axis=0)])
        links.append(binned.transpose(2, 1, 0))  # antenna x subcarrier x bins
    x = np.concatenate(links).reshape(-1, bins)
    x = (x-x.mean(dtype=np.float64))/max(float(x.std(dtype=np.float64, ddof=1)), 1e-6)
    return x.astype(np.float32)


def prepare_widar(raw, out, stream):
    """Instances (user, gesture, location, orientation, repetition) with all six receivers; repetitions 1-3/4/5 form the
    train/validation/test partitions; instances with fewer than 300 valid packets in a receiver are skipped."""
    ref = reference(stream); out.mkdir(parents=True, exist_ok=True)
    files = {}
    for path in sorted(Path(raw).rglob('*.dat')):
        m = WIDAR_NAME.search(path.name)
        if m: files.setdefault('user{}-{}-{}-{}-{}'.format(*map(int, m.groups()[:5])), {})[int(m.group(6))] = path
    for file, entry in ref['files'].items():
        skipped = {s['recording'] for s in ref['skipped'].get(file, [])}
        windows, ids = [], []
        for rid in [*entry['recordings'], *skipped]:
            receivers = files.get(rid, {})
            if sorted(receivers) != list(range(1, 7)): raise FileNotFoundError(f'{rid}: receivers {sorted(receivers)}')
            try: x = widar_recording([receivers[r] for r in range(1, 7)])
            except ValueError:
                assert rid in skipped, rid; continue
            windows.append(x); ids.append(rid)
        order = [ids.index(r) for r in entry['recordings']]
        np.savez_compressed(out/file, windows=np.stack([windows[i] for i in order]), labels=np.array(entry['labels'], np.int64),
                            recordings=np.array(entry['recordings']))
        print('PREPARED', file, len(order), flush=True)
    check_digests(stream, out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('dataset', choices=('3do', 'wimans-split', 'wimans', 'widar'))
    p.add_argument('--raw', type=Path); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--annotation', type=Path); p.add_argument('--stream', default='widar', choices=('widar', 'widar_r2', 'widar_r3'))
    a = p.parse_args(argv)
    if a.dataset == '3do': prepare_3do(a.raw, a.out)
    elif a.dataset == 'wimans-split': wimans_split(a.annotation, a.out)
    elif a.dataset == 'wimans': prepare_wimans(a.raw, a.out)
    else: prepare_widar(a.raw, a.out, a.stream)


if __name__ == '__main__':
    main()
