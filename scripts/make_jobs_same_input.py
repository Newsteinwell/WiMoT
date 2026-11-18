"""Writes configs/jobs_sameall/<study>.tsv (study name sameall_<study> in wimot.study): the baseline trajectories of the main, orders, fewlabel, widarext and backbone job lists with
the 9728-D input (Mantis block and amplitude descriptor, as read by the proposed method). The method's own lines are not
repeated, and the coefficient column is '-' so that the coefficient comes from the locks file named by WIMOT_LOCKS
(configs/locks_sameinput_all.json). The main list also holds the joint-training reference with the same input.
"""
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JOBS = REPO/'configs'/'jobs'
OUT = REPO/'configs'/'jobs_sameall'


def read(study):
    with open(JOBS/f'{study}.tsv') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def write(study, rows):
    OUT.mkdir(exist_ok=True)
    with open(OUT/f'{study}.tsv', 'w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n'); w.writeheader(); w.writerows(rows)
    print(f'sameall_{study}', len(rows))


def main():
    for study in ('main', 'orders', 'fewlabel', 'widarext', 'backbone'):
        rows = [dict(r, input='cat9728', coefficient='-') for r in read(study) if r['family'] != 'method']
        if study == 'main':
            rows += [dict(r) for r in read('sameinput') if r['family'] == 'joint' and r['input'] == 'cat9728']
        write(study, rows)


if __name__ == '__main__':
    main()
