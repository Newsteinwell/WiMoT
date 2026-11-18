Loss-weight sensitivity analysis (test partitions, mean±sample SD over seeds; AFR in percentage points).
Paper setting: lambda_sc = 0.5 (later domains and first domain), lambda_h = 1 (`main`). Sensitivity analysis only: no coefficient is changed.

### 3DO

**SupCon weight in the later domains (first domain: paper setting), lambda_h = 1**

| lambda_sc | n | AFA (%) | AFR (pp) | New-domain acc. (%) | Earlier-domain acc. (%) |
|---|---|---|---|---|---|
| 0 | 3 | 94.11±0.47 | -0.10±0.74 | 92.96±0.76 | 96.92±0.42 |
| 0.1 | 3 | 93.95±0.27 | 0.10±0.10 | 92.81±0.58 | 96.92±0.36 |
| 0.25 | 3 | 93.94±0.24 | 0.22±0.24 | 92.90±0.43 | 96.82±0.39 |
| 0.5 | 3 | 94.02±0.07 | 0.32±0.33 | 93.07±0.10 | 96.82±0.46 |
| 1 | 3 | 94.12±0.03 | 0.17±0.15 | 93.08±0.11 | 96.96±0.18 |
| 2 | 3 | 93.99±0.16 | 0.21±0.31 | 92.90±0.35 | 96.94±0.23 |

**L_hist weight, lambda_sc = 0.5**

| lambda_h | n | AFA (%) | AFR (pp) | New-domain acc. (%) | Earlier-domain acc. (%) |
|---|---|---|---|---|---|
| 0 | 3 | 93.94±0.60 | 0.43±0.30 | 92.97±0.85 | 96.76±0.28 |
| 0.25 | 3 | 93.99±0.16 | 0.21±0.31 | 92.90±0.35 | 96.94±0.23 |
| 0.5 | 3 | 94.12±0.03 | 0.17±0.15 | 93.08±0.11 | 96.96±0.18 |
| 1 | 3 | 94.02±0.07 | 0.32±0.33 | 93.07±0.10 | 96.82±0.46 |
| 2 | 3 | 93.94±0.24 | 0.22±0.24 | 92.90±0.43 | 96.82±0.39 |
| 4 | 3 | 93.94±0.28 | 0.10±0.15 | 92.79±0.53 | 96.92±0.40 |

**SupCon weight in all domains including the first (lambda_h = 1)**

| lambda_sc (all domains) | n | AFA (%) | AFR (pp) | New-domain acc. (%) | Earlier-domain acc. (%) |
|---|---|---|---|---|---|
| 0 | 3 | 91.85±0.57 | -0.38±0.07 | 88.29±0.81 | 97.40±0.38 |
| 0.1 | 3 | 94.05±0.24 | 0.18±0.20 | 93.00±0.59 | 96.92±0.41 |
| 0.25 | 3 | 94.00±0.21 | 0.18±0.25 | 92.94±0.38 | 96.85±0.41 |
| 1 | 3 | 94.13±0.06 | 0.28±0.09 | 93.14±0.13 | 96.96±0.22 |
| 2 | 3 | 93.94±0.14 | 0.21±0.19 | 92.82±0.28 | 96.96±0.22 |

**3 x 3 grid**

| lambda_sc / lambda_h | n | AFA (%) | AFR (pp) | New-domain acc. (%) | Earlier-domain acc. (%) |
|---|---|---|---|---|---|
| 0.25 / 0.5 | 3 | 94.02±0.07 | 0.32±0.33 | 93.07±0.10 | 96.82±0.46 |
| 0.25 / 1 | 3 | 93.94±0.24 | 0.22±0.24 | 92.90±0.43 | 96.82±0.39 |
| 0.25 / 2 | 3 | 93.94±0.28 | 0.10±0.15 | 92.79±0.53 | 96.92±0.40 |
| 0.5 / 0.5 | 3 | 94.12±0.03 | 0.17±0.15 | 93.08±0.11 | 96.96±0.18 |
| 0.5 / 1 | 3 | 94.02±0.07 | 0.32±0.33 | 93.07±0.10 | 96.82±0.46 |
| 0.5 / 2 | 3 | 93.94±0.24 | 0.22±0.24 | 92.90±0.43 | 96.82±0.39 |
| 1 / 0.5 | 3 | 93.99±0.16 | 0.21±0.31 | 92.90±0.35 | 96.94±0.23 |
| 1 / 1 | 3 | 94.12±0.03 | 0.17±0.15 | 93.08±0.11 | 96.96±0.18 |
| 1 / 2 | 3 | 94.02±0.07 | 0.32±0.33 | 93.07±0.10 | 96.82±0.46 |

**Paper ablation rows for comparison** (no_supcon: SupCon removed in every domain, first domain untrained; no_hist: L_hist removed)

| row | n | AFA (%) | AFR (pp) | New-domain acc. (%) | Earlier-domain acc. (%) |
|---|---|---|---|---|---|
| no_supcon (paper, reference results) | 3 | 91.85±0.57 | -0.38±0.07 | 88.29±0.81 | 97.40±0.38 |
| no_hist (paper, reference results) | 3 | 93.94±0.60 | 0.43±0.30 | 92.97±0.85 | 96.76±0.28 |

### WiMANS

**SupCon weight in the later domains (first domain: paper setting), lambda_h = 1**

| lambda_sc | n | AFA (%) | AFR (pp) | New-domain acc. (%) | Earlier-domain acc. (%) |
|---|---|---|---|---|---|
| 0 | 3 | 88.68±0.47 | -0.93±0.46 | 85.65±1.67 | 89.97±0.27 |
| 0.1 | 3 | 89.09±1.17 | -0.62±0.96 | 86.57±1.22 | 90.28±0.80 |
| 0.25 | 3 | 89.30±0.71 | 0.15±0.27 | 87.19±1.49 | 90.28±0.46 |
| 0.5 | 3 | 89.61±0.47 | 0.00±0.46 | 87.50±1.22 | 90.74±0.00 |
| 1 | 3 | 89.30±1.08 | -0.15±1.41 | 87.04±0.93 | 90.43±0.96 |
| 2 | 3 | 89.30±1.58 | -0.46±1.67 | 86.73±0.96 | 90.28±1.67 |

**L_hist weight, lambda_sc = 0.5**

| lambda_h | n | AFA (%) | AFR (pp) | New-domain acc. (%) | Earlier-domain acc. (%) |
|---|---|---|---|---|---|
| 0 | 3 | 88.58±1.11 | 0.31±0.71 | 86.57±1.22 | 89.35±0.93 |
| 0.25 | 3 | 89.30±1.58 | -0.46±1.67 | 86.73±0.96 | 90.28±1.67 |
| 0.5 | 3 | 89.30±1.08 | -0.15±1.41 | 87.04±0.93 | 90.43±0.96 |
| 1 | 3 | 89.61±0.47 | 0.00±0.46 | 87.50±1.22 | 90.74±0.00 |
| 2 | 3 | 89.30±0.71 | 0.15±0.27 | 87.19±1.49 | 90.28±0.46 |
| 4 | 3 | 89.20±1.35 | -0.15±1.17 | 86.73±1.17 | 90.28±1.22 |

**SupCon weight in all domains including the first (lambda_h = 1)**

| lambda_sc (all domains) | n | AFA (%) | AFR (pp) | New-domain acc. (%) | Earlier-domain acc. (%) |
|---|---|---|---|---|---|
| 0 | 3 | 84.47±2.59 | -4.63±2.58 | 80.25±4.20 | 85.34±1.93 |
| 0.1 | 3 | 89.30±1.28 | -0.46±1.39 | 86.88±0.96 | 90.59±0.96 |
| 0.25 | 3 | 89.51±1.11 | -0.00±0.00 | 87.35±1.87 | 90.43±0.96 |
| 1 | 3 | 89.30±1.08 | 0.00±1.67 | 87.19±0.96 | 90.43±0.96 |
| 2 | 3 | 88.99±1.28 | 0.15±1.41 | 86.73±0.96 | 89.97±1.17 |

**3 x 3 grid**

| lambda_sc / lambda_h | n | AFA (%) | AFR (pp) | New-domain acc. (%) | Earlier-domain acc. (%) |
|---|---|---|---|---|---|
| 0.25 / 0.5 | 3 | 89.61±0.47 | 0.00±0.46 | 87.50±1.22 | 90.74±0.00 |
| 0.25 / 1 | 3 | 89.30±0.71 | 0.15±0.27 | 87.19±1.49 | 90.28±0.46 |
| 0.25 / 2 | 3 | 89.20±1.35 | -0.15±1.17 | 86.73±1.17 | 90.28±1.22 |
| 0.5 / 0.5 | 3 | 89.30±1.08 | -0.15±1.41 | 87.04±0.93 | 90.43±0.96 |
| 0.5 / 1 | 3 | 89.61±0.47 | 0.00±0.46 | 87.50±1.22 | 90.74±0.00 |
| 0.5 / 2 | 3 | 89.30±0.71 | 0.15±0.27 | 87.19±1.49 | 90.28±0.46 |
| 1 / 0.5 | 3 | 89.30±1.58 | -0.46±1.67 | 86.73±0.96 | 90.28±1.67 |
| 1 / 1 | 3 | 89.30±1.08 | -0.15±1.41 | 87.04±0.93 | 90.43±0.96 |
| 1 / 2 | 3 | 89.61±0.47 | 0.00±0.46 | 87.50±1.22 | 90.74±0.00 |

**Paper ablation rows for comparison** (no_supcon: SupCon removed in every domain, first domain untrained; no_hist: L_hist removed)

| row | n | AFA (%) | AFR (pp) | New-domain acc. (%) | Earlier-domain acc. (%) |
|---|---|---|---|---|---|
| no_supcon (paper, reference results) | 3 | 84.47±2.59 | -4.63±2.58 | 80.25±4.20 | 85.34±1.93 |
| no_hist (paper, reference results) | 3 | 88.58±1.11 | 0.31±0.71 | 86.57±1.22 | 89.35±0.93 |

### Widar3.0

**SupCon weight in the later domains (first domain: paper setting), lambda_h = 1**

| lambda_sc | n | AFA (%) | AFR (pp) | New-domain acc. (%) | Earlier-domain acc. (%) |
|---|---|---|---|---|---|
| 0 | 3 | 93.80±0.54 | -0.14±0.21 | 90.19±0.59 | 95.42±0.38 |
| 0.1 | 3 | 94.96±0.71 | 0.42±0.25 | 94.56±0.25 | 96.14±0.82 |
| 0.25 | 3 | 95.21±0.72 | 0.31±0.19 | 94.86±0.32 | 96.36±0.84 |
| 0.5 | 3 | 95.16±1.23 | 0.58±0.58 | 95.08±0.42 | 96.31±1.13 |
| 1 | 3 | 94.99±1.37 | 0.89±0.51 | 95.03±0.72 | 96.11±1.23 |
| 2 | 3 | 95.06±1.22 | 1.08±0.22 | 95.22±0.59 | 96.08±1.17 |

**L_hist weight, lambda_sc = 0.5**

| lambda_h | n | AFA (%) | AFR (pp) | New-domain acc. (%) | Earlier-domain acc. (%) |
|---|---|---|---|---|---|
| 0 | 3 | 94.25±0.70 | 1.33±0.22 | 94.86±0.67 | 95.33±0.66 |
| 0.25 | 3 | 95.06±1.22 | 1.08±0.22 | 95.22±0.59 | 96.08±1.17 |
| 0.5 | 3 | 94.99±1.37 | 0.89±0.51 | 95.03±0.72 | 96.11±1.23 |
| 1 | 3 | 95.16±1.23 | 0.58±0.58 | 95.08±0.42 | 96.31±1.13 |
| 2 | 3 | 95.21±0.72 | 0.31±0.19 | 94.86±0.32 | 96.36±0.84 |
| 4 | 3 | 95.06±0.83 | 0.36±0.38 | 94.64±0.35 | 96.28±0.92 |

**SupCon weight in all domains including the first (lambda_h = 1)**

| lambda_sc (all domains) | n | AFA (%) | AFR (pp) | New-domain acc. (%) | Earlier-domain acc. (%) |
|---|---|---|---|---|---|
| 0 | 3 | 92.27±0.56 | -0.06±0.17 | 88.08±0.79 | 94.06±0.41 |
| 0.1 | 3 | 95.04±0.80 | 0.36±0.32 | 94.58±0.29 | 96.19±0.91 |
| 0.25 | 3 | 95.21±0.72 | 0.25±0.17 | 94.83±0.29 | 96.39±0.82 |
| 1 | 3 | 95.09±1.28 | 0.86±0.29 | 95.11±0.77 | 96.19±1.06 |
| 2 | 3 | 95.11±1.22 | 1.06±0.29 | 95.25±0.52 | 96.14±1.13 |

**Paper ablation rows for comparison** (no_supcon: SupCon removed in every domain, first domain untrained; no_hist: L_hist removed)

| row | n | AFA (%) | AFR (pp) | New-domain acc. (%) | Earlier-domain acc. (%) |
|---|---|---|---|---|---|
| no_supcon (paper, reference results) | 3 | 92.27±0.56 | -0.06±0.17 | 88.08±0.79 | 94.06±0.41 |
| no_hist (paper, reference results) | 3 | 94.25±0.70 | 1.33±0.22 | 94.86±0.67 | 95.33±0.66 |

**Consistency checks against the paper ablation rows** (same seeds; identical accuracy matrices expected on the same GPU model)

* 3DO, L_hist weight 0 vs no_hist: max |difference| of the accuracy matrix per seed (pp) = 0.00, 0.00, 0.00
* WiMANS, L_hist weight 0 vs no_hist: max |difference| of the accuracy matrix per seed (pp) = 0.00, 0.00, 0.00
* Widar3.0, L_hist weight 0 vs no_hist: max |difference| of the accuracy matrix per seed (pp) = 0.00, 0.00, 0.00
* 3DO, lambda_sc = 0 in all domains vs no_supcon: max |difference| of the accuracy matrix per seed (pp) = 0.00, 0.00, 0.00
* WiMANS, lambda_sc = 0 in all domains vs no_supcon: max |difference| of the accuracy matrix per seed (pp) = 0.00, 0.00, 0.00
* Widar3.0, lambda_sc = 0 in all domains vs no_supcon: max |difference| of the accuracy matrix per seed (pp) = 0.00, 0.00, 0.00

**Joint rescaling of both weights** (AdamW is invariant to the scale of the loss, so only lambda_sc / lambda_h should matter): max |difference| of the accuracy matrices of settings with the same ratio (later-domain weights only, all seeds), in pp

* 3DO, ratio 0.125: (0.25, 2), (0.5, 4) -> 0.00
* 3DO, ratio 0.25: (0.25, 1), (0.5, 2) -> 0.00
* 3DO, ratio 0.5: (0.5, 1), (0.25, 0.5), (1, 2) -> 0.00
* 3DO, ratio 1: (0.5, 0.5), (1, 1) -> 0.00
* 3DO, ratio 2: (0.5, 0.25), (1, 0.5), (2, 1) -> 0.00
* WiMANS, ratio 0.125: (0.25, 2), (0.5, 4) -> 0.00
* WiMANS, ratio 0.25: (0.25, 1), (0.5, 2) -> 0.00
* WiMANS, ratio 0.5: (0.5, 1), (0.25, 0.5), (1, 2) -> 0.00
* WiMANS, ratio 1: (0.5, 0.5), (1, 1) -> 0.00
* WiMANS, ratio 2: (0.5, 0.25), (1, 0.5), (2, 1) -> 0.00
* Widar3.0, ratio 0.25: (0.25, 1), (0.5, 2) -> 0.00
* Widar3.0, ratio 1: (0.5, 0.5), (1, 1) -> 0.00
* Widar3.0, ratio 2: (0.5, 0.25), (2, 1) -> 0.00
