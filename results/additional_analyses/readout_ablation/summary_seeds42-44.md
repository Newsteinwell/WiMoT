Readout ablation, seeds 42-44 (mean±sample SD over seeds; AFR in percentage points)

**3DO** (seeds [42, 43, 44])

| Readout | AFA (%) | AFR (pp) | New-domain acc. (%) | Earlier-domain acc. (%) |
|---|---|---|---|---|
| OSCL (WiMoT) | 94.02±0.07 | 0.32±0.33 | 93.07±0.10 | 96.82±0.46 |
| Nearest class mean | 81.27±0.94 | 13.99±1.06 | 89.22±1.23 | 79.01±0.21 |
| LDA | 91.86±0.36 | 2.19±0.67 | 91.11±1.02 | 95.44±0.76 |
| QDA (no domain gate) | 90.75±0.83 | 1.28±0.31 | 89.11±1.06 | 95.43±0.31 |
| Linear probe W | 85.56±0.56 | -9.99±0.39 | 70.44±1.05 | 98.32±0.36 |
| LDA, ridge 1e-3 (supplementary) | 91.85±0.42 | 2.21±0.72 | 91.11±1.05 | 95.46±0.79 |
| QDA, ridge 1e-3 (supplementary) | 89.30±0.77 | 0.49±0.45 | 86.64±1.02 | 95.36±0.24 |

Paired difference OSCL minus readout over the same seeds (mean in pp; seeds with OSCL strictly better / equal / worse):

| Readout | ΔAFA | ΔAFR | ΔNew | ΔEarlier | AFA better/equal/worse |
|---|---|---|---|---|---|
| Nearest class mean | +12.75 | -13.67 | +3.85 | +17.81 | 3/0/0 |
| LDA | +2.16 | -1.87 | +1.96 | +1.38 | 3/0/0 |
| QDA (no domain gate) | +3.27 | -0.96 | +3.96 | +1.39 | 3/0/0 |
| Linear probe W | +8.46 | +10.31 | +22.63 | -1.50 | 3/0/0 |
| LDA, ridge 1e-3 (supplementary) | +2.17 | -1.89 | +1.96 | +1.36 | 3/0/0 |
| QDA, ridge 1e-3 (supplementary) | +4.72 | -0.17 | +6.43 | +1.46 | 3/0/0 |

**WiMANS** (seeds [42, 43, 44])

| Readout | AFA (%) | AFR (pp) | New-domain acc. (%) | Earlier-domain acc. (%) |
|---|---|---|---|---|
| OSCL (WiMoT) | 89.61±0.47 | 0.00±0.46 | 87.50±1.22 | 90.74±0.00 |
| Nearest class mean | 87.35±0.53 | -0.31±0.71 | 84.57±1.17 | 88.12±0.71 |
| LDA | 88.07±1.17 | 0.00±1.39 | 85.19±0.93 | 89.97±1.41 |
| QDA (no domain gate) | 81.28±1.70 | 0.15±1.34 | 76.70±3.60 | 82.87±1.22 |
| Linear probe W | 85.49±1.41 | -9.10±3.28 | 73.15±2.45 | 88.58±1.41 |
| LDA, ridge 1e-3 (supplementary) | 87.96±0.31 | 0.15±0.53 | 85.34±0.71 | 89.81±0.46 |
| QDA, ridge 1e-3 (supplementary) | 63.79±2.01 | 10.34±5.48 | 61.42±2.38 | 63.73±3.60 |

Paired difference OSCL minus readout over the same seeds (mean in pp; seeds with OSCL strictly better / equal / worse):

| Readout | ΔAFA | ΔAFR | ΔNew | ΔEarlier | AFA better/equal/worse |
|---|---|---|---|---|---|
| Nearest class mean | +2.26 | +0.31 | +2.93 | +2.62 | 3/0/0 |
| LDA | +1.54 | +0.00 | +2.31 | +0.77 | 3/0/0 |
| QDA (no domain gate) | +8.33 | -0.15 | +10.80 | +7.87 | 3/0/0 |
| Linear probe W | +4.12 | +9.10 | +14.35 | +2.16 | 3/0/0 |
| LDA, ridge 1e-3 (supplementary) | +1.65 | -0.15 | +2.16 | +0.93 | 3/0/0 |
| QDA, ridge 1e-3 (supplementary) | +25.82 | -10.34 | +26.08 | +27.01 | 3/0/0 |

**Widar3.0** (seeds [42, 43, 44])

| Readout | AFA (%) | AFR (pp) | New-domain acc. (%) | Earlier-domain acc. (%) |
|---|---|---|---|---|
| OSCL (WiMoT) | 95.16±1.23 | 0.58±0.58 | 95.08±0.42 | 96.31±1.13 |
| Nearest class mean | 66.22±0.64 | 19.03±1.32 | 81.39±0.63 | 67.31±1.04 |
| LDA | 84.32±0.96 | 2.03±0.63 | 82.53±0.63 | 87.00±1.21 |
| QDA (no domain gate) | 91.63±0.39 | 2.42±0.38 | 92.78±0.61 | 93.75±0.65 |
| Linear probe W | 81.90±0.59 | 0.03±0.17 | 48.75±0.95 | 86.31±0.77 |
| LDA, ridge 1e-3 (supplementary) | 84.32±0.75 | 2.19±0.49 | 82.67±0.65 | 86.92±1.08 |
| QDA, ridge 1e-3 (supplementary) | 90.81±0.61 | 2.22±0.27 | 90.28±0.79 | 92.89±0.92 |

Paired difference OSCL minus readout over the same seeds (mean in pp; seeds with OSCL strictly better / equal / worse):

| Readout | ΔAFA | ΔAFR | ΔNew | ΔEarlier | AFA better/equal/worse |
|---|---|---|---|---|---|
| Nearest class mean | +28.94 | -18.44 | +13.69 | +29.00 | 3/0/0 |
| LDA | +10.84 | -1.44 | +12.56 | +9.31 | 3/0/0 |
| QDA (no domain gate) | +3.53 | -1.83 | +2.31 | +2.56 | 3/0/0 |
| Linear probe W | +13.26 | +0.56 | +46.33 | +10.00 | 3/0/0 |
| LDA, ridge 1e-3 (supplementary) | +10.84 | -1.61 | +12.42 | +9.39 | 3/0/0 |
| QDA, ridge 1e-3 (supplementary) | +4.35 | -1.64 | +4.81 | +3.42 | 3/0/0 |

