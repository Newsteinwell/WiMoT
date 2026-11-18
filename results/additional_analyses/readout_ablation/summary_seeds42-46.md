Readout ablation, seeds 42-46 (mean±sample SD over seeds; AFR in percentage points)

**3DO** (seeds [42, 43, 44, 45, 46])

| Readout | AFA (%) | AFR (pp) | New-domain acc. (%) | Earlier-domain acc. (%) |
|---|---|---|---|---|
| OSCL (WiMoT) | 93.98±0.15 | 0.26±0.31 | 93.10±0.13 | 96.75±0.34 |
| Nearest class mean | 80.89±0.89 | 14.04±0.98 | 88.91±0.97 | 78.95±0.41 |
| LDA | 92.06±0.49 | 1.64±0.90 | 90.98±0.89 | 95.69±0.63 |
| QDA (no domain gate) | 91.00±0.70 | 1.23±0.26 | 89.53±0.98 | 95.47±0.23 |
| Linear probe W | 85.77±0.49 | -9.97±0.48 | 70.80±0.97 | 98.22±0.35 |
| LDA, ridge 1e-3 (supplementary) | 92.04±0.47 | 1.61±0.99 | 90.95±0.91 | 95.72±0.67 |
| QDA, ridge 1e-3 (supplementary) | 89.64±0.73 | 0.21±0.59 | 86.94±0.85 | 95.57±0.34 |

Paired difference OSCL minus readout over the same seeds (mean in pp; seeds with OSCL strictly better / equal / worse):

| Readout | ΔAFA | ΔAFR | ΔNew | ΔEarlier | AFA better/equal/worse |
|---|---|---|---|---|---|
| Nearest class mean | +13.09 | -13.78 | +4.19 | +17.80 | 5/0/0 |
| LDA | +1.93 | -1.38 | +2.12 | +1.06 | 5/0/0 |
| QDA (no domain gate) | +2.98 | -0.97 | +3.57 | +1.27 | 5/0/0 |
| Linear probe W | +8.22 | +10.23 | +22.30 | -1.47 | 5/0/0 |
| LDA, ridge 1e-3 (supplementary) | +1.94 | -1.35 | +2.15 | +1.03 | 5/0/0 |
| QDA, ridge 1e-3 (supplementary) | +4.34 | +0.05 | +6.16 | +1.18 | 5/0/0 |

**WiMANS** (seeds [42, 43, 44, 45, 46])

| Readout | AFA (%) | AFR (pp) | New-domain acc. (%) | Earlier-domain acc. (%) |
|---|---|---|---|---|
| OSCL (WiMoT) | 89.57±0.55 | -0.28±0.53 | 87.22±1.25 | 90.83±0.21 |
| Nearest class mean | 87.84±0.80 | 0.56±1.28 | 86.11±2.29 | 88.06±0.60 |
| LDA | 88.83±1.33 | -0.56±1.40 | 85.83±1.21 | 90.56±1.37 |
| QDA (no domain gate) | 80.93±1.33 | 1.11±1.81 | 77.50±2.85 | 81.94±1.57 |
| Linear probe W | 85.62±1.10 | -8.80±2.45 | 73.89±2.23 | 88.52±1.06 |
| LDA, ridge 1e-3 (supplementary) | 88.89±1.33 | -0.65±1.16 | 86.02±1.11 | 90.56±1.11 |
| QDA, ridge 1e-3 (supplementary) | 63.46±1.85 | 11.76±4.76 | 62.59±2.52 | 63.98±3.91 |

Paired difference OSCL minus readout over the same seeds (mean in pp; seeds with OSCL strictly better / equal / worse):

| Readout | ΔAFA | ΔAFR | ΔNew | ΔEarlier | AFA better/equal/worse |
|---|---|---|---|---|---|
| Nearest class mean | +1.73 | -0.83 | +1.11 | +2.78 | 5/0/0 |
| LDA | +0.74 | +0.28 | +1.39 | +0.28 | 3/1/1 |
| QDA (no domain gate) | +8.64 | -1.39 | +9.72 | +8.89 | 5/0/0 |
| Linear probe W | +3.95 | +8.52 | +13.33 | +2.31 | 5/0/0 |
| LDA, ridge 1e-3 (supplementary) | +0.68 | +0.37 | +1.20 | +0.28 | 3/0/2 |
| QDA, ridge 1e-3 (supplementary) | +26.11 | -12.04 | +24.63 | +26.85 | 5/0/0 |

**Widar3.0** (seeds [42, 43, 44, 45, 46])

| Readout | AFA (%) | AFR (pp) | New-domain acc. (%) | Earlier-domain acc. (%) |
|---|---|---|---|---|
| OSCL (WiMoT) | 95.24±0.88 | 0.42±0.49 | 94.97±0.35 | 96.38±0.82 |
| Nearest class mean | 67.05±1.28 | 18.20±1.76 | 81.43±0.70 | 68.32±1.67 |
| LDA | 84.43±0.94 | 2.18±0.53 | 82.58±0.57 | 87.08±0.97 |
| QDA (no domain gate) | 91.41±0.50 | 2.48±0.34 | 92.58±0.57 | 93.58±0.57 |
| Linear probe W | 81.94±0.80 | 0.27±0.41 | 48.77±1.59 | 86.35±0.69 |
| LDA, ridge 1e-3 (supplementary) | 84.37±0.99 | 2.20±0.47 | 82.63±0.62 | 86.95±1.00 |
| QDA, ridge 1e-3 (supplementary) | 90.62±0.52 | 2.00±0.55 | 89.88±0.78 | 92.73±0.69 |

Paired difference OSCL minus readout over the same seeds (mean in pp; seeds with OSCL strictly better / equal / worse):

| Readout | ΔAFA | ΔAFR | ΔNew | ΔEarlier | AFA better/equal/worse |
|---|---|---|---|---|---|
| Nearest class mean | +28.19 | -17.78 | +13.53 | +28.07 | 5/0/0 |
| LDA | +10.81 | -1.77 | +12.38 | +9.30 | 5/0/0 |
| QDA (no domain gate) | +3.84 | -2.07 | +2.38 | +2.80 | 5/0/0 |
| Linear probe W | +13.30 | +0.15 | +46.20 | +10.03 | 5/0/0 |
| LDA, ridge 1e-3 (supplementary) | +10.87 | -1.78 | +12.33 | +9.43 | 5/0/0 |
| QDA, ridge 1e-3 (supplementary) | +4.62 | -1.58 | +5.08 | +3.65 | 5/0/0 |

