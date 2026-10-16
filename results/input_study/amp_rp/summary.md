## Input family 2b: random projection of the amplitude descriptor (128-D)

AFA (%) ↑ and AFR (pp) ↓, mean ± sample SD over seeds 42–44; sorted by the mean AFA over the three datasets. Rows marked `n<3` have fewer than three seeds.

| Method | 3DO AFA | 3DO AFR | WiMANS AFA | WiMANS AFR | Widar3.0 AFA | Widar3.0 AFR | Mean AFA |
|---|---:|---:|---:|---:|---:|---:|---:|
| WiMoT (amp_full, primary) | 98.64 ± 0.24 | 0.10 ± 0.28 | 33.74 ± 0.99 | 0.31 ± 0.53 | 88.20 ± 0.28 | 1.67 ± 0.36 | 73.53 |
| Joint training (non-continual reference) | 99.31 ± 0.06 | – | 30.25 ± 1.72 | – | 87.36 ± 0.57 | – | 72.30 |
| WiMoT (amp_only) | 97.98 ± 0.14 | 0.49 ± 0.09 | 31.28 ± 0.78 | 0.62 ± 1.07 | 78.84 ± 0.26 | 1.81 ± 0.27 | 69.37 |
| VanillaCP+ER | 96.53 ± 0.12 | 2.06 ± 0.13 | 27.47 ± 0.31 | 1.54 ± 0.96 | 63.95 ± 0.75 | 13.67 ± 0.87 | 62.65 |
| DualCP+ER | 95.72 ± 1.03 | 2.42 ± 1.26 | 29.84 ± 0.36 | -0.00 ± 1.22 | 56.62 ± 1.58 | 13.19 ± 0.67 | 60.72 |
| VanillaCP | 91.37 ± 0.30 | 9.61 ± 0.35 | 27.47 ± 0.82 | 0.31 ± 0.27 | 61.90 ± 0.15 | 15.42 ± 0.08 | 60.25 |
| DualCP | 91.58 ± 0.31 | 8.86 ± 0.30 | 27.67 ± 1.46 | 0.77 ± 1.87 | 56.15 ± 1.50 | 15.06 ± 0.27 | 58.47 |
| DER++ | 96.60 ± 0.88 | 2.12 ± 0.87 | 28.09 ± 2.69 | 3.55 ± 2.14 | 42.74 ± 1.44 | 50.11 ± 2.06 | 55.81 |
| ER | 98.19 ± 0.31 | 1.74 ± 0.52 | 26.85 ± 2.74 | 5.40 ± 3.08 | 40.59 ± 1.49 | 53.94 ± 1.78 | 55.21 |
| EWC+ER | 98.20 ± 0.32 | 1.78 ± 0.52 | 27.06 ± 1.25 | 4.63 ± 2.12 | 40.32 ± 1.37 | 54.31 ± 1.74 | 55.19 |
| SI+ER | 98.30 ± 0.31 | 1.63 ± 0.46 | 26.65 ± 2.36 | 5.56 ± 3.34 | 40.47 ± 1.31 | 54.03 ± 1.53 | 55.14 |
| LwF+ER | 95.38 ± 0.98 | -3.58 ± 0.49 | 26.23 ± 1.54 | 1.70 ± 1.93 | 43.09 ± 0.23 | 33.03 ± 0.49 | 54.90 |
| CN-DPM | 97.73 ± 0.44 | 1.88 ± 0.62 | 27.67 ± 3.58 | -2.62 ± 4.20 | 38.81 ± 0.96 | 41.64 ± 1.90 | 54.74 |
| MAS+ER | 93.56 ± 0.46 | 6.25 ± 0.81 | 28.09 ± 1.60 | 2.93 ± 1.41 | 41.70 ± 0.97 | 35.92 ± 1.63 | 54.45 |
| EdgeCL | 97.50 ± 0.78 | 2.56 ± 1.06 | 29.94 ± 1.72 | 4.01 ± 2.09 | 33.68 ± 0.42 | 58.94 ± 1.21 | 53.71 |
| LwF | 74.92 ± 0.35 | 31.06 ± 0.52 | 25.00 ± 0.31 | 1.23 ± 1.17 | 36.69 ± 0.35 | 42.86 ± 0.27 | 45.54 |
| ConSense | 76.94 ± 1.78 | 33.82 ± 2.63 | 27.67 ± 1.55 | 6.02 ± 1.22 | 30.17 ± 0.45 | 65.64 ± 0.98 | 44.93 |
| SI | 69.93 ± 0.83 | 43.82 ± 1.19 | 25.31 ± 1.93 | 6.79 ± 1.49 | 36.17 ± 1.01 | 55.53 ± 1.99 | 43.80 |
| A-GEM | 69.81 ± 0.95 | 44.42 ± 1.59 | 25.21 ± 1.55 | 5.86 ± 2.79 | 33.68 ± 0.31 | 62.00 ± 0.36 | 42.90 |
| MAS | 83.74 ± 1.80 | 12.14 ± 2.87 | 22.43 ± 0.78 | 4.01 ± 0.27 | 22.10 ± 1.27 | 9.22 ± 1.80 | 42.76 |
| EWC | 67.59 ± 0.21 | 47.71 ± 0.23 | 25.21 ± 0.78 | 7.25 ± 0.96 | 35.19 ± 0.41 | 59.75 ± 0.76 | 42.66 |
| Fine-tuning | 67.61 ± 0.60 | 47.74 ± 0.90 | 26.44 ± 0.94 | 5.25 ± 0.27 | 33.78 ± 0.77 | 62.06 ± 0.70 | 42.61 |

### Ranking by AFA on each dataset (all methods and the reference)

| Rank | 3DO | WiMANS | Widar3.0 |
|---:|---|---|---|
| 1 | Joint training (non-continual reference) (99.31) | WiMoT (amp_full, primary) (33.74) | WiMoT (amp_full, primary) (88.20) |
| 2 | WiMoT (amp_full, primary) (98.64) | WiMoT (amp_only) (31.28) | Joint training (non-continual reference) (87.36) |
| 3 | SI+ER (98.30) | Joint training (non-continual reference) (30.25) | WiMoT (amp_only) (78.84) |
| 4 | EWC+ER (98.20) | EdgeCL (29.94) | VanillaCP+ER (63.95) |
| 5 | ER (98.19) | DualCP+ER (29.84) | VanillaCP (61.90) |
| 6 | WiMoT (amp_only) (97.98) | MAS+ER (28.09) | DualCP+ER (56.62) |
| 7 | CN-DPM (97.73) | DER++ (28.09) | DualCP (56.15) |
| 8 | EdgeCL (97.50) | DualCP (27.67) | LwF+ER (43.09) |
| 9 | DER++ (96.60) | CN-DPM (27.67) | DER++ (42.74) |
| 10 | VanillaCP+ER (96.53) | ConSense (27.67) | MAS+ER (41.70) |
| 11 | DualCP+ER (95.72) | VanillaCP+ER (27.47) | ER (40.59) |
| 12 | LwF+ER (95.38) | VanillaCP (27.47) | SI+ER (40.47) |
| 13 | MAS+ER (93.56) | EWC+ER (27.06) | EWC+ER (40.32) |
| 14 | DualCP (91.58) | ER (26.85) | CN-DPM (38.81) |
| 15 | VanillaCP (91.37) | SI+ER (26.65) | LwF (36.69) |
| 16 | MAS (83.74) | Fine-tuning (26.44) | SI (36.17) |
| 17 | ConSense (76.94) | LwF+ER (26.23) | EWC (35.19) |
| 18 | LwF (74.92) | SI (25.31) | Fine-tuning (33.78) |
| 19 | SI (69.93) | A-GEM (25.21) | EdgeCL (33.68) |
| 20 | A-GEM (69.81) | EWC (25.21) | A-GEM (33.68) |
| 21 | Fine-tuning (67.61) | LwF (25.00) | ConSense (30.17) |
| 22 | EWC (67.59) | MAS (22.43) | MAS (22.10) |

### WiMoT versus the strongest baseline on each dataset (paired by seed)

| Dataset | Strongest baseline (mean AFA) | Baseline AFA | WiMoT amp_full AFA | Gap, pp (mean ± SD of per-seed differences; seeds with WiMoT ahead) | Mantis-input gap of the paper, pp |
|---|---|---:|---:|---|---:|
| 3DO | SI+ER | 98.30 | 98.64 | +0.34 ± 0.49 (2/3) | +3.56 |
| WiMANS | EdgeCL | 29.94 | 33.74 | +3.81 ± 0.99 (3/3) | +1.03 |
| Widar3.0 | VanillaCP+ER | 63.95 | 88.20 | +24.25 ± 0.97 (3/3) | +13.28 |

### Mantis-input reference (paper Table I; WiMoT with 9728-D input, baselines with the 9216-D Mantis block)

| Method | 3DO AFA | WiMANS AFA | Widar3.0 AFA |
|---|---:|---:|---:|
| WiMoT | 94.02 ± 0.07 | 89.61 ± 0.47 | 95.16 ± 1.23 |
| Fine-tuning | 82.98 ± 0.66 | 86.52 ± 0.18 | 72.47 ± 1.57 |
| ER | 89.76 ± 0.77 | 87.45 ± 0.18 | 76.49 ± 0.67 |
| EWC | 82.97 ± 0.61 | 86.32 ± 0.18 | 72.47 ± 1.24 |
| EWC+ER | 89.73 ± 0.81 | 87.35 ± 0.31 | 76.52 ± 0.78 |
| SI | 89.69 ± 0.55 | 87.14 ± 0.71 | 77.53 ± 0.69 |
| SI+ER | 90.36 ± 0.20 | 88.17 ± 0.47 | 81.88 ± 0.76 |
| MAS | 87.55 ± 0.69 | 86.93 ± 0.18 | 68.77 ± 0.31 |
| MAS+ER | 87.79 ± 0.42 | 87.65 ± 0.31 | 71.14 ± 0.11 |
| LwF | 84.45 ± 0.29 | 87.24 ± 0.78 | 79.46 ± 0.42 |
| LwF+ER | 87.20 ± 0.66 | 88.37 ± 0.18 | 81.41 ± 0.34 |
| DER++ | 89.86 ± 0.81 | 88.58 ± 0.82 | 80.37 ± 0.22 |
| A-GEM | 85.51 ± 0.33 | 86.73 ± 0.62 | 73.06 ± 0.95 |
| ConSense | 77.05 ± 1.28 | 81.17 ± 2.23 | 56.00 ± 1.37 |
| EdgeCL | 89.23 ± 0.98 | 86.93 ± 1.58 | 74.89 ± 0.83 |
| DUCT | 76.82 ± 12.55 | 79.84 ± 2.62 | 42.10 ± 1.32 |
| DualCP | 77.59 ± 0.31 | 68.31 ± 0.94 | 53.51 ± 0.64 |
| VanillaCP | 78.24 ± 0.11 | 73.05 ± 1.39 | 69.23 ± 0.57 |
| CN-DPM | 90.45 ± 0.11 | 85.60 ± 0.64 | 75.80 ± 0.85 |
| DUCT+ER | 84.95 ± 0.82 | 79.01 ± 3.21 | 37.63 ± 7.16 |
| DualCP+ER | 81.42 ± 0.49 | 71.09 ± 1.52 | 53.28 ± 0.11 |
| VanillaCP+ER | 82.69 ± 0.06 | 73.87 ± 1.39 | 69.28 ± 0.54 |

### Coefficients selected on seed-42 validation (3DO and WiMANS)

| Method | Selected | Candidates | Mean validation AFA per candidate |
|---|---:|---|---|
| Fine-tuning | None | – | single candidate |
| ER | None | – | single candidate |
| EWC | 100.0 | [1.0, 100.0, 10000.0] | [0.4558, 0.4706, 0.4623] |
| EWC+ER | 1.0 | [1.0, 100.0, 10000.0] | [0.6275, 0.6184, 0.6081] |
| SI | 10.0 | [0.1, 1.0, 10.0] | [0.4557, 0.4598, 0.471] |
| SI+ER | 0.1 | [0.1, 1.0, 10.0] | [0.6264, 0.6213, 0.6213] |
| MAS | 10.0 | [0.1, 1.0, 10.0] | [0.4987, 0.5269, 0.5515] |
| MAS+ER | 0.1 | [0.1, 1.0, 10.0] | [0.6095, 0.5862, 0.5661] |
| LwF | 0.5 | [0.5, 1.0, 2.0] | [0.4863, 0.4352, 0.4269] |
| LwF+ER | 0.5 | [0.5, 1.0, 2.0] | [0.6025, 0.5646, 0.5515] |
| DER++ | None | – | single candidate |
| A-GEM | None | – | single candidate |
| ConSense | 0.01 | [0.01, 0.1, 1.0] | [0.5474, 0.5448, 0.5302] |
| EdgeCL | 0.03 | [0.01, 0.03, 0.1] | [0.6276, 0.6403, 0.6254] |
| DUCT | None | [0.1, 0.25, 0.5] | not runnable on this input: RuntimeError('Sinkhorn failed to satisfy marginal masses: error=1.287699882390747e-06') |
| DualCP | 0.1 | [0.1, 0.5] | [0.6303, 0.6195] |
| VanillaCP | 1.0 | – | single candidate |
| CN-DPM | 0.0 | [-10.0, 0.0, 10.0] | [0.6152, 0.6165, 0.6014] |
| DUCT+ER | None | [0.1, 0.25, 0.5] | not runnable on this input: RuntimeError('Sinkhorn failed to satisfy marginal masses: error=1.287699882390747e-06') |
| DualCP+ER | 0.1 | [0.1, 0.5] | [0.6453, 0.6391] |
| VanillaCP+ER | 1.0 | – | single candidate |
