## Input family 1: 512-D amplitude descriptor

AFA (%) ↑ and AFR (pp) ↓, mean ± sample SD over seeds 42–44; sorted by the mean AFA over the three datasets. Rows marked `n<3` have fewer than three seeds.

| Method | 3DO AFA | 3DO AFR | WiMANS AFA | WiMANS AFR | Widar3.0 AFA | Widar3.0 AFR | Mean AFA |
|---|---:|---:|---:|---:|---:|---:|---:|
| WiMoT (amp_full, primary) | 99.02 ± 0.16 | -0.01 ± 0.13 | 32.51 ± 2.06 | 1.70 ± 3.15 | 94.00 ± 0.15 | 0.64 ± 0.48 | 75.18 |
| Joint training (non-continual reference) | 99.09 ± 0.23 | – | 27.88 ± 0.89 | – | 94.81 ± 0.20 | – | 73.93 |
| WiMoT (amp_only) | 97.37 ± 0.23 | 0.31 ± 0.43 | 25.82 ± 1.25 | -0.62 ± 0.71 | 85.90 ± 0.37 | 1.08 ± 0.25 | 69.70 |
| SI+ER | 97.86 ± 0.59 | 1.65 ± 1.13 | 23.97 ± 1.39 | 5.56 ± 5.26 | 52.54 ± 1.37 | 43.92 ± 1.30 | 58.13 |
| EWC+ER | 98.26 ± 0.60 | 1.07 ± 0.99 | 25.72 ± 1.55 | 4.94 ± 3.99 | 47.93 ± 0.80 | 50.94 ± 0.82 | 57.30 |
| DER++ | 95.89 ± 1.04 | 2.14 ± 1.41 | 24.90 ± 1.39 | 5.71 ± 0.27 | 46.79 ± 1.30 | 51.56 ± 0.94 | 55.86 |
| CN-DPM | 97.44 ± 0.32 | 2.32 ± 0.02 | 27.88 ± 1.70 | 1.54 ± 2.28 | 42.02 ± 0.15 | 45.17 ± 0.25 | 55.78 |
| ER | 98.22 ± 0.39 | 1.37 ± 0.74 | 23.46 ± 1.72 | 6.94 ± 5.01 | 45.04 ± 1.85 | 54.64 ± 2.28 | 55.57 |
| LwF+ER | 92.12 ± 1.43 | -4.26 ± 1.27 | 23.66 ± 1.46 | 1.23 ± 0.27 | 48.30 ± 2.36 | 35.81 ± 2.38 | 54.69 |
| MAS+ER | 91.31 ± 0.45 | 7.62 ± 0.76 | 24.07 ± 1.11 | 5.40 ± 2.33 | 45.75 ± 1.53 | 36.53 ± 2.04 | 53.71 |
| VanillaCP | 61.06 ± 11.67 | 13.43 ± 5.73 | 21.40 ± 0.36 | 2.93 ± 0.71 | 59.73 ± 2.14 | 6.42 ± 0.14 | 47.40 |
| EdgeCL | 82.81 ± 0.94 | 17.56 ± 1.93 | 31.17 ± 2.23 | 3.86 ± 2.67 | 25.68 ± 0.23 | 53.03 ± 1.29 | 46.56 |
| VanillaCP+ER | 56.41 ± 10.59 | 12.69 ± 9.21 | 21.91 ± 0.82 | 1.54 ± 1.41 | 60.22 ± 0.15 | 6.39 ± 0.42 | 46.18 |
| SI | 72.88 ± 0.39 | 39.15 ± 0.66 | 22.74 ± 0.71 | 8.49 ± 0.96 | 42.17 ± 1.52 | 55.81 ± 1.94 | 45.93 |
| LwF | 74.66 ± 0.23 | 29.62 ± 1.15 | 20.88 ± 1.86 | 5.25 ± 1.17 | 40.59 ± 0.32 | 49.86 ± 0.59 | 45.38 |
| A-GEM | 71.31 ± 1.08 | 41.79 ± 1.66 | 22.22 ± 1.35 | 8.64 ± 2.19 | 38.72 ± 1.71 | 61.75 ± 1.97 | 44.08 |
| EWC | 72.86 ± 0.51 | 39.32 ± 0.69 | 20.78 ± 1.39 | 10.80 ± 1.49 | 38.44 ± 0.46 | 61.67 ± 0.67 | 44.03 |
| Fine-tuning | 69.24 ± 0.60 | 44.88 ± 0.98 | 22.94 ± 0.47 | 9.10 ± 0.96 | 36.99 ± 0.17 | 63.47 ± 0.34 | 43.06 |
| ConSense | 70.03 ± 0.29 | 38.03 ± 0.55 | 25.21 ± 1.70 | 6.33 ± 4.45 | 27.80 ± 1.01 | 67.69 ± 1.13 | 41.01 |
| MAS | 83.02 ± 1.37 | 8.04 ± 2.11 | 16.98 ± 1.41 | 8.64 ± 1.49 | 22.52 ± 0.46 | 7.03 ± 1.04 | 40.84 |
| DUCT+ER | 55.93 ± 2.78 | -18.10 ± 9.09 | 15.64 ± 2.66 | 7.56 ± 3.35 | 17.68 ± 1.23 | 11.39 ± 0.34 | 29.75 |
| DUCT | 33.33 ± 0.00 | 4.24 ± 7.34 | 23.46 ± 2.14 | -1.85 ± 3.24 | 17.43 ± 1.54 | 7.81 ± 5.56 | 24.74 |

### Ranking by AFA on each dataset (all methods and the reference)

| Rank | 3DO | WiMANS | Widar3.0 |
|---:|---|---|---|
| 1 | Joint training (non-continual reference) (99.09) | WiMoT (amp_full, primary) (32.51) | Joint training (non-continual reference) (94.81) |
| 2 | WiMoT (amp_full, primary) (99.02) | EdgeCL (31.17) | WiMoT (amp_full, primary) (94.00) |
| 3 | EWC+ER (98.26) | Joint training (non-continual reference) (27.88) | WiMoT (amp_only) (85.90) |
| 4 | ER (98.22) | CN-DPM (27.88) | VanillaCP+ER (60.22) |
| 5 | SI+ER (97.86) | WiMoT (amp_only) (25.82) | VanillaCP (59.73) |
| 6 | CN-DPM (97.44) | EWC+ER (25.72) | SI+ER (52.54) |
| 7 | WiMoT (amp_only) (97.37) | ConSense (25.21) | LwF+ER (48.30) |
| 8 | DER++ (95.89) | DER++ (24.90) | EWC+ER (47.93) |
| 9 | LwF+ER (92.12) | MAS+ER (24.07) | DER++ (46.79) |
| 10 | MAS+ER (91.31) | SI+ER (23.97) | MAS+ER (45.75) |
| 11 | MAS (83.02) | LwF+ER (23.66) | ER (45.04) |
| 12 | EdgeCL (82.81) | ER (23.46) | SI (42.17) |
| 13 | LwF (74.66) | DUCT (23.46) | CN-DPM (42.02) |
| 14 | SI (72.88) | Fine-tuning (22.94) | LwF (40.59) |
| 15 | EWC (72.86) | SI (22.74) | A-GEM (38.72) |
| 16 | A-GEM (71.31) | A-GEM (22.22) | EWC (38.44) |
| 17 | ConSense (70.03) | VanillaCP+ER (21.91) | Fine-tuning (36.99) |
| 18 | Fine-tuning (69.24) | VanillaCP (21.40) | ConSense (27.80) |
| 19 | VanillaCP (61.06) | LwF (20.88) | EdgeCL (25.68) |
| 20 | VanillaCP+ER (56.41) | EWC (20.78) | MAS (22.52) |
| 21 | DUCT+ER (55.93) | MAS (16.98) | DUCT+ER (17.68) |
| 22 | DUCT (33.33) | DUCT+ER (15.64) | DUCT (17.43) |

### WiMoT versus the strongest baseline on each dataset (paired by seed)

| Dataset | Strongest baseline (mean AFA) | Baseline AFA | WiMoT amp_full AFA | Gap, pp (mean ± SD of per-seed differences; seeds with WiMoT ahead) | Mantis-input gap of the paper, pp |
|---|---|---:|---:|---|---:|
| 3DO | EWC+ER | 98.26 | 99.02 | +0.76 ± 0.52 (3/3) | +3.56 |
| WiMANS | EdgeCL | 31.17 | 32.51 | +1.34 ± 1.25 (3/3) | +1.03 |
| Widar3.0 | VanillaCP+ER | 60.22 | 94.00 | +33.78 ± 0.30 (3/3) | +13.28 |

### Mantis-input reference (WiMoT with 9728-D input, baselines with the 9216-D Mantis block; paper Supplementary Table S9, 9216 column)

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
| EWC | 100.0 | [1.0, 100.0, 10000.0] | [0.4621, 0.4837, 0.4815] |
| EWC+ER | 100.0 | [1.0, 100.0, 10000.0] | [0.6126, 0.6317, 0.6091] |
| SI | 10.0 | [0.1, 1.0, 10.0] | [0.4581, 0.4703, 0.474] |
| SI+ER | 10.0 | [0.1, 1.0, 10.0] | [0.6084, 0.6108, 0.6164] |
| MAS | 10.0 | [0.1, 1.0, 10.0] | [0.5009, 0.519, 0.533] |
| MAS+ER | 0.1 | [0.1, 1.0, 10.0] | [0.6065, 0.5796, 0.5322] |
| LwF | 0.5 | [0.5, 1.0, 2.0] | [0.4784, 0.4376, 0.4296] |
| LwF+ER | 0.5 | [0.5, 1.0, 2.0] | [0.601, 0.5553, 0.5503] |
| DER++ | None | – | single candidate |
| A-GEM | None | – | single candidate |
| ConSense | 0.1 | [0.01, 0.1, 1.0] | [0.4906, 0.4983, 0.4976] |
| EdgeCL | 0.03 | [0.01, 0.03, 0.1] | [0.5557, 0.5673, 0.5304] |
| DUCT | 0.1 | [0.1, 0.25, 0.5] | [0.2731, 0.2716, 0.2531] |
| DualCP | None | [0.1, 0.5] | not runnable on this input: AssertionError('Not a dual hierarchy') |
| VanillaCP | 1.0 | – | single candidate |
| CN-DPM | -10.0 | [-10.0, 0.0, 10.0] | [0.6186, 0.6115, 0.612] |
| DUCT+ER | 0.25 | [0.1, 0.25, 0.5] | [0.2454, 0.4156, 0.3825] |
| DualCP+ER | None | [0.1, 0.5] | not runnable on this input: AssertionError('Not a dual hierarchy') |
| VanillaCP+ER | 1.0 | – | single candidate |
