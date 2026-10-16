## Input family 2: PCA-whitened amplitude descriptor

AFA (%) ↑ and AFR (pp) ↓, mean ± sample SD over seeds 42–44; sorted by the mean AFA over the three datasets. Rows marked `n<3` have fewer than three seeds.

| Method | 3DO AFA | 3DO AFR | WiMANS AFA | WiMANS AFR | Widar3.0 AFA | Widar3.0 AFR | Mean AFA |
|---|---:|---:|---:|---:|---:|---:|---:|
| Joint training (non-continual reference) | 96.03 ± 0.11 | – | 19.65 ± 0.47 | – | 96.96 ± 0.45 | – | 70.88 |
| WiMoT (amp_full, primary) | 92.38 ± 1.05 | 1.39 ± 1.92 | 16.26 ± 1.25 | 1.39 ± 1.67 | 91.48 ± 1.08 | 0.92 ± 0.38 | 66.71 |
| VanillaCP+ER | 92.68 ± 0.39 | 3.92 ± 0.42 | 13.58 ± 1.88 | 1.23 ± 1.17 | 75.31 ± 0.24 | 11.33 ± 0.08 | 60.52 |
| WiMoT (amp_only) | 83.40 ± 0.06 | 0.67 ± 0.86 | 10.91 ± 1.52 | 0.93 ± 0.93 | 84.79 ± 0.49 | 1.36 ± 0.39 | 59.70 |
| VanillaCP | 84.10 ± 0.11 | 15.46 ± 0.11 | 13.68 ± 1.58 | 1.70 ± 0.27 | 74.47 ± 0.41 | 13.11 ± 0.05 | 57.42 |
| SI+ER | 87.35 ± 0.16 | 9.28 ± 0.25 | 20.78 ± 0.78 | -3.40 ± 0.27 | 58.10 ± 2.24 | 41.72 ± 2.63 | 55.41 |
| EWC+ER | 88.60 ± 1.29 | 10.00 ± 1.92 | 21.09 ± 1.25 | -4.01 ± 1.63 | 52.47 ± 1.33 | 50.44 ± 1.45 | 54.05 |
| ER | 88.60 ± 1.29 | 10.00 ± 1.92 | 20.58 ± 1.08 | -3.24 ± 1.22 | 52.64 ± 1.78 | 50.28 ± 1.95 | 53.94 |
| DER++ | 88.80 ± 0.17 | 5.99 ± 1.27 | 17.70 ± 1.70 | -1.08 ± 1.41 | 53.09 ± 2.47 | 48.36 ± 2.72 | 53.19 |
| EdgeCL | 91.08 ± 1.03 | 5.14 ± 0.75 | 20.58 ± 2.48 | -3.86 ± 3.60 | 46.35 ± 0.82 | 50.72 ± 0.88 | 52.67 |
| CN-DPM | 92.79 ± 0.24 | 4.44 ± 0.69 | 12.14 ± 0.89 | 3.40 ± 1.63 | 49.56 ± 3.81 | 42.81 ± 2.98 | 51.49 |
| LwF+ER | 83.27 ± 0.53 | -1.89 ± 0.88 | 18.42 ± 1.55 | -0.46 ± 0.80 | 51.63 ± 2.02 | 31.64 ± 3.06 | 51.10 |
| MAS+ER | 81.67 ± 0.99 | 12.00 ± 0.74 | 19.34 ± 1.58 | -2.62 ± 0.27 | 49.75 ± 2.61 | 42.17 ± 2.13 | 50.25 |
| ConSense | 84.85 ± 2.31 | 17.60 ± 2.67 | 17.59 ± 1.07 | -0.77 ± 2.28 | 31.41 ± 2.19 | 61.61 ± 3.12 | 44.62 |
| EWC | 82.30 ± 0.59 | 19.79 ± 0.65 | 16.36 ± 0.00 | 1.85 ± 1.60 | 33.78 ± 2.26 | 71.75 ± 2.43 | 44.14 |
| Fine-tuning | 82.30 ± 0.59 | 19.79 ± 0.65 | 15.43 ± 0.31 | 2.31 ± 1.60 | 33.90 ± 2.16 | 71.61 ± 2.23 | 43.88 |
| A-GEM | 82.24 ± 0.65 | 19.67 ± 1.07 | 16.05 ± 1.85 | 3.09 ± 0.96 | 32.59 ± 1.29 | 72.83 ± 1.40 | 43.63 |
| SI | 82.03 ± 0.49 | 17.11 ± 1.03 | 15.12 ± 0.31 | 2.47 ± 1.17 | 33.04 ± 2.44 | 70.36 ± 2.57 | 43.40 |
| DUCT | 78.39 ± 0.99 | 3.53 ± 0.43 | 14.09 ± 0.36 | -0.00 ± 0.93 | 35.93 ± 2.59 | 19.50 ± 3.56 | 42.80 |
| DUCT+ER | 82.54 ± 1.13 | 0.28 ± 1.76 | 13.99 ± 2.70 | 0.15 ± 3.08 | 28.22 ± 2.78 | 28.42 ± 3.15 | 41.58 |
| LwF | 79.18 ± 0.74 | 7.11 ± 0.23 | 16.26 ± 0.89 | 1.08 ± 1.41 | 24.42 ± 2.36 | 51.64 ± 0.79 | 39.95 |
| MAS | 74.02 ± 2.23 | 23.46 ± 2.84 | 14.51 ± 1.63 | 2.47 ± 1.34 | 31.31 ± 3.68 | 63.33 ± 5.09 | 39.94 |

### Ranking by AFA on each dataset (all methods and the reference)

| Rank | 3DO | WiMANS | Widar3.0 |
|---:|---|---|---|
| 1 | Joint training (non-continual reference) (96.03) | EWC+ER (21.09) | Joint training (non-continual reference) (96.96) |
| 2 | CN-DPM (92.79) | SI+ER (20.78) | WiMoT (amp_full, primary) (91.48) |
| 3 | VanillaCP+ER (92.68) | EdgeCL (20.58) | WiMoT (amp_only) (84.79) |
| 4 | WiMoT (amp_full, primary) (92.38) | ER (20.58) | VanillaCP+ER (75.31) |
| 5 | EdgeCL (91.08) | Joint training (non-continual reference) (19.65) | VanillaCP (74.47) |
| 6 | DER++ (88.80) | MAS+ER (19.34) | SI+ER (58.10) |
| 7 | EWC+ER (88.60) | LwF+ER (18.42) | DER++ (53.09) |
| 8 | ER (88.60) | DER++ (17.70) | ER (52.64) |
| 9 | SI+ER (87.35) | ConSense (17.59) | EWC+ER (52.47) |
| 10 | ConSense (84.85) | EWC (16.36) | LwF+ER (51.63) |
| 11 | VanillaCP (84.10) | WiMoT (amp_full, primary) (16.26) | MAS+ER (49.75) |
| 12 | WiMoT (amp_only) (83.40) | LwF (16.26) | CN-DPM (49.56) |
| 13 | LwF+ER (83.27) | A-GEM (16.05) | EdgeCL (46.35) |
| 14 | DUCT+ER (82.54) | Fine-tuning (15.43) | DUCT (35.93) |
| 15 | Fine-tuning (82.30) | SI (15.12) | Fine-tuning (33.90) |
| 16 | EWC (82.30) | MAS (14.51) | EWC (33.78) |
| 17 | A-GEM (82.24) | DUCT (14.09) | SI (33.04) |
| 18 | SI (82.03) | DUCT+ER (13.99) | A-GEM (32.59) |
| 19 | MAS+ER (81.67) | VanillaCP (13.68) | ConSense (31.41) |
| 20 | LwF (79.18) | VanillaCP+ER (13.58) | MAS (31.31) |
| 21 | DUCT (78.39) | CN-DPM (12.14) | DUCT+ER (28.22) |
| 22 | MAS (74.02) | WiMoT (amp_only) (10.91) | LwF (24.42) |

### WiMoT versus the strongest baseline on each dataset (paired by seed)

| Dataset | Strongest baseline (mean AFA) | Baseline AFA | WiMoT amp_full AFA | Gap, pp (mean ± SD of per-seed differences; seeds with WiMoT ahead) | Mantis-input gap of the paper, pp |
|---|---|---:|---:|---|---:|
| 3DO | CN-DPM | 92.79 | 92.38 | -0.41 ± 0.82 (1/3) | +3.56 |
| WiMANS | EWC+ER | 21.09 | 16.26 | -4.84 ± 1.89 (0/3) | +1.03 |
| Widar3.0 | VanillaCP+ER | 75.31 | 91.48 | +16.17 ± 1.15 (3/3) | +13.28 |

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
| EWC | 1.0 | [1.0, 100.0, 10000.0] | [0.5138, 0.4999, 0.486] |
| EWC+ER | 1.0 | [1.0, 100.0, 10000.0] | [0.5766, 0.5735, 0.545] |
| SI | 10.0 | [0.1, 1.0, 10.0] | [0.5083, 0.511, 0.5158] |
| SI+ER | 10.0 | [0.1, 1.0, 10.0] | [0.5738, 0.5743, 0.5775] |
| MAS | 0.1 | [0.1, 1.0, 10.0] | [0.4823, 0.3969, 0.3401] |
| MAS+ER | 0.1 | [0.1, 1.0, 10.0] | [0.5468, 0.4302, 0.3619] |
| LwF | 0.5 | [0.5, 1.0, 2.0] | [0.5123, 0.4746, 0.4613] |
| LwF+ER | 0.5 | [0.5, 1.0, 2.0] | [0.5565, 0.5237, 0.5086] |
| DER++ | None | – | single candidate |
| A-GEM | None | – | single candidate |
| ConSense | 1.0 | [0.01, 0.1, 1.0] | [0.4942, 0.5046, 0.5193] |
| EdgeCL | 0.03 | [0.01, 0.03, 0.1] | [0.5642, 0.5739, 0.5723] |
| DUCT | 0.25 | [0.1, 0.25, 0.5] | [0.49, 0.4946, 0.4639] |
| DualCP | None | [0.1, 0.5] | not runnable on this input: AssertionError('Not a dual hierarchy') |
| VanillaCP | 1.0 | – | single candidate |
| CN-DPM | 10.0 | [-10.0, 0.0, 10.0] | [0.5401, 0.5299, 0.5427] |
| DUCT+ER | 0.5 | [0.1, 0.25, 0.5] | [0.4737, 0.4711, 0.4809] |
| DualCP+ER | None | [0.1, 0.5] | not runnable on this input: AssertionError('Not a dual hierarchy') |
| VanillaCP+ER | 1.0 | – | single candidate |
