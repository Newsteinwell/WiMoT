# Third-party code and models

## Code included in this repository

| File | Origin | License |
|---|---|---|
| `wimot/baselines/duct.py` | Adapted from the official DUCT code, [Estrella-fugaz/CVPR25-DUCT](https://github.com/Estrella-fugaz/CVPR25-DUCT) at commit `ecfb579f418b555b029a6ae11e74e5dd47ee6b59` | MIT, Copyright (c) 2025 Pluto; the notice is kept in the file header |
| `wimot/baselines/cn_dpm.py` | Adapted from the official CN-DPM code, [soochan-lee/CN-DPM](https://github.com/soochan-lee/CN-DPM) at commit `d592c5d794e4bddc4947f2ed61d6a9e50c6da277` | MIT, Copyright (c) 2020 Soochan Lee; the notice is kept in the file header |

All other baselines (EWC, SI, MAS, LwF, ER, DER++, A-GEM, DualCP/VanillaCP, ConSense, EdgeCL) are written for this
repository from the respective papers; no third-party source is copied.

## Dependencies installed separately (not redistributed)

| Package | Use | License |
|---|---|---|
| `mantis-tsfm==1.1.0` ([vfeofanov/mantis](https://github.com/vfeofanov/mantis)) | MantisV2 encoder (feature extraction only) | Apache-2.0 |
| `csiread==1.4.1` | Reading Widar3.0 Intel 5300 `.dat` files | MIT |
| `momentfm`, `granite-tsfm==0.3.10`, `chronos-forecasting==2.3.2`, `timesfm==3.0.2` | Alternative encoders of the backbone study | see the respective packages |

## Model weights (downloaded by the user, not redistributed)

| Encoder | Hugging Face repository | Revision |
|---|---|---|
| MantisV2 | `paris-noah/MantisV2` | `8f6ca35cb54ab14b120618943c6fca5ddf5a76a6` (Apache-2.0) |
| TSPulse | `ibm-granite/granite-timeseries-tspulse-r1` | `2e64fcdc2a06d3565dfadaf0065c0ab5055f80f2` |
| Chronos-2 | `amazon/chronos-2` | `29ec3766d36d6f73f0696f85560a422f50e8498c` |
| TimesFM 2.5 | `google/timesfm-2.5-200m-pytorch` | `1d952420fba87f3c6dee4f240de0f1a0fbc790e3` |
| TimesFM 3.0 | `google/timesfm-3.0-pytorch` | `43046b85ec22d584a13f8098c2ed39c889e129c2` |
| MOMENT | `AutonLab/MOMENT-1-large` | `ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc` (MIT) |

The license terms of each checkpoint are those stated on its model card.

## Datasets (not redistributed)

3DO, WiMANS, and Widar3.0 are used under the terms of their providers; see docs/REPRODUCE.md, "Data".
