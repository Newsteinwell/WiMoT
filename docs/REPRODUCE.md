# Reproducing the paper

Everything needed to regenerate the numbers of the paper from the official raw data: data preparation, feature extraction,
the job lists of every table, reference results, and the determinism notes. The [README](../README.md) covers installation,
the quick start, and the smoke test. Commands are run from the repository root.

The repository contains

* data preparation from the official raw files to the fixed partitions used in the paper (`data/splits/`),
* frozen-encoder feature extraction (MantisV2 and the alternative encoders of the backbone study),
* the method, its ablation and storage variants,
* the 21 baseline configurations (12 classic, CN-DPM, DUCT, DualCP/VanillaCP with and without replay, ConSense, EdgeCL)
  and the joint-training reference,
* one job list per table (`configs/jobs/*.tsv`: dataset, order, label fraction, seed, method, coefficient) and the
  locked coefficients (`configs/locks.json`),
* the per-seed reference results of every reported trajectory: `results/reference_results_same_input.json` (the records of the
  paper tables, in which the baselines of the main, orders, fewlabel, widarext, and backbone studies receive the 9728-D input of the method) and
  `results/reference_results.json` (the baselines with the 9216-D Mantis block alone), and a script that
  rebuilds the tables and compares a reproduction with them,
* the stage-wise test predictions of the reported trajectories (`results/stagewise_test_predictions.npz`) and the
  coefficient-selection runs (`results/coefficient_selection.json`).

Raw datasets, prepared arrays, features, and model weights are not redistributed.

## Environment

All reported numbers were produced with Python 3.12, PyTorch 2.8.0 (CUDA 12.8, cuDNN 9.10), NumPy 2.3.2, and SciPy 1.18.1
on Linux with an NVIDIA GPU.

```bash
conda env create -f environment.yml && conda activate wimot          # Linux + CUDA 12.8, versions used for the paper
# or
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128    # Linux + NVIDIA GPU
pip install -r requirements.txt                                      # core: torch, numpy, scipy, pytest
pip install -r requirements-data.txt                                 # only for data preparation: pandas, csiread
pip install -r requirements-features.txt                             # only for feature extraction: mantis-tsfm
python -m pytest tests
```

On macOS or without a GPU, skip the first `pip` line (the default PyTorch wheel is used) and add `--device cpu` to the
commands below (`DEVICE=cpu` for `experiments/run_table.sh`); the smoke test in the README needs only the core requirements.
The code was also run with Python 3.13 and the same pinned versions on macOS (Apple silicon). NumPy builds that use Apple
Accelerate print spurious floating-point `RuntimeWarning`s in float64 matrix products; the results are finite, and
`PYTHONWARNINGS=ignore::RuntimeWarning` silences them. `fcntl` file locks make Windows unsupported.

`mantis-tsfm` is needed only for feature extraction, `pandas` and `csiread` only for data preparation. The alternative
encoders of the backbone study need further packages with conflicting pins; we used one environment per encoder:
TSPulse with `granite-tsfm==0.3.10` and `transformers==5.19.0`; Chronos-2 with `chronos-forecasting==2.3.2` and
`transformers==5.19.0`; TimesFM 2.5 and 3.0 with `timesfm==3.0.2` (which provides the `timesfm3` module); MOMENT with
`momentfm`.

## Data


Download the datasets from their providers and accept their terms of use.

| Stream | Source | Files used |
|---|---|---|
| 3DO (3 days, 3 activities) | J. Strohmayer and M. Kampel, Zenodo, doi:[10.5281/zenodo.10925351](https://doi.org/10.5281/zenodo.10925351) | `d1/`, `d2/`, `d3/` session folders `<activity><session>/csiposreg.csv` |
| WiMANS (3 rooms, 9 activities) | S. Huang et al., arXiv:2402.09430; Kaggle dataset `shuokanghuang/wimans` | `annotation.csv`, `wifi_csi/amp/<recording>.npy` |
| Widar3.0 Room 1 (9 users, 6 gestures) | Y. Zheng et al., MobiSys 2019, doi:[10.1145/3307334.3326081](https://doi.org/10.1145/3307334.3326081); official Tsinghua download | `CSI/20181130_user5_10_11.zip`, `CSI/20181130_user12_13_14.zip`, `CSI/20181130_user15_16_17.zip` |
| Widar3.0 Rooms 2 and 3 | same | `CSI/20181204.zip`, `CSI/20181209.zip` (Room 2), `CSI/20181211.zip` (Room 3) |

Gestures 1-6 of Widar3.0 (Push&Pull, Sweep, Clap, Slide, Draw-O (horizontal), Draw-Zigzag (horizontal)) are used.
Extract the Widar3.0 archives; the preparation searches them recursively for `*.dat`.

```bash
RAW_3DO=/data/3do RAW_WIMANS=/data/wimans/wifi_csi/amp WIMANS_ANNOTATION=/data/wimans/annotation.csv \
RAW_WIDAR=/data/widar/room1 RAW_WIDAR_ROOMS=/data/widar/rooms PREPARED=/data/prepared experiments/prepare_data.sh
```

The partitions are fixed by `data/splits/<stream>.json`: 3DO lists the session and window start of every window and the
SHA-256 of every source CSV; WiMANS and Widar3.0 list the recording identities and labels of every partition (and the
Widar3.0 instances skipped for having fewer than 300 valid packets in a receiver). The preparation stops if a source file
or a partition differs, and compares every output array with the SHA-256 of the original preparation
(`configs/digests.json`).

## Features


```bash
PREPARED=/data/prepared FEATURES=/data/features WEIGHTS=/data/weights experiments/extract_features.sh mantis
```

`WEIGHTS/mantis` is a local snapshot of the Hugging Face repository `paris-noah/MantisV2` at revision
`8f6ca35cb54ab14b120618943c6fca5ddf5a76a6` (`model.safetensors` SHA-256
`49d46d9a49cccdc87c46f4e0088fa52c0a6ef7eb4c13de5cc9815426b7b17ab1`), e.g.

```bash
huggingface-cli download paris-noah/MantisV2 --revision 8f6ca35cb54ab14b120618943c6fca5ddf5a76a6 --local-dir /data/weights/mantis
```

Each output `<domain>_<split>.npz` holds `features` (N x 9728) and `labels`; `passport.json` records the file and array
digests, and the extraction compares the arrays with those of the paper (`FEATURES_MATCH_ORIGINAL`). The backbone study
additionally uses `moment`, `tspulse`, `chronos2`, `timesfm25`, and `timesfm30` (revisions in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md))
and `mantis_random`, the MantisV2 architecture with random weights (`torch.manual_seed(20261014)`, no download).

## Reproducing the tables


```bash
FEATURES=/data/features RUNS=/data/runs SHARDS=4 experiments/run_table.sh <study>
python scripts/build_tables.py --runs /data/runs --study <study>       # comparison with the paper
python scripts/build_tables.py --reference-only --study <study>        # the paper numbers only
```

| Study | Paper | Trajectories |
|---|---|---|
| `main` | Tables I, III; Fig. 3 (stage-wise accuracies); Supplementary Tables S2, S7, S9 (baselines: 9728-D input, see below) | 198 |
| `ablation` | Table IV; Supplementary Table S10 (seeds 42-46) | 120 |
| `sameinput` | Supplementary Table S8 (with the joint-training reference) | 180 |
| `backbone` | Table VIII; Supplementary Tables S4, S5 (needs `FEATURES/<encoder>/`) | 324 |
| `widarext` | Table VI; Fig. 4 | 90 |
| `orders` | Table VII; Supplementary Table S1; Fig. 4 | 450 |
| `fewlabel` | Supplementary Table S3; Fig. 4 | 198 |
| `memory` | Table IX (storage); Supplementary Table S6 | 45 |
| `extra_seeds` | Supplementary Section S-IV-A, "Additional Seeds and Reproducibility" (seeds 45, 46) | 56 |

The baselines of `main`, `orders`, `fewlabel`, `widarext` (including Widar3.0 rooms 2 and 3), and `backbone` (encoder study) in the paper tables use the 9728-D input described below; the trajectory counts
above are those of the job lists of `configs/jobs/` (WiMoT, the ablations, and the baselines with the 9216-D Mantis block). The storage study (`memory`) and the five-seed study (`extra_seeds`) use the 9216-D input.

The plasticity and stability measures of Table III and Supplementary Table S10 (new-domain accuracy, earlier-domain accuracy after the
last update, AFR) are recomputed from the stored accuracy matrices, without retraining, by
`python scripts/plasticity_stability.py` (studies `main` and `ablation`; `--results results/reference_results.json` for the 9216-D baselines).
`python scripts/readme_numbers.py` prints every number quoted in the README, including the leads against the 9216-D baselines, and
`python scripts/plot_stagewise.py` and `python scripts/plot_loss_sweep.py` redraw the figures of the README.

### Classifier ablation and loss-weight sweep

Table IV (rows OSCL → NCM/LDA/QDA) and Supplementary Section S-VIII fit alternative classifiers on the adapters and stored moments of the
trained full method, without retraining the adapters; Supplementary Section S-IX retrains the method with other loss weights. Both write to
`results/additional_analyses/`.

```bash
python scripts/ablation_readout.py eval --features /data/features/mantis --runs /data/runs --out OUT [--dataset 3do,wimans,widar]
python scripts/ablation_readout.py summarize --out OUT
python scripts/ablation_loss_sweep.py jobs --out DIR --shards 12          # job lists for the grid, then run them with wimot.study
python scripts/ablation_loss_sweep.py summarize --runs RUNS --out OUT
```

The sweep reproduces the accuracy matrices of the ablation rows `no_hist` (weight 0 of the historical loss) and `no_supcon` (SupCon weight 0 in every domain)
identically on the same GPU model, and settings with the same ratio of the two weights give identical matrices.

`study.py` trains all trajectories of a table and evaluates on TEST only afterwards. Every trajectory writes `receipt.json`
(training record, validation matrix) and `test.json` (test accuracy matrix, AFA, AFR). The deployment costs of Table IX
are analytic; the storage column is the `statistic_bytes` field of the `memory` runs.

The coefficients in `configs/locks.json` were selected before any test evaluation from the candidates in
`configs/protocol.json` (seed 42, mean validation AFA over 3DO and WiMANS). `results/coefficient_selection.json` holds the
validation AFA and validation matrix of every candidate run of the paper (132 runs, 24 selections, all reproducing the
locked values). They can be recomputed with

```bash
python scripts/select_coefficients.py --features /data/features/mantis --out /data/runs/selection
```

## Baselines with the 9728-D input

All baselines of the main, orders, fewlabel, widarext (and backbone) tables can also be run on the input of the method, the
9216-D Mantis block concatenated with the 512-D amplitude descriptor (`--input cat9728`). The job lists are in
`configs/jobs_sameall/` (baselines only; `python scripts/make_jobs_same_input.py` rewrites them), the coefficients re-selected
for this input in `configs/locks_sameinput_all.json`, which `WIMOT_LOCKS` selects at run time.

```bash
# coefficient selection (seed 42, validation AFA, 3DO and WiMANS), then the table; ConSense/EdgeCL dropout seeded per trajectory
python scripts/select_coefficients.py --features FEATURES/mantis --out RUNS/selection --input cat9728 --global-seed 20261015 \
    --reseed-per-job --extra-candidates dualcp=0.85,dualcp_er=0.85 --aggregate --write-locks configs/locks_sameinput_all.json  # after the runs
WIMOT_LOCKS=configs/locks_sameinput_all.json python -m wimot.study sameall_main --features FEATURES --out RUNS \
    --global-seed 20261015 --reseed-per-job --cleanup --shard 0/4        # likewise sameall_orders, sameall_fewlabel, sameall_widarext
python scripts/same_input_report.py --runs RUNS --out OUT                # tables and old-versus-new numbers
```

The encoder study and Widar3.0 rooms 2 and 3 read `FEATURES/<encoder>/<dataset>` for `<encoder>` in `mantis`, `mantis_random`, `moment`, `tspulse`,
`chronos2`, `timesfm25`, `timesfm30` and `<dataset>` in `3do`, `wimans`, `widar`, and (for `mantis`) `widar_r2`, `widar_r3`; each directory holds the
`<domain>_<split>.npz` files and the `passport.json` of the feature extraction (symlinks are sufficient). The coefficients are those selected on the
MantisV2 features. Each directory contains the encoder's 9216-D embedding block followed by the same 512-D amplitude descriptor.

```bash
WIMOT_LOCKS=configs/locks_sameinput_all.json python -m wimot.study sameall_backbone --features FEATURES --out RUNS \
    --global-seed 20261015 --reseed-per-job --cleanup --shard 0/5        # 270 trajectories
WIMOT_LOCKS=configs/locks_sameinput_all.json python -m wimot.study sameall_widarext --features FEATURES --out RUNS --dataset widar_r2,widar_r3 \
    --global-seed 20261015 --reseed-per-job --cleanup                    # 30 trajectories
```

`--cleanup` trains, tests, and deletes the checkpoints of every trajectory in turn (disk budget of the 9-domain Widar3.0
streams). ConSense reads 19 tokens (18 Mantis tokens and the amplitude descriptor) in this setting, as EdgeCL does. DualCP
needs a class-similarity graph with a dual hierarchy: with this input the protocol thresholds 0.1 and 0.5 do not give one on
3DO (and 0.1 not on WiMANS), so the threshold 0.85 is added to the candidates and is the only eligible one.

## Inputs without a foundation model

The study of four inputs that do not read any MantisV2 output (the 512-D amplitude descriptor and three transforms of it) uses
`scripts/make_derived_inputs.py` (transforms fitted on the first-domain training partition only), `scripts/input_study.py`
(seed-42 coefficient selection per input, then WiMoT, 21 baselines, and the joint-training reference on seeds 42-44),
`scripts/summarize_input_study.py` (one family) and `scripts/combine_input_families.py` (all families).

```bash
python scripts/make_derived_inputs.py --features FEATURES/mantis --out FEATURES_DERIVED --kinds amp_pca,amp_rp,amp_hand
for TAG in amp_pca amp_rp amp_hand; do               # the 512-D descriptor itself: --features FEATURES/mantis --tag amp512
  python scripts/input_study.py select-train --features FEATURES_DERIVED/$TAG --out RUNS/$TAG --tag $TAG --shard-name w0 &   # several workers may share RUNS/$TAG
  wait
  python scripts/input_study.py select --features FEATURES_DERIVED/$TAG --out RUNS/$TAG --tag $TAG --cleanup
  python scripts/input_study.py final  --features FEATURES_DERIVED/$TAG --out RUNS/$TAG --tag $TAG --shard-name w0 --cleanup
  python scripts/summarize_input_study.py --raw RUNS/$TAG --out OUT/$TAG
done
python scripts/combine_input_families.py --root OUT --families amp512,amp_pca,amp_rp,amp_hand
```

A candidate that cannot be trained on an input (DualCP: no dual hierarchy on amp512 and amp_pca; DUCT and DUCT+ER: the Sinkhorn
solver of the implementation does not reach its marginal tolerance on WiMANS with amp_rp) has no seed-42 validation score; the
method is then not run on that input and is listed as not runnable in the summary.

## Time and hardware


The reported runs used one NVIDIA RTX PRO 6000 Blackwell Server Edition GPU per machine (two CPU threads per process).
On this GPU, a 3DO trajectory takes about 30 s for the method and the classic baselines and about 40 s for EdgeCL.
Measured with four processes sharing one GPU: the `main` table (198 trajectories) took 32 min of wall-clock time
(training 30 min, testing 3 min; summed training time 1.8 h, of which 1.0 h for CN-DPM, DUCT, and DualCP), and the
`ablation` table (120 trajectories) 1.8 h. Data preparation runs on CPU: about 1 min per WiMANS room and about
2 min per Widar3.0 user.

## Determinism


* The method, the classic baselines, CN-DPM, DUCT, and DualCP draw all random numbers from seeded generators, and TF32
  is disabled. On the same GPU model and software stack, a second run reproduces the reference accuracy matrix bitwise; we
  verified this for SI+ER with the Mantis block and with the concatenated input (3DO, seed 42) and for the method,
  whose repeated runs of Tables I and IV matched 127 of 129 reference matrices bitwise (the other two differ by 0.01 pp).
* Floating-point results depend on the GPU model. The baselines with the 9216-D Mantis block were trained on a different machine than the 9728-D baselines;
  on the GPU above the same code yields trajectories that differ in the last digits from the first epoch on: for SI+ER, the AFA of the nine
  stream-and-seed trajectories differs by −0.62 to +0.08 pp, with table means within the seed variation. The nine main-setting trajectories of the method are bitwise identical on the two machines.
* The WiMoT records of the encoder study (`backbone`, input `m9216`) are the values of the paper. Run on the machine that trained the 9728-D baselines, the method reproduced 33 of the 54
  encoder trajectories bitwise and changed 21 (randomly initialized MantisV2 9, TimesFM 3.0 6, Chronos-2 3, MOMENT-1-large 3) by −0.31 to +0.93 pp AFA per seed; no lead changed its sign and no strongest
  baseline changed (largest change of a mean lead: 0.62 pp, WiMANS with the randomly initialized MantisV2). The six WiMoT trajectories of Widar3.0 rooms 2 and 3 were reproduced bitwise.
  EdgeCL is not bitwise reproducible across machines: with identical code and seeds its per-seed AFA differed by up to 3.06 pp between two machines
  (Supplementary Section S-V), so differences of this size between EdgeCL runs are not interpreted.
* ConSense and EdgeCL initialize their weights from the seed but draw dropout masks from the global PyTorch generator,
  which the paper runs did not seed (PyTorch seeds it non-deterministically in every process). Their trajectories are
  therefore reproducible only in distribution. `--global-seed` makes repeated runs deterministic.
* DUCT is unstable across seeds (SD up to 7.9 pp for DUCT and 31.6 pp for DUCT+ER on WiMANS in Table I), so last-digit differences can change its outcome by
  several points.
* The reduced-label baseline runs were resumed after interruptions and partly ran on CPU; they are reproducible in
  distribution only.

## Reference results


`results/reference_results_same_input.json` holds the records of the paper tables and `results/reference_results.json` the records with the 9216-D baselines (both with the layout below); `results/same_input/<study>_baselines_9728.json` are the baseline records with the 9728-D input, and `scripts/merge_same_input.py` writes the merged file. Each record holds, for every trajectory reported in the paper, its study, dataset, order, label
fraction, encoder, input, family, method, coefficient, seed, AFA, AFR, and the full test accuracy matrix (row *s*: test
accuracies on the domains seen after update *s*).

`results/stagewise_test_predictions.npz` (5 MB) holds, for 1337 of the 1652 records, the predicted class, true class,
and stream position of the domain of every TEST window after every update; record *n* is entry *n* of
`reference_results.json` (the baselines with the 9216-D input; the stage-wise predictions of the 9728-D baselines were not stored), and `results/stagewise_test_predictions.json` lists the covered records and the reasons for the
others (the joint-training reference has no stages; the reduced-label baseline runs and the MOMENT and random-initialization
backbone runs did not keep their predictions). The class scores (softmax outputs or logits, 97 MB) are an archive attachment,
`stagewise_test_scores.npz`, whose SHA-256 is given in the same file.

```bash
python scripts/check_predictions.py [--scores stagewise_test_scores.npz]   # predictions -> reference matrices
```
