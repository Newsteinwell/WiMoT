# Foundation-model-free input study: all input families

AFA (%) ↑ / AFR (pp) ↓, mean ± sample SD over seeds 42–44 (`n<3`: fewer seeds). Every family is evaluated with the protocol of the paper; coefficients of the baselines were selected on seed-42 validation (3DO and WiMANS) separately for each input. No input family reads a foundation-model output; WiMoT runs as `amp_full` (primary; all stems and the PCA shortcut read the family) and `amp_only` (amplitude stem only).

## 3DO

| Input | WiMoT amp_full AFA | AFR | WiMoT amp_only AFA | Joint AFA | Strongest baseline | Baseline AFA | AFR | Gap, pp (seeds ahead) | Baselines below WiMoT | WiMoT rank (1 = best, among WiMoT and baselines) |
|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|
| amp512 | 99.02 ± 0.16 | -0.01 ± 0.13 | 97.37 ± 0.23 | 99.09 ± 0.23 | EWC+ER | 98.26 ± 0.60 | 1.07 ± 0.99 | +0.76 ± 0.52 (3/3) | 19/19 | 1 |
| amp_pca | 92.38 ± 1.05 | 1.39 ± 1.92 | 83.40 ± 0.06 | 96.03 ± 0.11 | CN-DPM | 92.79 ± 0.24 | 4.44 ± 0.69 | -0.41 ± 0.82 (1/3) | 17/19 | 3 |
| amp_rp | 98.64 ± 0.24 | 0.10 ± 0.28 | 97.98 ± 0.14 | 99.31 ± 0.06 | SI+ER | 98.30 ± 0.31 | 1.63 ± 0.46 | +0.34 ± 0.49 (2/3) | 19/19 | 1 |
| amp_hand | 97.19 ± 0.28 | 0.58 ± 0.13 | 95.72 ± 0.29 | 97.44 ± 0.35 | EdgeCL | 95.96 ± 0.24 | 1.64 ± 0.52 | +1.22 ± 0.31 (3/3) | 21/21 | 1 |
| Mantis block + amplitude (paper, reference) | 94.02 ± 0.07 | | | | CN-DPM | 90.45 | | +3.56 (mean) | | |

## WiMANS

| Input | WiMoT amp_full AFA | AFR | WiMoT amp_only AFA | Joint AFA | Strongest baseline | Baseline AFA | AFR | Gap, pp (seeds ahead) | Baselines below WiMoT | WiMoT rank (1 = best, among WiMoT and baselines) |
|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|
| amp512 | 32.51 ± 2.06 | 1.70 ± 3.15 | 25.82 ± 1.25 | 27.88 ± 0.89 | EdgeCL | 31.17 ± 2.23 | 3.86 ± 2.67 | +1.34 ± 1.25 (3/3) | 19/19 | 1 |
| amp_pca | 16.26 ± 1.25 | 1.39 ± 1.67 | 10.91 ± 1.52 | 19.65 ± 0.47 | EWC+ER | 21.09 ± 1.25 | -4.01 ± 1.63 | -4.84 ± 1.89 (0/3) | 10/19 | 10 |
| amp_rp | 33.74 ± 0.99 | 0.31 ± 0.53 | 31.28 ± 0.78 | 30.25 ± 1.72 | EdgeCL | 29.94 ± 1.72 | 4.01 ± 2.09 | +3.81 ± 0.99 (3/3) | 19/19 | 1 |
| amp_hand | 37.24 ± 2.32 | 3.55 ± 1.63 | 33.64 ± 1.85 | 32.82 ± 0.78 | DualCP+ER | 33.74 ± 1.52 | 0.77 ± 1.41 | +3.50 ± 1.89 (3/3) | 21/21 | 1 |
| Mantis block + amplitude (paper, reference) | 89.61 ± 0.47 | | | | DER++ | 88.58 | | +1.03 (mean) | | |

## Widar3.0

| Input | WiMoT amp_full AFA | AFR | WiMoT amp_only AFA | Joint AFA | Strongest baseline | Baseline AFA | AFR | Gap, pp (seeds ahead) | Baselines below WiMoT | WiMoT rank (1 = best, among WiMoT and baselines) |
|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|
| amp512 | 94.00 ± 0.15 | 0.64 ± 0.48 | 85.90 ± 0.37 | 94.81 ± 0.20 | VanillaCP+ER | 60.22 ± 0.15 | 6.39 ± 0.42 | +33.78 ± 0.30 (3/3) | 19/19 | 1 |
| amp_pca | 91.48 ± 1.08 | 0.92 ± 0.38 | 84.79 ± 0.49 | 96.96 ± 0.45 | VanillaCP+ER | 75.31 ± 0.24 | 11.33 ± 0.08 | +16.17 ± 1.15 (3/3) | 19/19 | 1 |
| amp_rp | 88.20 ± 0.28 | 1.67 ± 0.36 | 78.84 ± 0.26 | 87.36 ± 0.57 | VanillaCP+ER | 63.95 ± 0.75 | 13.67 ± 0.87 | +24.25 ± 0.97 (3/3) | 19/19 | 1 |
| amp_hand | 82.07 ± 0.71 | 1.53 ± 0.21 | 73.09 ± 0.21 | 78.44 ± 0.63 | VanillaCP+ER | 62.89 ± 0.71 | 12.00 ± 0.50 | +19.19 ± 0.97 (3/3) | 21/21 | 1 |
| Mantis block + amplitude (paper, reference) | 95.16 ± 1.23 | | | | SI+ER | 81.88 | | +13.28 (mean) | | |

## Lowest AFR among the baselines versus WiMoT (amp_full)

| Input | Dataset | WiMoT AFR | Lowest baseline AFR (method, its AFA) |
|---|---|---:|---|
| amp512 | 3DO | -0.01 ± 0.13 | DUCT+ER: -18.10 pp (AFA 55.93) |
| amp512 | WiMANS | 1.70 ± 3.15 | DUCT: -1.85 pp (AFA 23.46) |
| amp512 | Widar3.0 | 0.64 ± 0.48 | VanillaCP+ER: 6.39 pp (AFA 60.22) |
| amp_pca | 3DO | 1.39 ± 1.92 | LwF+ER: -1.89 pp (AFA 83.27) |
| amp_pca | WiMANS | 1.39 ± 1.67 | EWC+ER: -4.01 pp (AFA 21.09) |
| amp_pca | Widar3.0 | 0.92 ± 0.38 | VanillaCP+ER: 11.33 pp (AFA 75.31) |
| amp_rp | 3DO | 0.10 ± 0.28 | LwF+ER: -3.58 pp (AFA 95.38) |
| amp_rp | WiMANS | 0.31 ± 0.53 | CN-DPM: -2.62 pp (AFA 27.67) |
| amp_rp | Widar3.0 | 1.67 ± 0.36 | MAS: 9.22 pp (AFA 22.10) |
| amp_hand | 3DO | 0.58 ± 0.13 | DUCT+ER: -3.61 pp (AFA 82.96) |
| amp_hand | WiMANS | 3.55 ± 1.63 | CN-DPM: -0.15 pp (AFA 30.35) |
| amp_hand | Widar3.0 | 1.53 ± 0.21 | DUCT+ER: 9.86 pp (AFA 22.79) |

## Trajectories per family (finished test.json; seeds 42–44)

| Family | WiMoT amp_full | WiMoT amp_only | Joint | Baseline trajectories (expected / finished) | Baselines not runnable on the input |
|---|---:|---:|---:|---|---|
| amp512 | 9/9 | 9/9 | 9/9 | 171 / 171 | DualCP (AssertionError('Not a dual hierarchy')); DualCP+ER (AssertionError('Not a dual hierarchy')) |
| amp_pca | 9/9 | 9/9 | 9/9 | 171 / 171 | DualCP (AssertionError('Not a dual hierarchy')); DualCP+ER (AssertionError('Not a dual hierarchy')) |
| amp_rp | 9/9 | 9/9 | 9/9 | 171 / 171 | DUCT (RuntimeError('Sinkhorn failed to satisfy marginal masses: error=1.287699882390747e-06')); DUCT+ER (RuntimeError('Sinkhorn failed to satisfy marginal masses: error=1.287699882390747e-06')) |
| amp_hand | 9/9 | 9/9 | 9/9 | 189 / 189 | none |

## 与基础模型输入（论文主结果）的对比

基线使用 9216 维 Mantis 块时（论文补充材料 Table S9 的 9216 列）：WiMoT 94.02 / 89.61 / 95.16；最强基线 90.45（CN-DPM）/ 88.58（DER++）/ 81.88（SI+ER）。联合训练（9728 维）90.64 / 91.46 / 94.84。随机初始化的 MantisV2（`backbone` 研究，不是无编码器）：WiMoT 96.03 / 78.29 / 87.33，该研究中最强基线（9216 维输入）87.32 / 76.03 / 54.69；论文 Table VIII 与补充材料 Table S4 使用 9728 维输入。

| 数据集 | WiMoT：Mantis 输入 | WiMoT：无基础模型（四个族的范围；amp512） | 最强基线：Mantis 输入 | 最强基线：无基础模型（范围） |
|---|---:|---|---:|---|
| 3DO | 94.02 | 92.38–99.02（99.02） | 90.45 | 92.79–98.30 |
| WiMANS | 89.61 | 16.26–37.24（32.51） | 88.58 | 21.09–33.74 |
| Widar3.0 | 95.16 | 82.07–94.00（94.00） | 81.88 | 60.22–75.31 |

（范围是四个输入族各自的取值，仅供对照，不是按测试结果挑选的单一输入。）

## 结论（以实际数字为准）

1. **3DO：基础模型对两类方法都没有帮助。** 除 PCA 白化外，WiMoT 在无基础模型输入下为 97.19–99.02，高于 Mantis 输入的 94.02（amp512、amp_rp、amp_hand 分别高 5.0、4.6、3.2 pp，即基础模型的“贡献”为负；amp_pca 的 92.38 低 1.6 pp）；基线同样升至 95.96–98.30，高于 90.45（amp_pca 的 92.79 也略高）。WiMoT 在 amp512 上领先最强基线 +0.76（3/3 个种子），amp_rp 上 +0.34（2/3，在种子波动内），amp_hand 上 +1.22（3/3）；**amp_pca 上落后**：92.38 对 CN-DPM 的 92.79（−0.41，仅 1/3 个种子领先）。
2. **WiMANS：振幅描述子几乎不含类别信息，基础模型不可缺少。** 无基础模型时所有方法都只有 16–37%（随机猜测约 11%，联合训练 20–33%），WiMoT 从 89.61 降到 16.26–37.24，基线从 88.58 降到 ≤33.74。WiMoT 在 amp512、amp_rp、amp_hand 上名义上最高（领先 +1.34、+3.81、+3.50 pp，3/3 种子），**但在这个水平上差异不应被解读为方法优劣**；amp_pca 上 WiMoT（16.26）低于 9 个基线，最强基线 EWC+ER 为 21.09（−4.84，0/3）。
3. **Widar3.0：振幅描述子信息足够，WiMoT 几乎不需要基础模型，基线需要。** 联合训练在 amp512 上 94.81，WiMoT amp_full 94.00（与 Mantis 输入的 95.16 相差 1.16 pp）；其余族 82.07–91.48。基线在无基础模型输入下只有 60.22–75.31（最强者是 VanillaCP+ER；多数方法的 AFR 达 40–55 pp，基本发生灾难性遗忘），低于 Mantis 输入的 81.88。因此 WiMoT 对最强基线的领先在无基础模型输入下**比有基础模型时更大**：+33.78 / +16.17 / +24.25 / +19.19 pp，对 +13.28 pp，每个种子都领先。
4. **WiMoT 相对基线的总体情况：** 在 4 个输入族×3 个数据集共 12 个格子里，WiMoT amp_full 的 AFA 在 10 个格子里最高，2 个格子里落后（amp_pca 的 3DO 与 WiMANS）；其中 3DO 的 amp512/amp_rp 与 WiMANS 的全部领先量都在 1–4 pp，种子数只有 3，不应过度解读。WiMoT 的遗忘（AFR）在所有格子里都在 −0.01 到 3.55 pp 之间；基线的 AFR 因方法而异，少数方法（如 LwF+ER、DUCT+ER、CN-DPM 在某些格子）的 AFR 更低，但它们的 AFA 更低。详见每个族 `summary.md` 中的“最低 AFR”表。
5. **基础模型到底带来多少（Mantis 输入减去无基础模型输入的 AFA，pp，各输入族的范围）：** WiMANS 上收益很大（WiMoT +52.4 至 +73.4；最强基线 +54.8 至 +67.5）；Widar3.0 上对 WiMoT 只有 +1.2 至 +13.1，对最强基线是 +6.6 至 +21.7；3DO 上对 WiMoT 在 amp512/amp_rp/amp_hand 为 −5.0/−4.6/−3.2（amp_pca 为 +1.6），对最强基线为 −2.3 至 −7.9（无基础模型更好）。持续学习算法本身不依赖基础模型特征；使用基础模型的主要原因是 WiMANS 这类需要强特征的数据集，以及保护基线在 Widar3.0 上不崩溃。

## 局限

- 没有原始 CSI 窗口，所以没有真正的手工时频特征和从头训练的编码器；“手工特征”只来自 512 维描述子，3DO 描述子的数值秩只有约 100（`amp_pca` 的 D=104）。
- 只有 3 个种子；随机投影只抽一次；EdgeCL 在 GPU 上不确定；ConSense/EdgeCL 的 dropout 种子与论文不同。
- 部分基线在某些输入上无法运行，已从该族排除（见 README 第 5 节），“最强基线”是在可运行的基线中取的。
- 各族的系数只在 3DO 与 WiMANS 的验证集上选择；WiMANS 的验证 AFA 在无基础模型输入下接近随机，选择可能主要由噪声决定。
- WiMoT 的 `amp_full` 在 D<512 的族里，输入被零填充到 512 维。
