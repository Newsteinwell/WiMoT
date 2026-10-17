<div align="center">

# WiMoT：基于矩传输的免回放持续 Wi-Fi 感知

**在不保存任何历史 CSI 的前提下，持续学习新的房间、日期和用户。**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](requirements.txt)
[![PyTorch 2.8](https://img.shields.io/badge/PyTorch-2.8-EE4C2C.svg)](requirements.txt)
[![Paper](https://img.shields.io/badge/Paper-TO%20BE%20ADDED-lightgrey.svg)](#引用)

[论文（待补充）](#引用) · [补充材料（待补充）](#引用) · [引用](#引用) · [English](README.md)

</div>

<p align="center">
  <img src="assets/wimot_stagewise.png" width="100%" alt="三个数据集上逐阶段的已见域平均准确率">
</p>
<p align="center"><sub><b>数据流上的准确率</b>（论文 Fig. 3）。纵轴为每个训练阶段之后在已见域上的平均测试准确率（%），横轴为训练阶段，域顺序为原始顺序，三次运行的均值；
(a) 3DO（3 天）、(b) WiMANS（3 个房间）、(c) Widar3.0（9 个用户）。曲线依次为 WiMoT、SI+ER、MAS+ER、LwF+ER、DER++、EdgeCL、ConSense 和微调（基线使用 9728 维输入）；
微调、SI+ER、MAS+ER、LwF+ER、DER++ 共用首个域的网络，因此在阶段 1 重合。每条曲线的末点即下表的 AFA（WiMoT 为 94.02、89.61、95.16%）。</sub></p>

论文 *WiMoT: Replay-Free Continual Wi-Fi Sensing via Moment Transport*（IEEE Internet of Things Journal，[待补充]）的代码。
WiMoT 是一种免回放的持续学习算法：学习新采集域时不保存任何早期样本。它不存样本，而是在冻结特征基中为每个早期域的每个类别存储特征的均值与协方差。表示发生变化后，这些统计量会过时；WiMoT 把变化限制为仿射适配器，并对存储的统计量施加同一个仿射映射，使其保持有效：仿射映射把均值 μ 变为 Aμ + b、协方差 Σ 变为 AΣAᵀ，与它对样本本身的作用一致。监督对比学习与由存储统计量闭式计算的历史损失在可塑性（学习新域）与稳定性（保持早期域）之间取得平衡；OSCL 分类器（高斯判别分类器）每次更新后按各域等权重重新拟合，预测时无需域 ID。

<p align="center">
  <img src="assets/wimot_framework.png" width="92%" alt="WiMoT 框架">
</p>

<p align="center">
  <img src="assets/wimot_components.png" width="92%" alt="WiMoT 的三个组件">
</p>
<p align="center"><sub><b>三个组件</b>（论文 Fig. 2；二维合成点示意）。(a) SupCon 提供可塑性。(b) 历史损失提供表示稳定性：早期类别保持线性可分（圆环：最佳线性映射误判的样本）。(c) 矩传输提供记忆稳定性：适配器更新后，存储的统计量仍然有效。</sub></p>

## 亮点

- **同一组矩提供可塑性与稳定性。** 监督对比学习（SupCon）学习新域；由存储矩闭式计算的历史损失使早期标签保持线性可读；OSCL 分类器由全部域传输后的矩重新拟合。WiMoT 兼顾可塑性与稳定性：三个数据流上的遗忘率均不超过 0.58 pp，AFA 在 3DO 和 Widar3.0 上最高、在 WiMANS 上相当，新域精度比最佳基线低 0.67 至 3.49 pp。
- **不会过期的记忆。** 首个域之后只调整仿射映射，已存储的类别均值和协方差经同一仿射映射随表示变化，不估计漂移。
- **无回放，无需域 ID。** 更新后不保存任何 CSI 窗口或早期特征；每个域只保存 512 维基空间中各类别的样本数、均值和协方差，其大小不随样本数增长。OSCL 分类器从输入本身推断域权重。
- **冻结的基础模型加轻量适配器。** MantisV2 特征在提取后固定；后续域只训练 512→128 的仿射适配器、投影头和历史损失中用到的线性映射。
- **领先于 21 个持续学习基线（含回放方法）。** 在主设定下，3DO 上高 3.77 pp，Widar3.0 上高 13.11 pp，WiMANS 上与最强基线相当（平均最终准确率 +0.41 ± 0.64 pp，遗忘率 0.00 对 2.78 pp）；WiMoT 在三个数据流上的遗忘率均不超过 0.58 pp。

| 设计问题 | 组件 | 保护对象 | 去除实验的证据（论文表 IV 与补充材料 Table S10） |
|---|---|---|---|
| 如何适应而不使记忆失效？ | 冻结基上的仿射适配器；由同一映射传输矩 | 存储记忆的有效性 | 冻结适配器：AFA −1.07 / −3.33 / −5.93 pp |
| 没有样本时如何学习新域而不遗忘？ | SupCon（可塑性）与历史损失（稳定性） | 新域的学习；表示 | 去除 SupCon：新域精度 −5.60 至 −6.85 pp；去除历史损失：AFR +0.36 至 +1.11 pp |
| 如何在无域 ID 时预测并与表示一致？ | 由全部域传输后的矩在均匀先验下重新拟合的 OSCL | 分类器 | 去除历史矩：AFA −17.04 / −5.31 / −47.10 pp |

## 主要结果

平均最终准确率（AFA，%，越高越好）与平均遗忘率（AFR，百分点，越低越好），均值 ± 样本标准差，种子 42–44。所有方法接收 9728 维输入（冻结的 9216 维 MantisV2 特征加 512 维幅度描述子）。
数字由 [`results/reference_results_same_input.json`](results/reference_results_same_input.json) 经 [`scripts/readme_numbers.py`](scripts/readme_numbers.py) 计算，与论文 Table I 一致。

| 方法 | 回放 | 3DO AFA | 3DO AFR | WiMANS AFA | WiMANS AFR | Widar3.0 AFA | Widar3.0 AFR |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **WiMoT（本文）** | 否 | **94.02 ± 0.07** | **0.32 ± 0.33** | **89.61 ± 0.47** | **0.00 ± 0.46** | **95.16 ± 1.23** | **0.58 ± 0.58** |
| 各数据集最强基线¹ | 是 | 90.25 ± 0.67 | 5.64 ± 0.06 | 89.20 ± 1.07 | 2.78 ± 1.22 | 82.05 ± 1.07 | 10.25 ± 1.63 |
| SI+ER | 是 | 90.25 ± 0.67 | 5.64 ± 0.06 | 87.86 ± 0.64 | 4.01 ± 0.27 | 81.38 ± 0.95 | 13.67 ± 0.75 |
| MAS+ER | 是 | 89.28 ± 0.96 | 5.21 ± 0.40 | 89.20 ± 1.07 | 2.78 ± 1.22 | 76.17 ± 0.68 | 10.47 ± 0.88 |
| LwF+ER | 是 | 88.36 ± 0.29 | −1.17 ± 0.41 | 89.09 ± 0.47 | 1.23 ± 0.27 | 82.05 ± 1.07 | 10.25 ± 1.63 |
| DER++ | 是 | 89.81 ± 0.89 | 5.29 ± 0.51 | 88.27 ± 0.53 | 2.78 ± 1.22 | 80.25 ± 1.01 | 16.72 ± 1.39 |
| CN-DPM | 是 | 88.48 ± 0.50 | 8.18 ± 0.32 | 86.32 ± 1.25 | 1.54 ± 2.09 | 74.84 ± 2.22 | 18.97 ± 2.63 |
| 微调（Fine-tuning） | 否 | 84.19 ± 1.13 | 17.62 ± 1.70 | 86.93 ± 1.17 | 4.48 ± 0.96 | 70.02 ± 0.38 | 28.78 ± 0.34 |
| 联合训练² | – | 90.64 ± 0.76 | – | 91.46 ± 0.94 | – | 94.84 ± 0.41 | – |

<sub>¹ 3DO 为 SI+ER，WiMANS 为 MAS+ER，Widar3.0 为 LwF+ER（21 个基线中平均 AFA 最高者，均使用回放）。² 非持续学习的参照：基线网络在全部域的并集上、以相同的 9728 维输入训练（补充材料 Table S8），是参照而非上界。相对最强基线的领先为 +3.77、+0.41 ± 0.64 和 +13.11 pp。
若基线只接收 9216 维 Mantis 块（补充材料 Table S9），领先为 +3.56、+1.03 和 +13.28 pp；对每个基线取两种输入中较好者，则为 +3.56、+0.41 和 +13.11 pp。
**WiMoT 并非在所有设定下领先。** 在 WiMANS 上，WiMoT 的六顺序平均 AFA 最高（比 SI+ER 高 0.34 pp），六种顺序中四种以及 25% 和 10% 标签下 AFR 最低；主设定中 WiMoT 与最强基线的 AFA 未被区分（三个种子中有两个领先，+0.41 pp）；六种域顺序中有三种，MAS 或 SI+ER 高 0.93–1.95 pp；后续域只有 25% 或 10% 标签时 MAS 高 1.23 和 1.44 pp；换用另外五种基础模型编码器时（使用相应编码器的 9728 维输入），五个基线中的最强者（SI+ER，TSPulse-r1 下为 DER++）高 2.67–5.04 pp。</sub>

**可塑性与稳定性。** 新域精度（每次更新中所学域的平均精度，度量可塑性）、最后一次更新后在早期域上的精度，以及 AFR；由同一批运行保存的精度矩阵经 [`scripts/plasticity_stability.py`](scripts/plasticity_stability.py) 计算（论文：表 III）。

| 方法 | 3DO 新域 / 早期域 / AFR | WiMANS 新域 / 早期域 / AFR | Widar3.0 新域 / 早期域 / AFR |
|---|:-:|:-:|:-:|
| **WiMoT（本文）** | 93.07 / **96.82** / 0.32 | 87.50 / **90.74** / **0.00** | 95.08 / **96.31** / **0.58** |
| 微调（Fine-tuning） | 96.49 / 78.76 / 17.62 | 88.89 / 85.96 / 4.48 | 95.28 / 67.83 / 28.78 |
| SI+ER | 93.60 / 90.24 / 5.64 | 89.66 / 86.88 / 4.01 | 92.94 / 81.42 / 13.67 |
| MAS+ER | 91.71 / 90.24 / 5.21 | 90.12 / 88.43 / 2.78 | 83.31 / 76.72 / 10.47 |
| LwF+ER | 83.94 / 93.44 / −1.17 | 88.43 / 87.81 / 1.23 | 89.78 / 82.83 / 10.25 |
| DER++ | 92.39 / 90.39 / 5.29 | 89.20 / 87.50 / 2.78 | 94.72 / 79.19 / 16.72 |
| CN-DPM | 92.49 / 88.35 / 8.18 | 89.04 / 84.26 / 1.54 | 92.17 / 74.17 / 18.97 |

微调学得新域很好但遗忘严重；遗忘低的基线对新域学得较差（3DO 上 LwF+ER 和 DUCT+ER 的 AFR 低于 WiMoT（−1.17 和 −0.87 pp），但学习新域的精度低 9.13 和 14.00 pp）。最佳基线学习新域的精度在 3DO（A-GEM，AFR 14.64 pp）、WiMANS（MAS+ER，AFR 2.78 pp）和 Widar3.0（EWC+ER，AFR 19.25 pp）上分别比 WiMoT 高 3.49、2.62 和 0.67 pp。WiMoT 的遗忘率不超过 0.58 pp，兼顾新域的学习与早期域的保持。

主设定之外（相对每个设定中参与比较的最强基线，三个种子的平均 AFA；见论文 Fig. 4、Table VI–IX 和补充材料 Table S1–S6；除五种子研究的基线使用 9216 维 Mantis 块外，这些设定中的基线均使用 9728 维输入）：

- **域顺序：** 3DO 的全部六种域顺序均领先（+1.13 至 +6.81 pp），Widar3.0 另外三种用户顺序均领先（+9.58 至 +14.52 pp）；WiMANS 六种顺序中三种领先（+0.41 至 +1.85 pp），另外三种有基线更高（0.93 至 1.95 pp）。Widar3.0 房间 2 和 3：+0.45 pp（SI+ER；三个种子中两个领先，处于种子波动范围内）和 +9.92 pp（DER++）。
- **编码器：** WiMoT 相对五个基线（微调、LwF、SI+ER、DER++、EdgeCL）中的最强者：在全部七种编码器（预训练 MantisV2、随机初始化 MantisV2、MOMENT-1-large、TSPulse-r1、Chronos-2、TimesFM 2.5、TimesFM 3.0）下，3DO（+3.77 至 +9.56 pp）和 Widar3.0（+13.70 至 +30.99 pp）上均在每个种子领先；WiMANS 上在两种 MantisV2 编码器下领先（+1.03 和 +1.44 pp），在其余五个模型下最强基线更高（2.67 至 5.04 pp）。
- **少标签：** 后续域仅保留 25% 或 10% 标签时，3DO 领先 +2.40 至 +2.81 pp，Widar3.0 领先 +9.85 至 +10.57 pp；WiMANS 上 AFR 最低（−0.46 和 −0.62 pp），AFA 比 MAS 低 1.23 和 1.44 pp。
- **存储：** 秩 32 的低秩类别矩仅需完整协方差 6.4% 的存储，配对 AFA 变化在 −0.49 至 +1.07 pp 之间（至多损失 0.49 pp）；只存上三角可使存储减半且预测不变。

## 补充分析

对设计的两项分析（论文表 IV、补充材料 S-VIII 与 S-IX）；它们不改变主结果的任何系数。

**分类器消融。** 各分类器均在完整方法的同一适配器和已存储的矩上拟合（AFA，%，均值 ± 样本标准差，种子 42–46；
[`scripts/ablation_readout.py`](scripts/ablation_readout.py)，结果见 [`results/additional_analyses/readout_ablation/`](results/additional_analyses/readout_ablation/)）。

| 分类器 | 3DO | WiMANS | Widar3.0 |
|---|:-:|:-:|:-:|
| **OSCL（WiMoT）** | **93.98 ± 0.15** | **89.57 ± 0.55** | **95.24 ± 0.88** |
| 最近类均值 | 80.89 ± 0.89 | 87.84 ± 0.80 | 67.05 ± 1.28 |
| LDA | 92.06 ± 0.49 | 88.83 ± 1.33 | 84.43 ± 0.94 |
| QDA（无域门控） | 91.00 ± 0.70 | 80.93 ± 1.33 | 91.41 ± 0.50 |
| 线性探针 W | 85.77 ± 0.49 | 85.62 ± 1.10 | 81.94 ± 0.80 |

WiMANS 上 OSCL 与 LDA 的配对差为 +0.74 ± 1.39 pp（OSCL 在三个种子中更高、一个相等、一个更低），不足以区分二者；其余各对中 OSCL 在每个种子上都更高。

**损失权重扫描。** WiMoT 的 AFA 随 SupCon 权重（λ<sub>sc</sub>，后续域）与历史损失权重（λ<sub>h</sub>）的变化，种子 42–44
（[`scripts/ablation_loss_sweep.py`](scripts/ablation_loss_sweep.py)、[`scripts/plot_loss_sweep.py`](scripts/plot_loss_sweep.py)，
结果见 [`results/additional_analyses/loss_weight_sweep/`](results/additional_analyses/loss_weight_sweep/)）。

<p align="center">
  <img src="assets/loss_weight_sweep.png" width="100%" alt="三个数据集上 WiMoT 的 AFA 随 SupCon 权重与历史损失权重的变化">
</p>

权重大于零时（λ<sub>sc</sub> 取 0.1–2，λ<sub>h</sub> 取 0.25–4），AFA 在 3DO 为 93.94–94.12%，WiMANS 为 89.09–89.61%，Widar3.0 为 94.96–95.21%，与论文设定（λ<sub>sc</sub> = 0.5，λ<sub>h</sub> = 1）相差约一个种子标准差以内。
把某个权重置零会使 WiMANS（−0.93 和 −1.03 pp）与 Widar3.0（−1.36 和 −0.91 pp）的 AFA 下降，但 3DO 不然（后续域去掉 SupCon 为 +0.09 pp，去掉历史损失为 −0.08 pp，均在种子标准差内）；
3DO 上去掉历史损失的影响体现在 AFR（0.43 对 0.32 pp），在所有域中去掉 SupCon 则使 AFA 下降 2.17 pp。由于 AdamW 对损失的整体缩放不变，只有比值 λ<sub>sc</sub>/λ<sub>h</sub> 起作用：比值相同的设置给出相同的精度矩阵（最大差 0.00 pp）。

## 方法概要

**矛盾。** 不存储样本的持续学习必须同时满足两项要求：拟合新域（可塑性）并继续识别早期域（稳定性）。冻结表示是稳定的但无法学习；不加约束的微调能学习但会遗忘；更新表示的同时保存类别统计量，则统计量描述的是旧表示，而估计其漂移只是近似。遗忘发生在两个层面：在*表示*层面，更新可能丢弃区分早期类别的方向，任何分类器都无法恢复；在*分类器*层面，存储的记忆或决策规则可能落后于表示，或偏向最新域。

**三个问题，三种机制。** WiMoT 保留一份紧凑的记忆，即冻结特征基 φ 中每个域各类别的样本数、均值与协方差，并为每种失效配置一种机制。

1. *如何适应新域而不使记忆失效？* 首个域之后只训练仿射适配器，把改变表示的同一个仿射映射施加到存储的类别均值与协方差上，记忆保持有效，无需漂移估计。
2. *没有样本时如何学习新域而不遗忘？* SupCon 提供可塑性；由存储矩计算的历史损失在表示层面提供稳定性，它惩罚使早期标签线性可读性下降的更新（这是分类精度的代理，而非保证）。
3. *如何在没有域 ID 的情况下预测并与表示保持一致？* 每次更新后，OSCL 分类器由全部域传输后的矩在均匀的域先验和类别先验下重新拟合，使分类器与表示一致，且不偏向最新域。

**公式。** 首个域用 MantisV2 特征上的语义分支、首域 PCA 捷径和固定随机的幅度分支，以监督对比损失训练后冻结为 512 维基空间 φ。之后每个域只训练仿射适配器（以及投影头和历史损失所用的线性映射 W），保存各类别的矩后丢弃新到达的样本。适配器把基空间映射到分类器所用的表示：

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/equations/eq_adapter_dark.png"><img src="assets/equations/eq_adapter_light.png" width="112" alt="h = A_t phi + b_t"></picture></p>

φ 坐标下保存的矩由标准的仿射矩规则传输到当前表示（CASTER 和 DPCR 也以估计得到的映射使用它）。此处的映射就是模型自身的适配器，因此在固定基和仿射适配器下，存储的矩随实际用于预测的模型变化：

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/equations/eq_transport_dark.png"><img src="assets/equations/eq_transport_light.png" width="301" alt="mu^h = A_t mu^phi + b_t, Sigma^h = A_t Sigma^phi A_t^T"></picture></p>

对线性评分映射 F = [W A_t, W b_t + d]（类中心化并按首域范数缩放），每个已存储的域 s 以闭式给出 F 相对类中心化 one-hot 标签的平方误差，仅用到存储的二阶矩 B_s 和 [φ, 1] 的标签交叉矩 K_s。训练损失为

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/equations/eq_loss_dark.png"><img src="assets/equations/eq_loss_light.png" width="538" alt="L_t = 0.5 L_SC for t=1; L_t = 0.5 L_SC + (1/(t-1)) sum_{s<t} ( tr(F B_s F^T) - 2<F,K_s>_F + 1 ) for t>1"></picture></p>

OSCL 分类器（Yang、Tiomoko 和 Wang，ICLR 2023）用 h 上的高斯密度模型给出域权重 g_s(x)，因此不需要域 ID：

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/equations/eq_readout_dark.png"><img src="assets/equations/eq_readout_light.png" width="272" alt="y(x) = argmax_c sum_{s<=t} g_s(x) p_s(c|x)"></picture></p>

## 安装与冒烟测试

```bash
git clone https://github.com/Newsteinwell/WiMoT.git && cd WiMoT
python -m venv .venv && source .venv/bin/activate
pip install torch==2.8.0            # CUDA 12.8 版本：加 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python scripts/smoke_test.py        # 合成特征、CPU，约 3 分钟；--quick 约 1 分钟
python -m pytest tests
```

论文数字在 Linux + GPU、Python 3.12 下得到；代码也可在 CPU 与 macOS（Python 3.13，Apple 芯片）上运行。冒烟测试用合成特征验证代码路径，不验证准确率。
特征提取与数据准备需额外依赖：`requirements-features.txt`（MantisV2）、`requirements-data.txt`（原始数据读取）。

## 快速上手

输入为 [`wimot.features.extract`](wimot/features/extract.py) 输出的冻结编码器特征：每个窗口 9728 维（9216 维语义坐标加 512 维幅度描述子，float32）。首个域需要超过 256 个窗口，每个域的每个类别至少 2 个窗口。

```python
import numpy as np
from wimot import WiMoT

load = lambda domain, split: np.load(f"features/{domain}_{split}.npz")   # 含 features (N, 9728) 与 labels (N,)
domains = ["d1", "d2", "d3"]                                              # 按到达顺序

model = WiMoT(num_classes=3, seed=42)             # 有 CUDA 则用 CUDA，否则用 CPU
for k, domain in enumerate(domains):
    train = load(domain, "train")
    model.fit_domain(train["features"], train["labels"])   # 首次调用训练基空间，之后为增量更新
    seen = [load(d, "test") for d in domains[: k + 1]]
    print(domain, [round(model.score(t["features"], t["labels"]), 4) for t in seen])

labels = model.predict(load("d3", "test")["features"])     # 预测时不需要域 ID
```

`WiMoT` 封装的就是 `python -m wimot.run` 执行的同一份 `Learner` 代码；`tests/test_api.py` 在 CPU 上验证两者的准确率矩阵逐位一致。

## 复现论文

数据准备、特征提取、各表的任务列表与锁定系数、每条已报告轨迹的逐种子参考结果（`results/reference_results_same_input.json`，1661 条，基线读取 9728 维输入，对应论文各表；`results/reference_results.json`，1652 条，基线使用 9216 维 Mantis 块）、逐阶段测试预测、确定性说明等见 **[docs/REPRODUCE.md](docs/REPRODUCE.md)**（英文）。

## 引用

```bibtex
@article{wimot,
  title   = {{WiMoT}: Replay-Free Continual {Wi-Fi} Sensing via Moment Transport},
  author  = {Yang, Chunchun and Cao, Yang and Chen, Zhitang},
  journal = {IEEE Internet of Things Journal},
  year    = {2026},
  note    = {Under review}
}
```

## 许可与致谢

MIT 许可（见 [LICENSE](LICENSE)）；`duct.py` 与 `cn_dpm.py` 改编自 MIT 许可的代码，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
感谢数据集提供者：3DO（J. Strohmayer 与 M. Kampel）、WiMANS（S. Huang 等）、Widar3.0（Y. Zheng 等），以及提供冻结编码器的 MantisV2
（[`paris-noah/MantisV2`](https://huggingface.co/paris-noah/MantisV2)，[`mantis-tsfm`](https://github.com/vfeofanov/mantis)）的作者。使用数据集和模型权重前请接受各自的使用条款。
