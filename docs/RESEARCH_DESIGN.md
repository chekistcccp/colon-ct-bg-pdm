# BG-PDM：基于边界保护与患者均衡双原型记忆库的低标注结直肠 CT 肿瘤异常定位

- 文档版本：v1.0
- 日期：2026-10-18
- 项目状态：核心代码已实现；方法收益尚未经过真实数据实验验证
- 目标期刊：医学图像分析/医学影像计算相关二区期刊（具体分区以投稿年度与期刊目录为准）
- 数据集：仅使用 Medical Segmentation Decathlon（MSD）Task10 Colon
- 硬件约束：单张 NVIDIA RTX 4090 48GB
- 核心方法代号：BG-PDM（Boundary-Guarded, Patient-balanced Dual Prototype Memory）

> 本文件锁定当前项目主线。核心实验代码已写入 src/bgpdm/，可通过 run.sh 启动；nnU-Net v2 作为可选强监督参考，需单独安装并设置环境变量。BG-PDM 是候选研究方法，不代表其新颖性或性能已经得到验证。论文中的创新性主张必须由最新文献比较和严格实验结果支持。

## 1. 执行摘要

本项目研究在有限肿瘤标注预算下，如何使用记忆库方法定位结直肠 CT 中的肿瘤区域。项目不以开发大型新骨干网络为目标，而是围绕记忆库质量设计并验证三项机制：

1. **边界保护（Boundary Guard）**：利用训练掩膜的三维物理距离场，将肿瘤核心、非肿瘤参考区和边界保护带分开，降低单一标注边界误差对原型库的污染。
2. **患者均衡原型选择（Patient-balanced Prototype Selection）**：限制单个病例贡献的候选特征数量，降低大体积病例或切片多的病例对记忆库的支配，并在全局候选集上进行代表性压缩。
3. **双原型记忆库（Dual Prototype Memory）**：分别构建非肿瘤参考库和肿瘤核心原型库，联合使用特征到非肿瘤库的距离与双库相对距离生成异常分数。

核心研究问题是：在相同标注预算、特征编码器和计算预算下，BG-PDM 是否比标准 PatchCore、PaDiM、单库最近邻、普通双库以及同等标注预算下的监督分割模型，更有效地定位肿瘤，并在低标注条件下保持竞争力。

**重要任务定义**：MSD Colon 所有病例均来自原发性结肠癌队列，掩膜外区域只代表“未被该肿瘤掩膜标注的区域”，不代表健康人或经确认的正常组织。BG-PDM 主实验利用有限的肿瘤掩膜构建双库，因此属于**低标注/弱监督异常定位**，不能宣称为严格的 normal-only、无监督患者筛查或正常人与患者分类。

## 2. 研究目标与范围

### 2.1 主要目标

在仅使用 MSD Task10 Colon、单张 RTX 4090 48GB、不增加人工勾画的前提下，完成一套可复现的低标注结直肠 CT 肿瘤异常定位研究，验证记忆库构建质量是否影响像素级和病灶级定位效果。

### 2.2 研究问题

- RQ1：在不同标注预算下，记忆库异常定位与监督式分割的性能差距如何变化？
- RQ2：边界保护是否减少训练标签边界不确定性对记忆库的污染？
- RQ3：患者均衡候选采样与原型选择是否优于不区分病例来源的全局采样？
- RQ4：双库相对距离是否比单一非肿瘤库的最近邻距离更能区分肿瘤与非肿瘤区域？
- RQ5：方法收益能否在五折患者级验证中稳定复现，并且不依赖单一随机划分或少数病例？

### 2.3 明确不做的事情

- 不使用 MSD 以外的新数据集作为主实验数据。
- 不增加人工勾画、多医生重复标注或额外临床标签。
- 不把患者级癌症筛查、良恶性分类、复发预测或生存预测作为研究目标。
- 不把 MC Dropout、扩散模型、VLM、Transformer 大模型或复杂生成网络加入当前主线。
- 不把边界保护、双库、原型压缩中的任何一个模块单独宣称为首创。
- 不使用官方测试集不可公开的标签进行本地指标计算。
- 不根据测试折表现选择阈值、特征层、扰动宽度、融合权重或后处理参数。

## 3. 数据集与数据边界

### 3.1 MSD Task10 Colon

MSD Colon 是门静脉期 CT 的三维结肠癌原发灶分割数据集。官方描述为 190 例：126 例训练病例和 64 例隐藏测试病例。官方隐藏测试标签不可直接用于本地定量评价。

参考：
- Antonelli M, Reinke A, Bakas S, et al. The Medical Segmentation Decathlon. *Nature Communications*. 2022;13:4128. https://doi.org/10.1038/s41467-022-30695-9
- 官方网站：https://medicaldecathlon.com/

### 3.2 主实验可用数据

- 使用有公开标签的 126 例进行患者级五折交叉验证。
- 官方 64 例测试病例如无可访问的参考标签，不计算本地 Dice、AUPRC、AUPRO 或错误检测指标。
- 如未来通过官方挑战提交获得聚合结果，只作为补充外部评估，不替代本地可复现实验。

### 3.3 患者级数据划分

1. 先按患者/体积建立固定的五折划分。
2. 每折约 20% 病例作为外层测试折；其余病例作为开发集。
3. 在开发集内划分验证集（建议约 10% 开发病例）；验证集用于阈值、分数尺度、超参数和模型选择。
4. 标注预算只从当前折的训练子集中抽取，不能从验证或测试病例中抽取。
5. 对同一病例的全部切片、派生图像块、增强版本和特征都必须留在同一数据分区。
6. 将患者 ID 划分保存为可复现的 JSON/CSV 元数据文件；不要把患者影像或掩膜提交到 GitHub。

### 3.4 标注预算

主预算：10%、20%、40%、100%。5% 作为极低标注压力测试。每个比例表示当前外层折训练子集中的**病例比例**，不是切片比例。

- 5% 数据量极少，必须至少使用 3 个随机病例子集重复抽样，并单独报告方差。
- 10%、20%、40%、100% 使用固定的可复现子集；关键低预算结果建议再用 3 个子集随机种子做敏感性验证。
- 每个方法在相同外层折、相同标注子集上比较。
- 不得先切片再随机划分，否则相邻切片泄漏会造成虚高结果。

## 4. 方法概述：BG-PDM

### 4.1 特征编码器

主实现采用冻结的 ImageNet 预训练 ResNet-18，以控制训练成本并让研究焦点集中在记忆库。对中心切片及其前后相邻切片构造三通道 2.5D 输入：

\[
X_z = [I_{z-1}, I_z, I_{z+1}]
\]

体积首尾切片使用边界复制或明确记录的 padding 规则。图像方向、切片顺序和掩膜空间信息必须先校验一致。

从 ResNet-18 的 layer2 与 layer3 提取多尺度特征；将 layer3 特征插值到 layer2 的空间分辨率后拼接，再进行通道归一化。主实验不微调骨干，以免把训练策略差异混入记忆库机制的评价。若后续测试医学域自监督特征，只作为独立扩展，不替换主线，除非用户明确批准。

### 4.2 图像预处理

- 使用 NIfTI 的物理空间信息检查图像和掩膜是否对齐。
- 统一方向；图像重采样使用连续插值，掩膜使用最近邻插值。
- 初始 CT 强度窗可使用 [-200, 300] HU 并线性归一化；同时保留基于训练集统计的稳健归一化作为预注册消融。
- 物理距离变换必须使用毫米间距，而不是未经校正的体素单位。
- 可使用基于 HU 的人体区域掩膜排除患者外空气，但不能利用真实肿瘤掩膜裁剪测试图像。
- 对原始数据、预处理后体积和掩膜做尺寸、方向、spacing、值域、空掩膜和连通分量审计。

### 4.3 边界保护

对训练掩膜 \(M_i\) 计算三维带符号距离场 \(d_i(x)\)，约定肿瘤内部为正、外部为负：

\[
d_i(x)=
\begin{cases}
+\operatorname{dist}(x,\partial M_i), & x \in M_i\\
-\operatorname{dist}(x,\partial M_i), & x \notin M_i
\end{cases}
\]

设边界保护宽度为 \(r\) mm：

- 肿瘤核心候选：\(d_i(x)>r\)；
- 非肿瘤参考候选：\(d_i(x)<-r\)；
- 保护带：\(|d_i(x)|\leq r\)，不进入任何记忆库。

默认先测试 \(r=2\) mm；消融比较 \(r\in\{0,1,2,3\}\) mm。若体素 spacing、肿瘤尺寸或重采样设置不适合这些数值，应在查看数据统计后调整，并记录原因。

该操作只能减少几何边界附近的潜在标签污染；它不能证明这些位置具有真实的临床标注歧义，也不能替代多医生标注。

### 4.4 候选特征采样与患者均衡

对每个训练病例独立构建候选特征集：

- 非肿瘤参考候选与肿瘤核心候选分别采样。
- 在切片和空间网格上分层采样，避免大量相邻背景特征占据候选集。
- 初始每例上限：非肿瘤候选 2,000 个、肿瘤候选 1,000 个；这是工程起点，需根据显存、特征数量和病例数审查。
- 全局原型选择前，先限制每例候选贡献，确保病例数量与体积大小不会直接决定其原型份额。
- 全局压缩采用可复现的 k-center/greedy coreset 或明确实现的同类代表性选择。随机采样作为必要对照。
- 默认最终记忆库容量：非肿瘤库 5,000 个原型、肿瘤库 2,000 个原型；在可行时比较更小/更大容量（如 1k/2k、5k/2k、10k/5k）。
- 如果低标注预算下可用候选少于容量，使用所有合格候选，不重复复制原型填充容量。

候选数量和原型选择只使用当前训练折的数据。验证和测试特征不能参与记忆库构建或 coreset 选择。

### 4.5 双原型记忆库

构建两个库：

- \(\mathcal M_N\)：非肿瘤参考记忆库。它只表示当前标注方案中掩膜外的参考区域，不代表健康患者组织。
- \(\mathcal M_T\)：肿瘤核心原型库。它来自掩膜内部、且远离边界保护带的训练特征。

对查询特征 \(f(x)\)，使用 L2 归一化特征与欧氏距离，计算到两个库的 k 近邻平均距离（默认 \(k=3\)）：

\[
d_N(x)=\operatorname{MeanKNN}(f(x),\mathcal M_N),\quad
d_T(x)=\operatorname{MeanKNN}(f(x),\mathcal M_T)
\]

用仅由训练数据拟合的稳健统计量（如中位数与 IQR）对距离进行尺度归一化，避免不同库的距离尺度不一致。初始异常分数：

\[
S(x)=\alpha\,\widetilde d_N(x)
+(1-\alpha)\,\widetilde{\log\frac{d_N(x)+\epsilon}{d_T(x)+\epsilon}}
\]

其中 \(\epsilon\) 防止数值不稳定；默认 \(\alpha=0.5\)，主结果使用预设值，验证集可用于有限的敏感性分析。测试集不得用于选择 \(\alpha\)、阈值或归一化参数。

直觉上，偏离非肿瘤参考库且相对更接近肿瘤原型的特征应获得更高分数。但这个分数不是天然概率；在没有额外校准时，只称为 anomaly score，不称为肿瘤概率。

### 4.6 三维跨层处理

特征输入已经包含相邻切片信息。为获得三维定位图，将每层分数映射回相应空间后重建三维体积。主实验先报告不做后处理的结果。

可选的轻量跨层平滑/连通域处理只作为单独消融：
- 仅允许基于图像和预测分数的后处理；
- 不得用真实测试掩膜确定连通域阈值；
- 不得默认删除所有小连通区域；
- 需报告后处理前后指标，并对所有基线使用同等后处理规则。

## 5. 对照方法与实验矩阵

### 5.1 必需基线

1. **Nearest-neighbour single-bank**：仅使用非肿瘤参考库的 kNN 距离。
2. **PatchCore-style memory bank**：使用代表性 patch 特征与 coreset 选择。
3. **PaDiM-style feature distribution**：按位置估计特征分布并使用 Mahalanobis 距离；如高维协方差不稳定，使用明确记录的降维或正则化。
4. **Unconstrained dual bank**：双库但不做边界保护与患者均衡，用于隔离机制贡献。
5. **Boundary-guard only**：仅加边界保护。
6. **Patient-balanced only**：仅做患者均衡候选采样/选择。
7. **BG-PDM full**：完整方法。
8. **nnU-Net v2 supervised segmentation**：至少在 100% 标注预算下作为强分割参考；低标注预算下尽可能在同样子集上训练，以保证公平比较。

PatchCore 原论文：Roth K, Pemula L, Zepeda J, et al. Towards Total Recall in Industrial Anomaly Detection. CVPR 2022. https://doi.org/10.1109/CVPR52688.2022.01392

PaDiM 原论文：Defard T, Setkov A, Loesch A, Audigier R. PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization. ICPR 2021. https://arxiv.org/abs/2011.08785

### 5.2 标注预算实验

- 主预算：10%、20%、40%、100%。
- 5% 为压力测试，使用至少 3 次独立患者子集抽样。
- 主要表格按标注预算列出各方法的定位性能、错误率与资源成本。
- 绘制标注比例—性能曲线；曲线下面积可作为补充汇总，不替代各预算点结果。

### 5.3 消融实验

| ID | 设置 | 验证内容 |
|---|---|---|
| A1 | 无边界保护 vs. \(r=1,2,3\) mm | 边界候选污染的影响 |
| A2 | 全局采样 vs. 患者均衡候选采样 | 病例份额失衡的影响 |
| A3 | 单库 vs. 双库 | 肿瘤原型是否提供增益 |
| A4 | 随机采样 vs. coreset | 原型代表性与计算成本 |
| A5 | 不同库容量 | 性能—内存—速度权衡 |
| A6 | 不同 kNN 数量 | 距离估计稳定性 |
| A7 | 不做 vs. 跨层后处理 | 三维后处理的真实贡献 |
| A8 | 特征层选择 | layer2、layer3、多层拼接的贡献 |

### 5.4 记忆库污染压力测试

人为向非肿瘤参考候选中混入一定比例的肿瘤核心特征，比较单库、无边界保护双库和 BG-PDM 的性能下降幅度。

建议污染比例为 1%、5%、10%（只作为可控的合成压力测试）。应明确说明它不是对真实数据污染率的估计，也不能代替真实标注误差研究。

## 6. 指标与统计分析

### 6.1 定位指标

- Pixel AUROC；
- Pixel AUPRC（类别不平衡时重点报告）；
- AUPRO/PRO；
- 采用验证集阈值后计算 Dice、NSD、HD95；
- 病灶级敏感度；
- 每例假阳性连通区域数、假阳性体积；
- 若有多病灶病例，补充按病灶匹配的召回率。

像素级 AUROC 可能被大量背景像素抬高，因此不能单独作为主要终点。建议将 AUPRC 与病灶级敏感度列为共同主要结果。

### 6.2 资源指标

- 原型总数、磁盘/内存占用；
- 特征缓存体积；
- 单例推理时间；
- 记忆库构建时间；
- 峰值 GPU 显存；
- 总实验耗时。

### 6.3 统计方案

- 患者是统计独立单位，不能将像素当作独立样本做显著性检验。
- 报告每例指标的均值、标准差、中位数及四分位数。
- 主要方法比较使用患者级配对 Bootstrap（建议 2,000 次重采样）给出 95% 置信区间。
- 多方法比较应控制多重检验。
- 固定随机种子并记录每折划分、标注子集种子和训练配置。
- 仅在验证集上选择阈值、\(\alpha\)、后处理参数和归一化参数。
- 结果若未显著优于强基线，应如实报告；不得只挑选最有利的预算、折或指标。

## 7. 成功标准与停止条件

### 继续推进的条件

- BG-PDM 在多个患者级测试折上稳定优于单库和无约束双库基线。
- 改善不仅体现在 AUROC，也体现在 AUPRC、病灶级敏感度或假阳性负担上。
- 边界保护和患者均衡至少各自带来可重复的增益，或者存在清晰的性能—资源优势。
- 在低标注预算下相对监督分割基线显示出有意义的竞争力。
- 增益不依赖单个病例、单个随机种子或测试集调参。

### 停止/重新设计的条件

- 双库没有稳定优于单库；
- 边界保护对性能无益或明显损害召回率；
- 患者均衡选择与普通 coreset 的差异不稳定；
- 只有 Pixel AUROC 上升，而 AUPRC、病灶级召回和假阳性负担没有改善；
- 结果与现有 PatchCore/PaDiM 基线相当，且没有计算成本、标注效率或鲁棒性优势。

## 8. 计算与软件实施约束

- 单 GPU：RTX 4090 48GB；先运行单折小规模试验并记录实际峰值显存。
- 主特征编码器冻结；优先缓存特征，避免每次消融重复提取特征。
- 大规模近邻距离计算必须分块，避免一次性构建超大距离矩阵。
- 优先使用 FAISS（若环境兼容）或分块 PyTorch kNN；两种实现需做距离数值一致性检查。
- 实验应可通过根目录单一 `run.sh` 顺序运行；支持断点续跑、日志和 checkpoint。
- 用户自行创建并激活 Conda 环境；脚本不得创建、删除或自动切换 Conda/venv。
- 原始数据由用户手动放入 `data/`；预处理程序可识别指定目录中的 MSD Task10 Colon 结构并生成审计报告。
- 本阶段仅完成设计文件；实现阶段先完成数据审计和基线，不要一开始就写全部复杂模块。

## 9. 二区发表定位与创新性边界

本项目的可辩护贡献应表述为：

1. 提出一个针对有限肿瘤标注场景的边界保护与患者均衡双原型记忆库构建流程；
2. 系统分析边界保护、病例份额控制和双库距离在结直肠 CT 肿瘤定位中的独立贡献；
3. 在患者级低标注预算下，比较异常定位与监督分割的性能、错误类型和计算成本。

不要声称：
- 首次提出记忆库异常检测、双库或 coreset；
- 首次处理标注污染、空间上下文或医学 CT 异常定位；
- 仅凭 MSD 内部五折就证明临床泛化；
- 掩膜外区域等同于健康组织；
- 合成边界扰动等同于真实医生分歧；
- 未经正式统计检验就声称显著优于已有方法。

### 关键相关文献

1. Antonelli M, Reinke A, Bakas S, et al. The Medical Segmentation Decathlon. *Nature Communications*. 2022;13:4128. https://doi.org/10.1038/s41467-022-30695-9
2. Roth K, Pemula L, Zepeda J, Schölkopf B, Brox T, Gehler P. Towards Total Recall in Industrial Anomaly Detection. *CVPR*. 2022:14318–14328. https://doi.org/10.1109/CVPR52688.2022.01392
3. Defard T, Setkov A, Loesch A, Audigier R. PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization. *ICPR*. 2021. https://arxiv.org/abs/2011.08785
4. Koshil M, Wegener T, Mentrup D, Frintrop S, Wilms C. AnomalousPatchCore: Exploring the Use of Anomalous Samples in Industrial Anomaly Detection. 2024. https://arxiv.org/abs/2408.15113
5. Cai Y, Zhang W, Chen H, Cheng K-T. MedIAnomaly: A comparative study of anomaly detection in medical images. *Medical Image Analysis*. 2025;102:103500. https://doi.org/10.1016/j.media.2025.103500
6. Wang H. Codebook-guided multi-cue normality modeling for pulmonary CT anomaly detection. *Frontiers in Medicine*. 2026;13:1927996. https://doi.org/10.3389/fmed.2026.1927996
7. Unsupervised Anomaly Detection in Medical Imaging: A Survey of Methods, Challenges, and Future Directions. 2026. https://pmc.ncbi.nlm.nih.gov/articles/PMC13203773/

文献列表是当前设计阶段的核心参考，而非穷尽式系统综述。正式投稿前应补充最近 12–24 个月的相关工作、检查代码可用性与实验设置，并重新判断创新性。

## 10. 交付与复现实验

预期代码仓库结构：

```text
colon-ct-bg-pdm/
├── README.md
├── AGENTS.md
├── run.sh
├── configs/
│   ├── default.yaml
│   └── splits/
├── data/                    # 本地数据，不提交
├── src/
│   ├── data/
│   ├── features/
│   ├── memory/
│   ├── scoring/
│   ├── evaluation/
│   └── utils/
├── scripts/
├── tests/
├── docs/
│   ├── RESEARCH_DESIGN.md
│   └── LITERATURE_AND_NOVELTY.md
├── checkpoints/             # 不提交
├── cache/                   # 不提交
└── outputs/                 # 不提交
```

结果管理：
- 完整实验指标汇总到本地 `outputs/full/experiment_results.json`。
- JSON 只存数值、文本摘要、配置、随机种子、划分 ID、运行时间与指标，不内嵌图像。
- 图表与预测图保存在本地 `outputs/figures/`，不提交。
- 数据、掩膜、特征缓存、模型权重、checkpoint、日志和实验结果不得同步到 GitHub。
- 代码仓库仅保存方法实现、配置模板、数据审计摘要（不含患者信息）、测试与文档。

## 11. 参考资源与获取

- MSD 官方页面：https://medicaldecathlon.com/
- MSD 原始论文 DOI：https://doi.org/10.1038/s41467-022-30695-9
- PatchCore 论文与代码：https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html ，https://github.com/amazon-research/patchcore-inspection
- PaDiM 论文：https://arxiv.org/abs/2011.08785
- MedIAnomaly 综述/比较研究：https://doi.org/10.1016/j.media.2025.103500
- 2026 肺部 CT 多线索正常性建模：https://doi.org/10.3389/fmed.2026.1927996