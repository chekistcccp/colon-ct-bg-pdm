# 文献与新颖性记录：BG-PDM

- 日期：2026-10-18
- 状态：初步针对性检索，不是系统综述；投稿前必须更新。
- 目的：明确哪些已有技术不能被误写成 BG-PDM 的独立创新。

## 1. 核心参考

### 1.1 Medical Segmentation Decathlon
Antonelli M, Reinke A, Bakas S, et al. The Medical Segmentation Decathlon. *Nature Communications*. 2022;13:4128.  
DOI: https://doi.org/10.1038/s41467-022-30695-9

意义：确认 MSD 任务、数据使用边界与官方隐藏测试集限制。Colon 任务为 190 例三维 CT，其中 126 例训练、64 例官方隐藏测试。

### 1.2 PatchCore
Roth K, Pemula L, Zepeda J, Schölkopf B, Brox T, Gehler P. Towards Total Recall in Industrial Anomaly Detection. *CVPR*. 2022:14318–14328.  
DOI: https://doi.org/10.1109/CVPR52688.2022.01392  
论文：https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html  
代码：https://github.com/amazon-research/patchcore-inspection

已覆盖：patch 特征记忆库、代表性 coreset 选择、近邻异常分数。BG-PDM 不能把“记忆库 + kNN + coreset”作为创新。

### 1.3 PaDiM
Defard T, Setkov A, Loesch A, Audigier R. PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization. *ICPR*. 2021.  
论文：https://arxiv.org/abs/2011.08785

已覆盖：位置相关的局部特征分布与 Mahalanobis 距离。必须纳入记忆库/分布式基线比较。

### 1.4 AnomalousPatchCore
Koshil M, Wegener T, Mentrup D, Frintrop S, Wilms C. AnomalousPatchCore: Exploring the Use of Anomalous Samples in Industrial Anomaly Detection. 2024.  
论文：https://arxiv.org/abs/2408.15113

已覆盖：在异常样本可用时，将异常样本用于特征学习，再使用记忆库检测。BG-PDM 的双库思路本身不能宣称首创。

### 1.5 MedIAnomaly
Cai Y, Zhang W, Chen H, Cheng K-T. MedIAnomaly: A comparative study of anomaly detection in medical images. *Medical Image Analysis*. 2025;102:103500.  
DOI: https://doi.org/10.1016/j.media.2025.103500

意义：医疗异常检测系统性比较研究，提醒本项目必须选择医学影像场景相关的强基线，并谨慎解释自然图像预训练特征的迁移效果。

### 1.6 2026 年肺部 CT 多线索正常性建模
Wang H. Codebook-guided multi-cue normality modeling for pulmonary CT anomaly detection. *Frontiers in Medicine*. 2026;13:1927996.  
DOI: https://doi.org/10.3389/fmed.2026.1927996

意义：该工作已将多个正常性线索、离散码本与 CT 异常热图结合。BG-PDM 不能以“多线索融合”或“应用于 CT”本身宣称新颖。本文的任务、数据与机制不同，但需要在论文中直接讨论其差异。

### 1.7 医学异常检测综述
Unsupervised Anomaly Detection in Medical Imaging: A Survey of Methods, Challenges, and Future Directions. 2026.  
https://pmc.ncbi.nlm.nih.gov/articles/PMC13203773/

意义：用于梳理记忆库、特征嵌入、正常性建模与医学异常检测的常见方法族。正式投稿前应核实每项相关方法的原始论文与公开实现。

## 2. 当前可提出、但尚未证实的新颖性假设

本项目不主张发明了新的记忆库范式，而是提出一个面向低标注结肠 CT 的组合研究假设：

1. 基于三维物理距离的边界保护可能降低单一掩膜边缘附近特征混入原型库的风险。
2. 患者均衡候选采样可能减少体积较大、切片较多病例对全局记忆库的过度代表。
3. 双库距离可能结合“偏离非肿瘤参考分布”和“相对接近肿瘤核心特征”两类证据。
4. 上述机制在相同标注预算下可能改善异常定位或降低误检负担。

这些均为待检验假设。只有当消融实验表明边界保护、患者均衡选择和双库分别产生稳定且可重复的收益，才可将其作为方法贡献。若标准 PatchCore、PaDiM 或监督 nnU-Net 已取得相同或更好结果，应收缩研究主张或停止该方案。

## 3. 不能宣称的贡献

- 首次提出记忆库异常检测、双库或 coreset。
- 首次利用异常样本或肿瘤样本辅助记忆库。
- 首次进行医学 CT 正常性建模。
- 掩膜外区域等同于健康组织。
- 合成边界保护带代表真实医生分歧。
- 仅凭 MSD 内部五折证明跨中心或临床泛化。
- 未经配对统计检验就声称显著优于已有方法。

## 4. 投稿前必须补充的文献检查

1. 检索近 24 个月的医学异常检测、低标注分割、记忆库污染和患者均衡原型选择论文。
2. 搜索“medical anomaly detection dual memory”, “patient-balanced prototype memory”, “boundary-aware memory bank”, “colon CT anomaly localization”等组合关键词。
3. 检查相关论文是否已使用带符号距离场排除边界、是否按病例配额构建记忆库、是否在同一病例数据集上使用肿瘤/非肿瘤双库。
4. 确认 PatchCore、PaDiM 及可行医学基线的代码许可、依赖和输入输出定义。
5. 保存检索日期、检索式、筛选标准和 DOI/URL，避免将二手综述误当作原始方法证据。