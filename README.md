# Colon-CT-BG-PDM

**BG-PDM: Boundary-Guarded, Patient-balanced Dual Prototype Memory for Label-Efficient Colorectal CT Tumor Localization**

本仓库用于研究有限肿瘤标注预算下的结直肠 CT 异常定位。研究主线围绕记忆库质量，而非设计大型新骨干网络。

## 当前状态

当前版本包含完整研究设计和 Codex/Codex Agent 执行约束；实验代码尚未实现，所有方法性能都属于待验证假设。

- [完整研究设计](docs/RESEARCH_DESIGN.md)
- [文献与新颖性分析](docs/LITERATURE_AND_NOVELTY.md)
- [AGENTS.md](AGENTS.md)

## 研究主线

BG-PDM 由三项机制组成：

1. **边界保护**：使用三维物理距离场，从记忆库候选中排除肿瘤边界保护带。
2. **患者均衡原型选择**：控制单一病例贡献的特征数量，再进行全局代表性压缩。
3. **双原型记忆库**：分别建立非肿瘤参考库与肿瘤核心原型库，联合距离生成异常分数。

主数据集仅为 MSD Task10 Colon。由于掩膜外区域不等于健康组织，且双库使用有限肿瘤标注，本项目定位为低标注/弱监督异常定位，而不是严格无监督患者筛查。

## 固定约束

- 单张 NVIDIA RTX 4090 48GB。
- 仅使用 MSD Task10 Colon。
- 患者/CT 体积级划分，五折交叉验证。
- 主标注预算：10%、20%、40%、100%；5% 为压力测试。
- 用户自行创建并激活 Conda 环境；脚本不创建环境。
- 完整实验计划通过根目录 `run.sh` 启动。
- 原始数据、模型权重、特征缓存、预测图和实验结果不得提交到 GitHub。
- 本地完整指标输出：`outputs/full/experiment_results.json`。

## 数据准备

将 MSD Task10 Colon 数据手动放在 `data/` 下。预期至少包含图像和对应标签 NIfTI 文件（例如 `imagesTr/`、`labelsTr/`）；实际目录名由实现阶段的数据审计脚本确认。

MSD Colon 官方有 126 例训练标签；64 例官方测试标签不公开，因此不得在没有真实参考标签的情况下报告其本地定量指标。

## 计划中的实验

- 单非肿瘤库 kNN
- PatchCore-style memory bank
- PaDiM-style feature distribution
- 无边界保护的双库
- 仅边界保护
- 仅患者均衡
- 完整 BG-PDM
- nnU-Net v2 监督分割参考
- 低标注预算、记忆库消融、污染压力测试和患者级统计分析

## 重要说明

BG-PDM 的新颖性和性能尚未得到验证。不要在论文中声称它必然优于 PatchCore、PaDiM 或 nnU-Net。请以 `docs/RESEARCH_DESIGN.md` 和 `AGENTS.md` 为规范来源。