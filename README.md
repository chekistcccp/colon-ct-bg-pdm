# Colon-CT-BG-PDM

**BG-PDM: Boundary-Guarded, Patient-balanced Dual Prototype Memory for Label-Efficient Colorectal CT Tumor Localization**

本仓库用于研究有限肿瘤标注预算下的结直肠 CT 肿瘤异常定位。当前代码实现了数据审计、冻结 2.5D ResNet-18 特征缓存、患者级五折、多个记忆库基线、BG-PDM、验证集阈值校准、患者级 Bootstrap 和单一 JSON 指标汇总。

## 当前状态

- 研究设计与文献边界：[docs/RESEARCH_DESIGN.md](docs/RESEARCH_DESIGN.md)
- Codex 执行规范：[AGENTS.md](AGENTS.md)
- 文献与新颖性记录：[docs/LITERATURE_AND_NOVELTY.md](docs/LITERATURE_AND_NOVELTY.md)
- 环境和运行说明：[docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md)

BG-PDM 的新颖性和性能仍需真实实验验证。仓库代码不代表方法已优于 PatchCore、PaDiM 或监督分割基线。

## 唯一研究主线

1. **边界保护**：使用三维物理距离场，在肿瘤核心和非肿瘤参考区之间设置保护带。
2. **患者均衡原型选择**：控制每例候选特征贡献，再进行代表性压缩。
3. **双原型记忆库**：分别建立非肿瘤参考库和肿瘤核心原型库，联合两类距离产生异常分数。

MSD Colon 的掩膜外区域不等同于健康组织；本项目利用有限肿瘤掩膜构建双库，因此任务定位为低标注/弱监督异常定位，而不是严格 normal-only 的无监督患者筛查。

## 已实现方法

- **knn_single_random**：单非肿瘤参考库随机采样 kNN；
- **patchcore_single_coreset**：PatchCore-style 代表性 coreset；
- **padim_diagonal**：PaDiM-style 空间位置对角高斯分布；
- **dual_unprotected**：无边界保护、无患者均衡的双库；
- **boundary_only**：仅边界保护；
- **patient_balanced_only**：仅患者均衡；
- **bgpdm**：完整方法。

## 运行要求

- 数据：MSD Task10 Colon，有公开标签的 imagesTr/ 和 labelsTr/；
- GPU：建议单张 RTX 4090 48GB；CPU 可运行但很慢；
- 用户自行创建并激活 Python/Conda 环境；
- 首次运行默认使用 torchvision ImageNet 预训练 ResNet-18，可能需要下载权重；
- 不需要手工修改代码中的绝对路径。

安装依赖：

    pip install -r requirements.txt

建议先做数据审计和小规模流程检查：

    BG_PDM_AUDIT_ONLY=1 bash run.sh
    BG_PDM_SMOKE=1 bash run.sh

完整实验：

    bash run.sh

完整实验包含五折患者级交叉验证、10/20/40/100% 标注预算、三次 5% 压力测试、七种方法、患者级置信区间与配对比较，可能需要数小时或更久。Smoke 模式结果不能用于论文结论。

## 输出

- outputs/audit/data_audit.json：数据审计；
- outputs/splits.json：固定患者划分；
- outputs/full/experiment_results.json：全部实验指标、配置、患者级结果、记忆库规模、阈值、运行时间和 BG-PDM 与基线的配对 Bootstrap 差值区间。

实验结果、影像、掩膜、特征缓存、模型权重与日志均被 .gitignore 排除，不会自动同步到 GitHub。

## 指标分辨率说明

Pixel AUROC、Pixel AUPRC 与 AUPRO 在特征网格计算；Dice 同时报告特征网格和分数上采样至输入分辨率后的结果；HD95/NSD 在特征网格上按物理 spacing 修正。上采样不会恢复特征网格未预测出的细节，报告时必须明确这一限制。

## 尚未整合的基线

nnU-Net v2 仍是研究设计中的强监督分割参考，但目前没有被自动集成到 run.sh。需要先实现与外层患者级五折完全一致的训练/验证/测试划分适配，再报告其结果；不能拿不同划分的公开数字替代当前实验。
