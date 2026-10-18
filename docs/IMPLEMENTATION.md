# 实现与运行说明

## 1. 运行环境

建议使用 Python 3.10 或 3.11、PyTorch 与匹配的 CUDA 版本。用户自行创建并激活 Conda 环境；仓库脚本不会创建、删除或自动切换环境。

安装依赖：

```bash
pip install -r requirements.txt
```

第一次运行默认使用 torchvision 的 ImageNet 预训练 ResNet-18，若本机没有缓存权重，torchvision 会尝试下载权重。若运行环境不能联网，可先在可联网机器准备 torchvision 权重缓存，或在 configs/default.yaml 中把 features.pretrained 改为 false（这会改变实验设置，必须在实验记录中说明）。

## 2. 数据目录

将 MSD Task10 Colon 解压到以下任一常见布局：

```text
data/
└── Task10_Colon/
    ├── imagesTr/
    │   ├── colon_001_0000.nii.gz
    │   └── ...
    ├── labelsTr/
    │   ├── colon_001.nii.gz
    │   └── ...
    └── dataset.json
```

也支持 data/ 目录本身直接包含 imagesTr/ 和 labelsTr/。图像和掩膜需一一对应，且 affine、方向和 shape 必须一致。程序不会静默重采样不匹配的标签。

## 3. 推荐运行顺序

先激活你手动创建的环境，然后执行：

```bash
bash run.sh
```

建议第一次只做小规模流程检查：

```bash
BG_PDM_SMOKE=1 bash run.sh
```

只审计数据而不提取特征或运行模型：

```bash
BG_PDM_AUDIT_ONLY=1 bash run.sh
```

如果需要重新提取特征，单独运行：

```PYTHONPATH=src python -m bgpdm.experiment --config configs/default.yaml --force-features```

## 4. 特征缓存与计算量

特征按病例缓存于 cache/features/。默认输入大小为 224，使用相邻三张切片构造 2.5D 输入；冻结 ResNet-18 layer2/layer3 多尺度特征并缓存为 float16。缓存会占用数 GB 磁盘空间。改变特征提取器、输入尺寸或强度窗时，应递增 features.cache_version 或清理旧缓存。

完整实验包括患者级五折、10/20/40/100% 标注预算、三次 5% 压力测试、七种记忆库/分布方法以及患者级 Bootstrap 置信区间，可能需要数小时或更久。Smoke 模式用于验证代码路径，不用于论文报告。

## 5. 输出

- 数据审计：outputs/audit/data_audit.json 和 data_audit.md
- 固定患者划分：outputs/splits.json
- 全部实验指标：outputs/full/experiment_results.json

实验结果 JSON 包含每折、每标注预算、每种方法的汇总指标、患者级指标、记忆库规模、阈值、运行时间和 BG-PDM 与基线的配对 Bootstrap 差值区间。图像和热图不是 JSON 的一部分。

Pixel AUROC、Pixel AUPRC 与 AUPRO 在特征网格计算；Dice 同时报告特征网格和上采样到输入分辨率后的结果；HD95/NSD 在特征网格上按物理 spacing 修正。报告时应明确这一分辨率限制，不要将上采样误称为恢复了模型未预测出的细节。

## 6. 基线说明与当前限制

当前代码已实现：
- 单非肿瘤参考库随机采样 kNN；
- PatchCore-style 代表性 coreset 记忆库；
- PaDiM-style 按空间位置的对角高斯分布；
- 无边界保护双库；
- 仅边界保护；
- 仅患者均衡；
- 完整 BG-PDM。

nnU-Net v2 是研究设计中的强监督分割参考，但其训练耗时较大，且需要为每个外层折配置严格一致的 train/validation/test 病例划分。当前自动运行脚本尚未将 nnU-Net v2 训练整合进 run.sh；在完成独立的数据划分适配和输出评估之前，不得把 nnU-Net 结果标记为已完成。它应作为下一阶段独立基线任务，而不能用不同划分的公开结果替代。

## 7. 结果解释

本方法使用有肿瘤标注病例中的掩膜外组织构建非肿瘤参考库，因此属于低标注/弱监督肿瘤定位，不是严格 normal-only 的无监督异常检测。MSD Colon 的官方隐藏测试集在没有真实标签的情况下不能计算本地定量指标。所有性能结论必须来自真实运行结果。
