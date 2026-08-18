## EDSR（Enhanced Deep Residual Networks）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
EDSR 是 CVPRW 2017 的经典单图超分（SISR）模型（Lim et al.）：堆叠去除 BatchNorm 的残差块（常规 16 或 32 块），仅用卷积+ReLU，避免 BN 在小 batch 归一化伪影与对超分任务的负作用，提升训练稳定性与表现；设计上支持多尺度（同一主干共享、仅尾部上采样模块按倍数切换）。SLICE 以 EDSR 作为 4× 上采样基准 SR 模型，全部推理在 Jetson AGX Orin GPU 上用 FP16 执行。论文实测 EDSR 推理延迟是 bicubic 插值的 120.7×——这个巨大计算差距正是 SLICE 选择性推理的核心动机。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 SLICE 管线中 EDSR 只对选中的 patch 批做 forward：
```
# inter 帧：270p(480×270) → 30×17 个 16×16 patch → TopK 选 ~35%≈178 个
#   → unfold 聚合成 (178, 3, 16, 16) batch → 一次 EDSR(FP16) forward
#   → 输出 4× 的 (178, 3, 64, 64) patch → 合并成 1080p 帧
# intra 帧（GOP 开头）：全帧 forward
```
EDSR 的残差学习结构（残差块：Conv→ReLU→Conv + 跳跃连接）与 SLICE 的"patch 级选择性调用"正交：模型参数/权重共享，SLICE 只决定何时何地调用该模型。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现众多（如 EDSR-PyTorch，基于 DIV2K 数据集训练，可下载预训练权重）；SLICE 以 PyTorch FP16 部署于 Jetson AGX Orin GPU。论文未明确说明其使用的具体仓库与权重来源（记为论文未明确说明）。与模型级高效 SR（APE 的 patch early-exit、量化/轻量 SR 模型）互补：模型效率降低单次推理成本，SLICE 降低推理次数。

涉及论文标题：
- SLICE A Selective Local Inference Framework with Codec Exploitation for Accelerating Video Super-Resolution
