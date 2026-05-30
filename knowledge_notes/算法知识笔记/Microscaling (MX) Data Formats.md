## Microscaling (MX) Data Formats

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Microscaling (MX) 是由 Open Compute Project (OCP) 在 2023 年发布的一种面向深度学习的块级缩放（block-scaled）数值格式规范（OCP Microscaling Formats V1.0 Specification）。MX 的基本单元是一个大小为 k 的块（默认 k=32），包含 k 个低精度标量元素和一个共享的 8-bit 缩放因子（E8M0 格式）。E8M0 是纯指数格式（8-bit 指数，0-bit 尾数），只能表示 2 的幂次，这使得反量化操作仅需移位运算而非乘法。MX 格式族包括 MXFP8（E4M3 或 E5M2，8-bit 元素）、MXFP6（E3M2 或 E2M3，6-bit 元素）、MXFP4（E2M1，4-bit 元素）和 MXINT8。与传统的 per-tensor 或 per-channel 量化不同，MX 在每个 32 元素块内共享一个 scale，实现更细粒度的量化误差控制。NVIDIA Blackwell GPU 原生支持 MXFP8 和 MXFP4 的 MMA 指令，AMD 和 Intel 也提供软件支持。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
给定一个 FP16 张量 X ∈ R^{L×I}，MXFP 量化流程：
```
# 1. 分块：将 X reshape 为 (L*I/32, 32) 的 block 序列
blocks = X.reshape(-1, 32)
# 2. 逐块量化
for block in blocks:
    s = 2^{floor(log2(max(abs(block)))) - b}  # E8M0 scale, b为格式特定偏置
    Q(block) = round(clip(block/s, -q_max, q_max))
    # 存储：Q(block) 的 32 个元素 + 1 个 E8M0 scale s
```
其中各格式参数：MXFP4 (E2M1): b=1, q_max=6；MXFP6 (E3M2): b=3, q_max=28；MXFP8 (E4M3): b=7, q_max=448。反量化：x = s × Q(x)，s 为 2 的幂次，仅需移位。在 MicroMix 中，同一层的不同通道组 G4/G6/G8 分别应用 MXFP4/MXFP6/MXFP8 的 block-wise 量化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MX 格式被 PyTorch (torchao.MXTensor)、NVIDIA Transformer Engine (v2.14+)、AMD Quark、Intel Neural Compressor、OpenVINO 等框架支持。硬件层面，NVIDIA Blackwell (SM 10.0+) 的第五代 Tensor Core 通过 `am16n8k64` 指令原生支持 MXFP4/MXFP8 的 MMA，block scale 反量化融合在 MMA 指令内部。MX 适用于 LLM 推理量化（如 MicroMix 的 ~5.5-bit 混合精度）、训练中的低精度前向（如 DeepSeek V3 使用 UE8M0 FP8-scaled）、以及需要细粒度量化控制的场景。

涉及论文标题：
- MicroMix Efficient Mixed-Precision Quantization with Microscaling Formats for Large Language Models

---
