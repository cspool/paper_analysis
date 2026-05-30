## MagR (Weight Magnitude Reduction / 权重大幅值衰减)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MagR (Weight Magnitude Reduction) 是 Zhang et al. (2024) 提出的一种 LLM 后训练量化（PTQ）预处理技术。核心思想：在量化前对模型权重做幅值衰减变换，降低异常值（outlier）的幅值，使权重的数值范围更集中，从而减少量化时的 clamping 误差和舍入误差。MagR 通过引入一个可学习的 per-channel scaling vector 来缩放权重矩阵：W' = diag(s) · W，其中 s 的元素 < 1 用于压缩异常值的幅值。缩放变换与 Hessian-guided 优化结合，目标是在最小化输出误差的前提下找到最优缩放向量。缩放后的权重 W' 再送入标准量化器（如 GPTQ）进行量化。推理时，缩放因子可以通过与前一层或后一层的权重提前融合（merge），从而不增加推理开销。在 QWHA 论文中，GPTQ_MagR 作为基础量化方案（即 GPTQ + MagR 预处理），用于所有 baseline 和 QWHA 的模型量化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MagR 在 GPTQ_MagR 量化流程中的使用：

```
# MagR 预处理 + GPTQ 量化流程
# Input: 预训练权重矩阵 W ∈ R^{d_out × d_in}
#        activation X, 校准集

# Step 1: MagR - 学习 per-channel 或 per-group scaling factors
# 目标: min_s ||WX - (s^{-1}·round(s·W))X||^2
for each channel/group:
    s_init = max(|W_channel|) / max_safe_value  # 初始估计
    optimize s to minimize output error:         # 梯度下降/网格搜索
        W_scaled = s * W
        W_q = round(W_scaled)                    # 量化
        W_dq = W_q / s                            # 反量化
        error = ||WX - W_dq @ X||^2
    # s < 1: 压缩异常值

# Step 2: GPTQ 逐列量化 (使用 MagR 缩放后的权重)
# Hessian H = 2XX^T (从校准集累积)
W_q = copy(W_scaled)
for col in 0..d_in-1:
    # 量化第 col 列
    W_q[:, col] = quantize(W_scaled[:, col], scale[col])
    # 补偿剩余列的误差
    error = (W_scaled[:, col] - W_q[:, col]) / H[col, col]
    W_scaled[:, col+1:] -= error * H[col, col+1:]

# Step 3: 推理时融合 MagR scaling
# Option A: 融合到上一层输出投影
# Option B: 融合到本层权重 W_q' = W_q / s
# Both options: 零推理开销

# QWHA 中 GPTQ_MagR 的使用:
W_Q = GPTQ_MagR(W_0, calibration_data=X_calib)
ΔW_Q = W_0 - W_Q  # 用于 QWHA 初始化
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MagR 的关键设计选择：(1) Scaling 粒度：per-channel 或 per-group（与量化 group size 对齐）；(2) 优化方法：网格搜索（brute-force，用于小规模）或梯度下降（用于大规模）；(3) 与 GPTQ 的集成方式：先确定 scaling factors，再对缩放后的权重执行 GPTQ 逐列量化。MagR 的有效性源于：LLM 权重的异常值（outlier）是量化的主要精度瓶颈——这些大值被 clamp 后产生巨大误差。通过 MagR 降低异常值幅值后再量化，clamping 边界能覆盖更大比例的权重。与其他量化预处理技术（如 QuaRot 的随机 Hadamard 旋转、SmoothQuant 的 per-channel scaling）的对比：MagR 专注于"幅值衰减"，而 QuaRot 专注于"incoherence processing"（使权重分布更均匀），两者可互补。在 QWHA 实验中，GPTQ_MagR 在 4-bit 下的 CSQA 准确率（LLaMA-3.1-8B）为 69.11%，比原始 GPTQ 更高。

涉及论文标题：
- QWHA: Quantization-Aware Walsh-Hadamard Adaptation for Parameter-Efficient Fine-Tuning

---
