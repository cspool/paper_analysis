## HQQ (Half Quadratic Quantization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

HQQ（Half Quadratic Quantization，半二次量化）是一种 data-free 的模型权重量化算法（Badri & Shaji, 2023），无需校准数据即可将模型权重压缩到低比特（4-bit、3-bit、2-bit）。与需要校准数据的 GPTQ、AWQ 不同，HQQ 通过半二次优化直接从权重分布求解量化参数，避免了校准数据依赖带来的部署复杂性。论文选择 HQQ 仅因为其对 Mixtral 模型已有良好测试，且算法选择不影响方法的核心结论。

Mixtral offloading 论文中的 HQQ 配置：
- 4-bit: group size 64, scale group size 256（用于 attention 层）
- 3-bit: group size 64, scale group size 128（用于 expert 层）
- 2-bit: group size 16, scale group size 128（用于 expert 层，实际约 2.6 bits/param 因大量 scale/zero-point overhead）

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

HQQ 对 expert 权重的量化流程：
```
# 对于每个 expert 权重矩阵 W ∈ R^{M×H}:
# Step 1: 按 group size G 分组
#   W 沿行维分成 M/G 个 group，每 group 独立量化

# Step 2: 对每个 group g 的半二次优化
#   min_{W_q, s, z} ||W_g - (W_q - z) * s||²  
#   其中 W_q 为 INT2/3/4 整数权重，s 为 scale，z 为 zero point
#   通过迭代交替优化求解

# Step 3: 存储格式
#   expert_weight_int[expert_id]  # INT3 packed
#   expert_scale[expert_id]       # FP16, per group
#   expert_zero[expert_id]        # FP16, per group

# Step 4: 推理时 dequantize
#   W_fp16 = (W_int.to(fp16) - zero) * scale
#   output = input @ W_fp16  # 或 fused dequant+GEMM
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 开源实现：https://github.com/mobiusml/hqq
- Data-free 特性优势：无需准备校准数据集、无校准分布偏差风险、部署即用
- 与 GPTQ/AWQ 的关键区别：GPTQ 逐列贪心量化需校准数据（128 样本），AWQ 需校准数据确定 per-channel scaling，HQQ 纯优化求解零校准
- 论文作者表示若换用 GPTQ 或 AWQ 结论应类似（因量化选择与 offloading 策略正交）
- 子 1-bit QMoE 在 Mixtral-8x7B 上导致过大的 perplexity 退化，不适用

涉及论文标题：
- Fast Inference of Mixture-of-Experts Language Models with Offloading
- FloE: On-the-Fly MoE Inference on Memory-constrained GPU
