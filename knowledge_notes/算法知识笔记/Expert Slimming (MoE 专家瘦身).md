## Expert Slimming (MoE 专家瘦身)

术语解释
Expert Slimming 是对 MoE 模型中单个 expert 内部权重进行压缩的技术，通过应用压缩变换 f(W) 减少每个 expert 的冗余，创建轻量化的"slim experts"。由 He et al. (2025) 在 MoE 统一压缩框架中提出，与 Expert Trimming（移除结构化模块）互补。

术语是什么？
Expert Slimming 专注于单个 expert 的权重变换，不改变 expert 数量。主要方法：
1. **Pruning（剪枝）**：f(W) = M ⊙ W，通过 binary mask M 置零不重要权重。分为 unstructured（任意位置，效果最好但硬件不友好）、semi-structured（如 2:4，每 4 个值中保留 2 个，硬件友好但性能损失大）、structured（整行/列移除）
2. **Quantization（量化）**：f(W) = Quant(W)，将 FP16/FP32 权重转换为 INT4/INT8 等低精度表示，减少内存但保持 FLOPs

论文对比发现：量化优于剪枝——unstructured pruning (50%) 虽能保持 >95% 性能但无法硬件加速，semi-structured (2:4) 硬件友好但性能损失显著。4-bit 量化 (AWQ) 实现 >98% 性能 + 5.08× speedup (Mixtral-8×7B) 且硬件可加速。

从算法pipeline角度拆解术语：
```
=== 统一框架中的 Expert Slimming ===
# 压缩后的 MoE 输出
y = Σ_{i∈T'} G_i · E_i(x | f(W_i))
# T': Expert Trimming 保留的 expert 子集
# f(W_i): Expert Slimming 压缩后的权重

# Pruning
W_i_pruned = M_i ⊙ W_i,   M_i ∈ {0,1}^{d×d_h}  # 二值 mask
半结构化 2:4: 每 4 个连续元素中最多 2 个非零

# Quantization (AWQ)
W_i_quant = AWQ_quantize(W_i, bits=4, group_size=128)
# 推理时: y = x @ W_i_quant, 使用 INT4 GEMM kernel
```

Shared Expert 不可压缩性发现：DeepSeek-MoE-16B 残差 MoE（2 shared + 64 routed）中，pruning 不包含 shared experts 时：Wanda +3.6%, SparseGPT +1.5% 平均精度提升。说明 shared experts 承载更关键的通用知识。

术语一般如何实现？如何使用？
- Pruning: 用 Wanda（activation-based magnitude）或 SparseGPT（Hessian-aware）在 calibration data（128 C4 samples）上一次性剪枝
- Quantization: 用 GPTQ/AWQ 进行 weight-only 4-bit 量化。GPTQ default: 128 Alpaca samples；AWQ default: 128 Pile samples。Group size: 128 (Mixtral) 或 64 (DeepSeek)
- 最佳组合：先 Expert Slimming 后 Expert Trimming ("S+T" order)，量化保持性能 + Layer/Block Drop 增加效率
- 综合效果：AWQ 4-bit + Block Drop B5/32, Mixtral-8×7B: 6.05× speedup, 20GB memory, 92.4% performance

涉及论文标题：
- Demystifying the Compression of Mixture-of-Experts Through a Unified Framework
