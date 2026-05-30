## SwiGLU FFN（SwiGLU 前馈网络）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

SwiGLU FFN 是现代 LLM（Llama, Qwen, GPT-oss 等）中替代传统 ReLU FFN 的激活门控前馈网络结构。与标准两层 FFN（Y = σ(XW_up)W_down）不同，SwiGLU 使用三个投影矩阵和一个门控机制：Y = (SiLU(XW_gate) ⊙ XW_up)W_down，其中 SiLU(x) = x·σ(x) 是 Swish 激活函数，⊙ 为逐元素乘法。中间维度 d_ff 通常设为 8d/3（而非传统 FFN 的 4d）。两个 up-projection 矩阵（W_gate, W_up）的参数在训练中可视为 2×d_ff 的总扩展，但因 gating 机制提供的非线性，在同等参数量下比 ReLU FFN 表现更好。

从算法pipeline角度拆解术语：

```
// SwiGLU FFN 前向计算
输入: h ∈ R^d (attention output)
// Step 1: Gate path — 带 SiLU 激活
x_gate = h @ W_gate^T    // [d] → [d_ff], W_gate ∈ R^{d_ff×d}
g = SiLU(x_gate)          // SiLU(x) = x / (1 + e^{-x})
// Step 2: Up path — 无激活
x_up = h @ W_up^T         // [d] → [d_ff], W_up ∈ R^{d_ff×d}
// Step 3: Element-wise 门控
y_inter = g ⊙ x_up        // [d_ff], gate × up
// Step 4: Down projection
output = y_inter @ W_down^T  // [d_ff] → [d], W_down ∈ R^{d×d_ff}
// 总参数: d × d_ff × 3 (W_gate, W_up, W_down)
// vs ReLU FFN: d × d_ff × 2 (W_up, W_down) + d_ff ≈ 4d
```

**SwiGLU 在 prefill 阶段的内存瓶颈**：在 Llama-3 风格模型中，中间维度 I = 4d（因为 W_gate 和 W_up 各输出 d_ff ≈ 8d/3，但中间激活 I_up 和 I_gate 都约为 2d_ff ≈ 4d 量级）。Prefill 阶段处理完整序列 S 时，峰值中间激活内存 = S × I ≈ S × 4d，远大于 attention 优化后的内存（FlashAttention 中 attention 峰值内存约 S × d）。因此 SwiGLU MLP 成为 prefill 阶段峰值内存的主导因素（MOM 论文的观察）。

术语一般如何实现？如何使用？

本文使用 SwiGLU FFN（遵循 Llama-3 惯例），d_ff ≈ 8d/3（而非 ReLU FFN 的 4d），round 至最近 32 的倍数。SwiGLU 的参数计数计入 time-invariant cost（不随 T 增长），因此优化模型时 N 的 scaling 同时影响 SwiGLU 参数量。与 RMSNorm 和 RoPE 一起构成现代 Llama-style 模型的标准组件。在 GPU 上通常用 cuBLAS GEMM 实现三个矩阵乘法，SiLU 和 element-wise 乘用 CUDA kernel。SwiGLU 的中间激活内存占 LLM prefill 阶段峰值内存的主要部分，是长上下文推理的内存瓶颈（MOM 和 MST 均基于此观察提出优化）。

涉及论文标题：
- Cost-Optimal Grouped-Query Attention for Long-Context LLMs
- MOM: Memory-Efficient Offloaded Mini-Sequence Inference for Long Context Language Models

---
