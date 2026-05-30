## Mini-Sequence Inference（Mini-Sequence 推理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Mini-Sequence Inference 是一种 LLM 推理阶段的 MLP 内存优化算法，将 MLP 层的输入序列沿 token 维度划分为多个较小的 "mini-sequences"（每个大小 N ≈ S/M），逐个通过 MLP 计算以降低峰值中间激活内存。其核心原理基于：MLP 层对每个 token 的计算是独立的（无 token 间信息交互），因此可以分批处理而不改变输出结果（数学等价性）。此算法是 Mini-Sequence Transformer (MST, NeurIPS 2024) 在推理场景的适配——MST 的训练版本需要梯度累加来支持 backward pass，而 Mini-Sequence Inference 只需要前向 pass，因此更简单、更高效。

从算法pipeline角度拆解术语：

```
// Mini-Sequence Inference for MLP layers (MOM Algorithm 1)
输入: A ∈ R^{S×d} (attention output, S=sequence_length)
超参: C (mini-sequence size), M = ceil(S/C)

// 非最后 MLP 层: 完整 mini-sequence 处理
if not last_mlp_layer:
    Partition A into {A_i}_{i=1}^M, A_i ∈ R^{B×N×d}, N ≈ C
    O = []  // 输出列表
    for i = 1 to M:
        // SwiGLU MLP:
        // gate = SiLU(A_i @ W_gate^T)       [N, I]
        // up   = A_i @ W_up^T                [N, I]
        // hidden = gate ⊙ up                 [N, I]
        // O_i = hidden @ W_down^T            [N, d]
        O.append(O_i)
        // 释放 I_up_i, I_gate_i, hidden_i 的中间内存
    return concat(O)  // [S, d]
// 最后 MLP 层: 仅处理最后一个 token
else:
    A_last = A[-1, :]        // [1, d]
    O_last = MLP(A_last)     // [1, d]
    logits = LM_Head(O_last) // [1, vocab_size]
    return logits
```

**内存节省分析**：
```
Standard: M_intermediate = S × I, I ≈ 4d ≈ 16384 (Llama-3-8B)
MOM:      M_intermediate = N × I = (S/M) × I, M = S/C

举例 (S=128K, C=8192, d=4096, I=16384, bf16):
Standard: 128K × 16384 × 2B = 4.2 GB per MLP layer
MOM:      8K × 16384 × 2B = 262 MB per mini-sequence (16× reduction)
```

**与 Chunked Prefill 的关键差异**：
- Chunked Prefill：将整个 transformer block（attention + MLP + LM Head）切分为多个 chunk，每个 chunk 串行执行完整 forward → 导致 attention 重复计算和 KV cache 重载
- Mini-Sequence：仅切分 MLP 层，attention 保持完整序列处理 → 无 attention 重复计算，单次 forward pass 完成

**Decode 阶段的化简**：decode 时每步仅 1 个 token，MLP 中间激活 = 1 × I，远非瓶颈。因此 Mini-Sequence 只作用于 prefill 阶段。

术语一般如何实现？如何使用？

HuggingFace Transformers 中实现：(1) 识别所有 MLP 层（通常为模型中的 SwiGLU 模块）；(2) 非最后一层的 MLP 层：将输入分块，循环执行 `SiLU(X_chunk @ W_gate^T) ⊙ (X_chunk @ W_up^T) @ W_down^T`，拼接输出；(3) 最后一层 MLP + LM Head：仅取 X[-1:] 进行投影。代码改动极小，仅修改 MLP 的 forward 方法。

兼容性：(a) 与 FlashAttention 完全兼容（attention 层不变）；(b) 与 GQA/MQA 完全兼容（attention 结构不变）；(c) 与 KV cache offloading 完全兼容（MOM 的核心贡献即是将两者结合）。与 HuggingFace 的 OffloadedCache 直接集成。由于 mini-sequence 尺寸更小，可更好地适配 GPU L2 cache，使 mini-sequence only 模式甚至能略微提升吞吐量（MOM 论文观察）。

涉及论文标题：
- MOM: Memory-Efficient Offloaded Mini-Sequence Inference for Long Context Language Models

---
