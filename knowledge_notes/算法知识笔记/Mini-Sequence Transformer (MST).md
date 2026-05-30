## Mini-Sequence Transformer (MST)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Mini-Sequence Transformer (MST, NeurIPS 2024) 是一种 LLM 训练优化方法，通过将输入序列在 MLP 和 LM-Head 层内部划分为 mini-sequences 来降低训练时的峰值中间激活内存。与 Mini-Sequence Inference（推理版本）不同，MST 需要处理 backward pass 中的梯度累加。MST 结合 gradient checkpointing 和 gradient accumulation：forward 时逐 mini-sequence 计算并释放中间激活，backward 时 recompute 激活并累加梯度。MST 在 LLM 训练中实现 12-24× 的序列长度扩展（如 Llama3-8B 从 5K 扩展到 60K on single A100）。

从算法pipeline角度拆解术语：

```
// MST Training (simplified)
输入: X ∈ R^{S×d}
// Forward pass (chunked)
M = S / C  // 划分 mini-sequences
for i = 1 to M:
    // 仅保留 attention output
    A_i = Attention(X_i)        // 使用 FlashAttention，forward 后仅保留 output
    // MLP mini-sequence
    O_i = MLP(A_i)              // forward only，不保留中间激活（类似 gradient checkpointing）
    // 释放 A_i, MLP 中间激活

// Loss computation
loss = CrossEntropy(LM_Head(O_M[-1]), target)

// Backward pass (recompute + gradient accumulation)
for i = M down to 1:
    A_i = recompute_Attention(X_i)
    recompute MLP forward with grad
    accumulate gradients into W_gate, W_up, W_down
```

中间激活内存节省：标准训练 $I_{mem} = S \times I$，MST $I_{mem} = S \times I / M$（M 为 mini-sequence 数量）。MST 也可与 activation recomputation 正交叠加：两者结合时中间内存进一步降至 $I_{mem} = S \times I / (M \times checkpoint\_segments)$。

术语一般如何实现？如何使用？

基于 PyTorch，开源代码 https://github.com/wdlctc/mini-s (MIT license)。实现为 HuggingFace 模型的替换 MLP/LM-Head forward 方法。关键实现细节：(1) chunk size C 通常设为 hidden dimension d，当 S < C 时不拆分（短序列无 overhead）；(2) 与 DeepSpeed-Ulysses 序列并行兼容（attention 用 all-to-all，MLP 用 mini-sequence）；(3) 支持 LoRA 等 PEFT 方法。训练吞吐量几乎无损失（因为长序列下 compute/IO 主导项 SI 和 SV 不变）。

涉及论文标题：
- MOM: Memory-Efficient Offloaded Mini-Sequence Inference for Long Context Language Models
