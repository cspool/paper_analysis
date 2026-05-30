## Streaming Attention Heads / Retrieval Heads (流式注意力头 / 检索头)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Transformer attention heads 的二分分类：Streaming Heads 仅关注最近 W token + 前 S attention sinks，KV cache 固定 W+S 大小，负责局部语法结构；Retrieval Heads 保留完整 KV cache，从远处检索关键信息，对 recall/RAG 任务至关重要。PruLong/DuoAttention 通过训练学习各 head 的二值分类 mask z_{i,j} ∈ {0,1}。

从算法pipeline角度拆解术语。

```
// 混合 Attention
if z_lh == 1:  // retrieval: full causal attention
    attn = FlashAttention(Q, K_full, V_full)
else:  // streaming: local + sinks
    K_attn = concat([K[:S], K[-W:]])
    attn = FlashAttention(Q, K_attn, V_attn)

// PruLong 训练后离散化：top k% log_α → z=1, 其余 z=0
```

术语一般如何实现？如何使用？

DuoAttention（Xiao et al., 2025）首次提出系统性的 retrieval/streaming head 二分分类方法和基于优化的识别方法。核心发现：retrieval heads 仅占总 head 的少数（MHA: ~25%, GQA: ~50%），但对其做 KV cache 压缩会显著损害长上下文能力；streaming heads 占多数，压缩其 KV cache 几乎无性能影响。识别方法：基于优化的 gate value training（合成 passkey retrieval 数据 + L2 distillation loss + L1 regularization），直接测量输出偏差而非依赖 attention score profiling。消融实验证明该方法优于 attention profiling-based 方法（FastGen, RazorAttention）和 language modeling-based 方法。实现代码：https://github.com/mit-han-lab/duo-attention。

StreamingLLM（Xiao et al., ICLR 2024）首次发现 streaming heads（attention sink）现象。PruLong 在 Llama-3.1-8B-Instruct 上 70% streaming heads 可在 recall 上保持 ≥90% 性能，critical KV footprint 约 30%。不同 task 对 retrieval/streaming 最优比例不同。代码：https://github.com/princeton-pli/PruLong

CompressKV（Lin et al., 2025）进一步细化了 Retrieval Head 的子类型：传统 Retrieval Head 识别标准要求 head 的 top-1 attention 精确落在正确答案 token 上（仅捕捉 copy-paste 行为），而 **Semantic Retrieval Head** 聚合 head 在整个 answer span 上的 attention scores 来评估语义检索能力。公式：SemanticRetrievalScore(h) = Σ_{t} I[y_t ∈ A] Σ_{j∈A} a_{t,j}^h。这种方法能捕捉到对答案周边语义相关 token（如 "eat", "a thing" 围绕 "sandwich"）有高 attention 的 head——这些 head 即使 top-1 attention 不落在正确答案 token 上，仍具有语义检索能力。在 GQA-based LLM 中，使用 Semantic Retrieval Head（而非全部 head 或传统 Retrieval Head）进行 KV cache eviction 的 token 选择，可避免 Streaming Head 主导 eviction 导致仅保留首尾 token 的问题。

涉及论文标题：
- DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads
- Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs
- CompressKV: Semantic Retrieval Heads Know What Tokens are Not Important Before Generation
- Elastic Attention: Test-time Adaptive Sparsity Ratios for Efficient Transformers

**Elastic Attention 的新贡献**：将 head 分类从静态（训练后固定）变为动态（test-time adaptive）。通过 Attention Router（每层轻量 MLP，0.27M 参数）在推理时根据输入 hidden states 实时决定每个 head 使用 FA 还是 SA 模式，而非像 DuoAttention/PruLong 那样训练后 head 分配固定不变。Router 使用 Gumbel-Sigmoid + STE 训练，backbone 冻结，仅训练 Router 参数（12h on 8×A800）。

---
