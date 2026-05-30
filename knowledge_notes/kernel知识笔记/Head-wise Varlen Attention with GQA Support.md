## Head-wise Varlen Attention with GQA Support

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Head-wise Varlen Attention 是支持每个 attention head 使用不同 sequence length（不同 budget）的 sparse attention kernel。由于 top-p 使不同 head 选择不同数量的 token（B1 因 head 而异），传统 padded attention（将所有 head pad 到 max B1）浪费计算；head-wise varlen attention 为每 head 分配恰好 B1(head) 个 token 的计算，消除 padding waste。对 GQA（Group Query Attention），升级为 group-wise varlen：同一 query group 内取各 head 选择 token 的 union，以 group 粒度做 varlen attention——平衡了实现效率与兼容性。

从kernel调度角度拆解术语，给出具体例子。
```
// MHA: head-wise varlen
// 不同 head 选不同 B1(head) token → 直接 varlen attention per head
// FlashInfer load balancing: flatten head dim, 按 B1 per head 分配 compute

// GQA: group-wise varlen (Twilight, Appendix B.2)
// e.g., LLaMA-3.1: 4 Q heads → 1 KV head per group
for each query_group in 1..H_kv:
    I_group = empty_set()
    for each q_head in query_group:
        I_group = I_group ∪ I_head  // union of selected tokens
    B_group = |I_group|
    // Attention: Q_group ∈ R^{4×d} @ K[I_group]^T → group_size × B_group
    // 所有 heads in group 共享 K/V[I_group]

// 方案对比 (Figure 13):
// Padded: 所有 head pad 到 max_budget → 大量浪费
// Head Varlen: 每 head 独立加载 K/V → 重复加载（GQA中多head共享同一KV）
// Group Varlen: union per group → 平衡计算浪费与重复加载
```

术语一般如何实现？如何使用？
基于 FlashInfer 的 varlen attention + load balancing。Flattened paged KV cache layout，通过 per-group 的 B_group 做 scatter-arrange。GQA group union 使同一 group 内共享 KV 的 heads 不产生重复加载。适用于所有需要动态 per-head budget 的 sparse attention 方法。

涉及论文标题：
- Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning


---
