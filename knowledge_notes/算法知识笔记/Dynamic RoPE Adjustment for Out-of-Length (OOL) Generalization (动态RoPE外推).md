## Dynamic RoPE Adjustment for Out-of-Length (OOL) Generalization (动态RoPE外推)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Dynamic RoPE Adjustment 是 InfiniteHiP 提出的训练无关的长上下文外推策略。核心观察：LLM 的不同层具有不同的 attention pattern——早期层（layer ≤ 5）呈现 dynamic sliding window-like attention（依赖相对位置信息），后期层依赖语义内容。基于此，InfiniteHiP 在不同层和不同计算阶段使用不同的 RoPE position ID 策略：(1) 前 3 层剪枝阶段使用 Chunk-indexed RoPE；(2) 第 4 层起剪枝阶段使用 Relative-style RoPE；(3) BSA 阶段使用 StreamingLLM-style RoPE。

从算法pipeline角度拆解术语：

**三种 RoPE 策略的具体实现**（ApplyRopeQ 和 ApplyRopeK，layer index l）：

```
// Chunk-indexed RoPE (l ≤ 3, 在剪枝阶段)
ApplyRopeQ_l(q) = ApplyRope(q, p[min(i_orig, l_c + n_stream)])
  // i_orig: q 的原始 position
  // 将 position 钳制在 chunk 级粒度，引导滑窗式 mask
ApplyRopeK_lj(k) = ApplyRope(k, p[c_orig])
  // c_orig: k 所在 chunk 的索引
  // 同 chunk 内所有 key 共享同一 position ID

// Relative-style RoPE (l > 3, 在剪枝阶段)
ApplyRopeQ_l(q) = ApplyRope(q, p[n_stream + 1])
  // 对 query 使用统一的相对偏移
ApplyRopeK_lj(k) = ApplyRope(k, p[j-1])
  // j∈{1,2}: SelectRep 中左分支(j=1)或右分支(j=2)
  // 两分支获得不同 position offset，实现层次化相对编码

// StreamingLLM-style RoPE (BSA 阶段，所有层)
  // 选中的 key（含 sink+streaming）按原始顺序排列
  // 最近 token 获得与当前 query 相同的 position ID
  // 等效于在原始 RoPE 空间中重新索引选中的 token
```

**消融实验结果（∞Bench En.MC, Llama 3.1 8B, T=128K）**：
- Chunk-indexed in pruning + StreamingLLM in BSA: 67.69%
- Relative in pruning + StreamingLLM in BSA: 70.31%（best for BSA）
- 混合（前3层 Chunk-indexed + 后续 Relative）+ StreamingLLM BSA: 74.23%（best overall）

术语一般如何实现？如何使用？

实现关键：(1) 在 Triton kernel 中根据 layer index l 选择 ApplyRopeQ/ApplyRopeK 的分支；(2) 预计算多组 cos/sin 向量以支持不同 position offset（增加额外 memory read overhead，使 prefill 慢约 1.6×）；(3) 可动态开关——当不需要 OOL generalization 时（如上下文在预训练长度内），可禁用动态 RoPE 以消除 overhead。Chunk-indexed RoPE 仅在 context pruning 的前 3 层使用，不要在全部层使用（全层 Chunk-indexed 导致显著性能退化）。

涉及论文标题：
- InfiniteHiP: Extending Language Model Context Up to 3 Million Tokens on a Single GPU
