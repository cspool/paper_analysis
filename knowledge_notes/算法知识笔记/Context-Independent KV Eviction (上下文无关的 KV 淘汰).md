## Context-Independent KV Eviction (上下文无关的 KV 淘汰)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Context-Independent KV Eviction 是 KVzip 支持的一种部署时零压缩开销的模式。与 context-dependent 模式（每个新 context 都需执行评分+淘汰，但有 ~2x prefill 开销）不同，context-independent 模式在部署前预计算 static head-level importance scores，推理时直接按 head 重要性分配固定 KV 容量。

该方法借用 DuoAttention 的 head-level eviction 框架（重要 head 保留 full context，不重要 head 使用 sliding window），但评分方式完全不同：DuoAttention 通过检索合成 passkey 优化 head scores（需数小时 8-GPU 优化），KVzip 通过上下文重建（更通用的 proxy task）在单个自然语言样本上计算 head scores（数次 forward pass，一分钟内完成），且性能更优（Figure 11）。

从算法pipeline角度拆解术语：

**预计算与部署流程**：

```
// === 预计算阶段（仅一次，per model） ===
context = single_book_sample   // 88K tokens 英文书（En.QA）
S = compute_scores(context)     // L×H×n_c，chunked scoring
S_head[l,h] = max_i S[l,h,i]   // L×H head-level scores

// 部署后策略
sorted_heads = argsort_desc(S_head)
for head in top_k(sorted_heads):
    // 保留更多 KV pairs（e.g., full context）
    budget[head] = high
for head in bottom_heads:
    // sliding window attention (e.g., 1K tokens)
    budget[head] = low

// 推理时：按固定 budget 执行 head-level eviction，零评分开销
```

术语一般如何实现？如何使用？

Head-level scores 使用通用文本样本预计算（KVzip 使用 SCBench En.QA 的英文书样本，88K tokens）。Figure 24 可视化显示 KVzip 的 head-score 分布比 DuoAttention 更均匀（因使用自然语言重建而非合成 passkey 检索），跨不同数据源（En.QA、En.MultiChoice、Retr.KV）的 head-score 模式高度重叠。部署后零评分开销，压缩比下限约 0.32-0.4（部分 head 仍需 sliding window）。适用于对压缩比要求适中但延迟敏感的场景。代码开源：https://github.com/snu-mllab/KVzip。

涉及论文标题：
- KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction

---
