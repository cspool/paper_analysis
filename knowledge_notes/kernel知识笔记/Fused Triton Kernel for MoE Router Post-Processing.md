## Fused Triton Kernel for MoE Router Post-Processing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Fused Triton Kernel for MoE Router Post-Processing 是 LYNX 用于在 MoE 推理的 critical path 中高效执行 batch 级 expert selection 的 GPU kernel 实现。LYNX 将 confidence analysis、adaptive expert scoring、expert pruning 和 expert remapping 四个步骤融合为 4 个 Triton kernel，替代原本需要超过 700 个 PyTorch 小算子的 naive implementation。

四个 kernel 的分工：
1. **Kernel 1 (Token-wise Binning)**：对 batch 中所有 token 并行计算 log-ratio 并做 AffinityBinning 离散化，同时计算 top-k weight sums
2. **Kernel 2-3 (Batch-wise Scoring & Expert Pruning)**：对每个 expert 做 batch 级别指数加权评分，基于分数分布动态确定 active expert 集
3. **Kernel 4 (Expert Remapping & Compaction)**：将 low-confidence token 重映射到 reduced expert set，compaction 重排映射表，renormalize weights 并重新计算 top-k

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
┌── LYNX Kernel Launch Sequence (per MoE layer, decode iteration) ─┐
│                                                                    │
│  [Kernel 1: Token-wise Binning]                                   │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ Grid: (B, ) 即 batch_size 个 thread blocks                    │ │
│  │ 每个 block 处理 1 个 token:                                    │ │
│  │                                                                │ │
│  │ %token_logits = load(router_logits + token_id * N)  // [N]    │ │
│  │ top1_logit = max(%token_logits)                    // scalar  │ │
│  │ for e in topk_indices[token_id]:                              │ │
│  │     log_ratio = %token_logits[e] - top1_logit                 │ │
│  │     bin[token_id][e] = clamp(floor(log_ratio * α), -β, 0)    │ │
│  │ store(bin_out + token_id * k, bin[token_id])                  │ │
│  │                                                                │ │
│  │ Fusion: subtract + multiply + floor + clamp → 1 kernel       │ │
│  │ Replaces: ~200 PyTorch ops (index_select, sub, mul,           │ │
│  │           floor, clamp, scatter)                              │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  [Kernel 2-3: Batch-wise Scoring & Expert Pruning]                │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ Kernel 2: Score accumulation                                  │ │
│  │ Grid: (N, ) 即 N 个 thread blocks (每个 expert 一个)          │ │
│  │ 每个 block:                                                     │ │
│  │   score = 0.0                                                  │ │
│  │   for t in range(B):                                          │ │
│  │       for rank in range(k):                                   │ │
│  │           if topk_idx[t][rank] == expert_id:                  │ │
│  │               score += pow(B, bin[t][rank])  // B^{bin}      │ │
│  │   store(scores_out + expert_id, score)                        │ │
│  │                                                                │ │
│  │ Kernel 3: Threshold & Pruning                                 │ │
│  │ Grid: (1, )  single block                                     │ │
│  │   sorted_scores = sort(scores, descending=True)               │ │
│  │   threshold = determine_by_distribution(sorted_scores,       │ │
│  │                bin_width, max_bins)                            │ │
│  │   active_mask = scores >= threshold                           │ │
│  │   store(active_mask_out, active_mask)                         │ │
│  │                                                                │ │
│  │ Fusion: reduce + pow + sort + threshold → 2 kernels          │ │
│  │ Replaces: ~300 PyTorch ops                                    │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  [Kernel 4: Expert Remapping & Compaction]                        │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ Grid: (B, )  batch_size 个 thread blocks                      │ │
│  │ 每个 block 处理 1 个 token:                                    │ │
│  │                                                                │ │
│  │   // 对 high-confidence token: 保留 origin top-k               │ │
│  │   // 对 low-confidence token: remap lower-ranked experts      │ │
│  │   for rank in range(k):                                       │ │
│  │       if confidence(token) >= threshold OR rank == 0:         │ │
│  │           new_expert[rank] = original_topk[rank]              │ │
│  │       else:                                                    │ │
│  │           new_expert[rank] = find_best_alt_in_active_set(...) │ │
│  │                                                                │ │
│  │   // Compaction: 将 sparse expert indices 映射为 dense        │ │
│  │   compact_expert = active_expert_map[new_expert]              │ │
│  │                                                                │ │
│  │   // Renormalize: 重新计算 softmax                            │ │
│  │   new_weights = softmax(router_logits[compact_expert])        │ │
│  │   store(mapping_out, compact_expert)                          │ │
│  │   store(weights_out, new_weights)                             │ │
│  │                                                                │ │
│  │ Fusion: gather + scatter + softmax + topk → 1 kernel         │ │
│  │ Replaces: ~250 PyTorch ops                                    │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  总计: 4 Triton kernels 替代 700+ PyTorch ops                      │
│  Overhead: <4% 总体 decode latency                                 │
│  CUDA Graph 兼容: 所有 kernel 保持静态控制流                        │
└────────────────────────────────────────────────────────────────────┘
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

LYNX 的 4 个 fused kernel 使用 Triton 语言编写，编译为 CUDA PTX。关键设计选择：(1) 每个 kernel 将数据保持在 registers 或 shared memory 中，消除 intermediate tensor 的 global memory 读写；(2) 静态控制流确保 CUDA Graph capture 兼容——这是 vLLM 等 serving engine 的关键优化需求；(3) 4 个 kernel launch 的开销远小于 700+ 个细粒度 PyTorch kernel launch 的累积开销（每个 launch ~5-10μs）；(4) Kernel 参数（α, β）在模型加载时计算一次，作为 kernel constant 传入。

涉及论文标题：
- LYNX: Enabling Efficient MoE Inference Through Dynamic Batch-Aware Expert Selection
