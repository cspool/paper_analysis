## Speculative Prefill (SPECPREFILL)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Speculative Prefill 是一种 training-free 的 LLM prefill 加速框架。核心思想：利用同系列中较小的"推测器模型"（speculator，如 8B）估计 prompt 中各 token 对主模型（如 70B/405B）的重要性，仅将筛选出的 token 子集（保留原 position IDs）送入主模型 prefill，跳过其余 token 的 attention + MLP 计算。加速比近似正比于 token 丢弃率（实测 405B 模型 10% 保持率下达 7.66× TTFT 加速、7× QPS 提升）。与 speculative decoding 天然兼容：speculator 同时服务 prefill token 选择和 decode draft proposal。

从算法pipeline角度拆解术语：

```
// Speculative Prefill Pipeline
Input: Base model M, Speculator S, prompt P
// Phase 1: Look-ahead (N=8 steps)
for i = 1 to N:
    Q_i, K_i, V_i = S.forward(P, store_q=True)
    P.append(argmax(S.lm_head(Q_last)))

// Phase 2: Attention score aggregation
A = compute_attention(Q_saved, K_saved)  // [N, L, S, H]
score = mean_over_N(max_over_L_H(A))     // → [S]

// Phase 3: Chunk selection
score_smoothed = AvgPool1D(score)
chunks = split_into_chunks(score_smoothed)
selected = TopK(chunk_avg(chunks))
T = tokens_in(selected)

// Phase 4: Main model forward (selected tokens only, with original pos IDs)
output = M.forward(T, original_positions[T])
```

术语一般如何实现？如何使用？

基于 vLLM 0.6.3.post1 monkey patch 实现。需要同模型家族的 speculator（如 Llama-3.1-8B → 70B/405B）。适用于长上下文可压缩 prompts；不适用信息密集短 prompts。开源代码：https://github.com/anonymous/speculative_prefill。

涉及论文标题：
- Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation

---
