## Causal Importance Score (CIS) / 因果重要性分数

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Causal Importance Score (CIS) 是 LOCRET 论文提出的 KV cache unit 重要性度量指标。其核心思想是：在 chunked prefill 的每步中，为每个 KV cache unit（单个 token 在单个 attention head 的 KV 向量对）分配一个分数，该分数**仅依赖当前及之前的 token**（causal），反映该 cache unit 对理解后续上下文的重要性。

CIS 的 ground truth 定义：对于训练样本 $d$，在第 $i$ 层第 $j$ 个 attention head，token $k$ 的 CIS 为所有 answer token 对该 prefix token 的最大 pre-softmax attention score：$\mathbf{S}[k]_j^{(i)} := \max_p (\mathbf{Q}_j^{(i)}\mathbf{K}_j^{(i)T})_{p,k}$，其中 $p \in [n_q(d)+1, n_q(d)+n_a(d)]$（answer token 范围），$k \in [1, n_q(d)]$（prefix token 范围）。对 GQA 模型，取同一 group 内不同 query head 的最大值。

CIS 的**因果性**（causality）是其区别于 H2O/SNAPKV 等 non-causal 评分的核心特征：CIS 在 token 出现时即可计算，不需要等待后续 token 的 attention score，因此天然兼容 chunked prefill。Non-causal 评分（如 H2O 的 A2S、SNAPKV 的 voting）需要完整序列的 attention scores 才能准确评估，在 chunked prefill 中只能看到当前 chunk 导致严重低估某些 token 的重要性（local-global discrepancy）。

从算法pipeline角度拆解术语。

**CIS 在 LOCRET 中的使用流程**：

```
// ============ Training: CIS ground truth 收集 ============
for layer l in 1..L:
    Q_l, K_l = qkv_proj(H_{l-1})
    // 计算 pre-softmax attention scores
    A_l = Q_l @ K_l^T / sqrt(d_k)        // [h, n_seq, n_seq]
    for head j in 1..h:
        for prefix_token k in 1..n_q:
            // 所有 answer token 对该 prefix token 的最大 pre-softmax score
            S[k]_j = max(A_l[j, n_q+1:n_q+n_a, k])

// ============ Inference: Retaining head 预测 CIS ============
for each chunked prefill step:
    Q_chunk, K_chunk, V_chunk = forward_attention(chunk, K_cache, V_cache)
    // Retaining head 预测 CIS（仅依赖当前及之前 token）
    score_chunk = R([Q_chunk, K_chunk, V_chunk])   // MLP forward
    score_cache = concat(score_cache, score_chunk)
    // CIS-based eviction: 保留 top-b 最高分
    indices = topk(score_cache, b)
    K_cache, V_cache = K_cache[indices], V_cache[indices]
```

**CIS 的数学保证**（LOCRET Theorem N.4）：用 Top-b CIS 选择 cache unit 等价于一个 **cache problem**（有预算限制的因果计算问题）。具体而言：若选择函数 $\text{Sel}(c_1, \dots, c_i) = \{c_{p_1}, \dots, c_{p_{b'}}\}$ 其中 $s_{p_1}, \dots, s_{p_{b'}} \in \text{Top-}b(s_1, \dots, s_i)$，则 $(f, b, \{c_i\})$ 是一个 cache problem。这意味着 CIS 可以形式化为一个在线缓存问题的最优解。

术语一般如何实现？如何使用？

CIS 的预测由 retaining head（小型 MLP）完成，推理开销可忽略。CIS 评分在 chunked prefill 的每个 chunk 步后执行一次 TopK eviction。CIS 一旦计算即不变（causal），无需重新评估。LOCRET 将 stabilizers（最后 $n_s$ 个 token）的 CIS 强制设为 $+\infty$ 以防止被 evict，缓解上下文不连续性。LOCRET-Q 变体训练时将 query token 前置使 CIS labels 感知 query，推理时将 query 置于序列首部。开源：https://github.com/huangyuxiang03/Locret。

涉及论文标题：
- LOCRET: Enhancing Eviction in Long-Context LLM Inference with Trained Retaining Heads on Consumer-Grade Devices

---
