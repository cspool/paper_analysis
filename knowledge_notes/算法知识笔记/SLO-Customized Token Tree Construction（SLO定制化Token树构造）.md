## SLO-Customized Token Tree Construction（SLO定制化Token树构造）

术语是什么？通过联网搜索让回答具体和精准。
SLO-Customized Token Tree Construction 是 AdaServe 的核心算法，将 multi-SLO serving 形式化为带 token budget 约束的 token tree 构造问题。目标：每次 decoding iteration 中，给定硬件可验证 token budget B，为 batch 中各请求构造 speculation token tree，使得 (1) 每个请求期望接受 token 数满足其 TPOT SLO；(2) 同时最大化总 expected accepted tokens。问题分解为两阶段贪心选择。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
Input: requests R, token budget B, candidate tree per request T_i,
       per-request SLO target min_tokens_i

// Phase 1: SLO-customized selection
for each r_i sorted by urgency (SLO slack ascending):
    remaining = min_tokens_i - expected_accepted(selected[r_i])
    while remaining > 0 and |selected[r_i]| < per_request_limit:
        best = argmax(T_i.unselected, key=path_prob)
        selected[r_i].add(best)
        remaining -= best.path_prob

// Phase 2: Throughput-optimized selection
remaining_budget = B - Σ|selected[r_i]|
all_remaining = ∪_i (T_i.nodes - selected[r_i])
top_nodes = argmax_k(all_remaining, k=remaining_budget, key=path_prob)
add top_nodes to respective request selected sets

// Submit all selected token trees for tree-based verification
```
min_tokens_i = max(0, latency_elapsed_i / TPOT_SLO_i - tokens_generated_i)。SLO phase 优先满足严格请求；throughput phase 用剩余 budget 最大化总体吞吐。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 AdaServe 的 CPU scheduler 中实现，用 draft model logits 近似真实 path probability。per-request token limit 防止某严格请求吞噬过多低概率 budget。CPU selection overhead 仅 0.41%（Llama-70B）/ 0.31%（Qwen-32B）。

涉及论文标题：
- AdaServe: Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding

