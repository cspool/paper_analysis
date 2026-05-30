## Similarity-Aware Expert Selection（相似度感知专家选择 / δ-threshold）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Similarity-Aware Expert Selection 是 FineMoE 的 expert prefetching 决策机制：根据检索到的 historical expert map 与当前 context 的 cosine similarity score 动态决定预取多少 experts。核心公式：δ_l = clip(1 - similarity_score, 0, 1)，从 searched expert map P_l 中按概率从高到低选择 experts，直到累积概率 Σp ≥ δ_l 且至少选择 K 个（MoE 模型每层需激活 K 个 experts）。直观逻辑：高 similarity → 高 confidence → 低 δ → 选少量 high-probability experts → 节省 GPU cache；低 similarity → 低 confidence → 高 δ → 选更多 experts → 增大 coverage 防 miss。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
Similarity-Aware Expert Selection 算法：

Input: searched probability distribution P_l ∈ R^J  (layer l, from best-match historical map)
       similarity_score ∈ [-1, 1]                  (cosine similarity from search)
       K ∈ Z+                                      (top-K required by MoE model, e.g., K=2 for Mixtral)

Algorithm:
δ_l = clip(1 - similarity_score, 0, 1)
# 例: score=0.9 → δ=0.1 (高 confidence, 少选)
#     score=0.3 → δ=0.7 (低 confidence, 多选)
#     score=-0.5 → δ=1.0 (极低 confidence, 全选)

sorted_experts = argsort(P_l, descending=True)  # 按概率降序
E_prefetch = []
cum_prob = 0

for j in sorted_experts:
    E_prefetch.append(j)
    cum_prob += P_l[j]
    if cum_prob >= δ_l and len(E_prefetch) >= K:
        break

# 约束条件 (Eq. 6-8):
#   Σ_{j ∈ E_prefetch} p_{l,j} ≥ δ_l   (累计概率达到阈值)
#   |E_prefetch| ≥ K                   (至少选 K 个)

return E_prefetch
```

该机制在 hit rate（减少 expert miss）和 GPU memory（减少 prefetch 量）之间实现连续可调的 trade-off。消融实验（图 14a）表明 Map(T+S+δ) 比 Map(T+S)（静态 top-K selection）进一步提升 expert hit rate。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FineMoE 中以 PyTorch ops 实现，与 Expert Map Searcher 集成。每个 inference iteration 的每个 target layer 都执行动态 selection，确保 prefetch 策略随 context 变化自适应调整。对比 baseline 的固定 stride（ProMoE）或 LFU（MoE-Infinity）策略，similarity-aware 方式使 GPU cache 容量利用更高效：高 confidence 时省出 cache 空间给更多 KV cache/batch tokens，低 confidence 时增大 coverage 保 latency。

涉及论文标题：
- Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading
