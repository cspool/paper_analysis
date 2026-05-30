## PiKV Scheduling (Query-Aware KV Scheduling / 查询感知KV调度)

术语是什么？
PiKV Scheduling 是 PiKV 的 KV cache 页面调度器，将传统 KV cache 驱逐从静态策略（LRU/FIFO）升级为 query-aware 的动态效用评分系统。每个 cached page i 被分配一个标量效用分数 u_i，基于多维特征（注意力强度 a_i、访问新近度 r_i、访问频率 f_i、年龄 t_i 等）计算。当内存不足时，低 u_i 的 page 被驱逐。PiKV 支持 8 种调度策略：H2O（u_i = a_i，仅注意力权重）、StreamingLLM（u_i = I[t_i > τ]，滑动窗口）、QUEST（MLP_θ 学习评分）、FlexGen（plan-based）、LRU/LRU+（新近度/新近度+频率）、AdaKV（多特征自适应：u_i = Σ_j α_j φ_j(i)，阈值 θ ← θ + γ(η*-η)）、Duo Attention（跨层注意力累加：u_i = Σ_ℓ a_i^(ℓ)）。

从系统架构角度拆解术语：
PiKV Scheduling 在 decoding loop 中的运转流程：

```
# === Per Decode Step ===
# Input: q_t, current cached pages P = {page_i}
# Memory budget: M_max pages

# Step 1: Score all cached pages
for page in P:
    # AdaKV: multi-feature scoring
    attention_score = page.avg_attention_weight  # φ1: attention intensity
    recency_score = -(current_step - page.last_access)  # φ2: recency
    freq_score = page.access_count                # φ3: frequency
    u_i = α1 * attention_score + α2 * recency_score + α3 * freq_score

# Step 2: Update adaptive eviction threshold
# η* = target hit rate, η = actual hit rate
θ = θ + γ * (η* - η)   # increase θ if hit rate below target

# Step 3: Evict low-utility pages
for page in P:
    if u_i < θ and len(P) > M_max:
        evict(page)
        P.remove(page)

# Step 4: Fetch query-relevant pages for g_t
for e in g_t:
    relevant = lookup(C[e], q_t)
    for page in relevant:
        if page not in P:
            load_to_cache(page)
            page.last_access = current_step
            page.access_count = 1
        else:
            page.access_count += 1

# Step 5: Update attention weights for next scoring
# After FlashAttention, record per-page attention avg
for page in P_accessed:
    page.avg_attention_weight = compute_avg_attn(page, q_t)
```

**最优 buffer 大小公式**：
$$S^* = \sqrt{\frac{L}{KG}}, \quad \mathcal{M}^*_{\text{total}} = \frac{4d}{\rho} \sqrt{\frac{KL}{G}}$$
其中 S* 平衡 sharding granularity 与 reuse coverage。

术语一般如何实现？如何使用？
- PiKV 实现：`core/single/kvcache_centric_system.py` 中的 `CacheAwarePrefillScheduler`（TTFT SLO 约束）+ `LoadBalanceDecodingScheduler`（TBT SLO 约束）。调度策略通过 `create_scheduler()` 选择。
- AdaKV 的 α_j 权重可在线学习或离线标定。QUEST 的 MLP_θ 需 training（以命中率为 reward）。
- 与 vLLM 默认 LRU 的对比：LRU 仅考虑 recency（最短访问间隔），在 MoE 场景中可能驱逐短期内不被某 expert 访问但未来另一 expert 需要的关键 page。PiKV 的多特征评分可捕获跨 expert 的复用模式。
- 参考方法：H2O (heavy-hitter oracle)、StreamingLLM (attention sink)、Duo Attention (cross-layer accumulation)。

涉及论文标题：
- PiKV KV Cache Management System for Mixture of Experts
