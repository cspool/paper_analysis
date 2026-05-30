## PiKV Routing（Cache-Aware Expert Routing / 缓存感知专家路由）

术语是什么？
PiKV Routing 是 PiKV 提出的面向 MoE KV cache 的稀疏专家路由机制，将传统 MoE router 的"仅考虑计算负载"扩展为"同时考虑 KV cache 局部性"。核心思想：路由决策不仅基于 token-expert 语义匹配度，还引入 cache miss penalty——对 KV cache 命中率低的 expert 施加惩罚项，使 router 倾向于选择 KV 已在本地 GPU 的 expert，减少 KV cache miss 和跨 GPU 数据传输。PiKV 支持 7 种路由策略：
- R_B（Base hash）：O(1) round-robin，无 cache 感知
- R_T（TopK softmax）：标准 TopK，O(E log k)
- R_LB（Load-Balanced TopK）：+ 负载均衡惩罚 -α(μ_e - μ̄)，O(E)
- R_P（Cache-Aware PiKVRouter）：+ cache miss 惩罚 -λ log(1+miss_e)，O(E)
- R_E（Entropy-Penalized LB）：+ 熵惩罚 -β H(p_e)，O(E)
- R_A（RL-Adaptive）：learned gating，O(k²)
- R_H（Hierarchical）：coarse→fine 两阶段 TopK，O(E + k log k)

从系统架构角度拆解术语：
PiKV Routing 在 MoE serving 中的运转流程：

```
# === PiKV Cache-Aware Router ===
# Input: q_t ∈ R^d, cache state {miss_e}_{e∈E}
# Output: g_t ⊆ E with |g_t| = k

# Step 1: Compute per-expert affinity scores
for e in range(E):
    # Standard routing score (semantic match)
    affinity[e] = dot(q_t, expert_embedding[e])
    # Add cache-aware penalty: lower score for high-miss experts
    # miss_e = number of cache misses for expert e in recent window
    affinity[e] -= λ * log(1 + miss_counter[e])

# Step 2: Load Balancing (optional, for R_LB / R_E)
for e in range(E):
    # Penalize overloaded experts
    affinity[e] -= α * (current_load[e] - mean_load)

# Step 3: TopK selection + softmax
g_t = topk(softmax(affinity), k)

# Step 4: Update miss counter after KV fetch
for e in g_t:
    if cache_miss(e, q_t):
        miss_counter[e] += 1
    else:
        miss_counter[e] = max(0, miss_counter[e] - 1)  # decay

# Theoretical benefit:
# Attention complexity: C_dense = BLhE → C_sparse = BLhk
# Memory traffic: M_dense = 2BLhE → M_sparse = 2BLhk
# Speedup factor: E/k (e.g., 16/2 = 8× for E=16, k=2)
```

**Reuse-distance 分析**：
$$RD_{\text{dense}} = \frac{L}{E}, \quad RD_{\text{sparse}} = \frac{L}{k} \implies \text{hit-rate} \approx \frac{k}{E}$$
缓存感知路由通过惩罚高 miss expert 进一步提高 hit-rate（超出 k/E 的理论值）。

术语一般如何实现？如何使用？
- PiKV 实现：`core/single/moe/` 中的 `create_moe()` 函数，支持 BaseRouter/EPLBRouter/HierarchicalRouter/FlexMoERouter/FasterMoERouter 等 10+ 路由器。
- Cache-Aware Router 的 λ 调优：λ 过大导致 router 过度倾向于 cache-hit expert（可能忽略语义匹配）；λ 过小退化为标准 TopK。推荐 λ ∈ [0.1, 0.5]。
- 与 DeepSeek-V2 的 router 对比：DeepSeek-V2 使用 device-level auxiliary loss 做负载均衡（纯计算视角）；PiKV 的 cache-aware penalty 是"数据局部性"视角的补充。
- 集成方式：替换 vLLM 中 MoE layer 的 gating function，在 `PiKVvLLMEngine` 中通过 `create_moe(router='cache_aware')` 配置。

涉及论文标题：
- PiKV KV Cache Management System for Mixture of Experts
