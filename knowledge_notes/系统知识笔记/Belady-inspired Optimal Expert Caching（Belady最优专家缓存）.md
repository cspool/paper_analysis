## Belady-inspired Optimal Expert Caching（Belady最优专家缓存）

术语是什么？
Belady-inspired Optimal Expert Caching 是 Read-ME 基于 pre-gating 实现的 expert 缓存替换策略。Belady 算法（1966）是理论最优的离线缓存替换算法——替换未来最远访问的对象（evict argmax F(e,t)）。传统上因无法预知未来访问而不可实现，但 Read-ME 的 pre-gating 使系统可预知所有 token 在所有层的 expert 需求序列 → 精确计算 F(e,t)（expert e 的下次访问时间）→ 实施近似最优缓存替换。在 multi-request 共享 cache 场景下，Belady 在 capacity=4 时 hit ratio 77.21%（vs LRU 66.95%，Random 67.52%），因 LRU 依赖的 per-request temporal locality 在跨请求时失效。

从系统架构角度拆解术语：

```
# Pre-gating 输出所有 future expert references
future_refs = []
for each pending request r:
    for each token t in r:
        # expert_assignment 跨所有层一致!
        future_refs.append((time=t, expert=expert[r][t]))

# 构造 F(e): expert e 的下次访问时间
for each expert e:
    F[e] = min{t | (t, e) in future_refs, t > current_time}

# Cache miss 时的 Belady 替换
if cache_full:
    e_evict = argmax_{e in cache} F[e]  # 替换未来最远的
    cache.replace(e_evict, new_expert)
```

术语一般如何实现？如何使用？
- Cache 在 GPU memory 维护 k 个 expert slots。仅第 1 层加载不可被 prefetch 隐藏——cache 主要服务首层。
- 每次 token 处理后 O(k) 更新 F[e]。若 pre-gating 100% 准确（Read-ME 保证 exact routing），Belady 即为真正的最优离线策略。
- 跨 request 共享优势显著（vs LRU +10.26% hit ratio at k=4），是目前唯一可工程实现的 Belady 近似系统。

涉及论文标题：
- Read-ME: Refactorizing LLMs as Router-Decoupled Mixture of Experts with System Co-Design
