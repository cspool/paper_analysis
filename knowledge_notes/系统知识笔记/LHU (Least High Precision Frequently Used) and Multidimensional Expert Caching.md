## LHU (Least High Precision Frequently Used) and Multidimensional Expert Caching

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

LHU（Least High Precision Frequently Used）是 HOBBIT 提出的面向混合精度 Expert Cache 的缓存淘汰策略。与标准 LFU（追踪每个 expert 的总使用频次 F_t）不同，LHU 单独追踪高精度版本的使用频次 H_t，因为高精度 expert 的 cache miss 代价是低精度的 4×（以 FP16 vs INT4 计）。Multidimensional Expert Caching 是将 LHU 与 LRU、LFU、FLD 四种策略通过加权线性组合实现的多维缓存管理方案：p_t = w_lru·(R_t/T) + w_lfu·(F_t/T) + w_lhu·(H_t/T) + w_fld·(1-|l_t-l_i|/l_n)，权重通过校准数据集最小化 miss penalty 确定。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

HOBBIT Multidimensional Cache Manager 的工作流程：

```
# 高/低精度 cache 分离管理
HighPrecisionCache: [expert_0, expert_3, ...]  # 较大容量
LowPrecisionCache:  [expert_1, ...]            # 较小容量

# Policy Performer 记录（per-sequence）
Records:
  LR[t]: last_used_token_id     # LRU 记录
  F[t]:  total_use_count        # LFU 记录
  H[t]:  high_precision_count   # LHU 记录
  layer[t]: expert_layer_id     # FLD 记录

# On cache miss for expert e_i (high precision):
1. 加载 e_i 的高精度权重到 GPU
2. 更新 LR[e_i], F[e_i], H[e_i]
3. for each cached expert e_j in HighPrecisionCache:
     p_j = w_lru*(LR[e_j]/T) + w_lfu*(F[e_j]/T)
         + w_lhu*(H[e_j]/T) + w_fld*(1 - layer_dist(e_j, e_i)/l_n)
4. evict e_j* = argmin(p_j)，replace with e_i
5. 若 evicted expert 有低精度版本 → 可选降级到 LowPrecisionCache

# On cache miss for expert e_x (low precision):
1. 加载 e_x 的低精度权重
2. 更新 LR[e_x], F[e_x]（不更新 H，因为低精度）
3. 在 LowPrecisionCache 中按优先级 evict + replace

# New sequence start:
1. 重置所有 LR, F, H 记录
2. Expert cache 内容保留（warm cache）
```

关键设计点：
- 高/低精度 cache 分离避免低精度 expert 挤占高精度 expert 空间
- 序列级 LFU/LHU（per-sequence 统计）vs 模型级（全局统计）：序列级 LFU 带来 4.5% hit ratio 提升
- LHU 相比 LFU 的 cache miss penalty 降低约 15%（因优先保留高精度高频 expert）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 四种权重 w_lru, w_lfu, w_lhu, w_fld 为超参数，通过在 calibration dataset 上最小化 total miss penalty 搜索（网格搜索或贝叶斯优化）
- 高精度 cache 建议容量 ≥ 低精度 cache 的 2-3 倍（因高精度 expert 更多且 miss 代价更高）
- 与 Adaptive Expert Prefetching 配合：预取的 expert 被 mask 保护不被 evict
- HOBBIT 实验：Multidimensional policy 相比 LRU 降低 4.69%-8.68% miss penalty，相比 LFU 降低 2.13%-4.19%
- 变体：MoE-APEX 提出 LCU (Least Costly Used) = H_t + (B_l/B_h)·L_t，将高低精度使用频次统一为 cost 值，结合 FR (forward reuse) 和 layer distance，公式更简洁

涉及论文标题：
- HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference
- MoE-APEX: An Efficient MoE Inference System with Adaptive Precision Expert Offloading
