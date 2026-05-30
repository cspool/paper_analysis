## Token Scheduler (TS) for MoE Inference (MoE 推理中的Token调度器)

术语解释
Token Scheduler 是一种基于 routing path 相似度的 token 重新分组机制，用于提高 MoE 推理中 expert 的计算利用率。在 decoding 阶段，token 分布到各 expert 可能极度不均衡（部分 expert 仅收到单 token），导致 expert kernel 在少量 token 下处于 memory-bound 区域（roofline 模型），计算效率低下。TS 通过将具有相似 routing path 的 token 聚集到同一 batch，减少每 batch 激活的 expert 数量并增加 per-expert token 负载。

术语是什么？
ExpertFlow 提出的 TS 核心机制：
- **输入**：两个相邻 batch 的 2T 个 token，每个 token 有 routing path $r_i \in \{0,1\}^{L \times E}$
- **目标**：最小化 batch 级 expert 激活总数 $\min_{\mathcal{T}_1, \mathcal{T}_2} \sum_{l=1}^{L} \sum_{e=1}^{E} (R_1^{l,e} + R_2^{l,e})$，其中 $R_k = \bigvee_{i \in \mathcal{T}_k} r_i$（batch 级 OR 操作）
- **方法**：K-means 风格聚类，构建 Hamming distance 相似度矩阵 $S_{ij} = 1 - d_{ij}/(LE)$，迭代分配 token 到最近的 cluster centroid 并更新 centroid
- **开销**：CPU 上 <10ms，通过 Dual-Batch Pipeline 与 GPU 计算重叠隐藏
- **KV-Cache 管理**：Merge（按全局 token 顺序重建 KV cache）+ Reindex（更新 token 索引到新布局）

从系统架构角度拆解术语：
TS 与 RPP、ECE 协同工作的 Dual-Batch Pipeline 流程：
```
=== Dual-Batch Inference Pipeline ===
每个 scheduling unit 包含两个 batch: batch_k, batch_{k+1}

Step 1 [CPU, 与上一 unit 的 MoE 执行并行]:
  RPP(batch_k_inputs, batch_{k+1}_inputs) → routing_paths  # shape: (2B, S, L, E)

Step 2 [CPU, <10ms]:
  tokens = concat(batch_k, batch_{k+1})  # 2T tokens
  S[i][j] = 1 - Hamming(r_i, r_j) / (L*E)  # 相似度矩阵
  while not converged:
    分配每个 token 到最近的 cluster centroid
    更新 centroids 为 intra-cluster 平均相似度最高的 token
  yield (T1, T2)  # 两个等大小新 batch
  
  # KV-Cache 维护确保 attention 语义正确
  Merge(T1_kv, T2_kv) → new_kv_cache
  Reindex(token_indices)

Step 3 [GPU]:
  按新 batch 组织执行 MoE 推理
  for batch in [T1, T2]:
    for layer in moe_layers:
      ECE.prefetch(predicted_experts[batch])
      expert_ffn(tokens, experts)
```

术语一般如何实现？如何使用？
- 适用于 expert 数量大的 MoE 模型（Switch-128 受益最大，TS 提升 1.17× throughput）
- token 越多、expert 越多，聚类收益越大（Switch-32: 1.03×, Switch-64: 1.15×, Switch-128: 1.17×）
- KV-Cache Merge + Reindex 是正确性的关键保证，需与底层 attention 实现协同
- 论文未开源，实现细节（聚类初始化策略、收敛条件、centroid 更新算法）需参考论文描述

涉及论文标题：
- ExpertFlow: Optimized Expert Activation and Token Allocation for Efficient Mixture-of-Experts Inference
