## Fast Similarity Measurement (快速相似度测量)

术语解释
Fast Similarity Measurement 是 LUFFY 系统中 Token Condensation 的核心子算法，用于在 MoE 训练中高效识别被路由到同一 expert 的相似 token。它通过三层过滤策略将 O(N²·d) 的全对余弦相似度计算降低到仅少量不确定 token 对需要真实计算。

术语是什么？
Naive 方法的计算复杂度为 O(N²·d)，其中 N 可达数千，d 为 token embedding 维度（如 1024），在每次 training iteration 的每个 MoE block 都执行是不可行的。Fast Similarity Measurement 利用两个关键观察：
1. 被路由到不同 expert 的 token 极不可能相似（不同 expert 设计为处理不同类型输入）
2. 在连续 block 间，极端相似（s_{b-1} > S₁）或极端不相似（s_{b-1} < S₂）的 token 对会维持其模式——约 90% 的高相似 token 对在后续 block 保持相似

三步过滤策略：
- **Step 1: Expert Activation Filter** → 过滤掉约 (E-1)/E 的 token 对（被路由到不同 expert 的）
- **Step 2: Historical Similarity Lookup** → O(1) 查找前一 block 的相似度缓存，极端情况直接判定
- **Step 3: Real Cosine Calculation** → 仅对剩余高度不确定的 token 对计算 cos(u,v) = (u·v) / (||u||·||v||)

从算法pipeline角度拆解术语：
```
Algorithm: Fast Similarity Measurement

Input:  tokens X ∈ R^{N×d}, expert_ids ∈ Z^N, 
        prev_sim_cache (from block b-1), params S1, S2

Output: similarity graph G with edge weights

1. 初始化全连接图 G: N nodes, N(N-1)/2 edges

2. for each edge (i, j) in parallel:
     # Layer 1: Expert Activation Filter
     if expert_ids[i] != expert_ids[j]:
         G[i][j].weight = 0
         continue  # ~(E-1)/E 的边在此过滤
     
     # Layer 2: Historical Similarity Lookup
     s_prev = prev_sim_cache.get((i, j))
     if s_prev is not None:
         if s_prev > S1:  # 极端相似 (e.g., S1=0.8)
             G[i][j].weight = 1
             continue
         if s_prev < S2:  # 极端不相似 (e.g., S2=0.2)
             G[i][j].weight = 0
             continue
     
     # Layer 3: Real Cosine Similarity (仅 ~10-20% 的剩余边)
     u, v = X[i], X[j]
     G[i][j].weight = dot(u, v) / (norm(u) * norm(v))

3. return G
```

术语一般如何实现？如何使用？
- 每个 GPU 维护独立 CUDA stream 运行相似度计算，与 expert computation 并行
- 历史相似度缓存在 GPU 内存中，跨 block 复用
- S₁ 和 S₂ 参数需根据模型调整：减小 S₁→更多 token 被标记为相似→更多凝聚但可能误判；增大 S₂→更多 token 被标记为不相似→更保守
- 适合 MoE 训练场景，因 expert activation 提供了天然的 token 分组信号

涉及论文标题：
- Communication-Efficient Sparsely-Activated Model Training via Sequence Migration and Token Condensation

---
