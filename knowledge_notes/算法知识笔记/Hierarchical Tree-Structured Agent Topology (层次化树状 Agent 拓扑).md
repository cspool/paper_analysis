## Hierarchical Tree-Structured Agent Topology (层次化树状 Agent 拓扑)

术语是什么？
将 MoA 的全连接 agent 交互图替换为层次化树结构。典型 9-3-1 三层配置：Layer 1 的 9 个 leaf agents 分为 3 clusters（每 cluster 含 4B/8B/32B 三模型），Layer 2 每 agent 仅连接对应 cluster（|C(a)|=3），Layer 3 root 聚合所有 Layer 2 输出。延迟优化核心：T_ℓ^tree ≈ max_{a_{ℓ,j}} max_{c∈C(a_{ℓ,j})} t_c，远小于 all-to-all 的 T_ℓ^all = max_i t_i；子树间互不阻塞，straggler 影响局限于其子树。

从算法pipeline角度拆解：
```
Input: query Q
Layer 1 (9 agents, 3 clusters):
  // 3 clusters 独立并发
  Cluster k: {a_{1,3k-2}(4B), a_{1,3k-1}(8B), a_{1,3k}(32B)} 并行

Layer 2 (3 agents):
  a_{2,k} ← 仅聚合 Cluster k 的 3 个输出
  // 输入上下文 = prefix + 3×output (vs all-to-all 的 9×output)

Layer 3 (root):
  a_{3,1} ← 聚合全部 Layer 2 输出 → final answer
```
优势：(1) 上下文缩短（prefill 成本线性降）；(2) 子树并发（互不阻塞）；(3) straggler 隔离。

术语一般如何实现？如何使用？
- 修改 MoA orchestration 层的 agent 依赖图
- 聚类可异构分组（每 cluster 含小/中/大模型）
- 需 Shell Router 或编排器管理依赖

涉及论文标题：
- Efficient Mixture-of-Agents Serving via Tree-Structured Routing, Adaptive Pruning, and Dependency-Aware Prefill-Decode Overlap
