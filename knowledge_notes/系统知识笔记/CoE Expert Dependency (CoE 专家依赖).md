## CoE Expert Dependency (CoE 专家依赖)

术语解释
CoE Expert Dependency 是 Collaboration-of-Experts 推理系统中独有的属性——专家之间存在两类依赖关系：(1) 请求间依赖：多个请求可能依赖同一 expert，但它们在队列中的位置可能被不相关请求分隔；(2) 专家间依赖（preliminary→subsequent）：后续 expert 的推理依赖前置 expert 的输出结果。

术语是什么？
在 CoE 推理中，expert 依赖关系是 **可预知的**——因为 CoE 的路由规则由用户预定义或独立训练的路由模块确定，不像 MoE 的 router 输出仅在推理时动态产生。这使得 CoE 系统可以：
1. 提前计算每个 expert 的使用概率（而非依赖 LRU 等历史统计）
2. 识别 expert 之间的 preliminary→subsequent 依赖链
3. 基于已知依赖关系优化请求调度和 expert 管理

从系统架构角度拆解术语：
Expert Dependency 在 CoServe 系统中的两种利用方式：

**请求间依赖（用于请求调度）**：
```
Request queue: [R1(Expert1), R2(Expert2), R3(Expert1)]
FCFS (Samba-CoE):
  Expert1 load → R1 → Expert2 load → R2 → Expert1 reload → R3
  (Expert1 被 R2 淘汰后又需重新加载 → 不必要 switching)

Dependency-aware (CoServe):
  识别 R1 和 R3 依赖同一 Expert1 → 重排为 [R1, R3, R2]
  Expert1 load → R1+R3 (batch) → Expert2 load → R2
  (Expert1 仅加载一次 → 减少 switching)
```

**专家间依赖（用于 expert 管理）**：
```
Expert chain: E1(preliminary) → E2(subsequent)
若 E1 尚未加载，则 E2 即使已在 GPU 中也无法执行
→ CoServe 优先淘汰 E2（无前置依赖的后续 expert）
→ 保留 E1 或当前有前置依赖已满足的 expert
```

术语一般如何实现？如何使用？
- CoServe 中通过 Dependency-aware Request Arranging 将依赖同 expert 的请求在队列中成组排列
- CoServe 中通过 Dependency-aware Expert Management 在 eviction 时优先淘汰无前置依赖的后续 expert
- 与 MoE 的 LRU/LFU 淘汰策略互补——CoE 独有的先验信息使淘汰比纯历史统计更准确
- 使用前提：CoE 系统的路由规则可离线分析

涉及论文标题：
- CoServe: Efficient Collaboration-of-Experts (CoE) Model Inference with Limited Memory
