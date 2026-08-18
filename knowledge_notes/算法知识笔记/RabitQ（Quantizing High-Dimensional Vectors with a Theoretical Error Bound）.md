## RabitQ（Quantizing High-Dimensional Vectors with a Theoretical Error Bound）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RabitQ 是一种带理论误差界的向量量化方法，用于 ANN 候选过滤：对高维向量做有理论保证的紧凑量化，使量化后的距离估计有可证明的误差上界，从而在候选过滤（filter）阶段安全地淘汰大部分非候选、且不损失精确性。它是论文内存流量对比的另一个压缩 baseline（vault 笔记 ANN 条目：/data3/paper_analysis/knowledge_notes/算法知识笔记/Approximate Nearest Neighbor (ANN) Search（近似最近邻搜索）.md 引用该论文）。局限（论文 Fig.20 分析）：RabitQ 加速候选过滤，但幸存候选仍需全维精确距离做 re-rank，因此内存流量仍高于 FEE-sPCA+Dfloat（后者通过特征级早退直接砍掉访问维度数、Dfloat 再压位宽）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
RabitQ 的 pipeline：离线对向量量化（带误差界），在线查询时对每个候选先算量化近似距离 d_q(x,q)（紧凑读入）→ 结合误差界判定：d_q − bound ≥ threshold 则安全淘汰（无需读原向量）→ 幸存少数候选读原向量做精确距离与精排。伪代码（过滤阶段）：
```
for x in candidates:
  d_q ← dist_quantized(x, q)            # 仅读量化码（紧凑）
  if d_q − err_bound(x) ≥ threshold: skip(x)   # 理论保证可安全淘汰
  else: d_all ← dist_exact(x, q)       # 幸存者读原向量精排
```
对比：RabitQ 的淘汰决策基于"量化码+误差界"但仍需为幸存者读全维原向量；FEE-sPCA 用 PCA 估计距离在部分维度内就完成淘汰、Dfloat 进一步减少每次读取的位数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：RabitQ 以其理论误差界为核心设计量化器（基于随机量化的低偏差估计），可用于 FAISS 风格索引；论文将其配置到 HNSW 上做 re-ranking 前的过滤。使用场景：需要保证 recall 的候选过滤、与图/IVF 索引结合。局限（论文角度）：过滤后仍需全维精排 → 流量高；本论文的 FEE-sPCA 在 NDP 上与之兼容且流量更低（Fig.20，与 PQ 归一化对比）。

涉及论文标题：
- NasZip Software and Hardware Co-design to Accelerate Approximate Nearest Neighbor Search with DIMM-based Near-Data Processing
