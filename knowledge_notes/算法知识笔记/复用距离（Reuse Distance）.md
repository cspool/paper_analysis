## 复用距离（Reuse Distance）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
复用距离是缓存/局部性分析的经典度量：一个数据项两次访问之间被访问的不同数据项数量。距离小=高时间局部性；距离超过片上容量则该复用必然 miss。TensorPrism 把它适配到稀疏张量收缩（论文 §III-B，受 cache 建模 reuse distance 分析启发）：以"稠密行 B[K,:]"为被复用对象，复用距离=两次取同一稠密行之间访问的不同稀疏行数。对 3 阶张量 $A_{i,j,k}$，max reuse distance 上界为 I+J（自由模式大小之和）；unfold (i,j) 成单维后膨胀到 I×J。此外 unfold 还减少相邻邻居（2D 中每非零最多 4 个结构相邻、3 阶张量 6 个），立即复用机会更少。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
trace 驱动统计：
```
stack = {}
for row in access_trace(B):          # 稠密行访问序列
    if row in stack:
        dist = len(rows between prev and now); record(dist)
    stack.push(row)
# 结论: unfold 使 max dist 从 I+J -> I*J
```
后果：复用距离膨胀超出 GLB/片上容量→稠密行被迫重复从 DRAM 取（uber 上 SPADE 91% 开销、2.09× 超额执行时间）；量化出 unfold 后循环变换丢失 50-60%（量子模拟 90%）复用。对策：CoGTP 把高共现顶点（短复用距离的稠密行复用）聚到同分区=直接缩短有效复用距离；式 5 在 GLB 容量 $M_{cap}$ 约束下选 tiling 因子 $M_t$ 使复用距离 ≤ 片上驻留能力。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：记录稠密行访问序列，用栈/树统计两次访问间的不同行数，得复用距离分布。论文用它做动机分析（Fig.3 数据复用分析）而非运行时开销。在加速器设计中对应 GLB 容量约束（式 5）与 CoGTP 分区目标；48KB/PE 局部存储决定哪些距离可命中。场景：任何稀疏张量收缩的局部性评估。

涉及论文标题：
- TensorPrism: Rethinking Sparse High-order Tensor Acceleration via Co-occurrence Graph
