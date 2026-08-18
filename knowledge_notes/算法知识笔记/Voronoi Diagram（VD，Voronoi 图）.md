## Voronoi Diagram（VD，Voronoi 图）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Voronoi 图（Voronoi Diagram，以数学家 Voronoi 命名）是计算几何的经典空间划分结构：给定点集 S={s_1,...,s_m}（称为 sites/种子点），把空间划分为 m 个 Voronoi cell V(s_i)={p∈R^3 | ∀j≠i, d(p,s_i)≤d(p,s_j)}，每个 cell 包含"离该 site 比离任何其他 site 更近"的所有点；cell 边界由相邻 site 的垂直平分面构成。Voronoi 图与 Delaunay 三角剖分互为对偶，广泛用于最近邻查询、路径规划、插值与覆盖分析。NS-FPS（ISCA'26）首次把 Voronoi 图与 FPS 联系起来：FPS 维护的"每点到已采样集最近距离"缓存 T 隐含地就是一个 Voronoi 图——每个未采样点被分配给离它最近的已采样点（其 Voronoi cell 中心），T 的值即该点到 cell 中心的距离。当新一轮采样出 s_{m+1} 时，只有落在新 cell V(s_{m+1})（即离 s_{m+1} 比离旧采样集更近）内的点需要更新距离缓存。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - NS-FPS 的 Voronoi 部分更新理论（Section III-A）：设第 k 轮采样集 S_k，最新点 s_k 的搜索半径 d_k=min_{s_i∈S_{k-1}}||s_k−s_i||²（s_k 到旧采样集的最近距离）。证明：任何需要更新距离的点 p 必满足 ||p−s_k||² < min_{s_i∈S_{k-1}}||p−s_i||²；若 p 在球 B(s_k,d_k) 外则 ||p−s_k||²>d_k，而由 s_k 是第 k-1 轮最远点知 min_{s_i}||s_k−s_i||² ≥ min_{s_i}||p−s_i||²，矛盾。因此更新区域被界定为球 B(s_k,d_k)（该球安全包住真实 Voronoi cell，Fig.4 黄球包绿 cell），复杂度：
```
# vanilla FPS 每轮: 更新全部 N 个点 + 全量找 max  -> O(N)
# NS-FPS 每轮:      只更新球内 O(N/k) 个点 + 层次 max -> O(N/k)
# 总复杂度:          O(Σ_{m=1..M} N/m) = O(N log M) ≈ O(N log N)
# 伪代码（核心替换）:
for k in 1..M-1:
    d_k = T[s_{k-1}]
    for p in ball_query(s_{k-1}, d_k):   # 只有 Voronoi cell 邻域内的点
        T[p] = min(T[p], dist2(p, s_{k-1}))
    s_k = hierarchical_argmax(T)          # 只刷新受影响块
```
  - 关键性质：部分更新机制同时缩小"求全局最大"的范围——传统 FPS 每轮要全量扫描 T 找最远点，NS-FPS 把 max 搜索限制在受影响的 Voronoi cell 内，配合层次缓存进一步降低开销；且该重述与原始 FPS 采样结果逐点一致（lossless）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 通用实现：Voronoi 图构造算法（Fortune 扫描线 O(n log n)、增量法、Bowyer-Watson 经 Delaunay）多在 CPU/GPU 计算几何库实现；但在 NS-FPS 中**不显式构造** Voronoi 图——精确构造 VD 边界计算代价过高（论文明确说明），而是用"可证明充分"的球形搜索区域 B(s_k,d_k) 松弛替代，配 Morton cube 划分快速枚举球内点。该思路把 VD 的几何洞察转化为可硬件化的部分更新 + 球查询原语；后续 k-NN/ball query 等下游点云操作可直接复用 Morton 重排布局。

涉及论文标题：
- NS-FPS: Accelerating Farthest Point Sampling via Neighbor Search in Large-Scale Point Clouds
