## Early Exiting（早退，EE / FEE）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Early Exiting 是在逐维累加的距离计算中，一旦部分距离超过阈值即提前终止剩余维计算的一种剪枝技术：给定 D 维向量 x 与查询 q，部分距离 d_part^k(x,q) 是前 k 维的累加距离（L2 下严格小于全维距离 d_all），当 d_part^k 超过当前候选队列最远距离 threshold 时，该向量不可能成为更近的候选，继续计算是浪费，因此触发退出。它把平均访问维度数从 D 降到 k<D，直接削减内存受限的 ANNS 中的 DRAM 访问量。本论文指出既有 EE 的局限：部分距离收敛到 threshold 的速度太慢，导致实际只省约 20% 计算（Fig.3-5 动机分析）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
EE 在距离计算 pipeline 中的位置（每候选向量）：
```
d_part ← 0
for k = 1 to D（按 DRAM burst 步进，如每次 2 维）:
  d_part ← d_part + Σ_{i∈burst(k)} (x_i − q_i)²      # 部分距离累加
  if d_part ≥ threshold: return REJECT              # 早退：丢弃该候选
return ACCEPT（d_all = d_part < threshold → 入候选队列）
```
例子（论文 Fig.6a）：候选队列阈值 2.5，邻居 s1/s2/s3 分别在第 2、4、… 维触发早退被拒绝，仅 s0 全维计算后入队。对比朴素 EE：若一个"应被拒绝"的向量要到第 109 维才满足 d_part≥threshold，而使用估计距离可在第 4 维就触发（见 FEE-sPCA 条目），早退效率的差别即在此。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现上 EE 需要：部分距离累加器、threshold（来自候选优先队列最远点）、逐 burst 比较逻辑。硬件上由 NDP 的 FEE 模块实现（每次累加器更新即比较）；软件 ANNS 库中常用"维度重排 + 阈值剪枝"近似实现。局限：仅用 d_part 收敛慢（论文动机），需结合 PCA 估计（FEE-sPCA）与位级压缩（Dfloat）才显著。论文按 burst 粒度触发（每 2 维/步），与 DRAM 突发访问天然对齐。

涉及论文标题：
- NasZip Software and Hardware Co-design to Accelerate Approximate Nearest Neighbor Search with DIMM-based Near-Data Processing
