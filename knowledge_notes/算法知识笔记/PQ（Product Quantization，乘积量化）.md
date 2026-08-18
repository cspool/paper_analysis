## PQ（Product Quantization，乘积量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PQ 是把 D 维向量切分为 M 个子向量、对每个子空间分别做 K-means 聚类并用聚类中心索引（codeword）编码的向量压缩方法：每个向量用 M 个 code（每 code 约 log2 K bit）表示，距离用查找表近似（ADC：子空间距离查表求和），可把向量存储压缩到 1/8~1/64。它是 ANNS 的经典压缩 baseline，也是本论文内存流量对比的压缩基线之一（Fig.20）。局限：压缩引入明显精度损失，为保持高 recall 只能降低压缩比，导致比 RabitQ/NASZIP 高约 2× 的内存流量。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
PQ 的 pipeline（vault 笔记 ANNS 条目亦概述 IVF+量化 pipeline）：(1) 索引构建：把每个向量 x∈R^D 切为 M 个子向量 x^(j)∈R^(D/M)，对每个子空间独立跑 K-means 得 K 个中心，x 编码为 M 个 codeword；(2) 查询：对查询 q 计算每个子空间到 K 个中心的距离表，候选向量的近似距离 = Σ_j Table[j][code_j(x)]。伪代码：
```
# 离线：per-subspace 聚类
for j in 1..M: C_j ← KMeans({x_i^(j)})   # K 中心
    code(x) ← [argmin_k ||x^(1)−C_1[k]||, ..., argmin_k ||x^(M)−C_M[k]||]
# 在线：近似距离
d_est(x,q) ← Σ_j T_j[code_j(x)],  T_j[k] ← ||q^(j) − C_j[k]||²
```
论文用法：作为压缩内存流量 baseline（HNSW on PQ 编码），recall@10≥90% 时流量约为 RabitQ/NASZIP 的 2 倍。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：FAISS（IndexPQ/IVFPQ）、ScaNN 等主流库均内置；参数为 M（子空间数）与 K（每子空间中心数）。使用：先训练码本（采样向量聚类）→ 编码向量 → 查询用查表近似距离，必要时精排（re-rank 用原向量）。局限（论文指出）：PQ 主打压缩、精度损失大，高 recall 下需弱压缩，内存流量反而更高；且不直接匹配 NDP 的 burst 访问模式。论文以其为对照，说明 FEE-sPCA+Dfloat 在同等 recall 下流量更低。

涉及论文标题：
- NasZip Software and Hardware Co-design to Accelerate Approximate Nearest Neighbor Search with DIMM-based Near-Data Processing
