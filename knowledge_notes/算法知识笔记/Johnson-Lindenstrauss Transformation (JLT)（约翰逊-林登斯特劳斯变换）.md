## Johnson-Lindenstrauss Transformation (JLT)（约翰逊-林登斯特劳斯变换）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
JL 引理 (Johnson & Lindenstrauss, 1984) 是降维理论基石：对于 N 个 D 维点集，存在映射 f:R^D→R^d, d=O(ε^{-2}log N)，使所有点对距离在 (1±ε) 因子内保持。JLT 是实现此映射的方法，最常用随机正交矩阵或随机高斯矩阵。已证明在维度-vs-误差权衡上渐近最优 [Larsen & Nelson, FOCS 2017]。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 RaBitQ 中 JLT 的角色：(1) 随机化——P 的随机性使量化误差的集中不等式成立；(2) 等距性——正交矩阵保内积，允许在变换后空间操作；(3) P 应用于数据为 o'=P^{-1}o，利用正交性 ⟨P·y/||y||, o⟩ = ⟨y/||y||, P^{-1}o⟩ 将旋转转移到查询向量。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
随机正交矩阵通过 D×D 高斯矩阵 QR 分解生成。P 仅需采样一次，所有向量共享。存储开销 D² FP32（D=3072 时 ~38MB）。应用场景：局部敏感哈希 (LSH)、随机投影树、维度约简。

涉及论文标题：
- RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
- Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search

---
