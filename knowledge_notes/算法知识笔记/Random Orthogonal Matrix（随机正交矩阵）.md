## Random Orthogonal Matrix（随机正交矩阵）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
从正交群 O(D)={Q∈R^{D×D}|Q^T Q=I} 上均匀采样的 D×D 矩阵。性质：(1) Q^T=Q^{-1}；(2) 保内积: ⟨Qx,Qy⟩=⟨x,y⟩；(3) 保范数: ||Qx||=||x||。生成：D×D 高斯矩阵 QR 分解取 Q。均匀分布保证 P 的随机性是估计器无偏性和误差界的概率来源——打破码本与数据分布的对齐，使坐标尾界 P[|x[i]|>t/√D]≤2exp(-c₀t²) 成立。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 Extended RaBitQ 中：o'=P^{-1}o, q'=P^{-1}q, 利用正交性 ⟨P·y/||y||, q⟩=⟨y/||y||, P^{-1}q⟩=⟨y/||y||, q'⟩，将旋转从码本向量转移到查询向量，实现高效计算。P 仅采样一次，所有向量共享。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
QR 分解生成（BLAS dgeqrf），矩阵乘向量用 BLAS gemv (O(D²))。大 D 时可用结构化随机矩阵近似。存储 D² FP32（D=3072 时 ~38MB）。应用：JLT 降维、RaBitQ 量化、随机投影等。

涉及论文标题：
- RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
- Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search
