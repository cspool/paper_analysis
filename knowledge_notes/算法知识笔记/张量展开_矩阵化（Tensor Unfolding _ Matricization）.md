## 张量展开/矩阵化（Tensor Unfolding / Matricization）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
张量展开（矩阵化，matricization / mode unfolding）把高阶张量重排成矩阵：选定若干模式合并为矩阵的行/列索引。例如 $A_{i,j,k}$ 把 (i,j) 合并为行得 (IJ×K) 矩阵，或把 (i,k) 合并得 (IK×J) 矩阵；这样张量收缩 $C_{f_1,f_2}=\sum_c A_{f_1,c}B_{c,f_2}$ 变成标准 SpMM $C_{M,L}=A_{M,K}B_{K,L}$（M=自由模式合并、K=收缩模式、L=另一自由模式）。这是 einsum 落地的常见 lowering，用以复用成熟的 SpMM 优化（inner/outer/Gustavson 数据流、自适应 tiling、矩阵重排如列置换与图 islandization）。TensorPrism 指出展开的代价：(1) 元数据膨胀——张量原生格式 O(I+J+K)，unfold 成 CSR/CSC 后 O(IJ+K)；(2) 复用距离膨胀——max reuse distance 从 I+J 变 I×J；(3) 相邻非零邻居减少——2D 中每非零最多 4 个结构相邻邻居，3 阶张量有 6 个；(4) 不同展开方式产生不同稀疏模式，优化 mode-dependent；(5) 把中间结果映射回原张量域时部分计算不可恢复。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
展开→执行→还原三步 pipeline（SPADE/HotTiles baseline 路线）：
```
# unfold: (i,j,k) -> (m=i*J+j, k)
A_mk = reshape_coo(A, order=(i,j,k), row_modes=(i,j))
C_ml = SpMM_rowwise(A_mk, B)    # for m: for k in nnz(A_mk[m]): C[m,:]+=A_mk[m,k]*B[k,:]
C = reshape_back(C_ml, out_modes=(i,j,l))   # 映射回张量域(此处信息部分不可恢复)
```
后果量化（论文 Fig.3）：unfold 后循环变换丢失 50-60% 潜在数据复用，量子模拟（高阶）达 90%；uber 上 SPADE 91% 开销来自稠密行重复取数（复用距离膨胀超过片上容量）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：COO 坐标按目标 mode 顺序重排 → CSR/CSC 压缩（行指针+列索引）。使用场景：所有"借用 SpMM 优化"的稀疏张量加速路线（ExTensor/SEXTANS/DRT/SPADE/HotTiles/Trapezoid/Misam）。TensorPrism 用它作为 baseline 执行模型（SPADE/HotTiles 按各自论文算法贡献 + matricization 扩展），并证明其局限后以共现图替代（划分前统一分析所有维度索引交叠）。

涉及论文标题：
- TensorPrism: Rethinking Sparse High-order Tensor Acceleration via Co-occurrence Graph
