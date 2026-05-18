## Non-reduction Dimension Mixed Precision Computation（非归约维度混合精度计算）

术语是什么？通过联网搜索让回答具体和精准。

在矩阵乘法C[M,N] = A[M,K] × B[K,N]中，K为reduction dimension（归约维度），M和N为non-reduction dimension（非归约维度）。Mixed precision quantization可根据outlier所在的维度分为两类：(1) Reduction dimension（channel-wise）——outlier在K维度，可沿K分解GEMM为两个独立dense GEMM；(2) Non-reduction dimension（token-wise）——outlier在M维度，无法沿M简单分解，产生sparse computation pattern。RoMeo的token-wise mixed precision属于Non-reduction dimension类型，需要专门的permutation-free系统设计来高效实现。

从kernel调度角度拆解术语：

Non-reduction dimension mixed precision的本质挑战（以token-wise outlier为例）：

```
// Activation X [M, K], token-wise outliers在M维度
// M=4 tokens (t0-t3), K=4 channels, t0和t2是outlier tokens

// Channel-wise (reduction dim, 可分解):
//   X = [t0: INT8, t1: INT4, t2: INT8, t3: INT4] 每行精度不同
//   沿K分解: 将outlier columns抽出独立计算→dense GEMM
//   C = INT4_GEMM(X_[:, normal_cols], W_[normal_cols, :]) 
//     + INT8_GEMM(X_[:, outlier_cols], W_[outlier_cols, :])
//   ✓ 两个dense GEMM，Tensor Core兼容

// Token-wise (non-reduction dim, 不可简单分解):
//   沿M分解会导致sparse pattern:
//   C[outlier_rows, :] = INT8_dot(outlier_row, W_col)  // sparse access
//   C[normal_rows, :]  = INT4_dot(normal_row, W_col)   // sparse access
//   ✗ sparse computation, incompatible with Tensor Core dense tile requirement

// RoMeo的Permutation-free方案:
//   1. 复制outlier tokens到dedicated buffer（redundant computation代价）
//   2. 所有子矩阵uniform precision → dense GEMM
//   A_int4 = [M, K] (all INT4, 含outlier rows的INT4版本)
//   A_int8 = [k_a, K] (仅outlier rows, INT8)
//   W_int4 = [K, N_normal] (normal columns, INT4)
//   W_int8 = [K, N_outlier] (outlier columns, INT8)
//   → W4A4: A_int4[M, K] × W_int4[K, N_normal] = dense ✓
//   → W4A8: A_int4[M, K] × W_int8[K, N_outlier] = dense ✓
//   → W8A4: A_int8[k_a, K] × W_int4[K, N_normal] = dense ✓
//   → W8A8: A_int8[k_a, K] × W_int8[K, N_outlier] = dense ✓
```

Non-reduction dimension的sparse pattern可视化：
```
Thread Block workload partitioning (M=3 output tiles):
  Tile 0: 2 INT4 rows + 1 INT8 row → heterogeneous → conditional branches
  Tile 1: 1 INT4 row + 2 INT8 rows → heterogeneous → conditional branches  
  Tile 2: 3 INT4 rows             → homogeneous   → optimal Tensor Core use
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现策略对比：
1. **Permutation-based**：将同精度数据重排到相邻位置使tile内精度统一。但permutation引入非平凡index computation和in-place swap overhead（RoMeo实验表明常超过计算收益）。
2. **Permutation-free（RoMeo方案）**：tolerate redundant computation——outlier token同时参与INT4和INT8计算→所有矩阵为dense uniform-precision→无需permutation。代价为~5%额外计算（outlier比例），但换取Tensor Core高效dense执行。Post-mul overwrite将高精度结果覆盖到最终输出。
3. **Reduction-dimension方案（传统）**：沿K维度分解→天然的dense computation，无需处理non-reduction sparse pattern。这是channel-wise方法（MixQ/LLM.int8()）天然高效的原因。

涉及论文标题：
- RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization

