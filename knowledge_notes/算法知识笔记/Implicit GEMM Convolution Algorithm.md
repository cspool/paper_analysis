## Implicit GEMM Convolution Algorithm

术语是什么？
Implicit GEMM Convolution Algorithm 是一种将 2D 卷积（Conv2D）操作映射为矩阵乘法（GEMM）而在 GPU 上高效执行的算法，由 NVIDIA CUTLASS 库实现。与传统的 im2col 方法不同（im2col 先将输入图像显式展开为大型卷积矩阵，再调用 GEMM），Implicit GEMM 在从 global memory 加载数据到 shared memory 时**即时**构造卷积矩阵的 tile，避免了 im2col 矩阵的额外内存分配和带宽开销。映射关系：输入 image tensor x(NHWC) 展开为矩阵 A[NHW × RSC]，filter tensor w(KRSC) 视为矩阵 B[RSC × K]，输出 y(NPQK) 对应矩阵 C[NPQ × K] = A × B。

从算法pipeline角度拆解术语：
Implicit GEMM 的计算映射过程：
```
// 2D Convolution参数:
//   Input:  x[N, H, W, C]      batch × height × width × in_channels
//   Filter: w[K, R, S, C]      out_channels × kernel_h × kernel_w × in_channels
//   Output: y[N, P, Q, K]      batch × out_h × out_w × out_channels
//   (P, Q由padding和stride决定)

// 映射为GEMM: C = A × B
//   A矩阵: [N*P*Q, C*R*S]  — 每个输出像素对应一行，每行含感受野所有输入channel元素
//   B矩阵: [C*R*S, K]      — 每个filter展开为一列
//   C矩阵: [N*P*Q, K]      — 输出

// 传统im2col方法:
float* A_col = im2col(x, N, H, W, C, R, S, P, Q);  // 显式构造，内存开销大
gemm(A_col, w_reshaped, y_reshaped, N*P*Q, K, C*R*S);

// Implicit GEMM方法 (CUTLASS):
// 不显式构造A_col，而是在tile加载时计算A的索引映射
// 对每个tile (m_tile, k_tile):
//   A[m_start..m_start+M_tile, k_start..k_start+K_tile]
//   通过反向索引计算源自x的哪个 (n, h, w, c) 位置:
//     n = m / (P * Q)
//     p = (m % (P * Q)) / Q
//     q = (m % (P * Q)) % Q
//     对filter的 (r, s, c) → k索引, 取x[n, p*stride+r-pad, q*stride+s-pad, c]
```
cuSync 中，两个依赖 Conv2D 经 Implicit GEMM 后的依赖关系：第二个 Conv2D 的 Implicit GEMM 的每个 consumer tile 依赖第一个 Conv2D 的 Implicit GEMM 的所有 column tile。这通过 DSL 描述为 `Dep dep({g2, Tile(x,y)}, {g1, Tile(x/(R*S), y)})`，cuSyncGen 据此生成 RowSync（每行 row 一个 semaphore）和 Conv2DTileSync（每 tile 一个 semaphore）。

术语一般如何实现？如何使用？
CUTLASS 中 Implicit GEMM 的实现使用专门的 iterator（`conv2d_fprop_activation_tile_access_iterator` 和 `conv2d_fprop_filter_tile_access_iterator`）在 tile 加载时计算地址偏移。推荐配置：所有 tensor 128-bit 对齐的 NHWC 布局，channel 数 C 和 K 为 32 的倍数，使用 `kOptimized` iterator 模式预计算指针增量。cuSync 通过修改 CUTLASS Conv2D kernel 的 tile 加载循环添加 wait/post 同步点来支持 Implicit GEMM 的细粒度同步，修改量约 22 行（0.6% 的 CUTLASS Conv2D 代码）。实验显示，对 ResNet-38 和 VGG-19，cuSync 同步 Conv2D kernel 后最多减少 22% 推理时间。

涉及论文标题：
- A Framework for Fine-Grained Synchronization of Dependent GPU Kernels
