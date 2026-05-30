## SwapAB MoE GEMM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

SwapAB MoE GEMM 是 LongCat-Flash 推理中针对小 batch MoE 解码场景的 GEMM 优化技术。传统 MoE GEMM 使用 token activations 作为左矩阵 A (M×K)、expert weights 作为右矩阵 B (K×N)，公式为 C = A × B。在 decoding 阶段，M（token 数）通常很小（每个 GPU 上可能只有几十个 token），需要 padding 到 M 维度的 64 元素最小对齐（Tensor Core 要求），padding overhead 显著。

SwapAB 反转矩阵角色：将 expert weights 作为左矩阵（N×K）、token activations 作为右矩阵（K×M），利用 N 维度（expert intermediate dim, 通常 2048）的 8 元素对齐粒度。因为 N >> M（在 small batch 下），N 维度的 padding overhead 可忽略。计算 $C^T = B^T \times A^T$ 而非 $C = A \times B$，结果在内存中按需 reinterpret。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// SwapAB MoE GEMM 原理

// 传统 MoE GEMM: C[m, n] = A[m, k] × B[k, n]
//   m = token_count (小 batch 时如 64), 需要 padding 到 64 对齐 → 浪费
//   k = expert_hidden_dim (e.g., 6144)
//   n = expert_intermediate_dim (e.g., 2048)

// SwapAB: C'[n, m] = B_T[n, k] × A_T[k, m]
//   B_T: [n, k] = transpose(expert_weights)  → 左矩阵
//   A_T: [k, m] = transpose(activations)     → 右矩阵
//   n 维度对齐粒度为 8 (vs m 维度的 64) → padding overhead 低得多

// 伪代码:
// 输入:
//   activations: [m, k] (BF16/FP8)
//   weights: [k, n] (BF16/FP8)

// 内存中 reinterpret (无物理转置):
B_T = reinterpret_as([n, k], weights)    // 形状变化，无数据拷贝
A_T = reinterpret_as([k, m], activations) // 形状变化，无数据拷贝

// Tensor Core GEMM: C' = B_T × A_T
C_T = tiled_gemm(B_T, A_T)  // [n, m]

// 内存中 reinterpret 回原始形状:
output = reinterpret_as([m, n], C_T)     // 形状变化，无数据拷贝
```

Swapping 的效果：当 m=64, n=2048 时，传统方法 M 维度无 padding 但需 exact 64；当 m=63 时需 padding 1 个 token（1.6% overhead）；当 m=32 时需 padding 32 个 token（50% overhead）。SwapAB 使 padding 粒度从 M 的 64 降到 N 的 8，大幅减少 small-batch 下的计算浪费。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
1. 基于 DeepGEMM (https://github.com/deepseek-ai/DeepGEMM) 修改实现。DeepGEMM 设计为 right-hand B matrix 为 weight，SwapAB 反转此约定。
2. 内存 reinterpret 而非物理转置：Python/PyTorch 层面用 `view()` / `as_strided()` 改变形状，GPU kernel 内用指针偏移访问。
3. 适用场景：MoE decoding 阶段（small token count per expert per GPU）。Prefilling 阶段（large token count）M 维度足够大，padding overhead 占比小，SwapAB 的收益递减。
4. 与 quantization 协同：LongCat-Flash 使用 FP8 block-wise quantization (activations [1,128], weights [128,128])，SwapAB 不影响量化方案——quantization/dequantization 发生在 GEMM 之前/之后，与矩阵维度无关。

涉及论文标题：
- LongCat-Flash Technical Report
