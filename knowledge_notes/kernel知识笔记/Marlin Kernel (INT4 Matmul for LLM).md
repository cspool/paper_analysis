## Marlin Kernel (INT4 Matmul for LLM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Marlin (Frantar et al., 2024, arXiv:2408.11743) 是一种针对 LLM 推理中 INT4 量化权重矩阵乘法的 GPU kernel，专门为 FP16×INT4 的混合精度矩阵乘法优化。与传统的 INT4 推理 kernel（如 GPTQ、AWQ 的 GEMM kernel）相比，Marlin 的核心优势是：(a) 使用 group-wise scaling（group size=128）平衡精度与性能；(b) 利用 GPU Tensor Core 的 mma.sp 指令加速 INT4 计算；(c) 优化的 warp 调度减少 bank conflict。

在 ReSA 中，Marlin 被用于 INT4 精度下的 end-to-end 推理评测（Section 3.5.2）——配合 ReSA 的 sparse attention kernel，256K context 下实现 2.44× end-to-end speedup（vs FP16 dense 的 2.28×）。

从kernel调度角度拆解术语：

```
// Marlin INT4 Matmul: Y = X (FP16) × W (INT4, group-wise scaled)
// X: [M, K] FP16, W: [K, N] INT4, scales: [K/group_size, N] FP16

// Kernel 执行流程:
for each tile (m_tile, n_tile) in output:
    acc[m_tile, n_tile] = 0  // FP32 accumulator
    for k in 0..K step group_size:
        // 1. Load INT4 weight tile + dequant via scale
        w_int4 ← load(W[k:k+group_size, n_tile])    // INT4
        scale ← load(scales[k/group_size, n_tile])   // FP16
        w_fp16 ← dequant(w_int4, scale)              // INT4 → FP16
        
        // 2. Load FP16 activation tile
        x_fp16 ← load(X[m_tile, k:k+group_size])
        
        // 3. Tensor Core mma (FP16 accumulation)
        acc += x_fp16 @ w_fp16  // via mma.sync on Tensor Cores
    
    Y[m_tile, n_tile] = acc  // FP16 output
```

术语一般如何实现？如何使用？

开源：https://github.com/IST-DASLab/marlin。API：`marlin::mul(X_fp16, W_int4, scales, ...)`，直接替代 PyTorch 的 F.linear。group_size=128 是常用配置（论文默认）。与 AWQ kernel 的关系：Marlin 在 AWQ 的 weight-only quantization 基础上做更激进的 warp-level 优化，通常比 AWQ kernel 快 20-30%。ReSA 使用 Marlin 评估 INT4 下的 end-to-end speedup。

涉及论文标题：
- Rectified Sparse Attention
