## Fused Online Rotations (for FP4 Quantization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fused Online Rotations 是 QuTLASS 中的一种 GPU kernel 优化技术，将激活端的 block-wise 旋转（如 Hadamard）与后续的 FP4 量化操作融合为单个 kernel，消除中间 DRAM 写入。核心原理：对 k < 256 的 block diagonal 变换，dense 矩阵乘法为 memory-bound（arithmetic intensity 极低），因此旋转本身跟 memory copy 几乎同开销。通过将量化（absmax/MSE scale calculation + E2M1 RTN）作为 custom epilogue function 直接融合进旋转 kernel（利用 CUTLASS epilogue fusion），旋转输出不写回 DRAM，直接生成 FP4 packed 格式和 scales。这使得 MR-GPTQ 的 "micro-rotation" 组件（激活端在线 Hadamard 旋转）几乎零额外开销——与标准 FP4 量化（无旋转）的延迟差异在测量噪声范围内。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// 传统方法（未融合，3 次 kernel launch + 2 次 DRAM 写入）:
// Kernel 1: X_rot = X @ H_k  → write X_rot to DRAM
// Kernel 2: s_G = absmax(X_rot per group) → write s_G to DRAM
// Kernel 3: X_q = E2M1_quantize(X_rot / s_G) → write X_q to DRAM

// Fused 方法（QuTLASS，1 次 kernel launch + 0 次 DRAM 写入）:
__global__ void fused_rotate_quantize(
    half* X, half* H_k,         // inputs from DRAM
    uint8_t* X_q_out,           // output: FP4 packed (direct to DRAM)
    uint8_t* scales_out         // output: E8M0/E4M3 scales (direct to DRAM)
) {
    // All intermediate results stay in registers / shared memory
    
    // Phase 1: Block rotation (memory-bound, k<256)
    // Thread block loads X tile and H_k tile from DRAM
    // Computes X_rot = X_tile @ H_k_tile in shared memory
    
    // Phase 2: Fused epilogue — quantization (no intermediate DRAM write)
    // Directly from X_rot in shared memory:
    //   s_G = absmax(X_rot per group of k elements)
    //   s_G_q = quantize_scale(s_G)  // E8M0 or E4M3
    //   x_norm = X_rot / s_G_q
    //   x_q = RTN_E2M1(x_norm)      // FP4 E2M1
    //   pack 2×4-bit into 1 byte
    
    // Write packed X_q and scales to DRAM (single write each)
}

// 关键参数: k < 256 → arithmetic intensity 极低
// 旋转 cost ≈ 加载 H_k 的 memory cost (not compute-bound)
// → 任意旋转矩阵（不限于 Hadamard）几乎同成本
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
基于 NVIDIA CUTLASS 的 epilogue fusion 机制实现：主循环执行 tile-based GEMM（X @ H_k），epilogue 直接调用量化函数处理输出 tile（不写回 global memory）。模板设计允许自定义 epilogue（MSE scale 优化、Abs-Max 量化等）。代码位于 QuTLASS（https://github.com/IST-DASLab/qutlass）。B200 实测效果："Actual"（含全部开销）与 "Ideal"（纯 matmul）的差距在 MXFP4 上仅 ~10%（3.6× vs 4× ideal），证明 fused online rotation 开销极小。

涉及论文标题：
- Bridging the Gap Between Promise and Performance for FP4 Quantization

---
