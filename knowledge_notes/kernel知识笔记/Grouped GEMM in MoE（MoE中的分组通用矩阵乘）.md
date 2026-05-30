## Grouped GEMM in MoE（MoE中的分组通用矩阵乘）

术语是什么？
Grouped GEMM 是 MoE 推理中 expert 计算的核心 kernel 操作。不同于标准批处理 GEMM（所有 token 用同一权重），Grouped GEMM 将不同 expert 的 token 分组，每组用不同权重矩阵执行矩阵乘法。在 CUDA 层面通常通过 CUTLASS grouped GEMM 或 cuBLAS 实现，支持多个独立 GEMM 在单次 kernel launch 中批量执行。

从kernel调度角度拆解术语：
Grouped GEMM 的 kernel 调度与 GEMM efficiency η_g 的关系：
```
// Rank r 上执行 expert e 的 Grouped GEMM
输入: tokens_e ∈ ℝ^{n_e × H}, weights W_e^{gate,up,down}

// SwiGLU FFN 的 3 次 GEMM:
h_gate = tokens_e @ W_e^gate    // GEMM 1: (n_e, H) @ (H, I)
h_up   = tokens_e @ W_e^up      // GEMM 2: (n_e, H) @ (H, I)  
h_act  = SiLU(h_gate) ⊙ h_up   // element-wise
output = h_act @ W_e^down       // GEMM 3: (n_e, I) @ (I, H)

// 性能受 η_g(n_e) 影响:
η_g(n_e) ∝ n_e  // 当 n_e 大时接近 peak FLOPS
η_g(n_e) ≪ 1   // 当 n_e 小时受 padding + low intensity 影响
```
PROBE 分析指出 DP (Data Parallelism) 的 fragmentation penalty 源于每个 replica 处理的 per-expert token 过少，GEMM 效率 η_g 极低。EP 通过聚合全局 token 维持高 η_g，但引入 load imbalance。Expert replication 在两者间 trade off。

术语一般如何实现？如何使用？
CUTLASS 3.x grouped GEMM (支持 Hopper SM90)、cuBLAS grouped GEMM API、Triton 自定义 grouped matmul kernel。在 vLLM/SGLang 中通过 `torch.bmm` 或框架封装的 fused MoE kernel 调用。关键优化：tile size 对齐 Tensor Core 的 M/N/K 维度和 expert 级 padding 策略。

"Who Says Elephants Can't Run" 使用的 CUTLASS Grouped GEMM 方法：在 token routing 完成后（CUB radix sort + permute），为每个 expert 构造子矩阵指针（start offset + token count），将 (sub_matrix_ptr, weight_ptr, bias_ptr) 组队传入 CUTLASS grouped GEMM，单次 kernel launch 并行执行所有 expert 的矩阵乘法。关键特点是 fused dequantize 在 GEMM weight load 阶段进行，对 V100 (Volta, SM70) 优化，使用 FP16 bit-trick 替代原生 I2F 指令加速 int→FP16 转换。

涉及论文标题：
- PROBE: Co-Balancing Computation and Communication in MoE Inference via Real-Time Predictive Prefetching
- Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production
