## Grouped GEMM for MoE Inference (面向MoE推理的分组矩阵乘法)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Grouped GEMM 是 NVIDIA cuBLAS 12.5 引入的 API（`cublasGemmGroupedBatchedEx`），允许在单次 kernel launch 中执行多个不同形状、不同转置、不同缩放因子的矩阵乘法。与传统的 batched GEMM（要求所有矩阵具有相同的尺寸和转置参数）不同，Grouped GEMM 专为 MoE 推理场景设计：每个 expert 接收不同数量的 token（不同 M 维度），但所有 expert 共享相同的权重矩阵形状（K×N）。使用 Grouped GEMM 可将所有 expert 的计算批量化为一个 kernel launch，避免逐 expert 循环带来的 kernel launch overhead 和 GPU 利用率低下。在 MoE decode 阶段（batch size 8-64, FP16），Grouped GEMM 相比 naive batched GEMM 循环可达到约 1.2× 加速。

从kernel调度角度拆解术语：
Grouped GEMM 在 MoE layer decode 中的执行流程：
```
输入: B个token, N个expert, 每expert权重W_e[K,N]
Router输出: token i → experts S_i, gate weights w_i

// Step 1: Token-to-Expert Dispatch
for each token i:
    for each expert e in S_i:
        dispatch token i → expert e (记录M_e++)

// Step 2: 构造 Grouped GEMM 参数
group_count = |{e : M_e > 0}|  // 有token的expert数
for each active expert e:
    A_desc[e] = {tokens_e, K}   // M_e × K (变长)
    B_desc[e] = W_e             // K × N (固定)
    C_desc[e] = output buffer   // M_e × N

// Step 3: 单次kernel launch执行所有expert的GEMM
cublasGemmGroupedBatchedEx(
    handle, &A_desc, &B_desc, &C_desc,
    group_count, ...)

// 等价于并行执行:
// for e in active_experts:  (并行, 单kernel)
//     C_e = tokens_e @ W_e   // [M_e, K] × [K, N]
```
Grouped GEMM 当前仅使用 warp-level MMA 指令（未使用 wgmma），但因减少了 kernel launch overhead 和提高了 GPU SM 占用率，实际性能优于逐 expert 调用 batched GEMM。在 memory-bound decode 下，T（唯一激活 expert 数）= group_count，每个 expert 的 GEMM 仍需其权重从 HBM→SRAM 加载。因此即使 Grouped GEMM 优化了计算调度，延迟仍与 T 成正比。

术语一般如何实现？如何使用？
- 在 cuBLAS 12.5+ 中通过 `cublasGemmGroupedBatchedEx` 使用，支持 FP16/BF16/FP32/FP64。
- 在 PyTorch 中可通过 `torch._C._cuda_grouped_gemm` 或自定义 CUDA 扩展调用。
- SGLang/vLLM 等 serving 框架在 MoE layer 实现中集成 Grouped GEMM 或 DeepGEMM（DeepSeek 的专用 fused MoE kernel）。
- 限制：Grouped GEMM 不改变每个 expert 的权重仍需从 HBM 加载的事实。在 memory-bound decode 下，优化的重点仍是减少 T（如 OEA），而非 Grouped GEMM 本身。

涉及论文标题：
- Opportunistic Expert Activation: Batch-Aware Expert Routing for Faster Decode Without Retraining
