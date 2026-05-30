## CPU Attention Kernel with AVX Intrinsics (for MoE Offloading)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CPU Attention Kernel with AVX Intrinsics 是 MoE-GEN 实现的高性能 CPU 端 self-attention 计算内核，用于将 MoE 推理中的 attention mechanism（$QK^T$ 和 score-V 乘法）卸载到 CPU 执行。该 kernel 使用 Intel AVX（Advanced Vector Extensions）SIMD 指令集实现 Grouped Query Attention（GQA），采用 BF16 数据格式。关键设计点：(1) BF16 数据在 FP32 中表示，显式清零低 16 位尾数，所有计算和累加在 FP32 精度，每次点积累加后按 BF16 舍入规则舍入，保证与 PyTorch GPU attention 数值一致；(2) 优化 CPU cache 局部性，类似 FlashAttention CPU 版的设计思想；(3) 针对 GEMV（matrix-vector multiplication，解码阶段 attention 的算术特征）的算术强度进行优化，使 CPU 处理速度达到与 PCIe 4.0 传输 KV-cache + GPU 计算时间的可比水平。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CPU Attention Kernel 的伪代码（Grouped Query Attention, BF16 with AVX）：

```
// BF16 数据表示: 使用 FP32，尾数低16位置零
function bf16_mul_add(A_bf16[], B_bf16[], accum_fp32):
    // 加载 BF16 数据到 AVX 寄存器（256-bit YMM = 8×FP32）
    for i in 0..len step 8:  // AVX 一次处理 8 个 FP32
        a = _mm256_loadu_ps(&A_bf16[i])   // BF16 as FP32 (low 16 bits zero)
        b = _mm256_loadu_ps(&B_bf16[i])   
        accum_fp32 = _mm256_fmadd_ps(a, b, accum_fp32)  // FMA: a*b + accum
    
    // 舍入到 BF16: 保留高16位，低16位清零
    result_bf16 = accum_fp32 & 0xFFFF0000

// GQA Self-Attention（解码阶段，单 token，CPU path）:
function cpu_attention_gqa(Q, K_cache, V_cache, num_kv_heads, num_q_heads):
    // Q:    [num_q_heads, head_dim]    — 当前 token 的 single query
    // K_cache: [seq_len, num_kv_heads, head_dim]  — 历史 KV-cache
    // V_cache: [seq_len, num_kv_heads, head_dim]
    
    heads_per_kv = num_q_heads / num_kv_heads
    
    for g in 0..num_kv_heads:  // 每组 GQA group
        // Step 1: QK^T — score computation (GEMV)
        scores[0..seq_len] = 0
        for h in 0..heads_per_kv:
            q_head = Q[g*heads_per_kv + h]  // [head_dim]
            for pos in 0..seq_len:
                // AVX dot product: q_head · K_cache[pos, g]
                scores[pos] += bf16_mul_add(q_head, K_cache[pos,g], scores[pos])
        scores /= sqrt(head_dim)  // scaling
        
        // Step 2: Softmax
        max_score = max(scores)
        exp_sum = 0
        for pos in 0..seq_len:
            scores[pos] = exp(scores[pos] - max_score)
            exp_sum += scores[pos]
        for pos in 0..seq_len:
            scores[pos] /= exp_sum
        
        // Step 3: Score-V multiplication (GEMV)
        for h in 0..heads_per_kv:
            output[g*heads_per_kv + h] = 0  // [head_dim]
            for pos in 0..seq_len:
                // AVX dot-add: output += scores[pos] * V_cache[pos, g]
                output[g*heads_per_kv + h] = bf16_mul_add(
                    scores[pos], V_cache[pos,g], output[g*heads_per_kv + h])
    
    return output  // [num_q_heads, head_dim]
```

关键调度特性：
- **数据局部性**：KV-cache 在 host memory 中连续布局，CPU 顺序访问，L2/L3 cache 命中率高。
- **零拷贝**：CPU kernel 直接读取 host memory 中的 KV-cache，无需 PCIe HtoD copy。
- **与 GPU 并行**：GPU 处理 $(1-\omega) \cdot b_a$ 个 token 的 attention（需等 HtoD KV-cache copy），CPU 同时处理 $\omega \cdot b_a$ 个 token（无需等待 copy）。结果在 Post-Attention 阶段 concatenate。
- **Overlap 收益**：CPU attention 节省的 HtoD 带宽被 expert weight prefetch 利用，减少 GPU idle time。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实际实现和使用：
- **MoE-GEN 实现**：C++ 编写，AVX intrinsics（`_mm256_*` 指令），BF16 格式。约 3000 行 C++ 后端代码的一部分。当前仅支持 greedy decoding。
- **类似工作**：
  - **PowerInfer**：使用 CPU 执行部分 attention 计算，利用 consumer GPU + CPU 混合推理。
  - **Fiddler**：CPU-GPU orchestration for MoE，CPU 执行 attention/experts 以缓解 I/O 瓶颈。
  - **KTransformers**：Intel AMX（Advanced Matrix Extensions）CPU kernel，在 Sapphire Rapids 及以上 Xeon 上实现 21.3 TFLOPS（3.9× PyTorch native），相比 AVX 支持的 BF16 更高效。
  - **llama.cpp**：通过 BLAS backend 支持 CPU attention，但未针对 MoE 解码的 GEMV 特性优化 cache 行为。
- **何时使用**：CPU attention kernel 适用于 memory-bound 的 MoE offloading 场景（PCIe 带宽饱和时）。MoE-GEN 的 search procedure 自动决定最优 $\omega$：若 CPU kernel 执行时间 < GPU attention + KV-cache HtoD copy 时间，则 $\omega > 0$；若 CPU 计算能力弱（如 C3 的 16-Core CPU），$\omega$ 调低或为零。
- **数值精度**：BF16 在 FP32 中模拟（低 16 位清零），保证与 GPU BF16 attention 数值一致性，无需特殊 CPU 硬件支持（兼容旧 CPU）。

涉及论文标题：
- MoE-Gen: High-Throughput MoE Inference on a Single GPU with Module-Based Batching
- MoE-Lens: Towards the Hardware Limit of High-Throughput MoE LLM Serving Under Resource Constraints

**MoE-Lens 补充**：MoE-Lens 同样实现了手工优化的 CPU decode attention kernel，使用 **AVX512 SIMD intrinsics**（512-bit ZMM registers，一次处理 16 个 BF16 元素，upconvert 到 FP32 后为 8 elements per register）。相比 MoE-GEN 的 AVX（256-bit YMM），AVX512 提供双倍寄存器宽度。MoE-Lens 的优化包括：(1) manual vectorization 替代编译器自动向量化；(2) loop unrolling 减少分支和循环开销；(3) data prefetching 指令提前将下一轮 KV cache 数据加载到 CPU cache；(4) BF16→FP32 upconvert 和 FP32→BF16 rounding 每一步显式处理。在 Intel Platinum 8380 CPU 上，单线程 throughput 是 auto-vectorized baseline 的 4.7×，全线程为 3.1×（>20 threads 后因 memory controller contention 饱和）。该 kernel 在 VSLPipe 的 CPU Task (C) 阶段执行，与 GPU Task B 的 GEMM 并行。Kernel 的 throughput requirement 来自 Equation 6：$T_{CPU} = 2 \cdot s \cdot I_{CPU\_attn} \cdot B_{KV}$，需达到数百 GFLOPs 以满足 system target（当 KV cache = 2× model size 时）。
