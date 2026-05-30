## CPU GQA Attention Kernel for MoE Decode Offloading (MoE Decode卸载的CPU GQA Attention Kernel)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CPU GQA (Grouped Query Attention) Kernel 是 MoE-Lightning 在 CPU 端实现的 decode 阶段 attention 计算 kernel，基于 Intel MKL 库。其核心设计动机来自 HRM 分析：在 GPU 内存受限场景下，decode attention 的 operational intensity 极低（< 1 FLOP/Byte，属于 GEMV 模式），低于 HRM 的 P1 turning point 对应的 critical intensity——因此将 KV cache 从 CPU H2D 传输到 GPU 再做 attention 是不划算的（KV cache transfer 时间 > attention 计算时间）。替代方案：直接在 CPU 上执行 attention（利用 CPU DRAM 高带宽和 MKL 加速），仅将 attention 结果的 hidden states（远小于 KV cache）H2D 传输到 GPU 用于后续 O projection 和 MoE FFN。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CPU GQA Kernel 计算流程（伪代码）：
```
Input: Q [μ, n_q, d] in CPU pinned memory
       K_cache [s, n_kv, d] in CPU memory
       V_cache [s, n_kv, d] in CPU memory
       GQA_group_size = n_q / n_kv
Output: AttnOut [μ, n_q, d] in CPU pinned memory

for each batch in range(μ):
    for each kv_head in range(n_kv):
        // Step 1: QK dot product (MKL SGEMM)
        // Q_group [GQA_group_size, d] × K[kv_head]^T [d, s]
        scores = MKL_SGEMM(Q[batch, kv_head*gs:(kv_head+1)*gs, :],
                           K_cache[:, kv_head, :].T)  // [gs, s]
        
        // Step 2: Softmax over sequence dimension
        scores = softmax(scores / sqrt(d), dim=-1)  // vectorized
        
        // Step 3: Attention-weighted value sum (MKL SGEMM)
        // scores [gs, s] × V_cache[:, kv_head, :]
        AttnOut[batch, kv_head*gs:(kv_head+1)*gs, :] = 
            MKL_SGEMM(scores, V_cache[:, kv_head, :])  // [gs, d]
```
关键优化：(1) 利用 MKL batch GEMM 批量处理多个 attention heads；(2) GQA 共享 KV heads 减少 MKL 调用次数（n_kv < n_q）；(3) 使用 CPU pinned memory 存放 Q 和 output（与 GPU 共享地址空间，便于 H2D/D2H 直接传输）；(4) AttnOut 作为 PostAttn 的 input 通过 cudaMemcpyAsync H2D 回 GPU。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：MoE-Lightning 中为 PyTorch C++ extension，调用 Intel MKL SGEMM（单精度通用矩阵乘法）。编译时链接 libmkl_rt.so。
- 性能：CPU GQA kernel 比 KV cache H2D 到 GPU 快 3-4×（Fig. 9），接近 CPU BW (~200 GB/s) / PCIe BW (~50 GB/s) ≈ 4× 的理论比值。
- 瓶颈：当微批次 μ 和 context length s 增大时，CPU attention 可能成为 bottleneck（需要更多 CPU DRAM BW 和 compute），此时需要更高 CPU scaling ratio 或考虑 GPU attention（A_g=1）。
- 与 MoE-Lens 的 AVX512 intrinsics kernel 对比：MoE-Lightning 使用 MKL（高层库），MoE-Lens 使用手工 AVX512 SIMD + loop unrolling + prefetching（底层优化）。

涉及论文标题：
- MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs
