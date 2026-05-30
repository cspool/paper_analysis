## Compact Asynchronous Transfer (SIMD Multi-threaded CPU-to-GPU Data Transfer)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Compact Asynchronous Transfer 是 FloE 提出的高效 CPU-to-GPU 数据传输机制，用于将 DRAM 中压缩后的 expert 权重高效传输到 GPU VRAM。包含三个协同优化：(1) **Compact Weights Layout**：在 DRAM 中将 gate projection 的列和 down projection 的对应行（转置为列）co-locate 到连续内存，chunk 大小从 d_hidden×num_bytes 增加到 2×d_hidden×num_bytes，减少内存碎片和 DMA 请求数量；(2) **AVX-512 SIMD 打包**：CPU 端使用 AVX-512 指令集并行处理多个权重 group 的打包（解量化+拷贝到 pinned memory），利用 512-bit 寄存器一次处理 16 个 FP32 或 32 个 FP16 元素；(3) **Multi-threaded + Multi-stream 异步传输**：多 CPU 线程并行打包不同 expert 的权重到 pinned memory，使用多个 CUDA stream 异步发起 cudaMemcpyAsync 传输请求，最小化 PCIe 总线空闲时间。在 RTX 3090 + PCIe 4.0 上，达到峰值带宽的 88%，比 PyTorch 原生实现快 12.6×。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Compact Asynchronous Transfer 流程
// DRAM 布局 (Co-located Gate column + Down row):
//   Expert E_ij 在 DRAM 中按 chunk 组织:
//     Chunk_k = [W_gate[:, k] | W_down^T[:, k]]  // 2 * d_hidden * num_bytes
//     (仅 mask[k]==True 的列才被组织为 chunk)
//   每个 chunk 在 DRAM 中连续, 大小 = 2 * 4096 * 2 = 16KB (FP16)

// CPU 端多线程打包 (per thread):
function pack_and_transfer(expert_indices, mask, DRAM_base, pin_buffers, streams):
    for each thread t in parallel:
        thread_chunks = partition(chunks_of_expert, t, num_threads)
        for chunk_idx in thread_chunks:
            // AVX-512 SIMD 拷贝 gate 列和 down 列到 pinned memory
            src_gate = DRAM_base + chunk_idx * chunk_size
            src_down = DRAM_base + chunk_idx * chunk_size + d_hidden * num_bytes
            dst = pin_buffers[t] + offset
            
            // AVX-512: 每次 512-bit (64 bytes) = 32 × FP16
            for i in 0..(chunk_size/64):
                _mm512_store_si512(dst + i*64, _mm512_load_si512(src_gate + i*64))
            for i in 0..(chunk_size/64):
                _mm512_store_si512(dst + chunk_size/2 + i*64, 
                                   _mm512_load_si512(src_down + i*64))
            offset += chunk_size
        
        // 异步传输该线程的 pinned buffer 到 GPU
        cudaMemcpyAsync(GPU_buf + thread_offset, pin_buffers[t],
                        thread_chunks_size, cudaMemcpyHostToDevice, streams[t])

// 传输延迟分析 (FloE Figure 7):
// Chunk size 1:  高延迟 (~1.2ms) — 大量小 DMA 请求, API/cudaLaunch overhead
// Chunk size 50: 最低延迟 (~0.37ms) — API overhead 与 DRAM 打包时间平衡
// Chunk size 200: 高延迟 (~0.52ms) — DRAM 打包时间超过传输重叠收益
// 最优 chunk size = 50 (在 FloE 的硬件配置下)

// PyTorch 原生实现:
//   for col in selected_cols:
//       gate_col = W_gate[:, col].contiguous()  // 多次小内存拷贝
//       down_col = W_down[col, :].contiguous()
//       cudaMemcpyAsync(GPU_gate + offset, gate_col, ...)
//       cudaMemcpyAsync(GPU_down + offset, down_col, ...)
//   → 大量非连续小传输, ~7% PCIe 带宽利用率
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- AVX-512: Intel CPU (Skylake-X 及更新) 支持，512-bit 向量寄存器，一条指令处理 16×FP32 或 32×FP16。在 FloE 的 64 核 CPU 上，多线程 + AVX-512 提供足够 CPU 吞吐来喂饱 PCIe 4.0 带宽
- Pinned (page-locked) memory: 使用 `cudaHostAlloc()` 或 `torch.tensor.pin_memory()` 分配，GPU DMA engine 可直接访问，无需 CPU staging。pinned memory 大小需谨慎——过大会挤占 OS 可用内存
- 多 CUDA stream: 每个 stream 维护独立的命令队列和内存拷贝引擎，允许 PCIe 传输与 GPU kernel 计算重叠。FloE 使用多 stream 使得不同 expert 的传输可并行
- 紧凑布局的 trade-off: co-locate gate 列和 down 列需要 DRAM 中重新组织权重——这可在模型加载时一次性完成（offline），不增加推理运行时开销
- 自 PyTorch 2.0+ 起，`torch.compile` 和 CUDA graphs 可以部分实现类似的传输优化，但 FloE 的手动 AVX-512 实现仍显著优于 PyTorch 原生

涉及论文标题：
- FloE: On-the-Fly MoE Inference on Memory-constrained GPU
