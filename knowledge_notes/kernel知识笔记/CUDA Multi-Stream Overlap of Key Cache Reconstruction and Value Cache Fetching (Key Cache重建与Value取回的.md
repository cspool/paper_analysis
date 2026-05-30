## CUDA Multi-Stream Overlap of Key Cache Reconstruction and Value Cache Fetching (Key Cache重建与Value取回的多Stream重叠)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

CUDA Multi-Stream Overlap 是 ShadowKV 系统中用于隐藏 PCIe 数据传输延迟的 kernel 调度策略。在 decoding 的每个 step，系统需要：(a) 从低秩投影重建选中的 sparse key cache（GPU 计算密集，Tensor Core GEMM），(b) 从 CPU pinned memory 取回对应的 value cache（PCIe H2D 传输，带宽受限）。由于 GPU 计算引擎和 PCIe DMA 引擎是独立的硬件单元，通过 CUDA multiple streams 将两者并行执行，使 net latency = max(T_compute, T_transfer) 而非 T_compute + T_transfer。

从kernel调度角度拆解术语——伪代码和具体计算过程：

```
// CUDA Multi-Stream Overlap 伪代码
cudaStream_t stream_compute, stream_transfer;
cudaStreamCreate(&stream_compute);
cudaStreamCreate(&stream_transfer);

// Step 1: 在 default stream 上计算 chunk selection
AttentionApproxKernel<<<grid, block>>>(Q, L, I_out, scores);
TopKKernel<<<grid, block>>>(scores, I);  // I: [b, h_kv, k]

// Step 2: 并行执行 key 重建和 value 取回
// Stream 1 (GPU compute): 低秩 key 重建
cudaMemcpyAsync(d_K_selected, d_A, gather_indices, 
                cudaMemcpyDeviceToDevice, stream_compute);
GEMMKernel<<<grid, block, 0, stream_compute>>>(
    d_A_selected, d_B, d_K_sparse);  // [k*c, d] = [k*c, r] x [r, d]
cudaEventRecord(event_compute_done, stream_compute);

// Stream 2 (PCIe H2D): value 取回
cudaMemcpyAsync(d_V_sparse, h_V_CPU + offset, k*c*d*sizeof(half),
                cudaMemcpyHostToDevice, stream_transfer);
cudaEventRecord(event_transfer_done, stream_transfer);

// Step 3: 同步两个 stream 后执行 attention
cudaStreamWaitEvent(default_stream, event_compute_done, 0);
cudaStreamWaitEvent(default_stream, event_transfer_done, 0);
FlashAttentionKernel<<<grid, block>>>(
    d_Q, d_K_combined, d_V_combined, d_O);
```

**时间线图**：

```
Time →
Default Stream:  [QxL Approx][TopK][── wait ──][FlashAttn on sparse KV]
Stream Compute:                       [Gather A][GEMM A_sel @ B = K_sparse]
Stream Transfer:                      [PCIe H2D: V_CPU → V_sparse on GPU]
                                      ├────────── 重叠执行 ──────────┤
Net latency = max(T_GEMM, T_PCIe) ≈ T_PCIe (1.84ms) for 48×64K
vs. Naive sequential: T_GEMM(1.25ms) + T_PCIe(1.84ms) = 3.09ms
```

术语一般如何实现？如何使用？

实现依赖 CUDA Runtime API 的 stream 管理和 event 同步机制。Key 点：(1) `cudaMemcpyAsync` 必须是 Host-to-Device 方向且源为 pinned memory（`cudaMallocHost` 分配），否则无法与 kernel 执行重叠；(2) 使用 `cudaEventRecord` + `cudaStreamWaitEvent` 实现跨 stream 同步，而非全局 `cudaDeviceSynchronize`；(3) 重叠效果受限于 GPU 的 copy engine 数量（A100 有 1 个 H2D copy engine），若多个 stream 同时做 H2D 则串行化。

ShadowKV 中该策略与 temporal locality cache 结合：仅对 cache miss 的 chunk 执行重建和取回（~40%），进一步减少重叠的总工作量。在 48×64K 配置下，Fetch V 延迟 1.84ms 为瓶颈，Recompute K 1.25ms 完全被掩盖。

涉及论文标题：
- ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference

---
