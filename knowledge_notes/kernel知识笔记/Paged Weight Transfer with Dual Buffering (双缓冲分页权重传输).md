## Paged Weight Transfer with Dual Buffering (双缓冲分页权重传输)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Paged Weight Transfer with Dual Buffering 是 MoE-Lightning 提出的 GPU memory 受限场景下的权重传输机制。将 MoE 每层所有 experts FFN weights 从 CPU memory 分页传输到 GPU，利用双缓冲（2 × per-layer-weight-buffer-size）重叠当前层计算与下一层权重预取。传输采用两阶段流水线：CPU DRAM → CPU pinned memory (memcpy) → GPU HBM (cudaMemcpyAsync)，连续 pages 的 Stage 1 和 Stage 2 重叠执行。GPU expert FFN kernel 通过 page table（映射 expert_id × page_id → GPU buffer offset）访问正确的 weight pages。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Paged Weight Transfer 与 GPU kernel 的交互流程：
```
// 初始化：分配两个 GPU weight buffers
buf_A = cudaMalloc(sizeof_per_layer_weights)
buf_B = cudaMalloc(sizeof_per_layer_weights)

// 主循环 (CGOPipe 中每一层)
for i = 1 to num_layers:
    for j = 1 to num_micro_batches:
        // Page j of layer i weights: pinned→GPU (cudaMemcpyAsync on stream_w)
        cudaMemcpyAsync(buf_A + page_offset[j], pinned_weights[i][j], 
                        page_size, HtoD, stream_w)
        
        // PostAttn(i, j): GPU kernel accesses weights via page table
        // page_table[expert_id][j] → GPU address in buf_A
        post_attn_kernel<<<..., stream_c>>>(
            hidden_states, page_table, buf_A)
        
        // Concurrently: page j+1 of layer i+1 weights: CPU→pinned
        memcpy(pinned_weights[i+1][j+1], cpu_weights[i+1][j+1], page_size)
        
        // Swap buffers for next layer
        swap(buf_A, buf_B)

// GPU Expert FFN kernel 内部:
__global__ void moe_ffn_kernel(hidden, page_table, weight_buffer):
    expert_id = gate_routing(token)
    page_id = micro_batch_id
    weight_ptr = page_table[expert_id][page_id]  // lookup
    // GEMM using weight_ptr...
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：MoE-Lightning Memory Manager (Appendix A.1)。(1) 使用 CUDA streams 分离计算和传输；(2) cudaMemcpyAsync 用于异步 H2D；(3) CUDA events 用于 stream synchronization；(4) Page table 实现为 GPU 端的 simple lookup array。
- 优势：消除整层 weights 一次性传输导致的后续微批次 H2D 阻塞（FlexGen 的主要问题）。在 GPU memory 极受限时（如 T4 16GB running Mixtral 8x7B），paged transfer 是维持 GPU utilization 的关键。
- 参数：分页数 n_pages = num_micro_batches = N/μ，由 HRM policy optimizer 确定。

涉及论文标题：
- MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs
