## Recompute-Decode Fused Kernel (K1+K2 Fusion)

术语是什么？通过联网搜索让回答具体和精准。

Recompute-Decode Fused Kernel（K1+K2融合kernel）是eLLM系统在layer-level execution中提出的CUDA kernel优化技术。它将两个原本串行执行的kernel融合为单一kernel launch：(1) K1——Recomputation kernel，为layer i+1的uncached旧token执行KV投影（GEMM操作，compute-intensive）；(2) K2——Decode kernel，用layer i的完整历史KV（cached+recomputed）对当前新token执行decode attention（memory-intensive）。融合后减少kernel launch overhead，提高SM utilization，并使两类操作的计算资源可以在thread block级别动态分配。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

K1+K2 Fused Kernel的调度伪代码（以Llama2-13B MHA decode为例）：

```
// === Pre-compilation ===
// 预编译多组.so: 32种thread配置 (32..1024 step 32)
for num_threads in range(32, 1024+1, 32):
    compile_fused_variant(num_threads)
    // 每个variant包含K1和K2子区域，thread block数按计算量比分

// === Runtime per-layer execution ===
function fused_k1_k2_launch(layer_i, batch_requests, b, r):
    // 计算K1和K2的工作量
    num_uncached_tokens = sum(req.seq_len * req.r for req in batch_requests)
    // K1 FLOPs: GEMM QKV projection for old tokens
    flops_K1 = 2 * hidden_size * head_dim * num_heads * num_uncached_tokens
    // K2 FLOPs: attention decode for current token  
    flops_K2 = hidden_size * (num_cached_tokens + num_uncached_tokens)
    
    // 计算FLOP ratio
    ratio = flops_K1 / (flops_K1 + flops_K2)  // e.g. 0.7 → K1占70%
    
    // 选择最接近的预编译variant
    target_threads = int(ratio * max_threads)  // e.g. 0.7*1024 = 717
    target_threads = round_to_multiple(target_threads, 32)  // 对齐warp
    fused_kernel = load_library(target_threads)
    
    // Stream A: 异步host↔GPU KV传输
    cudaStream_t stream_A = streams["transfer"]
    for token in uncached_tokens:
        cudaMemcpyAsync(gpu_kv_buf, host_kv_buf, ..., stream_A)
    
    // Stream B: fused K1+K2在单一kernel launch中执行
    cudaStream_t stream_B = streams["compute"]
    fused_kernel<<<grid, block, 0, stream_B>>>(
        // K1子kernel参数
        K1_Q, K1_K, K1_V,        // old token QKV (query投影可选)
        K1_uncached_token_ids,    // 需要重算的token indices
        K1_output_KV,             // 输出：重算的KV (存入临时workspace)
        
        // K2子kernel参数  
        K2_Q_current,             // 当前新token的query
        K2_KV_cached,             // 已缓存的KV (从GPU显存)
        K2_KV_recomputed,         // K1刚产生的KV (从临时workspace)
        K2_output,                // 输出：当前token的attention output
        
        thread_ratio              // K1/K2 thread block分配比例
    )
    
    // K1完成后释放临时KV workspace
    // 同步两条stream
    cudaStreamSynchronize(stream_A)
    cudaStreamSynchronize(stream_B)

// === Fused Kernel CUDA内部 ===
__global__ void fused_k1_k2_kernel(...):
    block_id = blockIdx.x
    total_blocks = gridDim.x
    
    if block_id < K1_blocks:  // 前ratio*N个block执行K1
        // K1: GEMM for KV recomputation
        token_idx = K1_uncached_token_ids[block_id % num_uncached_tokens]
        // X[token_idx] · W_K → computed_K
        // X[token_idx] · W_V → computed_V  
        computed_K[...] = matmul(X[token_idx], W_K)
        computed_V[...] = matmul(X[token_idx], W_V)
        __syncthreads()
        // 写入临时workspace供K2使用
        K1_output_KV[token_idx] = {computed_K, computed_V}
        
    else:  // 剩余(1-ratio)*N个block执行K2
        // K2: attention decode
        head_idx = (block_id - K1_blocks) % num_heads
        // concat(cached_KV, recomputed_KV)
        full_K = concat(K2_KV_cached.K, K2_KV_recomputed.K)
        full_V = concat(K2_KV_cached.V, K2_KV_recomputed.V)
        // scaled dot-product attention
        Q_head = K2_Q_current[head_idx]
        scores = Q_head @ full_K^T / sqrt(head_dim)
        attn_weights = softmax(scores)
        K2_output[head_idx] = attn_weights @ full_V
```

**关键设计要点**：
1. **thread block分区**：不按warp而是按thread block划分K1/K2——K1和K2使用独立的thread block组，避免warp divergence。grid中的前`ratio * total_blocks`个block执行K1 GEMM，剩余block执行K2 attention。
2. **临时KV workspace**：K1产生的旧token KV暂存于临时workspace buffer（约1 layer KV大小），K2读取后立即释放，避免持久占用显存。
3. **线程数对齐**：总线程数对齐到32的倍数（warp granularity），确保无idle warp lane。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在eLLM中的实现方式：
1. **离线编译**：对Llama2-13B MHA和Llama2-70B GQA分别预编译32组fused kernel variant（tuned for 32..1024 threads），每组为独立的CUDA .so shared library。
2. **运行时加载**：layer-level scheduler在每层执行前根据当前b×r估算K1/K2 FLOP ratio → 在32组中选最接近的thread配置 → 动态加载对应.so → 配置grid/block dim → launch。
3. **workspace管理**：为K1临时KV输出预分配workspace buffer（大小 = max_uncached_tokens × per_token_kv_size_in_layer_group），每层复用。
4. **与Comm-Com Overlap协同**：K1+K2在Stream B上执行期间，Stream A异步传输下一层的host-GPU cached KV（对swapped token），两者通过cudaEvent同步。
5. **消融验证**：论文禁用Kernel Fusion后TPOT和throughput退化。融合的代价是额外workspace显存约1 layer KV，通过closed-loop adaptation中的Mo参数反馈控制。

术语的通用性：虽然K1+K2 fusion是eLLM的具体实现，但这种"将不同latency-bound/compute-bound特性的kernel融合以减少launch overhead并共享SM"的思路在其他LLM serving kernel优化中也有应用（如FlashAttention的forward+backward融合、FlashInfer的prefill+decode融合）。

涉及论文标题：
- High Throughput and Low Latency LLM Serving via Adaptive KV Caching

