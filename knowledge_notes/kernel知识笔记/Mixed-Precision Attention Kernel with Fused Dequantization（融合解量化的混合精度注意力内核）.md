## Mixed-Precision Attention Kernel with Fused Dequantization（融合解量化的混合精度注意力内核）

术语是什么？通过联网搜索让回答具体和精准。

Mixed-Precision Attention Kernel with Fused Dequantization是JanusQuant提出的自定义CUDA attention kernel，在同一kernel内同时处理2-bit quantized historical KV cache和FP16 recent KV tokens，将INT2 dequantization与attention computation融合消除独立dequantization kernel的memory-bound开销。在baseline准确率导向方法（SKVQ/KIVI/KVQuant）中，dequantization作为独立kernel先于attention执行，造成额外的kernel launch overhead和全局内存往返（论文SKVQ breakdown：dequantization占decode 80% runtime）。JanusQuant kernel利用RtSmooth保留的positional alignment特性——2-bit KV和FP16 KV在hidden dimension排列一致，可直接分段由不同thread block处理——在单一kernel内完成unpack、dequantize、attention三个步骤。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Mixed-Precision Attention Kernel的执行流程（以Llama2-7B decoding为例）：

```
// === Mixed-Precision Attention Kernel（单次decoding step）===
// 输入：
//   Q: FP16 query token (1 × num_heads × head_dim)
//   history_KV_quantized: INT2 packed KV cache (seq_len_hist tokens)
//   recent_KV_FP16: FP16 ring buffer KV (≤ n*g tokens)
//   param_blocks: unified parameter blocks per thread block segment

// Grid: 2D thread blocks — dim 0沿KV sequence, dim 1沿heads
// 每个thread block处理一段KV + 一个head

__global__ void mixed_precision_attention(
    FP16* Q,                     // [num_heads, head_dim]
    INT2_packed* K_hist,        // [seq_hist, num_kv_heads, head_dim] (packed)
    INT2_packed* V_hist,        // [seq_hist, num_kv_heads, head_dim] (packed)
    FP16* K_recent,             // [num_recent, num_kv_heads, head_dim]
    FP16* V_recent,             // [num_recent, num_kv_heads, head_dim]
    ParamBlock* param_blocks,   // unified parameter blocks aligned to thread blocks
    FP16* output                // [num_heads, head_dim]
) {
    int block_id = blockIdx.x;   // KV segment index
    int head_id = blockIdx.y;    // attention head index
    int kv_head_id = head_id_to_kv_head(head_id);  // GQA mapping

    // === Phase 1: 加载Q到shared memory ===
    __shared__ FP16 Q_shared[head_dim];
    if (threadIdx.x < head_dim) Q_shared[threadIdx.x] = Q[head_id * head_dim + threadIdx.x];
    __syncthreads();

    // === Phase 2: 处理2-bit historical KV segment ===
    if (block_id < num_hist_segments):
        // Step 2a: 加载unified parameter block（scale, zero_point, smoothing_factor, offsets）
        ParamBlock pb = param_blocks[block_id * num_kv_heads + kv_head_id];
        
        // Step 2b: 从global memory加载INT2 packed KV segment到shared memory
        // 注意：unified parameter layout将4类参数合并对齐 → 减少memory transactions
        load_packed_KV_to_smem(K_hist, V_hist, block_id, kv_head_id, pb);
        
        // Step 2c: INT2-to-FP16 unpacking + dequantization（在register中）
        for each value in thread's KV chunk:
            // 高效unpacking: 利用FP16 exponent manipulation
            FP16 k_val = int2_to_fp16_unpack(packed_K_chunk, thread_local_idx);
            FP16 v_val = int2_to_fp16_unpack(packed_V_chunk, thread_local_idx);
            
            // Dequantization: v_hat = q * s + z
            k_val = k_val * pb.K_scale + pb.K_zero_point;
            v_val = v_val * pb.V_scale + pb.V_zero_point;
            
            // Inverse smoothing: 恢复RtSmooth的缩放
            k_val = k_val * pb.smoothing_factor;
        
        // Step 2d: 计算QK attention scores + softmax（in register/shared memory）
        FP16 attn_score = dot_product(Q_shared, k_vals) / sqrt(head_dim);
        // ... softmax across all KV tokens (online softmax with running max/sum)
        
        // Step 2e: 加权求和V
        output_acc += attn_weight * v_vals;

    // === Phase 3: 处理FP16 recent KV segment（同一kernel内） ===
    else if (block_id >= num_hist_segments && block_id < num_hist_segments + num_recent_segments):
        // 直接加载FP16 KV，无需dequantization
        load_FP16_KV_to_smem(K_recent, V_recent, recent_block_id, kv_head_id);
        
        // 标准attention计算
        attn_score = dot_product(Q_shared, k_fp16) / sqrt(head_dim);
        // ... softmax ...
        output_acc += attn_weight * v_fp16;

    // === Phase 4: 输出 ===
    output[head_id * head_dim + threadIdx.x] = output_acc;
}

// === INT2-to-FP16 高效unpacking（3条指令处理2个值）===
// 利用FP16格式特性：[1024, 2047)区间共享exponent=1024
// mantissa直接编码offset
__device__ FP16 int2_to_fp16_unpack(uint32_t packed, int idx) {
    // Step 1: 提取2-bit值 (lop3 bitwise extract)
    uint32_t val_2bit = (packed >> (idx * 2)) & 0x3;
    
    // Step 2: 放入FP16 mantissa + 设置exponent (or with 0x64006400)
    // 0x6400 = exponent 25 (1024) + sign 0 → FP16 exponent field
    uint32_t fp16_repr = (val_2bit << 10) | 0x6400;  // int in mantissa, exp=1024
    
    // Step 3: 减去1024得到最终FP16 (sub)
    FP16 result = __int2float_rn(fp16_repr) - 1024.0f;
    // 等价于：result = (FP16)(int_val)，但避免了通用INT→FP转换指令
    return result;
}
```

关键thread block调度：kernel以task parallelism组织——不同thread block分别处理quantized和FP16 segments，异步执行重叠计算与访存。由于quantized segment的thread block做更多计算（unpack+dequantize），而FP16 segment的thread block做更少计算，这种分工自然平衡了block间的工作量。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现需约3500行CUDA/C++，包含：① fused dequantization-attention CUDA kernel（如上），通过FlashInfer API包装为Python extension；② INT2-to-FP16 unpacking的PTX inline assembly（利用lop3/or/sub指令）；③ unified parameter block layout（scale/zero_point/smoothing_factor/offsets按thread block KV segment访问模式重排）；④ thread block到KV segment的mapping逻辑（区分quantized/FP16区域）。Kernel可编译为standalone .so，通过Pybind暴露为Python compatible torch.nn.Module。使用Nsight Compute Roofline分析指导优化：初始kernel memory-bound → dequantization fused后变为compute-bound → INT2 unpacking优化降低compute intensity → 参数block layout减少memory transactions。论文Figure 14/15b显示：128K seq/hidden 4096/32 KV heads下JanusQuant attention kernel speedup 6.17× over KIVI、1.69× over QServe、平均1.99× over FA2。

涉及论文标题：
- JanusQuant: Accurate and Efficient 2-bit KV Cache Quantization for Long-context Inference

