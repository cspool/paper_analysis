## RtSmooth（Runtime Smoothing Quantization Algorithm / 运行时平滑量化算法）

术语是什么？通过联网搜索让回答具体和精准。

RtSmooth是JanusQuant提出的运行时per-token smoothing量化算法，核心思想是在量化前对K cache执行per-token平滑变换（smoothing transformation），动态缩小每个token内值的范围，降低2-bit分组量化的scale因子，从而减少outlier channel对同组其他值的量化误差放大。与SmoothQuant（离线校准per-channel smoothing factor并融入前层权重）不同，RtSmooth在推理时为每个请求、每个decoding step动态计算smoothing factor，适应不同输入和序列长度下KV cache分布的变化。K cache使用per-token smoothing + per-channel group quantization，V cache（无显著outlier）使用per-token group quantization。Decoding中每g=32步将FP16缓冲的recent tokens批量量化。平滑因子计算为max(|K_i|)^λ，λ=0.5由经验选定。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

RtSmooth在2-bit KV cache量化pipeline中的执行流程：

```
// === JanusQuant Decoding Step（RtSmooth核心路径）===
// 输入：新生成的token t，其K_t, V_t ∈ R^{hidden_dim}
// 常量：lambda = 0.5, group_size g = 32, recent_buffer 容量 = n*g

// Step 1: 将新token写入ring buffer（FP16精度）
recent_buffer[write_ptr] = (K_t, V_t)
write_ptr = (write_ptr + 1) % buffer_size

// Step 2: 若当前segment满g个token，触发量化
if tokens_since_last_quantize == g:
    quantize_segment = recent_buffer[oldest_segment_start : oldest_segment_start + g]
    
    // Step 2a: 对K cache执行RtSmooth + per-channel group quantization
    for each token K_i in quantize_segment:
        candidate_channels = FAVP_channel_set[layer_id]  // <2% of total channels
        absmax_i = max(|K_i[ch]| for ch in candidate_channels)
        
        // RtSmooth smoothing factor
        gamma_i = absmax_i ^ lambda  // lambda = 0.5
        if gamma_i == 0: gamma_i = 1.0
        
        // Per-token smoothing transformation
        K_i_smooth = K_i / gamma_i  // 缩放整行 → 缩小组内值域

    // Step 2b: 对smoothed K执行per-channel group quantization
    for each channel_group in K_smooth:
        for each group of g consecutive tokens within channel_group:
            s = (max(group) - min(group)) / 3
            z = min(group)
            K_quantized[group] = clamp(round((group - z) / s), 0, 3)
            store(K_quantized, s, z, gamma_i)

    // Step 2c: 对V cache执行per-token group quantization（无smoothing）
    for each token_group of g tokens in V_segment:
        for each value_group of g consecutive values within token:
            s = (max(value_group) - min(value_group)) / 3
            z = min(value_group)
            V_quantized[value_group] = clamp(round((value_group - z) / s), 0, 3)
            store(V_quantized, s, z)

    // Step 2d: 追加量化后的segment到历史KV cache
    history_KV_cache.append(quantize_segment_quantized)
    oldest_segment_start = (oldest_segment_start + g) % buffer_size

// Step 3: Fused Mixed-Precision Attention
output_token = fused_mixed_precision_attention(Q_t, history_KV_cache, recent_buffer)
```

核心特性：(1) smoothing因子从absmax导出而非全量统计，FAVP将其扫描成本降至O(0.02×hidden_dim)；(2) per-channel group quantization使outlier effects沿channel局部化，不跨token传播；(3) smoothing scale缩小后，误差上界从s/2降至s_smooth/2（公式6）；(4) 量化频率为每g步一次，而非每步一次，amortize量化开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

RtSmooth实现约为3500行CUDA/C++中的fused quantization kernel。该kernel在单一CUDA kernel中完成：扫描FAVP channel→计算absmax→生成smoothing factor→per-token smoothing→scale/zero_point计算→INT2 packing→参数重排(unified parameter block)。Smoothing factor gamma_i与scale/zero_point一起存入统一参数块供后续mixed-precision attention kernel使用。与SmoothQuant的offline calibration融合权重不同，RtSmooth的smoothing在运行时执行，因此需dequantization时做逆smoothing（K_hat = K_quantized_deq × gamma_i），这一操作在fused attention kernel内完成。Decoding开始时新KV token不做smoothing直接存入FP16 ring buffer，待segment达到g后统一量化。

涉及论文标题：
- JanusQuant: Accurate and Efficient 2-bit KV Cache Quantization for Long-context Inference

