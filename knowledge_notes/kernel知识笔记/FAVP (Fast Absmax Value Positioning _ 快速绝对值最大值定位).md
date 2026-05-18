## FAVP (Fast Absmax Value Positioning / 快速绝对值最大值定位)

术语是什么？通过联网搜索让回答具体和精准。

FAVP是JanusQuant提出的快速absmax值定位技术，用于在运行时以极低开销定位每个token中绝对值最大的channel值，以计算RtSmooth的smoothing factor。其核心insight是：K cache的per-token absmax值出现的channel具有跨token的稀疏性和规律性——少数channel持续持有每个token的absmax值（论文Figure 8显示超过90%的层仅涉及少于2%的channels）。FAVP通过部署前一次性离线校准（数分钟）识别每层这些稀疏channel集，运行时仅扫描这些候选channel（而非全量hidden_dim个channel）来获取每个token的absmax，将absmax计算开销降低超过50倍。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FAVP在JanusQuant fused quantization kernel中的执行流程：

```
// === 离线校准阶段（deployment前，一次性，数分钟）===
// 输入：128个WikiText2 8K samples, model
FAVP_channel_sets = {}  // 每层一个稀疏channel索引集
for each layer l in model.layers:
    absmax_channel_counter = zeros(hidden_dim)  // 统计每个channel作为absmax的频率
    for each sample in calibration_samples:
        K_cache = forward_model_to_layer_l(sample)  // 获取该层K cache
        for each token t in K_cache:
            absmax_ch = argmax(|K_cache[t, :]|)  // 该token absmax所在的channel
            absmax_channel_counter[absmax_ch] += 1
    
    // 选择覆盖大多数absmax的稀疏channel集
    sorted_channels = argsort_descending(absmax_channel_counter)
    cumulative = cumsum(absmax_channel_counter[sorted_channels]) / total_tokens
    // 论文：>90%的层仅需<2% channels即可覆盖大多数absmax
    FAVP_channel_sets[l] = sorted_channels[0 : top_k]  // top_k ≈ 0.02 * hidden_dim

// === 运行时：Fused Smoothing + Quantization Kernel ===
// 每g=32 token触发一次（而非每token每step）
__global__ void fused_smoothing_quantization_kernel(
    FP16* K_segment,      // g × hidden_dim FP16 K values
    int* FAVP_channels,   // 预校准的稀疏channel索引（~2% × hidden_dim）
    int num_candidate_ch,  // 候选channel数量
    FP16* K_quantized,    // 输出：INT2 packed K values
    ParamBlock* params    // 输出：unified parameter block
) {
    for each token t in block's assigned range:
        // Step 1: FAVP - 仅扫描稀疏候选channel
        FP16 absmax_val = 0.0;
        for each ch in FAVP_channels[0:num_candidate_ch]:
            absmax_val = max(absmax_val, abs(K_segment[t][ch]));
        
        // Step 2: Smoothing factor（无需扫描全部hidden_dim channels）
        FP16 gamma = pow(absmax_val, 0.5);  // lambda = 0.5
        gamma = (gamma == 0) ? 1.0 : gamma;
        
        // Step 3: Per-token smoothing
        for each ch in range(hidden_dim):
            K_segment[t][ch] = K_segment[t][ch] / gamma;
        
        // Step 4: Per-channel group quantization + INT2 packing
        // ... (后续量化步骤，smoothing factor存入param block)
        params[t].smoothing_factor = gamma;
}

// 无FAVP的naive runtime smoothing: absmax需扫描全部hidden_dim
// cost: O(g × hidden_dim) reads → O(g × 0.02 × hidden_dim) with FAVP
// 论文量化kernel breakdown (Figure 15a):
//   - naive smoothing: 4.43× overhead @ 64K seq len vs no-smoothing baseline
//   - RtSmooth + FAVP: overhead降至接近1× (near no-smoothing baseline)
```

关键实现细节：(1) FAVP依赖于absmax channel的跨token稳定性，论文在128个8K样本上验证——跨样本absmax分布保持稳定；(2) 候选channel集按per-layer存储，推理时作为constant memory或read-only buffer传入kernel；(3) FAVP不要求精确匹配真实absmax（它只是smoothing factor的近似计算），论文Figure 11显示predicted-to-actual ratio的偏差在极小比例token中出现，对perplexity无负面影响。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FAVP使用时需要：① 部署前用代表性calibration数据运行一次离线校准（论文用128个WikiText2 8K样本，数分钟完成）；② 为每个attention layer生成并存储稀疏channel索引集（存储开销可忽略，仅<2% × hidden_dim × sizeof(int) per layer）；③ 在fused quantization kernel中，以calibrated indices作为gather操作的索引，通过coalesced或random memory access读取候选channel值。由于候选集极小且across tokens重复使用，L1 cache命中率高。FAVP的最大优势在于将quantization kernel中占比>80%的absmax计算开销降至可忽略水平（论文Figure 15a）。其局限性是依赖cross-token absmax channel稳定性，对新架构/非标准attention的迁移需要重新校准。

涉及论文标题：
- JanusQuant: Accurate and Efficient 2-bit KV Cache Quantization for Long-context Inference

