## 2-bit KV Cache Quantization（2比特KV缓存量化）

术语是什么？通过联网搜索让回答具体和精准。

2-bit KV Cache Quantization是将长上下文LLM推理中的键值（Key-Value）缓存激活值量化到2位精度（即每个值用2比特表示，共4个离散值）的技术。在transformer decoder的自回归推理中，KV cache随序列长度线性增长，成为memory和bandwidth瓶颈。2-bit量化将每个FP16值（16位）映射到2位整数（{0,1,2,3}），提供理论8×压缩比。然而，2位精度下可表示值极少，outlier channel的极值会显著放大同组内其他非outlier值的量化误差。JanusQuant论文中将2-bit分组量化group size设为g=32，配合per-token smoothing transformation控制outlier影响。与4-bit量化（16个离散值，更鲁棒但对超长序列仍显不足）和KV cache selection/eviction（丢弃token有精度损失）不同，2-bit量化追求"保留所有token但用极低精度存储"的路线。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

2-bit KV cache quantization在长上下文LLM推理算法pipeline中的执行流程（以JanusQuant解码阶段为例）：

```
// === 2-bit 量化参数计算（per-group） ===
// group_size g = 32, num_bits n = 2, num_levels = 2^n - 1 = 3
for each quantization group in KV_cache:
    // Step 1: 计算scale和zero_point（公式2）
    s = (max(group_values) - min(group_values)) / 3
    z = min(group_values)
    
    // Step 2: 量化到2-bit整数（公式3）
    for each value v in group:
        q = clamp(round((v - z) / s), 0, 3)  // 2-bit: 仅4个离散值{0,1,2,3}
        store q as INT2 (2 bits in packed format)

    // Step 3: 解量化恢复（公式4，仅在attention前执行）
    for each quantized value q:
        v_hat = q * s + z  // 近似的FP16值
```

2-bit量化的特殊性：每组仅4个表示值（0→z, 1→s+z, 2→2s+z, 3→3s+z），量化scale s = (max-min)/3 由组内极值完全决定。若组内存在outlier channel的值远超其他channel，s会异常大→非outlier值的量化粒度极粗→MSE激增。JanusQuant中：Atom-2bit直接per-token group quantization→K cache MSE 1.0352；QServe-2bit加入per-channel smoothing→MSE 0.5552；理想情况（移除最极端outlier channel后量化）→MSE 0.3734。这解释了为何2-bit需要配套smoothing/outlier管理。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实践中，2-bit KV cache量化需要：① 量化粒度选择——per-channel（对K cache，outlier沿channel集中）或per-token（对V cache，无明显outlier pattern）；② group size trade-off——更小的g提高精度但增加scale/zero_point参数存储（每g个值存一对FP16参数），JanusQuant选g=32使2-bit uniform quantization平均bit-width约3.008（含参数）；③ outlier管理策略——或分离存储outlier（KVQuant dense-and-sparse）、或预保留recent token为FP16（KIVI/SKVQ）、或smoothing transformation（JanusQuant RtSmooth）。2-bit inferencing需要custom kernel支持INT2 unpacking和fused dequantization-attention。典型pipeline：prefill阶段仍可用full-precision KV token做attention保accuracy，decoding阶段逐步将历史KV token量化到2-bit、recent token保持FP16、需要时通过fused kernel解量化参与attention。

涉及论文标题：
- JanusQuant: Accurate and Efficient 2-bit KV Cache Quantization for Long-context Inference

