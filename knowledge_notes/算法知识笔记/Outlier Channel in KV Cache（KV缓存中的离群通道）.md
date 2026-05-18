## Outlier Channel in KV Cache（KV缓存中的离群通道）

术语是什么？通过联网搜索让回答具体和精准。

在transformer模型KV cache中，outlier channel（离群通道）指K cache某些hidden dimension channel中值幅度显著大于其他channel的现象。论文Figure 2展示了Llama2-13B的K cache各通道值幅度热力图：少数channel呈现强outlier特征（值可达正常channel的6倍以上），而V cache无此模式。这种channel-wise outlier集中性来自attention层中K投影权重和输入hidden states的特定交互。在2-bit分组量化中，当per-token group quantization将不同channel的值混在同一group内时，outlier channel的极端max值会扩大整个group的量化scale s = (max-min)/3，使同组非outlier channel值被粗糙量化（仅4个量化bin）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Outlier channel在2-bit量化中的量化误差放大机制：

```
// 假设per-token group quantization: 每token内g=4个连续值为一组
// 组内包含3个正常channel值 + 1个outlier channel值

// 正常值范围: [0.5, 2.0]，可被2-bit精细量化
normal_values = [0.5, 1.2, 0.8]

// Outlier channel值
outlier = 15.0

// 混合后group = [0.5, 1.2, 0.8, 15.0]
group = normal_values + [outlier]

// 量化参数被outlier主导
s = (15.0 - 0.5) / 3 ≈ 4.833
z = 0.5

// 2-bit量化表示
quantized_levels = {0→0.5, 1→5.333, 2→10.167, 3→15.0}

// 正常值0.5→0 (精确=0.5)，1.2→0 (恢复=0.5, error=-0.7)，0.8→0 (恢复=0.5, error=-0.3)
// 所有正常值被映射到同一量化level 0，完全丢失区分度
// MSE ≈ (0² + 0.7² + 0.3² + 0²)/4 = 0.145 per element
```

对应三种缓解策略：① Per-channel group quantization——沿channel维分组，同一group内各值来自不同token的同一channel，避免outlier channel与非outlier channel混合（JanusQuant在RtSmooth后对K cache采用）；② Per-token smoothing——缩小每个token内的值域→outlier channel值被smoothing factor压缩（RtSmooth核心）；③ Dense-and-sparse分离——检测outlier值并存入稀疏high-precision buffer，剩余值量化（KVQuant方法）。JanusQuant采用①②组合，避免③的稀疏路径额外overhead。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实践中outlier channel的识别和处理策略选择取决于calibration分析和目标精度。JanusQuant的FAVP离线校准用128个WikiText2 8K样本分析每层的absmax channel分布，识别出每层最可能持有absmax的稀疏channel集（>90%层涉及<2% channels）。RtSmooth通过per-token smoothing间接处理outlier：不直接检测或抽取outlier值，而是用smoothing factor = max(|K_i|)^0.5缩放整个token→缩小所有channel值域→降低outlier对group scale的影响。这种方法比explicit outlier detection/extraction更快（消除memory-bound的outlier handling步骤），参考论文中SKVQ attention layer breakdown：outlier handling在prefill占~20% runtime、decode占~2% runtime。

涉及论文标题：
- JanusQuant: Accurate and Efficient 2-bit KV Cache Quantization for Long-context Inference

