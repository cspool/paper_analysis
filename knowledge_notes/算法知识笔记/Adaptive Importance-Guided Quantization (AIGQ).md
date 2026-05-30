## Adaptive Importance-Guided Quantization (AIGQ)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Adaptive Importance-Guided Quantization (AIGQ) 是 QuantCache 论文提出的自适应混合精度量化框架，针对 Diffusion Transformers (DiTs) 视频生成的两阶段动态精度分配：(1) **权重量化（Weight Quantization）**：offline 评估每层的 sensitivity（综合考虑 numerical error、perceptual distortion 和 temporal dynamics），在总 bit-width 预算 B_total 约束下迭代分配每层 bit-width（Σ_l B(l) ≤ B_total）——关键层（高 sensitivity，对纹理重建和运动连续性影响大）分配高精度，冗余层分配低精度。同时引入 channel-balancing mechanism：scaling（修正 pretrained scale shift tables 的静态 imbalance）+ rotation（修正 timestep embeddings 引起的动态变化），确保更均匀的跨 channel 数据分布以减少量化 outlier。(2) **激活量化（Activation Quantization）**：提出 timestep-wise content-adaptive bit allocation function：bit-width(t) = Bit_max（D_t < θ_1，低冗余，关键 timestep）、Bit_mid（θ_1 ≤ D_t < θ_2）、Bit_min（D_t ≥ θ_2，高冗余 timestep，如连续帧变化极小），其中 D_t 为 timestep 冗余度度量（从相邻 feature map 距离推导）。核心洞见：不是所有 timestep 对输出质量贡献相同——早期/高冗余 intermediate step 用低精度即可，关键过渡阶段（细节涌现/场景切换）需要高精度。AIGQ 联合 HLC 实现 6.33× speedup。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
AIGQ 的完整 pipeline 流程：
```python
# Offline Phase: Weight Sensitivity Analysis & Budget Allocation
for layer in model.layers:
    # 用小规模校准集评估每层 sensitivity
    layer_sensitivity[l] = compute_sensitivity(
        numerical_error(W_quant[l], W_fp16[l]),
        perceptual_distortion(output_quant[l], output_fp16[l]),
        temporal_variance(feature_map[l])
    )
# 在总预算 B_total 下迭代分配 bit-width
B_remaining = B_total
sorted_layers = sort_by_sensitivity_desc(layers)
for layer in sorted_layers:
    if B_remaining >= 8:
        bit_width_W[l] = 8  # 高 sensitivity → 8-bit
    elif B_remaining >= 6:
        bit_width_W[l] = 6  # 中 sensitivity → 6-bit
    else:
        bit_width_W[l] = 4  # 低 sensitivity → 4-bit
    B_remaining -= bit_width_W[l]

# Channel Balancing
for layer in model.layers:
    S = compute_scaling_correction(scale_shift_tables[l])  # 静态修正
    R = compute_rotation_correction(timestep_embeddings[l])  # 动态修正
    # offline absorb scaling into preceding weights: W_prev = W_prev * S

# Online Phase: Timestep-wise Activation Bit Allocation
for t in range(T, 0, -1):
    D_t = compute_timestep_redundancy(x_t, x_{t-1})  # 相邻 feature map 距离
    if D_t < theta_1:      bit_width_A = Bit_max   # 低冗余，保持高精度
    elif D_t < theta_2:    bit_width_A = Bit_mid   # 中等冗余
    else:                  bit_width_A = Bit_min   # 高冗余，激进量化
    # 推理时动态量化激活 + 低精度 GEMM
    x_quant = uniform_quantize(x_t, bit_width_A, per_layer=True)
    for layer in model.layers:
        w_quant = load_quantized_weight(layer, bit_width_W[l])
        output = low_precision_gemm(w_quant, x_quant)  # 如 W4A6 GEMM
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
AIGQ 基于 uniform min-max quantization（per-channel weight + dynamic per-layer activation），激活量化参数 online 计算（minimal overhead），混合精度权重量化 offline 通过小规模校准数据集确定。具体实现时：(1) 将 channel-balancing scaling factors offline 吸收到前层权重中（受 QServe/SmoothQuant/ViDiT-Q 启发），消除推理时额外开销；(2) rotation transformation 通过 CUDA kernel fusion 与量化操作融合；(3) bit-width 分配阈值 θ_1、θ_2 和 B_total 为超参数，论文通过经验实验确定。AIGQ 在 Open-Sora 上 W4A6 配置下仍保持 competitive quality（VBench 指标接近 FP16 baseline），显著优于同 bit-width 下的 Q-DiT、PTQ4DiT、SmoothQuant 等 uniform quantization 方法。开源代码：https://github.com/JunyiWuCode/QuantCache。

涉及论文标题：
- QuantCache Adaptive Importance-Guided Quantization with Hierarchical Latent and Layer Caching for Video Generation

---
