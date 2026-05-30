## Hierarchical Latent Caching (HLC) for Diffusion Transformers

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hierarchical Latent Caching (HLC) 是 QuantCache 论文提出的针对 Diffusion Transformers (DiTs) 视频生成的训练无关（training-free）自适应特征缓存机制。其核心思想：不同于传统缓存方法（如 DeepCache、AdaCache、Δ-Cache）使用静态的固定缓存间隔，HLC 基于 inter-step feature divergence（跨步特征散度）动态决定每个 layer 在每个 timestep 是否需要刷新缓存。具体地，对 timestep t 和 layer l，计算 timestep-wise feature divergence score：D_t^(l) = ||p_t^(l) - p_{t-k}^(l)||_1 / k · ||∇_t m_t^(l)||，其中 p_t^(l) 为 layer l 在 timestep t 的激活值，k 为距离上次缓存刷新的步数，∇_t m_t^(l) 为帧间 feature map 的运动梯度。D_t^(l) 综合衡量了：(1) 激活值的变化幅度（L1 distance），(2) 帧间运动变化速度（inter-frame gradient）。基于 D_t^(l) 与预设阈值 δ_1、δ_2 的比较，HLC 决定三档缓存刷新间隔：τ_t^(l) = τ_max（D_t^(l) < δ_1，内容变化极小，长间隔缓存）、τ_mid（δ_1 ≤ D_t^(l) < δ_2，中等变化）、τ_min（D_t^(l) ≥ δ_2，剧烈变化，频繁刷新）。HLC 专门针对 DiT 架构设计——DiT 缺乏 U-Net 的 skip connections，传统 feature map caching 在 DiT 上效果差。HLC 在 QuantCache 中单独实现 4.12× speedup（on Open-Sora, A800-80GB）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
HLC 在 DiT 推理 pipeline 中的运行逻辑：
```python
# HLC: Hierarchical Latent Caching for DiT
# 每个 timestep t 和每个 DiT layer l
k = last_cached_step[l]  # 上次缓存刷新的 timestep
if t - k < tau[l]:        # 缓存未过期
    p_t[l] = cache_buffer[l]  # 直接复用缓存特征
    skip_computation(l)        # 跳过该层完整计算
else:
    p_t[l] = compute_layer(l, x_t)  # 正常前向计算
    D_t_l = norm(p_t[l] - p_k[l], 1) / (t - k) * norm(grad_m_t[l])
    # 三档决策更新刷新间隔
    if D_t_l < delta_1:
        tau[l] = tau_max     # 长间隔 (如 5-10 steps)
    elif D_t_l < delta_2:
        tau[l] = tau_mid     # 中等间隔 (如 2-3 steps)
    else:
        tau[l] = tau_min     # 短间隔 (每步刷新)
    cache_buffer[l] = p_t[l]
    last_cached_step[l] = t
```
HLC 与 AIGQ 联合优化：小 skip（低 τ_t^(l)）时用较小的 bit-width 利用高冗余性加速；大 skip（高 τ_t^(l)）后在 post-skip step 应用较小 bit-width 增强精度补偿缓存 drift。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
HLC 实现不需要额外训练或微调，完全在推理时在线运行：(1) 在 GPU 上分配 dedicated cache buffer 存储每层缓存的 feature map（FP16/量化格式）；(2) 每个 timestep 推理前先计算各层的 D_t^(l)，与阈值比较决定是否从 cache buffer 读取或重新计算；(3) 阈值 δ_1、δ_2 和 τ_max/τ_mid/τ_min 为超参数，论文通过经验实验确定。HLC 适用于所有 DiT-based 视频/图像生成模型（如 Open-Sora、Flux、CogVideoX），尤其对长时序视频（如 64+ 帧）加速效果显著。当前开源代码见 https://github.com/JunyiWuCode/QuantCache。

涉及论文标题：
- QuantCache Adaptive Importance-Guided Quantization with Hierarchical Latent and Layer Caching for Video Generation

---
