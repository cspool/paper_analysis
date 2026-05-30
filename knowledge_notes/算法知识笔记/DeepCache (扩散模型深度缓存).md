## DeepCache (扩散模型深度缓存)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DeepCache 是 Ma et al. (CVPR 2024) 提出的扩散模型加速方法，通过缓存和复用 U-Net 中间层的上采样特征来跳过冗余计算。核心观察：扩散模型去噪过程中相邻时间步的特征图高度相似，尤其在 U-Net 的 high-level（深层）特征。DeepCache 在 U-Net 的跳跃连接处缓存上采样特征，每 N 步更新一次缓存，中间 N−1 步直接复用缓存特征，跳过对应的下采样和中间块计算。N 越大加速越多但生成质量下降越严重。MoDiff 论文的初步研究（图1a）显示 DeepCache 的缓存复用策略会导致误差累积：即使 N=3（每3步更新），最终步的特征与标准扩散的相对 ℓ₂ 距离可达 40%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
DeepCache 在 U-Net 中的应用流程：

```
# U-Net 结构: Encoder(下采样) → Middle Block → Decoder(上采样+跳跃连接)
# DeepCache 策略:
cache = None
for t in range(T, 0, -1):
    if t % N == 0 or cache is None:
        # 完整前向：执行 Encoder + Middle + Decoder
        features = UNet_full(x_t, t)
        cache = features.up_sample_block_1  # 缓存上采样第一块特征
    else:
        # 加速前向：跳过 Encoder 和 Middle block
        features_low = UNet.shallow(x_t, t)  # 仅执行浅层
        features = concat(features_low, cache)  # 复用缓存的上采样特征
    x_{t-1} = sampler_step(x_t, features)
```

MoDiff 与 DeepCache 的关系：MoDiff 的调制量化在差值范围低于阈值时可为差值分配 0-bit（即跳过计算），此时 MoDiff 等价于 DeepCache 的行为。但 MoDiff 通过统一的理论框架（调制量化 + 误差补偿）提供了更灵活的控制和更严格的误差保证。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源代码：https://github.com/horseee/DeepCache（CVPR 2024）。基于 Stable Diffusion 和 DDIM/DDPM sampler 实现。主要修改 U-Net 的 forward 函数，插入缓存检查逻辑。局限性：(1) N 值需手动调参，无自适应机制；(2) 仅适用于 U-Net 架构，不适用于 DiT 等 Transformer 架构；(3) 缓存误差累积严重，不能支持极低比特或大幅加速。

涉及论文标题：
- Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization
