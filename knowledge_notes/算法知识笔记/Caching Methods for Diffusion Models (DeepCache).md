## Caching Methods for Diffusion Models (DeepCache)

术语是什么？
Caching methods for diffusion models（扩散模型缓存方法）是一类利用扩散采样过程中相邻时间步特征高度相似性来跳过高计算代价特征重算的加速技术。代表性方法是DeepCache (Ma et al., CVPR 2024)，它每隔N个去噪步骤在U-Net的高层缓存一次high-level features，在中间的N-1步直接复用cached features，跳过这些层在该步的重计算。其他方法包括Cache Me If You Can (Wimbauer et al., CVPR 2024)、Δ-DiT (Chen et al., 2024) 和 Learning-to-Cache (Ma et al., NeurIPS 2024)。

核心缺陷（如MoDiff论文§3.1/Figure 1a所示）：缓存复用引入approximation error——cached features是过去时间步的近似值而非当前步的精确值——该误差在迭代中累积：每步的偏差传到下一步后被放大。Relative ℓ₂ distance在最终step可达40%（即使每隔3步更新cache），导致生成质量下降。此外，最优的cache更新频率N需要通过heuristic search或retraining确定，泛化性差。

从算法pipeline角度拆解术语：
```
// DeepCache pipeline（以U-Net为例）：
schedule = [1, 0, 0, 1, 0, 0, ...]  // 1=完整计算, 0=复用缓存

for t = T..1:
    if schedule[t] == 1:
        h_low = encoder_layers(x_t, t)      // 低层：新鲜计算
        h_mid = middle_block(h_low, t)       // 中层：计算并缓存
        h_high = decoder_layers(h_mid, t)    // 高层
        cache = h_mid                        // 保存缓存
    else:  // schedule[t] == 0
        h_low = encoder_layers(x_t, t)       // 低层：新鲜计算
        h_mid = cache                        // 直接复用缓存（近似！）
        h_high = decoder_layers(h_mid, t)    // 使用近似中间特征

// 误差分析：
// 复用步：h_mid(实际值) ≠ cache(上次计算)
// 偏差通过decoder传播 → 迭代累积 → 最终step L2 distance达40%
```

术语一般如何实现？如何使用？
DeepCache开源：https://github.com/horseee/DeepCache。集成方式：对预训练扩散模型的U-Net包装DeepCacheModule，指定cache更新间隔N。适用于所有使用U-Net架构的扩散模型（DDPM、DDIM、LDM等）。MoDiff证明了caching方法是MoDiff在0-bit差分时的特例（Remark 4.1）：当时序差分幅度低于可容忍阈值时，MoDiff可分配0-bit跳过计算——此时等价于cache-and-reuse。MoDiff的优势在于通过误差补偿消除了cache的误差累积问题。

涉及论文标题：
- Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization
