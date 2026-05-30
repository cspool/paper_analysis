## Modulated Quantization (MoDiff)

术语是什么？
Modulated Quantization（调制量化）是MoDiff论文提出的核心算法技术，通过利用扩散采样过程中相邻时间步之间激活的相似性，将每层线性算子的计算从直接量化激活 $Q(\mathbf{a}_t)$ 重构为量化时序差分 $Q(\mathbf{a}_t - \mathbf{a}_{t+1})$，然后将差分计算结果累加到前一时间步的输出上：$\hat{\mathbf{o}}_t = \mathcal{A}(Q(\mathbf{a}_t - \mathbf{a}_{t+1})) + \hat{\mathbf{o}}_{t+1}$。其数学正确性源于线性算子 $\mathcal{A}$ 的加法分解性质：$\mathcal{A}(\mathbf{a}_t) = \mathcal{A}(\mathbf{a}_t - \mathbf{a}_{t+1}) + \mathcal{A}(\mathbf{a}_{t+1})$。

核心洞察：时序差分 $\mathbf{a}_t - \mathbf{a}_{t+1}$ 的分布范围比原始激活小10×以上，且更集中、异常值更少（Figure 1b中橙色vs蓝色violin plot），因此相同位宽的量化误差大幅降低。Theorem 4.3证明了量化误差界 $\|\mathbf{x} - Q(\mathbf{x})\|_2^2 \le (\max(\mathbf{x}) - \min(\mathbf{x}))^2 d / (2^b - 1)^2$，即误差正比于输入范围平方——范围缩小10×意味着误差降低100×。

从算法pipeline角度拆解术语：
```
// 标准PTQ方法（baseline）：逐层独立计算
for t = T, T-1, ..., 1:  // T个diffusion steps
    for layer l in 1..L:
        â_t^{(l)} = Q(a_t^{(l)})                      // 量化原始激活
        ô_t^{(l)} = A^{(l)}(â_t^{(l)})                 // 矩阵乘法

// MoDiff调制量化：逐层增量计算
for t = T, T-1, ..., 1:
    for layer l in 1..L:
        // Step 1: 量化时序差分（而非原始激活）
        diff = a_t^{(l)} - â_{t+1}^{(l)}               // 计算差分
        diff_q = Q(diff)                               // 量化差分（范围小→误差小）
        // Step 2: 增量计算输出
        ô_t^{(l)} = A^{(l)}(diff_q) + ô_{t+1}^{(l)}    // 累加到前一步输出
```

时序差分性质的量化误差分析：
```
激活分布:          范围变化大, 异常值多, 长尾分布    → 低bit量化困难
时序差分分布:      范围一致(小), 集中, 几乎无异常值  → 低bit量化容易

Err(原始激活) ≈ (range_a)² / (2^b - 1)²
Err(时序差分) ≈ (range_diff)² / (2^b - 1)²
若 range_diff < range_a / 10 → Err(差分) < Err(原始) / 100
等效：可用低3-4 bits达到相同误差界
```

术语一般如何实现？如何使用？
论文实现于PyTorch框架：对每个线性层（Conv2d、Linear），在forward pass中用MoDiff替换标准量化计算。需从原始权重中去除bias项（保证纯线性以允许加法分解）。第一时间步T使用全精度激活作为warm-up（4-5步收敛到可忽略误差）。MoDiff与具体量化方法（Q-Diffusion、动态per-channel/tensor量化）正交，可直接叠加于现有PTQ方法。在DDIM CIFAR-10上W8A3时FID=4.14（vs Q-Diff FID=143.39，10× Bops节省vs FP32）。支持0-bit skipping：当时序差分幅度低于阈值时跳过计算（此时等价于caching方法的特例）。开源：https://github.com/WeizhiGao/MoDiff。

涉及论文标题：
- Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization

---
