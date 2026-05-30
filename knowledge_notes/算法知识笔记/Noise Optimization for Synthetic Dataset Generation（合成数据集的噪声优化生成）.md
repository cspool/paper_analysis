## Noise Optimization for Synthetic Dataset Generation（合成数据集的噪声优化生成）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Noise Optimization 是 Zero-shot Quantization（ZSQ）中合成数据集生成的主流方法之一。其核心思想是：不训练额外的生成器网络，而是直接从随机高斯噪声出发，通过梯度下降迭代优化噪声本身以匹配预训练模型的内部统计信息（如 Batch Normalization 层的 running mean/variance）和分类行为（使预训练模型对优化后的噪声做出指定类别的预测）。与 Generator-based 方法相比，Noise Optimization 的优势在于：(1) 无需额外训练生成器，节省计算和存储；(2) 生成样本数量可控（如 5120 张）；(3) 灵活性高，可与其他优化目标（如硬样本生成、纹理校准）组合。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SynQ 论文中使用的 Noise Optimization pipeline（结合 TexQ 的 calibration center synthesis 和 HAST 的 hard sample generation）：
```
// Stage 1: Calibration Center Synthesis (TexQ)
for each class c in [1..C]:
    x_center[c] = random_noise(3, H, W)            // 每类一个校准中心
    optimize x_center[c] to minimize L_IL + α^C * L_BNS  // 纹理特征校准

// Stage 2: 批量合成样本生成
Initialize {x_i}_{i=1}^N with N(0,1)
for iter in 1..1000:
    // 三项损失
    L_BNS: (1/L) Σ || (μ_l, σ_l) - (μ_l({x_i}), σ_l({x_i})) ||²
    L_IL: (1/N) Σ CE(q(x_i), y_i)                  // Inception Loss
    L_HIL: (1/N) Σ δ(x_i) * CE(q(x_i), y_i)        // HAST: Hard-sample-enhanced IL
    L_Total = L_IL + α_1 * L_HIL + α_2 * L_BNS
    x_i -= η * ∇ L_Total                            // 更新噪声样本

// Stage 3: Sample Difficulty Promotion (HAST)
for each x_i:
    perturbation ε ~ N(0, σ) scaled by λ_P
    x_i = x_i + ε                                   // 附加扰动增加难度
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 PyTorch 中实现：(1) 将噪声张量包装为 nn.Parameter；(2) 使用 Adam 优化器（lr=0.5, momentum=0.9）直接优化噪声参数，学习率每 50 步不下降时衰减 0.1；(3) 合成图像使用与原始模型相同的归一化预处理（mean/std）；(4) BNS loss 计算每层 BN 统计量的 L2 距离，Inception Loss 计算预训练模型对合成样本预测分布的交叉熵；(5) 总共生成 N=5120 张图像，batch size=256，共 20 个 batch。主要优势：仅需几分钟到几十分钟即可完成数据生成，远快于 Generator-based 的几小时训练。

涉及论文标题：
- SynQ Accurate Zero-shot Quantization by Synthesis-aware Fine-tuning

---
