## Gumbel-Softmax

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Gumbel-Softmax 是由 Jang, Gu, Poole 在 ICLR 2017 提出的可微分离散采样方法，允许神经网络通过离散分类变量进行端到端梯度反向传播。其核心思想：在 forward pass 中，向未归一化的类别 logits $g_i$ 添加标准 Gumbel 分布噪声 $\sigma_i \sim \text{Gumbel}(0,1)$，然后取 argmax 获得离散选择 $\theta = \arg\max_i (g_i + \sigma_i)$；在 backward pass 中，使用 softmax 的连续近似概率 $p_i = \frac{\exp((g_i + \sigma_i)/\tau)}{\sum_j \exp((g_j + \sigma_j)/\tau)}$ 计算梯度，其中温度系数 $\tau$ 控制 softmax 的锐度（$\tau \to 0$ 时接近 one-hot，$\tau \to \infty$ 时趋向均匀分布）。这种"前向离散 + 反向连续"的 Straight-Through Estimator (STE) 模式使得网络可以端到端学习离散决策。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

在 Granular-DQ 的 Granularity-Bit Controller (GBC) 中，Gumbel-Softmax 用于为每个图像 patch 选择量化的 bit-width：

```
# GBC 中 Gumbel-Softmax bit-width 选择流程
输入: patch 特征 → 线性层 → 门控 logits g ∈ R^N (N=3, 候选 bit: [4,6,8])

# Forward pass (离散采样)
σ = sample_gumbel(shape=g.shape)  # σ_n = -log(-log(U)), U ~ Uniform(0,1)
θ = argmax(g + σ)                  # 离散门控索引 ∈ {1, 2, 3}

# 计算门控分数 (连续近似, 用于梯度传播)
p_i = exp((g_i + σ_i) / τ) / Σ_n exp((g_n + σ_n) / τ)   # τ=1

# Backward pass (STE): 前向使用离散 θ, 反向使用连续 p 的梯度
∇_{W_g} L = ∂L/∂p · ∂p/∂g · ∂g/∂W_g
```

GBC 接收 D 层多粒度特征融合后的通道统计量 S，线性层 W_g ∈ R^{(N×D)×N} 映射为 N 维门控 logits，对每个 patch 独立采样门控分数 p_i（衡量 patch 对整张图像的贡献比例），映射到对应 bit code。GBC 置于 SR 网络输入端，仅引入可忽略的计算开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

1. **温度退火**：训练初期用较大 τ (1-5) 鼓励探索，后期降低 τ (0.1-0.5) 稳定决策。
2. **Gumbel(0,1) 采样**：通过逆变换 $\sigma = -\log(-\log(U))$, $U \sim \text{Uniform}(0,1)$。
3. **PyTorch 内置支持**：`F.gumbel_softmax(logits, tau=1.0, hard=True)` 实现 hard Gumbel-Softmax（forward 返回 one-hot，backward 用 softmax 梯度）。
4. **应用场景**：动态量化 bit-width 选择、NAS 架构搜索、动态 routing、离散隐变量生成模型 (VAE)、RL 动作选择。

涉及论文标题：
- Thinking in Granularity Dynamic Quantization for Image Super-Resolution by Intriguing Multi-Granularity Clues
