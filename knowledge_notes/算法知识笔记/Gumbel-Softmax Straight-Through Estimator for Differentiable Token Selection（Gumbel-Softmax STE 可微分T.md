## Gumbel-Softmax Straight-Through Estimator for Differentiable Token Selection（Gumbel-Softmax STE 可微分Token选择）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Gumbel-Softmax + STE 是 GroundVTS 中用于实现端到端可训练 top-K token 选择的核心技术。问题: hard top-K selection (argmax/top-K) 是非可微操作，无法通过梯度下降优化。方案: (1) Gumbel-Softmax — 向 log-probabilities 添加 Gumbel 噪声并通过 softmax 产生连续松弛，近似 categorical 采样: z_i = softmax((log w_i + g_i) / τ_g)；(2) STE — forward pass 使用 hard (0/1) mask，backward pass 通过 soft 松弛传播梯度: \tilde{z}_i = z_i^hard + z_i - stopgrad(z_i)。这使离散 token 选择可端到端训练。GroundVTS 中该技术用在 VTS 模块的 top-K selection 步骤，backward 梯度通过 z_i（连续松弛）流入 w_i → V' → W_v 和 W_q → VTS 所有可学习参数。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === Gumbel-Softmax STE ===
# 输入: logits = log w (token 相关性)

# Gumbel 噪声采样 (reparameterization trick)
u ~ Uniform(0, 1)
g = -log(-log(u))  # Gumbel(0,1)

# 连续松弛 (Eq.4)
z_soft = softmax((logits + g) / τ_g)
# τ_g 控制松弛平滑度: τ_g→0 → z_soft→one-hot; τ_g→∞ → z_soft→uniform

# Hard Top-K (Eq.5, forward only)
z_hard = 1[i ∈ TopK(logits, K)]

# STE (Eq.6)
z_out = z_hard + z_soft - z_soft.detach()
# forward: z_out = z_hard (离散)
# backward: ∂L/∂z_out = ∂L/∂z_soft (连续梯度)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch 中通过 F.gumbel_softmax(logits, tau=τ_g, hard=True) 直接实现 (hard=True 时内置 STE)。训练时使用 Gumbel noise + STE，推理时直接 hard top-K (无 noise, 确定性)。GroundVTS 中 τ_g 为可调超参数，τ_g 过小导致训练不稳定（梯度方差大），τ_g 过大导致与 hard selection 偏差大（训练-推理 gap）。与 RL-based token selection 的区别：Gumbel-Softmax STE 直接通过梯度下降优化，无需 reward 设计或 policy gradient 的方差问题。CVPR 2026。

涉及论文标题：
- GroundVTS__Visual_Token_Sampling_in_Multimodal_Large_Language_Models_for_Video_Temporal_Grounding
- TSPO__Temporal_Sampling_Policy_Optimization_for_Long-form_Video_Language_Understanding

TSPO 中的 Gumbel-Softmax 使用方式：TSPO 使用 Gumbel-Softmax（不含 STE）进行概率化关键帧选择。流程：对 cross-modal similarity scores S ∈ R^{T_c}，注入 Gumbel(0,1) 噪声 γ，计算 softmax(S/τ + γ) 得到概率分布 P，再 TopK 选择 T_s 帧。训练时通过 τ annealing（0.025→0.01）控制探索-利用平衡；推理时去除 Gumbel 噪声，直接确定性 Softmax + TopK 采样。与 GroundVTS 的 STE 变体不同，TSPO 不通过梯度下降优化帧选择器，而是通过 GRPO 的 policy gradient 优化——Gumbel-Softmax 在此仅用于提供可探索的离散动作空间（概率化采样），梯度传播由 GRPO 的 importance sampling ratio 而非 STE 处理。

VisionSelector 的 DTS 作为替代方案：与 Gumbel-Softmax STE 不同，DTS 使用 sigmoid 连续松弛 + 隐函数微分实现可微分 Top-K，无需 Gumbel 噪声，具有确定性、严格单调性（s_i > s_j ⇔ M_i > M_j），梯度为闭式精确解（而非 STE 近似）。且 DTS 不通过 τ annealing 桥接训练-推理 gap，而是通过 Curriculum Annealing Strategy (CAS) 在损失权重空间渐进。详见 「Differentiable Top-K Selection（DTS）」条目。

涉及论文标题：
- GroundVTS__Visual_Token_Sampling_in_Multimodal_Large_Language_Models_for_Video_Temporal_Grounding
- TSPO__Temporal_Sampling_Policy_Optimization_for_Long-form_Video_Language_Understanding
- VisionSelector__End-to-End_Learnable_Visual_Token_Compression_for_Efficient_Multimodal_LLMs
