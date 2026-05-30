## Dynamic Token-aware Router (DTR)

术语解释
EvoMoE 提出的一种基于 hypernetwork 的 MoE 路由机制：使用两个独立的 hypernetwork 分别处理视觉 token 和文本 token，动态生成每个 token 专用的路由参数，替代传统的静态线性 router。

术语是什么？
DTR 解决 MoE-tuning 中的 **Router Rigidity** 问题：传统 MoE-tuning 使用共享的线性 router (`W_r * h`) 对所有 token 做统一的 top-k expert 选择，无法区分视觉 token 和文本 token 的模态差异，导致 router 输出对输入不敏感（KDE 图显示两种模态的 logit 分布高度重叠）。

DTR 的核心由三个组件构成：
1. **Hypernetwork H_V 和 H_T**：各含两个 MLP layer，接收 token hidden state 作为输入，动态生成 up-sampling 和 down-sampling 层的权重矩阵
2. **动态投影层**：使用 hypernetwork 生成的权重做 down-projection → SwiGLU → up-projection
3. **最终 router φ**：一个 MLP layer，将投影后的特征映射为 expert 概率分布

$$\Theta_{\text{up}}^{\tau}, \Theta_{\text{down}}^{\tau} = \mathcal{H}^{\tau}(z^{\tau\prime}), \quad \tau \in \{V, T\}$$

$$\mathcal{E}^{\tau} = \Theta_{\text{up}}^{\tau} \left( \text{SwiGLU} \left( \Theta_{\text{down}}^{\tau} \left( z^{\tau \prime} \right) \right) \right)$$

$$\rho^{\tau} = \phi(\mathcal{E}^{\tau})$$

其中 $\mathcal{H}^V$ 处理视觉 token，$\mathcal{H}^T$ 处理文本 token。训练时仅 $\mathcal{H}^V$、$\mathcal{H}^T$ 和 $\phi$ 可训练，expert 参数冻结。每个 token 通过 top-1 选择激活概率最高的 expert。

消融实验显示：(1) 模态特定 router（无共享）优于单 router，(2) HyperNet 进一步改善注意力于输入分布，(3) 添加加权共享 router 反而降低性能，(4) DTR（HyperNet + 无共享）结构最优。

从算法pipeline角度拆解术语：
```
# Stage III: DTR Training（每个 MoE decoder layer）
# 输入：MSA 输出 z'（visual tokens V 和 text tokens T 分别处理）
# experts=4, top-k=1，experts 参数冻结

for τ in {V, T}:
    # Step 1: Hypernetwork 生成动态参数
    # H^τ 含两个 MLP：(w1, b1) 和 (w2, b2)
    Θ_up^τ, Θ_down^τ = H^τ(z'^τ)  # 输出 δ/2 → δ 和 δ → δ/2 维矩阵

    # Step 2: Token-aware 动态投影
    h_down = Θ_down^τ @ z'^τ          # down-projection
    h_act = SwiGLU(h_down)           # 门控激活
    h_up = Θ_up^τ @ h_act             # up-projection
    E^τ = h_up                        # Token-specific feature

    # Step 3: 最终 router 预测
    ρ^τ = softmax(φ(E^τ))            # [batch, seq, num_experts]

    # Step 4: Top-1 expert selection
    for each token:
        expert_idx = argmax(ρ^τ)
        output = FFN_{expert_idx}(LN(z'^τ)) + z'^τ

# 损失：L_total = L_regressive + 0.001 * L_aux
```

术语一般如何实现？如何使用？
- 在 MoE decoder layer 中将原始 linear router 替换为 DTR 模块
- DTR 引入的参数增量极小（约 34760 额外参数，仅占模型总参数 < 0.5%）
- 需要两张 embedding table 区分视觉/文本 token 的 modality routing
- 训练使用 DeepSpeed ZeRO-2_offload（Stage III 显存开销较高，因 expert 参数虽冻结仍需驻留）
- 可与 Expert Evolution 的自由演化 expert 配合（Stage II 的产品作为 Stage III 的初始化）

涉及论文标题：
- EvoMoE: Expert Evolution in Mixture of Experts for Multimodal Large Language Models

---
