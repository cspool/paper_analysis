## Router Rigidity

术语解释
MoE-tuning 中因使用共享静态线性 router 导致的 router 输出僵化现象：router 对所有类型的输入 token（视觉/文本）产生几乎相同的 expert 分布，无法根据输入模态和内容做针对性路由，限制了 MoE 在多模态场景的适应性。

术语是什么？
Router Rigidity 由 EvoMoE 论文通过 KDE（核密度估计）分析揭示：对线性 router 产生的视觉 token logits 和文本 token logits 分别做密度估计，发现两种模态的 logit 分布高度重叠，表明 router 对模态变化不敏感——无论输入是图像 token 还是文本 token，router 给出的 expert 分配几乎一样。

根本原因：传统 MoE-tuning 使用单一线性层 `router = Linear(hidden_dim, num_experts)` 做 expert selection。该线性 router 的参数在训练后固化，对所有 token 使用相同的 W_r 矩阵——没有机制区分 token 来自视觉编码器还是文本 tokenizer。

对 MLLM 的影响：
- 多模态 MoE 的核心价值在于"不同的 expert 处理不同类型的输入"——视觉 expert 处理图像、语言 expert 处理文本
- Router Rigidity 使得这一目标无法实现，所有 token 被均匀分配到所有 expert
- DTR 解决 Router Rigidity 后，可视化显示 visual expert 和 text expert 的激活模式明显分化

从算法pipeline角度拆解术语：
```
# Router Rigidity 诊断（原始 MoE-tuning 的线性 router）
# 输入：visual tokens V ∈ R^{P×C}，text tokens T ∈ R^{M×C}

# 线性 router forward：
logits_V = W_r @ V  # [P, N_experts]  ← 使用相同的 W_r
logits_T = W_r @ T  # [M, N_experts]  ← 使用相同的 W_r

# KDE 分析：
# 对 logits_V 和 logits_T 分别做核密度估计
# 发现两个分布几乎完全重叠 → Router Rigidity
# Expert 分配在不同模态间无差异

# DTR 解决方式（对比）：
Θ_up^V, Θ_down^V = H_V(V)  # 视觉专用 hypernetwork
Θ_up^T, Θ_down^T = H_T(T)  # 文本专用 hypernetwork
# 不同模态走不同的参数生成路径，router 输出自然分化
```

Router Rigidity 不同于 Router Collapse（router 总是选择同一 expert），也不同于 Expert Uniformity（expert 参数趋同）。三者可同时存在但属于不同层面的问题：Router Rigidity 是 router 层的问题，Expert Uniformity 是 expert 层的问题。

术语一般如何实现？如何使用？
- Router Rigidity 是需要诊断和避免的问题
- 诊断：KDE plot（论文方法）、模态间 expert 分配的 Jensen-Shannon 距离、token 来源和 expert 选择的互信息
- 解决：DTR（EvoMoE，hypernetwork 动态生成 router 参数）、modality-specific router（无 hypernetwork 也有改善）、RoE [ICLR 2025]（adapter-based layer skipping）
- 在纯文本 LLM MoE 中不存在此问题（所有 token 同质），仅在 MLLM 中有意义

涉及论文标题：
- EvoMoE: Expert Evolution in Mixture of Experts for Multimodal Large Language Models
