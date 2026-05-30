## KTO (Kahneman-Tversky Optimization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
KTO (Kahneman-Tversky Optimization) 是一种基于前景理论（Prospect Theory）的 LLM 对齐方法，由 Ethayarajh et al. (2024, ICML 2024) 提出。KTO 的核心创新在于：仅需单样本偏好标签（"chosen" 或 "rejected"，如 👍/👎 二值信号），而不像 DPO 那样需要 pairwise preference data（chosen vs rejected 成对数据）。KTO 直接最大化生成的效用（utility），借鉴 Kahneman & Tversky (1992) 的前景理论——特别是损失厌恶（loss aversion）——人类对损失的敏感度高于等量收益。KTO 属于 HALOs（Human-Aware Losses）损失函数家族。在 EVA 中，KTO 被用于三阶段训练的第二阶段（SFT → KTO → GRPO），作用是：SFT 训练后的模型学会了 tool-call 格式但仍有典型失败模式（如视觉证据不足时猜测、欠采样、过采样），KTO 通过 63% chosen + 37% rejected 的数据让模型学习 fine-grained 策略偏好，使其偏好有效策略而避免已知失败模式。相比 DPO，KTO 不需要多轮对话共享回合的前提（这在 EVA 的多轮交互设置中会截断策略），更适合 EVA 的 multi-turn 场景。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
KTO loss 的核心计算：
```
# KTO Loss (简化形式)
# 输入: (x, y, label) where label ∈ {chosen, rejected}
# π_θ: 当前策略, π_ref: 参考策略 (SFT模型)

# 计算 log-ratio
r_θ(x,y) = log(π_θ(y|x) / π_ref(y|x))

# KTO loss per sample
if label == chosen:
    L_KTO = -λ_chosen * σ(β * r_θ(x,y) - z_ref)
else:  # rejected
    L_KTO = -λ_rejected * σ(z_ref - β * r_θ(x,y))

# σ: sigmoid, z_ref: 参考点(人类对收益/损失的不对称感知)
# β: 控制对 reference model 偏离的惩罚强度
# λ: chosen/rejected 样本的权重超参数
```

在 EVA 中使用时：
- 63% chosen（高质量成功轨迹）+ 37% rejected（SFT 构建过程中收集的错误轨迹）
- chosen data: LLM-as-Judge 筛选推理过程有足够 visual tokens 且正确回答的轨迹
- rejected data: LLM-as-Judge 筛选 visual tokens 不足但仍强行生成答案的轨迹（guessing 模式）+ 重新采样的高质量成功轨迹
- β=0.1, lr=2e-6, 1 epoch

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
KTO 在 HuggingFace TRL 库中有标准实现（`KTOTrainer`）。核心优势：(1) 不需要成对偏好数据，可以使用真实 chat logs (👍/👎)；(2) 适合 continual production fine-tuning；(3) 比 DPO 在非成对设置下更灵活。在 EVA 中，KTO 作为 GRPO 之前的"纠错阶段"，通过纠正已知 bad cases 来提升 GRPO 在线优化的收敛性、鲁棒性和稳定性。局限性：在成对偏好数据设置下 DPO 可能优于 KTO，β 超参数调优很关键。

涉及论文标题：
- EVA__Efficient_Reinforcement_Learning_for_End-to-End_Video_Agent
