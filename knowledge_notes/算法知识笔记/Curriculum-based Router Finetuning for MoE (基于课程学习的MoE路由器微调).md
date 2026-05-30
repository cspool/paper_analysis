## Curriculum-based Router Finetuning for MoE (基于课程学习的MoE路由器微调)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Curriculum-based Router Finetuning 是 MoE-Prism 提出的可选低成本微调策略，用于在 sub-expert 分解后提升 router 的 routing 质量。核心思路：(1) **参数效率极高**：仅微调 gating network/router（线性层，占总参数 <0.1%），冻结所有其他权重（expert FFN、attention、LayerNorm 等）；(2) **课程学习**：训练过程中逐步增加激活 sub-expert 数量 k（如从 k=8 递增到 k=24/32），而非固定 k 值。这使 router 学会在不同资源预算下做出高质量的 routing 决策，而非仅针对单一 k 值过拟合。

从算法pipeline角度拆解术语：
```
# Curriculum Router Finetuning in MoE-Prism
router = Linear(d_model, N_sub_experts)  # 仅此参数可训练
k_min, k_max = 8, 24  # 或 8→32
total_steps = len(dataloader) * epochs
# 冻结所有其他参数
for param in model.parameters():
    param.requires_grad = False
router.weight.requires_grad = True

for step, batch in enumerate(dataloader):
    # 课程调度: k随训练进度线性递增
    k_current = k_min + (k_max - k_min) * (step / total_steps)
    h = model.forward_to_router(batch)  # 冻结部分前向
    router_logits = router(h)  # [B, N_sub_experts]
    top_k_idx, top_k_probs = top_k(softmax(router_logits), k_current)
    # 仅选中的sub-experts计算输出
    output = weighted_sum(probs * sub_expert_ffn(h) for ...)
    loss = cross_entropy(output, labels)
    loss.backward()  # 仅更新router参数
```
关键设计：与标准 MoE 微调（固定 k）不同，课程训练使 router 暴露于多种 k 值，学习到在不同计算预算下的灵活 routing 策略。这与 Chen et al. (2023) "Sparse MoE as the New Dropout" 的渐进式训练理念一致。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- MoE-Prism 实现：LR=1e-5，训练集为 SlimPajama 的 200K 序列，batch_size=32（Deepseek/Qwen）或 64（OLMoE），k 从 8 线性递增到 24（Deepseek）或 32（OLMoE/Qwen）。
- 微调后 PPL 通常优于原始模型（如 OLMoE K=12: 原模型 15.72, LG w/FT 14.68），且下游任务（Winogrande, ARC-C, SciQ, BoolQ）保持或提升。
- 这一策略与全参数微调或 LoRA 等常见方法不同：它仅微调一个线性层的参数，比 LoRA（通常加 adapter 到 attention + FFN）更轻量。

涉及论文标题：
- MoE-Prism: Disentangling Monolithic Experts for Elastic MoE Services via Model-System Co-Designs

---
