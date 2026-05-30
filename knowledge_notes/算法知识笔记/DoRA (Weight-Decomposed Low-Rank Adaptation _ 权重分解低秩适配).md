## DoRA (Weight-Decomposed Low-Rank Adaptation / 权重分解低秩适配)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DoRA (Weight-Decomposed Low-Rank Adaptation) 由 Liu et al. (2024) 提出，是 LoRA 的变体。将预训练权重 W 分解为 magnitude（幅度 m）和 direction（W/||W||）分量独立更新：magnitude 通过可训练 scalar vector 直接更新（非低秩约束），direction 通过标准 LoRA（B·A 低秩分解）更新。相比标准 LoRA 仅用 BA 整体模拟 ΔW，DoRA 的 magnitude-direction 分解使学习模式更接近 full fine-tuning。在 MixLoRA 中，DoRA 被用作 expert 基础微调单元（替代 LoRA），构成 MixDoRA 变体。

从算法pipeline角度拆解术语：
```
// DoRA 线性层前向 (per expert)
输入: x [B, N, d_in], 预训练权重 W [d_out, d_in]
可训练参数: m [d_out], B [d_out, r], A [r, d_in]

W_norm = W / ||W||_c                           // column-wise L2 norm → unit direction
ΔW_dir = B · A                                  // [d_out, d_in], rank-r 方向更新
W_dir' = W_norm + ΔW_dir
W_updated = m · W_dir'                          // element-wise m 广播
y = W_updated · x
```
MixLoRA 中的 MixDoRA：每个 expert 的 FFN 权重按上式分解，8 experts, top-2, r=16。MixDoRA 对 load balance loss coefficient 更不敏感（禁用时仅降 ~1% vs MixLoRA 降 ~2.5%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 开源：HuggingFace PEFT `LoraConfig(use_dora=True)`。
- MixLoRA 实验：Gemma 2B 单任务 MixDoRA (71.6%) > MixLoRA (69.9%)；LLaMA-2 7B 多任务 MixLoRA (75.3%) ≈ MixDoRA (74.9%)。MoE 结构的微调多样性使 DoRA 的分解策略效果减弱。

涉及论文标题：
- MixLoRA: Enhancing Large Language Models Fine-Tuning with LoRA based Mixture of Experts

---
