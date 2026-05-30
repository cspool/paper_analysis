## KL Divergence Distillation for KV Cache Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

KL Divergence Distillation for KV Cache Compression 是 KV-Distill 论文提出的核心训练目标，通过 KL 散度直接匹配压缩前后 KV cache 产生的 next-token 概率分布。与传统 auto-encoding loss（重建被压缩 token 本身）不同，KL 散度蒸馏在 token 预测分布层面进行优化，消除了 pretraining-inference mismatch。使用加权组合 L(θ) = λ·D_KL(p||q_θ) + (1-λ)·D_KL(q_θ||p)，其中 p=完整 KV cache 的 next-token 分布（teacher），q_θ=压缩后分布（student）。Forward KL 为 mean-seeking（覆盖所有高概率输出），Reverse KL 为 mode-seeking（集中在高概率区域）。λ=0.6 偏 forward KL，因为 reverse KL 的 L1 梯度主导。

从算法pipeline角度拆解：

```
# Teacher (frozen LM, 完整 KV cache)
p = softmax(LM.decode(X_full))
# Student (LoRA-adapted LM_θ, 压缩 KV cache ˜X)
q_θ = softmax(LM_θ.decode(˜X))
# Loss
L = 0.6 * Σ p·log(p/q_θ) + 0.4 * Σ q_θ·log(q_θ/p)
```

LLAMA-3 SQuAD 20% retention: weighted KL 86.0% vs forward-only 83.4% vs reverse-only 82.7% vs AE+CE 79.1%。

术语一般如何实现？如何使用？

teacher forward 在 torch.no_grad() 下执行，仅需一次完整 KV cache 编码。反向传播仅更新 150M 参数（LoRA adapter + scorer）。训练时随机采样 retention ratio（0.1%-80%），单一模型支持任意压缩率推理。

涉及论文标题：
- KV-Distill: Nearly Lossless Learnable Context Compression for LLMs

**补充（来自 X-EcoMLA）**：X-EcoMLA 使用 KL 散度蒸馏将更大 teacher 模型的知识传递给 upcycle 后的 MLA student 模型，公式为 L_θ = Σ_{t=1}^T KL[p(·|y_{1:t}, x, θ_T) || p(·|y_{1:t}, x, θ)]。与 KV-Distill 的"压缩前后同模型分布匹配"不同，X-EcoMLA 的蒸馏是"跨模型架构的知识迁移"——teacher 可以是与 base model 不同的更大模型（如 8B teacher 指导 1B student）。消融实验（Table 12）表明纯 CE loss 导致性能大幅退化（48.54 vs 52.77），而纯 KL 蒸馏（50.84）或 KL+CE 混合（50.93-50.98）均远优于纯 CE，验证了 teacher dark knowledge 对 MLA upcycling 的关键作用。训练数据为 OpenHermes-2.5 + GenQA + Infinity-Instruct（6.8B tokens），AdamW optimizer with β=(0.9, 0.98)，batch size=96。

涉及论文标题：
- X-EcoMLA__Upcycling_Pre-Trained_Attention_into_MLA_for_Efficient_and_Extreme_KV_Compression

---
