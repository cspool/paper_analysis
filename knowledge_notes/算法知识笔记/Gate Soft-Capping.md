## Gate Soft-Capping

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Gate Soft-Capping 是 xLSTM 7B 用于训练稳定性的技术，对 mLSTM 的输入门和遗忘门 pre-activations 应用 tanh-based 软上限：softcap_a(x) = a · tanh(x/a)。其中 cap value a=15 用于 gate pre-activations（输入/遗忘门），a=30 用于输出 logits。该函数在输入值接近 0 时近似线性（梯度≈1），在 |x| 远大于 a 时渐近饱和于 ±a（梯度≈0）。与 hard clipping（直接截断）不同，soft-capping 提供了平滑的饱和行为，不会产生零梯度区域，在抑制异常值的同时保持了可训练性。xLSTM 7B 在 160B token 消融实验中证实：无 soft-capping 的训练表现出更高的梯度 norm 方差和更差的验证 loss。类似技术也用于 Gemma-2 模型（logit soft-capping）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// Soft-Capping 函数
def softcap_a(x):
    return a * tanh(x / a)

// 性质分析:
// |x| << a: tanh(x/a) ≈ x/a → softcap_a(x) ≈ x (线性区域, 梯度≈1)
// |x| >> a: tanh(x/a) ≈ sign(x) → softcap_a(x) ≈ ±a (饱和, 梯度≈0)
// x = 0:    softcap_a(0) = 0, 梯度 = 1
// x = a:    softcap_a(a) = a·tanh(1) ≈ 0.762a, 梯度 = sech²(1) ≈ 0.42

// 在 xLSTM 7B 中的应用
// 每层 mLSTM 的 gate 计算:
ĩ_sc = softcap_15(w_i^T @ RMSNorm(x) + b_i)   // input gate
f̃_sc = softcap_15(w_f^T @ RMSNorm(x) + b_f)   // forget gate

// 最终输出 logit:
logits_sc = softcap_30(W_lm_head @ h_final)  // logit soft-capping

// vs hard clipping:
// clip(x, -c, c) 在边界处梯度为 0，可能导致 dead neuron
// softcap 在边界处梯度非零但很小，允许缓慢逃离饱和
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- PyTorch 实现：`a * torch.tanh(x / a)`
- 需要放在 gate pre-activation 计算之后、指数激活之前（对 gate）或 logit 输出之前（对 logit）
- 典型配置：gate cap=15, logit cap=30（xLSTM 7B）；Gemma-2 使用类似配置
- 适用于任何使用指数或大范围值门控的递归架构（线性 RNN、SSM、LSTM variants）
- 注意：与 LayerNorm/RMSNorm 互补——Norm 在统计层面稳定激活分布，soft-capping 在个体值层面防止极端异常值

涉及论文标题：
- xLSTM_7B__A_Recurrent_LLM_for_Fast_and_Efficient_Inference

---
