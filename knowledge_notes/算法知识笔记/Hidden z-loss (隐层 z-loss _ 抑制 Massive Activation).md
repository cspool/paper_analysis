## Hidden z-loss (隐层 z-loss / 抑制 Massive Activation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Hidden z-loss 是 LongCat-Flash 提出的训练稳定性正则化技术，用于抑制 LLM 训练中出现的 massive activation 现象（某些 hidden state 元素幅度极大，达到 10^4-10^6 量级）。Massive activations 与训练中的 loss spike 强相关——Sun et al. [2024] 观察到这种相关性，LongCat-Flash 进一步确认并通过 hidden z-loss 解决。

公式：$$\mathcal{L}_Z = \frac{\lambda}{T} \sum_{t=1}^{T} \left( \log \sum_{i=1}^{|z_t|} \exp(\operatorname{abs}(z_t^i)) \right)^2$$ 其中 $\lambda$ 为极小的 loss coefficient，$z_t$ 为 final layer 输出（在 final layer norm 之前），$|z_t|$ 为 hidden state size，abs(\*) 为绝对值函数。

设计原理：通过在 LogSumExp(abs(z)) 上施加 L2 penalty，抑制 hidden state 中个别元素的 extreme magnitude。LogSumExp 近似 max(abs(z))——平滑且可微的 max 函数——因此 $\mathcal{L}_Z$ 惩罚 hidden state 的最大绝对值。内层 exp 放大极端值的影响，外层 log 平滑，平方使 penalty 随 max magnitude 超线性增长。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Hidden z-loss 计算 (per training step)

输入: z [batch, seq_len, d_model]  # final layer output, BEFORE final LayerNorm

# z-loss 计算:
abs_z = abs(z)                           # [batch, seq_len, d_model]
logsumexp_z = log(sum(exp(abs_z), dim=-1))  # [batch, seq_len], 近似 max(|z|)
z_loss_per_token = logsumexp_z ** 2       # [batch, seq_len]
L_Z = lambda * mean(z_loss_per_token)     # scalar, lambda 极小 (e.g., 1e-6)

# 总 loss:
total_loss = L_LM + alpha * L_LB + L_Z   # LM loss + Load Balance loss + z-loss
```

LongCat-Flash 实验（Figure 6）：使用极小的 λ（coefficient negligible）即可显著抑制 massive activation 现象（L2 norm of final layer hidden states 趋于稳定），且不 degrade training loss。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
1. λ 需极小（如 1e-6 或更小）以避免干扰主训练目标。即使 λ 很小，massive activation 的 magnitude 极大（10^4-10^6），loss contribution 仍足以驱动优化。
2. 与 Router z-loss [Zoph et al., 2022] 的区别：hidden z-loss 作用于 hidden states（所有层的最终输出），Router z-loss 作用于路由 logits。二者的共同点在于都用 LogSumExp 惩罚极端值。
3. 对于 BF16 训练，BF16 的动态范围有限（max ~3.4e38），massive activations 虽未直接溢出但增大数值误差风险。Hidden z-loss 降低 hidden state 的 magnitude，提高数值稳定性。

涉及论文标题：
- LongCat-Flash Technical Report
