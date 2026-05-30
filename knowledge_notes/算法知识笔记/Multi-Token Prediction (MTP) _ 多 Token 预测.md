## Multi-Token Prediction (MTP) / 多 Token 预测

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Multi-Token Prediction (MTP) 是一种辅助训练目标，使模型在每个位置预测多个 future token（而非仅 next token），在推理时可用作 speculative decoding 的 draft model。LongCat-Flash 采用 dense layer（非 MoE layer）作为 MTP head，在训练中期（而非全程）引入 MTP 训练。

LongCat-Flash 的 MTP 设计要点：
1. **Single dense layer head**：使用单个 dense FFN layer（非 ScMoE/MoE layer）作为 MTP head，参数量仅为主模型的 1.41%，接受率 92.1%（vs ScMoE head 的 4.17% params, 92.9% accept rate）。以微小的接受率损失换取大幅减少的 draft 计算开销。
2. **Late-phase training**：MTP head 在训练的中间阶段引入（非从零开始），因为 MTP loss 收敛极快。过早引入可能干扰主模型训练。
3. **Speculative decoding integration**：MTP head 作为 draft model，接受率 >90%，配合 C2T (Classifier-based Tree Construction) 过滤低接受概率 token，实现 expected accept length $\Omega(\gamma, \alpha)$ 约 1.8x。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# MTP Training (per training step)

输入: hidden_states [batch, seq_len, d_model]
      主模型: 28-layer ScMoE Transformer

# Forward:
h = MainModel(hidden_states)  # [batch, seq_len, d_model]

# MTP head 预测
logits_next1 = MTP_Head(h[:, :-1])    # 预测下一个 token
logits_next2 = MTP_Head(h[:, :-2])    # 预测下下个 token (如果 MTP depth=2)

# Loss
loss_next1 = CrossEntropy(logits_next1, tokens[:, 1:])
loss_next2 = CrossEntropy(logits_next2, tokens[:, 2:])  # 如果 MTP depth=2
total_loss = main_loss + lambda_mtp * (loss_next1 + loss_next2)

# MTP Head 结构 (LongCat-Flash 选择 single dense layer):
# MTP_Head(x) = LayerNorm(x) @ W_mtp.T
# W_mtp: [d_model, vocab_size=131072]
```

推理时 MTP + C2T Speculative Decoding：
- Draft stage: Target model forward → MTP head → 生成 γ 个 draft tokens
- Filter stage: C2T classification model → 过滤低接受概率 token
- Verify stage: Target model forward with all draft tokens → 接受/拒绝每个 token

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
1. MTP 最初由 Gloeckle et al., 2024 提出，DeepSeek-V3 将其从 independent output heads 改进为 sequential prediction（每个 MTP head 间有因果依赖），但 LongCat-Flash 采用 simpler single dense layer head。
2. LongCat-Flash 选择 dense layer 而非 MoE layer 的关键 tradeoff：dense head 参数量少但 GPU 利用率高（decode batch 小），MoE head 接受率略高但 draft cost 大（需要 all-to-all 通信）。
3. C2T (Huo et al., 2025) 是 classifier-based tree construction——训练一个轻量分类器判断 draft token 是否可能被 target model 接受，提前过滤可减少 verification 开销。
4. TVD fusion: Target forward + Verification + Draft forward 融合为单个 CUDA Graph 减少 kernel launch overhead。

涉及论文标题：
- LongCat-Flash Technical Report
- Better & Faster Large Language Models via Multi-Token Prediction
