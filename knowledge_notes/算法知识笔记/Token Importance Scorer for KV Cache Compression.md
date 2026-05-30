## Token Importance Scorer for KV Cache Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Token Importance Scorer 是 KV-Distill 中的可训练 FFN，在 prefill 阶段预测每个 token 对后续推理的重要性。输入为 layer η=6 的 hidden states X'_η ∈ R^{N×d}，输出为重要性分数 s ∈ R^N。取 top-k 作为保留 token。top-k 不可微，梯度通过 attention 衰减路径传播：α' = sigmoid(s) ⊙ α，被选 token sigmoid≈1 不变，未选 token sigmoid≈0 被衰减，梯度通过此路径传至 scorer。

从算法pipeline角度拆解：

```
s = FFN_scorer(LM.layer_6_output(context))  # d → d/4 → 1
indices = torch.topk(s, k).indices          # 不可微，梯度通过以下路径:
α' = sigmoid(s) ⊙ α                         # attention weights 衰减
# ∂L/∂s = (∂L/∂α') * α * sigmoid'(s)       # 梯度传播
```

术语一般如何实现？如何使用？

2 层 FFN（中间维度 d/4），约 1-2M 参数。与 LoRA adapter 联合优化（总 150M 参数）。推理时 scorer 仅执行一次，开销可忽略。若 scorer 错误评分，重要 token 信息在压缩 KV cache 中永久丢失。

涉及论文标题：
- KV-Distill: Nearly Lossless Learnable Context Compression for LLMs

---
