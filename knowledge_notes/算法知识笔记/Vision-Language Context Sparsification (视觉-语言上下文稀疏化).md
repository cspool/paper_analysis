## Vision-Language Context Sparsification (视觉-语言上下文稀疏化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Vision-Language Context Sparsification 是 Dynamic-LLaVA 提出的同时稀疏化多模态 LLM 推理中两种上下文（视觉 token 和语言 token）的框架。与仅稀疏化视觉 token 的方法（如 FastV）不同，Dynamic-LLaVA 使用两个可学习预测器在 prefill 阶段减少图像 token，在 decoding 阶段减少输出文本 token，实现整个 MLLM 生成过程的一致性高效推理。

核心动机（Eq. 4）：prefill 仅执行一次，image token 减少的收益在 decoding 阶段逐渐湮没——当输出文本 token 数量 |S_l^{OT}| → ∞ 时，Computation(Decoding_w/o_cache)_l ∝ |S_l^{OT}|，Memory(Decoding_w/cache)_l ∝ |S_l^{OT}|。仅减少 image token 无法在长生成中持续受益。Dynamic-LLaVA 是首个同时稀疏化 vision 和 language 上下文的 MLLM 高效推理框架。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Dynamic-LLaVA 三模式稀疏化推理 Pipeline**：

```
超参数: l=2, r^I=0.2, r^OT=0.5

=== Prefill 阶段 (Eq. 5)——仅 image token 稀疏化 ===
S_l^I, S_l^T = LLM_layers_1_to_l(image_tokens, text_tokens)
D^I = P^I(S_l^I)                          // Image Predictor: [N_l^I, d] → [N_l^I, 2]
M^I = argmax_j(D^I)                        // j=0 discard, j=1 keep
S_l^{I*} = {S_{l,i}^I | M_i^I == 1}       // 保留 r^I≈20% image tokens (~115/576)
S_l^{P*} = S_l^{I*} ∪ S_l^T               // 后续 L-l 层用缩减后 token 集

=== Decoding w/o KV Cache (Eq. 2 modified)——vision + language 同时稀疏化 ===
D^{OT} = P^{OT}(S_l^{OT})                 // Output Predictor: [N_l^{OT}, d] → [N_l^{OT}, 2]
M^{OT} = argmax_j(D^{OT})                 // M^{OT}_{N^{OT}} 强制=1（最后token始终保留）
S_l^{OT*} = {S_{l,i}^{OT} | M_i^{OT} == 1}  // 保留 r^OT≈50% 输出文本 token
S_{l+1} = LLM_layers_l+1_to_L(S_l^{P*} ∪ S_l^{OT*})  // 计算量减半

=== Decoding w/ KV Cache (Eq. 6)——在线 KV 压缩 ===
Q,K,V = W^{Q,K,V} · S_{l,N^{OT}}^{OT}
M^{OT}_{N^{OT}} = argmax(P^{OT}(S_{l,N^{OT}}^{OT}))  // 对当前token单点决策
O = W^O · Attention(Q, S_l^K ∪ K, S_l^V ∪ V)
if M^{OT}_{N^{OT}} == 1: S_l^K ∪= K, S_l^V ∪= V  // 保留 KV
else:                    S_l^K ∪= ∅, S_l^V ∪= ∅  // 丢弃 KV
S_{l+1,N^{OT}}^{OT} = FFN(O)
// 决策共享至所有后续层
```

术语一般如何实现？如何使用？

Predictor 架构：Image predictor 含 2 个 ViT blocks + MLP(512→256→128→2)；Output predictor 仅 MLP(512→256→128→2)。参数极小（<1% 总计算量）。一层预测，多层复用。训练时使用 MaskedSoftmax + Gumbel-Softmax(τ: 1.0→0.1) + STE 端到端优化。训练数据：LLaVA-1.5 656K Mixture（仅含图像样本）。代码开源：https://github.com/Osilly/dynamic_llava。

涉及论文标题：
- Dynamic-LLaVA: Efficient Multimodal Large Language Models via Dynamic Vision-language Context Sparsification

---
