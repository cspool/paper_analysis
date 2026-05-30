## Cross-Self Pruning (CSP) / Cross-Self Attention Decomposition for VLM KV Cache

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Cross-Self Pruning (CSP) 是一种 training-free 的 KV Cache 剪枝方法，专为多模态视觉语言模型（VLM）设计。核心创新是将原始注意力矩阵 A ∈ [0,1]^{L×L}（L = L_t + L_v，文本+视觉 token 长度）分解为四个子区域：(a) A^{st} ∈ [0,1]^{L_t×L_t}：文本→文本 self-attention；(b) A^{sv} ∈ [0,1]^{L_v×L_v}：视觉→视觉 self-attention；(c) A^{ct} ∈ [0,1]^{L_v×L_t}：视觉→文本 cross-attention；(d) A^{cv} ∈ [0,1]^{L_t×L_v}：文本→视觉 cross-attention。然后在 intra-modality（A^s = Σ_query A^{st} ⊕ Σ_query A^{sv}）和 inter-modality（A^c = Σ_query A^{ct} ⊕ Σ_query A^{cv}）两个维度上独立进行 top-K 选择，分别得到 binary mask M^s 和 M^c。最终保留的 token 必须在两个维度上都被判定为重要：M = M^s ∧ M^c（取交集）。

该设计解决了现有多模态 KV Cache 剪枝方法的核心缺陷：文本 token 的 self-attention scores 通常大于视觉 token，统一对待会导致关键视觉 token 被过度剪枝，破坏跨模态交互。通过独立评估 intra- 和 inter- 两个维度，CSP 确保视觉 token 即使在 self-attention 中得分较低，若在 cross-attention 中被文本 token 高度关注（说明跨模态信息重要），仍会被保留。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**CSP 在 VLM 推理中的伪代码流程**：

```
// 超参数：budget T, 最近窗口 R, 观察窗口 O
for each decoding iteration:
    if L_k < T: return K, V
    A = n-Softmax(O[-O:, :-R])      // n=1 平滑恢复
    A^{st}, A^{sv}, A^{ct}, A^{cv} = decompose(A)
    A^s = sum(A^{st}, axis=q) ⊕ sum(A^{sv}, axis=q)
    A^c = sum(A^{ct}, axis=q) ⊕ sum(A^{cv}, axis=q)
    M^s = TopK(A^s, K^s); M^c = TopK(A^c, K^c)
    M = M^s ∧ M^c                  // 双维度交集
    K = (K ⊙ M) ⊕ K[-R:]; V = (V ⊙ M) ⊕ V[-R:]
```

术语一般如何实现？如何使用？

CSP 以即插即用方式集成到 LLaVA 等 VLM 推理流程，仅修改 Attention 层的 token selection。默认 n=1, cross_ratio=0.5。在 MileBench 上 LLaVA-v1.5-13b 的 IR +9.6%、T-3 +8.3%。代码开源：https://github.com/TerryPei/CSP。

涉及论文标题：
- Cross-Self KV Cache Pruning for Efficient Vision-Language Inference

---
