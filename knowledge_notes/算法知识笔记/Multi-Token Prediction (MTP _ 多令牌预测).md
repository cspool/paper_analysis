## Multi-Token Prediction (MTP / 多令牌预测)

术语解释
Multi-Token Prediction (MTP) 是一种训练目标扩展技术，使 LLM 在每个 position 不仅预测下一个 token，还额外预测后续多个未来 token，从而稠化训练信号、提升数据效率。DeepSeek-V3 采用 D=1 depth 的 MTP 模块，不同于 Gloeckle et al. (2024) 的并行预测，DeepSeek 使用顺序预测并保持完整因果链。MTP 模块在推理时可丢弃或用于 speculative decoding（第二 token 接受率 85-90%，1.8× TPS 加速）。

术语是什么？
MTP 在 DeepSeek-V3 中的实现：(1) 1-depth MTP 模块（D=1），每个 position 额外预测第 2 个未来 token；(2) 每个 MTP 模块包含：shared embedding layer Emb(·)、shared output head OutHead(·)、独立 Transformer block TRM_k(·)、projection matrix M_k ∈ R^{d×2d}；(3) 保持完整 causal chain：h_i'^k = M_k[RMSNorm(h_i^{k-1}); RMSNorm(Emb(t_{i+k}))]；(4) 训练 loss：λ/D * Σ_k CrossEntropy(P_k, t)，其中 λ=0.3 (first 10T tokens) → 0.1 (last 4.8T tokens)。推理时可直接丢弃 MTP 模块，或保留用于 speculative decoding。

从算法pipeline角度拆解术语：
```
=== MTP Training Forward Pass (D=1) ===

Main Model:
  h_{1:T}^0 = MainTransformer(input[1:T])     // standard representation

MTP Module k=1:
  for i in 1..T-1:
    h_i'^1 = M_1 @ [RMSNorm(h_i^0); RMSNorm(Emb(t_{i+1}))]  // [d×2d] concat projection
  h_{1:T-1}^1 = TRM_1(h_{1:T-1}'^1)          // independent Transformer block
  P_{i+2}^1 = Softmax(OutHead(h_i^1))         // shared output head (with main model)

Loss:
  L_main = CrossEntropy(P_main[2:T+1], t[2:T+1])
  L_MTP^1 = CrossEntropy(P_{2+k:T+1}^1, t_{2+k:T+1})  // predict 2nd-next token
  L_total = L_main + (λ/D) * L_MTP^1
```

术语一般如何实现？如何使用？
消融实验（Table 4）：Small MoE (15.7B) 和 Large MoE (228.7B) 上 MTP 一致提升 benchmark 性能。推理时：(a) 直接丢弃 MTP 模块，主模型独立推理——MTP 的受益已融入主模型训练；(b) 保留 MTP 模块用于 speculative decoding——主模型预测 t_{n+1}，MTP 模块预测 t_{n+2}，接受率 85-90%，实现 1.8× TPS 加速。MTP 的思想也见于 EAGLE (Li et al. 2024b)，但 EAGLE 主要用于 speculative decoding，而 DeepSeek-V3 的 MTP 主要用于改善训练质量。

涉及论文标题：
- DeepSeek-V3 Technical Report
