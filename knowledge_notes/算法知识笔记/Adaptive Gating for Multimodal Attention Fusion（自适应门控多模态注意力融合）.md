## Adaptive Gating for Multimodal Attention Fusion（自适应门控多模态注意力融合）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Adaptive Gating 是 mPLUG-Owl3 提出的基于文本语义的门控机制，用于融合 self-attention（文本内信息）和 cross-attention（视觉补充信息）的输出。与 Flamingo 的固定 learnable scale 不同，Adaptive Gating 通过文本特征自身计算门控值 `g = Sigmoid(W_gate^T · H_text)`，使得每个 token 可以根据其语义需求动态决定从视觉模态摄取多少信息。例如：语义丰富的 token（名词、形容词）可能分配更低 g 值以获取更多视觉上下文，而功能词保持高 g 值。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# H_self  ∈ R^{L×D}:  self-attention 输出
# H_cross ∈ R^{L×D}:  cross-attention 输出
# W_gate  ∈ R^{D×1}:  可学习门控投影（每 HATB 层一个）

g = Sigmoid(H_text @ W_gate)       # g ∈ R^{L×1}, 逐 token 门控值
H_fused = H_self * g + H_cross * (1 - g)

# g ≈ 1.0: 信任文本内部信息（视觉与此 token 无关）
# g ≈ 0.0: 依赖视觉补充信息
# g ≈ 0.5: 均等融合
```
消融实验（Table 10）：
- 无 Adaptive Gating + 无 Shared LN + 无 MI-Rope: GQA 53.3, NLVR2 52.7, Mantis 41.9
- +Adaptive Gating: GQA 55.7 (+2.4), Mantis 47.9 (+6.0)
- +Adaptive Gating + Shared LN: GQA 58.1 (+4.8), TextVQA 49.7 (+5.1)
- 全配置 (+MI-Rope): NLVR2 59.5 (+6.8), Mantis 51.6 (+9.7)

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现为一个线性层 + Sigmoid：`nn.Linear(hidden_dim, 1) + nn.Sigmoid()`。W_gate 在 Stage 1 作为可训练模块（仅 ~D 参数），Xavier uniform 初始化。与固定 learnable scale 的区别：Adaptive Gating 是 per-token 动态的（每个 token 独立决策），而 learnable scale 是 per-layer 静态的（所有 token 同一权重）。梯度回传使模型学习"哪些文本 token 需要更多视觉信息"的语义判断。

涉及论文标题：
- mPLUG-Owl3__Towards_Long_Image-Sequence_Understanding_in_Multi-Modal_Large_Language_Models
