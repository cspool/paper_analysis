## Time Mixing and Channel Mixing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RWKV 每 residual block 两 Pre-LayerNorm 子层：Time Mixing（=Transformer attention，WKV 融合跨时间信息）和 Channel Mixing（=Transformer FFN，ReLU²+sigmoid gate 沿特征维变换）。原始 RWKV 论文（EMNLP 2023）首次提出这一架构：Time Mixing 使用 token shift→r/k/v 线性投影→WKV 带分母的 softmax-like 算子→Sigmoid(r)⊙wkv 输出门控；Channel Mixing 使用 token shift→r'/k'→Squared ReLU(k')=max(k',0)²→W'_v 线性投影→Sigmoid(r') 门控。两子块均输出 `o_t = W_o · (σ(r_t) ⊙ wkv_t)`（Time Mixing）和 `o'_t = σ(r'_t) ⊙ (W'_v · max(k'_t, 0)²)`（Channel Mixing）。借鉴 Gated MLP/MLP-Mixer 设计。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Time Mixing: Token Shift→r/k/v/g→WKV state→LayerNorm(r@wkv)→SiLU(g) gate→output。Channel Mixing: Token Shift→r'/k'→ReLU²(k')@W_v'→σ(r') gate→output。Eagle 缩 Channel Mixing hidden dim 至 3.5D 补偿新增 gate 参数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
两子层都用 residual connection。Finch 中两子层均用 ddlerp 替代 lerp。

涉及论文标题：
- RWKV__Reinventing_RNNs_for_the_Transformer_Era
- Eagle_and_Finch__RWKV_with_Matrix-Valued_States_and_Dynamic_Recurrence

---
