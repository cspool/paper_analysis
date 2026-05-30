## Matrix-Valued State in RNNs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
将 RNN hidden state 从向量 s∈R^D 扩展为矩阵 s∈R^{(D/h)×(D/h)} per head。每 head 维护 K^TV 矩阵记忆库：K 各行作 input gate，V 分配到 state 各行，每行独立 decay。RWKV-4 state 为向量（head=1）；Eagle/Finch 为矩阵（head=64），总 state 从 5DL→66DL（~13×）。矩阵 state 编码 key-value 间二阶交互。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
s_t = diag(w)·s_{t-1} + k_t^T·v_t   # s∈R^{(D/h)×(D/h)}
# s[i,j]: 第i key通道 × 第j value通道的加权和
# diag(w): row i 以 w[i] 衰减
```
vs 向量: s_t = w⊙s_{t-1} + k_t⊙v_t（无跨通道交互）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
WKV 用 float32 计算。O(D²/h) FLOPs per token。消融：RWKV6-Pile avg 50.7% > RWKV4-Pile 47.7%（Table 18）。

涉及论文标题：
- Eagle_and_Finch__RWKV_with_Matrix-Valued_States_and_Dynamic_Recurrence
- GoldFinch__High_Performance_RWKV_Transformer_Hybrid_with_Linear_Pre-Fill_and_Extreme_KV-Cache_Compression

---
