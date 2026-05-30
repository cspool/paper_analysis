## WKV (Weighted Key-Value) Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
WKV 是 RWKV 核心注意力原语：带 channel-wise learned exponential decay 的 linear attention。每 channel 有独立 decay w∈(0,1) 和 boost u。**原始论文（RWKV-4, EMNLP 2023）**首次提出：向量 state（head size=1），带分母的 softmax-like 归一化 + Sigmoid receptance gating。公式：`wkv_t = (Σ_{i=1}^{t-1} e^{-(t-1-i)w+k_i}⊙v_i + e^{u+k_t}⊙v_t) / (Σ_{i=1}^{t-1} e^{-(t-1-i)w+k_i} + e^{u+k_t})`，其中 w 为非负通道级时间衰减（`e^{-(t-i)w}≤1`，确保历史信息指数衰减），u 为当前 token 的 bonus 参数（独立于衰减路径，让当前 token 获得特殊权重）。递归形式：`a_t = e^{-w}⊙a_{t-1} + e^{k_t}⊙v_t; b_t = e^{-w}⊙b_{t-1} + e^{k_t}; wkv_t = a_t/b_t`。数值稳定实现使用共享指数技巧：维护 p_t 存储 a_t,b_t 的公共指数，避免 exp 溢出。内部状态共 5 部分（x_t, y_t, a'_t, b'_t, p_t），总大小 5DL。Eagle (RWKV-5) 升级为矩阵 state s∈R^{(D/h)×(D/h)}（head=64），LayerNorm 替代分母，SiLU gating + 线性 receptance。Finch (RWKV-6) 将静态 w 升级为 data-dependent w_t。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Eagle WKV: w=exp(-exp(ω)); s_t=diag(w)·s_{t-1}+k_t^T·v_t; wkv=diag(u)·k_t^T·v_t+s_{t-1}; o_t=LayerNorm(r_t@wkv)
Finch WKV: d_t=lora_d(ddlerp_d(x_t,x_{t-1})); w_t=exp(-exp(d_t)); s_t=diag(w_t)·s_{t-1}+k_t^T·v_t
GoldFinch (Finch-C2) WKV改进：k_t = ddlerp_k(x_t,x_{t-1})W^K·(1-w_t)，key乘以(1-decay)以保持kv-state行归一化；移除Gate（减参数）；LayerNorm across all heads替代GroupNorm；第二Value u'_t = u_t W^V + tanh(u_t W^{UD})W^{UU}替代Finch的静态u(bonus)项。Finch-C2参数更少但性能优于原Finch。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
推理 RNN 模式 O(1) per token。训练 custom CUDA kernel SRAM-resident state。16k 序列 Finch kernel 比 Flash Attn v2 快 4.2×（A100）。

涉及论文标题：
- RWKV__Reinventing_RNNs_for_the_Transformer_Era
- RWKV-X__A_Linear_Complexity_Hybrid_Language_Model
- Eagle_and_Finch__RWKV_with_Matrix-Valued_States_and_Dynamic_Recurrence
- GoldFinch__High_Performance_RWKV_Transformer_Hybrid_with_Linear_Pre-Fill_and_Extreme_KV-Cache_Compression
- VisualRWKV__Exploring_Recurrent_Neural_Networks_for_Visual_Language_Models

---
