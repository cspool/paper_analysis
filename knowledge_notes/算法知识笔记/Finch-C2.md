## Finch-C2

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Finch-C2是GoldFinch论文中提出的Finch (RWKV-6)时间混合器改进版，作为GoldFinch架构前2/3层的核心组件。四项改进：(1) 移除Gate（SiLU gating），用新的数据依赖第二Value (u'_t) 补偿性能损失，减少参数量；(2) 将per-head GroupNorm替换为跨所有head的LayerNorm；(3) key乘以(1-w_t)以保持kv-state行归一化（受HGRN2启发，HGRN2设key=1-decay，Finch-C2则乘而非设等）；(4) u'_t = u_t W^V + tanh(u_t W^{UD})W^{UU}，数据依赖的独立token-shifted第二Value，复用W^V权重（intentional参数节省）。Finch-C2在减少参数的同时性能略优于原Finch（消融L12 D768: Finch-C2 loss=2.7082 < Finch loss=2.7191）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// Finch-C2 Time Mixing (per-head):
d_t = lora_d(ddlerp_d(x_t, x_{t-1}))                  // data-dependent decay factor
w_t = exp(-exp(d_t))                                    // decay weight ∈ (0,1)
r_t = ddlerp_r(x_t, x_{t-1}) @ W^R                     // receptance
k_t = ddlerp_k(x_t, x_{t-1}) @ W^K · (1 - w_t)        // key × (1-decay) [创新]
v_t = ddlerp_{i,i}(x_t, x_{t-1}) @ W^V                 // first value
u_t = ddlerp_u(x_t, x_{t-1})                            // bonus raw
u'_t = u_t @ W^V + tanh(u_t @ W^{UD}) @ W^{UU}         // second value [创新]

// WKV linear attention:
wkv_t = diag(w_t) @ wkv_{t-1} + k_t^T @ v_t           // matrix state update
o_t = LayerNorm(concat(r_t @ wkv_t + u'_t)) @ W^O      // LayerNorm across heads [创新]
```
vs Finch原始: k_t无(1-w)乘积，gate applied to o_t (SiLU gating)，GroupNorm per head。Finch-C2可独立使用也可组合为GoldFinch。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Finch-C2每Time Mixing层5组LoRA (r/k/v/u rank-32, d rank-64)。Channel Mixing与Finch完全一致（lerp token shift + ReLU² FFN + σ(r) gate）。代码开源：https://github.com/recursal/GoldFinch-paper (Apache 2.0)。

涉及论文标题：
- GoldFinch__High_Performance_RWKV_Transformer_Hybrid_with_Linear_Pre-Fill_and_Extreme_KV-Cache_Compression

---
