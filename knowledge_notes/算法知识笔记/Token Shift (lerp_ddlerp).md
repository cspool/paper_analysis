## Token Shift (lerp/ddlerp)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token Shift 是 RWKV 轻量时序信息混合机制，类似 kernel size=2 causal conv 但复用参数。Eagle 用静态 lerp(a,b)=a+(b-a)⊙μ（learnable per-channel 混合比）。Finch 用 ddlerp(a,b)=a+(b-a)⊙lora(a+(b-a)⊙μ_x)，lora(x)=λ+tanh(xA)B（A∈R^{D×32},B∈R^{32×D}），使混合比依赖输入内容。允许单层形成 induction heads，替代显式位置编码。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Eagle lerp: r_t = (x_t+(x_{t-1}-x_t)⊙μ_r)@W_r
# Finch ddlerp: lora_r(x)=λ_r+tanh(x@A_r)@B_r
#   r_t = (x_t+(x_{t-1}-x_t)⊙lora_r(x_t+(x_{t-1}-x_t)⊙μ_x))@W_r
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
同时用于 Time Mixing(r,k,v,g) 和 Channel Mixing(r',k')。前一 token 存 state（每层 2D float）。替代显式位置编码使 RWKV 可处理任意长度。GoldFinch进一步引入DDLoRAdapt: loradapt_□(x)=x+tanh(xC_□)D_□，在ddlerp基础上再叠加data-dependent additive LoRA偏移，用于GOLD Attention中的key和value生成。

涉及论文标题：
- RWKV__Reinventing_RNNs_for_the_Transformer_Era
- Eagle_and_Finch__RWKV_with_Matrix-Valued_States_and_Dynamic_Recurrence
- GoldFinch__High_Performance_RWKV_Transformer_Hybrid_with_Linear_Pre-Fill_and_Extreme_KV-Cache_Compression

---
