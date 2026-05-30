## TokenCat (Token ConCatenation for KV-Cache Compression)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TokenCat是GoldFinch的KV-Cache压缩/解压机制，通过两步将Finch-C2层输出压缩至极小全局key cache。第一步（压缩）：取Finch-C2最终层输出x_t∈R^D，乘全局矩阵W^{KD}∈R^{D×(D/16)}压缩为c_t=x_t·W^{KD}∈R^{D/16}（16:1压缩），每token仅需D/16元素存储。第二步（解压）：拼接压缩key c_t与原始embedding x_t^0为concat(x_t^0,c_t)∈R^{D+D/16}，乘全局矩阵W^{KU}∈R^{(D+D/16)×D}并RMSNorm得proto-keys k_t^D供所有GOLD层共享。类似LoRA低秩分解思路。16:1 vs 1:1压缩loss差异可忽略（均为2.2762），验证几乎无损。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// 压缩 (after last Finch-C2 layer):
c_t = x_t @ W^{KD}           // W^{KD} ∈ R^{D×(D/16)}, global matrix

// 解压 (shared by all GOLD layers):
x_t^0 = embedding_lookup(idx_t)
k_t^D = RMSNorm(concat(x_t^0, c_t) @ W^{KU})  // W^{KU} ∈ R^{(D+D/16)×D}, global

// KV-Cache size:
// Traditional: 2·d_model·n_layer·ctx_len → Llama 256k ctx=128GB
// GoldFinch: (1+d_model/16)·ctx_len → 256k ctx=0.068GB (D=4096)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
W^{KD}和W^{KU}为全局参数（非per-layer），所有GOLD层共享同一proto-keys。C_t和idx_t常驻VRAM，keys on-the-fly解压。支持增量解压以降低VRAM峰值。开源实现：https://github.com/recursal/GoldFinch-paper。

涉及论文标题：
- GoldFinch__High_Performance_RWKV_Transformer_Hybrid_with_Linear_Pre-Fill_and_Extreme_KV-Cache_Compression

---
