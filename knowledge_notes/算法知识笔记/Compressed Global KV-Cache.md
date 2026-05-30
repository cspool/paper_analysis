## Compressed Global KV-Cache

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Compressed Global KV-Cache是混合RNN-Attention模型中通过三项设计实现极致缓存压缩的技术：(1) Global共享——所有attention层共享RNN最终层输出的单份压缩key cache（消除n_layer因子）；(2) 无Value Cache——value不缓存而由原始embedding按需重建（仅存token index≈2 bytes/token）；(3) Key低秩压缩——16:1压缩比（D→D/16），通过W^{KD}+TokenCat编码-解码。总cache=(1+D/16)元素per token。GoldFinch达到756-2550× cache缩小，256K context仅0.068GB vs Llama 128GB。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// Cache存储:
store: [c_t ∈ R^{D/16}, idx_t]  per token  // 总: (D/16+1) elements

// 解压重建:
K_all[t] = RMSNorm(concat(emb[idx_t], c_t) @ W^{KU})   // from compressed cache
V_all[t] = emb[idx_t]                                   // from token indices

// Size对比 (256k ctx, D=4096, 32 layers):
// LLlama:    2·4096·32·256K·2 = 128 GB
// GQA(8g):   8·128·32·256K·2   = 16.8 GB
// YOCO:      2·4096·256K·2     = 4 GB
// GoldFinch: (1+256)·256K·2    = 0.068 GB
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
解压和token shift可在contiguous region上增量执行降低VRAM峰值。长context fine-tuning仅更新GOLD层（冻结Finch-C2），约3× FLOPs节省。推理pre-fill O(1) per token（仅Finch-C2），decoding O(N)但通常很短。开源：https://github.com/recursal/GoldFinch-paper。

涉及论文标题：
- GoldFinch__High_Performance_RWKV_Transformer_Hybrid_with_Linear_Pre-Fill_and_Extreme_KV-Cache_Compression

---
