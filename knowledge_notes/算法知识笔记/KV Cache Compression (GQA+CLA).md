## KV Cache Compression (GQA+CLA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

KV Cache Compression 是 Hunyuan-Large 联合使用 GQA 和 CLA 两种技术，从两个维度压缩 KV cache 内存占用的策略：

1. **Head 维度（GQA）**：将 KV heads 数从 80 (MHA) 压缩到 8 (GQA)，10× 压缩
2. **Layer 维度（CLA）**：每 2 层共享 KV cache，2× 压缩
3. **联合效果**：KV cache 从 4×H×d_h×l 降至 2×G×d_h×l，仅为 MHA 的 `G/(2H) = 8/160 = 5%`

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Hunyuan-Large 每层 MoE block 的 KV cache 使用模式：

```
# 64 layers, GQA (G=8), CLA (share every 2 layers)
# KV cache slots: 32 (每2层1个)

for layer_id in range(64):
    hidden = RMSNorm(input)
    
    # CLA: 判断是否需要计算 KV
    if layer_id % 2 == 0:
        K, V = proj_kv(hidden)  # GQA: 仅8组KV
        cache_slot = layer_id // 2
        kv_cache[cache_slot] = (K, V)  # 存储KV
    else:
        K, V = kv_cache[layer_id // 2]  # 复用前一层KV
    
    # Attention with GQA (8 KV groups × 80 query heads = 10:1 ratio)
    Q = proj_q(hidden)        # [B, L, 80×d_k]
    attn_out = gqa_attention(Q, K, V, num_kv_groups=8)
    
    # MoE FFN
    moe_out = shared_expert(hidden) + top1_specialized_expert(hidden, router)
    
    input = input + attn_out + moe_out
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GQA+CLA 联合压缩无需特殊硬件支持——可在标准 PyTorch 中实现。GQA 通过修改 attention 的 KV 投影矩阵维度实现（`nn.Linear(d_model, G*d_head)` 替代 `nn.Linear(d_model, H*d_head)`）；CLA 通过在各层间共享 KV cache buffer 实现。两者组合的工程实现要点：(1) 验证 GQA 的 KV head 数量选择（太小编码能力下降），(2) 验证 CLA 的共享步长（太大影响层间表示多样性），(3) 与 FlashAttention 兼容。Hunyuan-Large 选择了保守的参数（G=8, s=2），在 ~95% KV cache 节省下无显著性能损失。

涉及论文标题：
- Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent
