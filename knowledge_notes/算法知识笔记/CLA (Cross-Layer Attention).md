## CLA (Cross-Layer Attention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Cross-Layer Attention (CLA) 是 Brandon et al. (2024) 提出的 KV cache 压缩技术：相邻的 Transformer layers 共享同一组 KV cache，而非每层独立维护。CLA 从 "layer 维度" 压缩 KV cache，将 KV cache 大小从 O(l) 降至 O(l/s)，其中 s 为共享步长（通常 s=2，即每 2 层共享）。与 GQA 从 "head 维度" 压缩互补。

KV cache 内存比较（bf16 bytes）：
- MHA: 4 × H × d_h × l
- CLA (s=2): 2 × H × d_h × l（减半）
- GQA+CLA (G=8, s=2): 2 × G × d_h × l（Hunyuan-Large 最终方案, ~5% MHA）

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# CLA with share interval=2 (每2层共享KV)
# 64 layers → 32 组共享 KV pairs

for block_idx in range(32):  # 32 个 CLA pairs
    # Layer 2*block_idx 和 Layer 2*block_idx+1 共享 KV
    for layer_offset in [0, 1]:
        layer_id = 2 * block_idx + layer_offset
        if layer_offset == 0:  # first layer in pair: 计算并缓存 KV
            K, V = self_attn_proj(hidden[layer_id])
            kv_cache[layer_id] = (K, V)      # 写入共享 KV cache slot
        else:  # second layer: 复用 KV
            K, V = kv_cache[layer_id - 1]    # 读取前一层（同一pair)的 KV
        
        # attention 计算
        Q = self.query_proj(hidden[layer_id])
        output[layer_id] = attention(Q, K, V)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

CLA 实现相对简单：(1) 在 attention 模块中，根据 layer_id 判断该层是否需要自己计算 KV（如 layer_id % share_interval == 0）或复用前一层的 KV。需要计算 KV 的层才分配 KV cache 空间。(2) 推理时只需维护共享的 KV cache entries。Hunyuan-Large 在 64 layers 中每 2 层共享，仅需 32 组 KV cache。CLA 带来的细微性能损失在实践中可忽略，而内存节省显著（~50% KV cache 减少，与 GQA 叠加后 ~95%）。

涉及论文标题：
- Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent
