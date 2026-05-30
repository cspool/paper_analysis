## K Compression Cache

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

K Compression Cache 是 SeerAttention-R 中为加速 AttnGate 推理而设计的压缩 key 缓存。类似于标准 KV Cache 缓存原始 key/value 以避免重复计算，K Compression Cache 缓存的是经过 pooling + 线性投影压缩后的 key 表示，用于 AttnGate 的 K 分支快速计算。

核心设计：
- 只缓存压缩后的 K_gate（经 Max/Min/Avg pooling + W_k_gate 线性层），而非原始 K
- 更新策略：accumulate 直到生成 block_size 个新 token，才计算这 block_size 个 token 的压缩表示并追加到 cache
- 在 accumulate 期间，最后一个不完整的 block 始终被标记为"选中"（补偿 K Compression Cache 信息滞后）
- block_size=64 时，K Compression Cache 内存仅占原始 KV cache 的 1/128 (<1%)：cache 中每个 block 存 d_gate 维向量（而非 block_size × d_head × num_kv_heads 维完整 K）

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# K Compression Cache 更新与使用
class KCompressionCache:
    def __init__(self, num_layers, num_kv_heads, d_gate, block_size):
        # 内存: num_layers × num_blocks × num_kv_heads × d_gate
        # vs KV cache: num_layers × seq_len × num_kv_heads × (d_head × 2)
        # 比例: (1/block_size) × (d_gate / (2*d_head)) ≈ 1/128 (block_size=64)
        self.cache = []           # list of [1, num_kv_heads, d_gate] per block
        self.accumulated_tokens = 0
    
    def update(self, new_K_tokens, K_gate_projector):
        # new_K_tokens: 自上次更新后新生成的 K (pe + nope)
        self.accumulated_tokens += len(new_K_tokens)
        
        if self.accumulated_tokens >= block_size:
            # 取最近 block_size 个 token 的 K
            K_block = new_K_tokens[-block_size:]
            # Pooling + 线性投影
            K_pooled = concat([
                MaxPool(K_block, kernel=block_size),
                MinPool(K_block, kernel=block_size),
                AvgPool(K_block, kernel=block_size)
            ])  # [1, num_kv_heads, 3*d_head]
            K_gate_new = K_gate_projector(K_pooled)  # [1, num_kv_heads, d_gate]
            self.cache.append(K_gate_new)
            self.accumulated_tokens = 0
            return True  # cache 已更新
        return False  # 还在 accumulate，cache 未更新
    
    def get_full_cache(self):
        return stack(self.cache)  # [num_blocks, num_kv_heads, d_gate]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

K Compression Cache 在推理时与 KV Cache 并行维护。每次 decode step 中：AttnGate 先读取 K Compression Cache 计算块级分数，Top-K 选择后，再仅从 KV Cache 中加载被选中的原始 K/V blocks 计算 attention。由于 K Compression Cache 极小，它可以始终驻留在 GPU 显存中，而完整的 KV Cache 可以 offload 到 CPU/SSD，按需加载被选中的 blocks。这为极端长序列推理（如 128K+）提供了高效的混合内存管理方案。

涉及论文标题：
- SeerAttention-R: Sparse Attention Adaptation for Long Reasoning

---
