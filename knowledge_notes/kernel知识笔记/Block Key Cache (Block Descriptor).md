## Block Key Cache (Block Descriptor)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Block Key Cache（或称 Block Descriptor）是 block-sparse attention 中用于快速近似匹配每个 KV block 与当前 query 相关性的元数据结构。每个 block 由一对向量 (k_block_min, k_block_max) ∈ R^{d×2} 描述——分别是该 block 内所有 token 的 key 向量的元素级最小值和最大值。这一设计基于 Quest 算法，核心思想是：用 min/max 界描述 block 内 key 分布，估计 query 与 block 的最大可能 attention score，从而在不加载完整 token-level KV 的情况下快速筛选相关 block。

Block key cache 的大小为 O(M·d)——M 个 block，每 block 2d 个值（min + max）。相比完整 KV cache O(n·d) （n = M·b），内存开销仅为 2/b ≈ 12.5%（b=16 时）。block key cache 的在线更新（新 token 追加后增量更新其所在 block 的 min/max）也是 O(d) per token。

从kernel调度角度拆解术语：

```
// Block Key Cache 的数据结构与在线更新
struct BlockDescriptor:
    k_min: float[d]  // element-wise minimum of all keys in this block
    k_max: float[d]  // element-wise maximum of all keys in this block

// 在线增量更新 (per decode step)
def update_block_key_cache(B, new_token_k, block_id):
    // new_token_k 加入 block block_id
    B[block_id].k_min = elementwise_min(B[block_id].k_min, new_token_k)
    B[block_id].k_max = elementwise_max(B[block_id].k_max, new_token_k)
    // O(d) per token, 无需求全部 block 内 token

// Block Selection 使用 Block Key Cache
def score_block(B[block_id], q):
    // upper-bound attention score estimation
    score = 0
    for dim j in 0..d:
        score += max(q[j] × B[block_id].k_max[j], q[j] × B[block_id].k_min[j])
    return score
    // O(d) per block, 而非 O(b·d) per block (if full token scan)
```

术语一般如何实现？如何使用？

在 GPU kernel 实现中，block key cache 通常存储在连续显存区域，与 KV cache 分离。Block selection 阶段每个 SM 加载全部 block descriptors（内存开销小：M·2d bytes），在 register 中计算 scores，然后对 scores 做 top-n（可用 warp-level reduction + shared memory sorting）。ReSA 的 Flash Decoding-based kernel 中，block key cache 的刷新与 KV cache rectification 同步进行——确保 rectify 后的 KV cache 变化反映在更新后的 block descriptors 中，否则新的 sparse decoding 阶段会基于过时的 descriptors 产生不准确的 block selection。

涉及论文标题：
- Rectified Sparse Attention
