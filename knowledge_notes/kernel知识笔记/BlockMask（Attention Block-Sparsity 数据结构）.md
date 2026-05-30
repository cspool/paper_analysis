## BlockMask（Attention Block-Sparsity 数据结构）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BlockMask 是 FlexAttention 中用于编码 attention score 矩阵 block 级稀疏性的紧凑数据结构。它将 score 矩阵 $S \in \mathbb{R}^{B \times H \times Q\_LEN \times KV\_LEN}$ 按固定 block size（默认 128）划分为 $\lceil Q\_LEN/BS \rceil \times \lceil KV\_LEN/BS \rceil$ 个 block，然后通过两个张量编码哪些 block 包含至少一个未被 mask 的 score 元素：
- `kv_num_blocks [B, H, Num_Row]`：每行的非 oblivious block 数量
- `kv_indices [B, H, Num_Row, Num_Col]`：每行非 oblivious block 的列索引

内存开销为 $O(\lceil Q\_LEN/BS \rceil \times \lceil KV\_LEN/BS \rceil)$，远小于完整 score 矩阵的 $O(Q\_LEN \times KV\_LEN)$ 或 itemized mask 的 $O(N^2)$。BlockMask 在编译时通过 `create_block_mask()` 利用 `torch.vmap` 对用户定义的 mask_mod 进行批量评估生成。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
BlockMask 在 kernel 调度中作为间接内存访问的索引结构：

```
# BlockMask-guided kernel scheduling (per SM, per Q tile)
num_rows = Q_LEN / Q_BLOCK_SIZE
num_cols = KV_LEN / KV_BLOCK_SIZE

for row in range(num_rows):
    nz_blocks = kv_num_blocks[b, h, row]  # 该行非 oblivious block 数
    for i in range(nz_blocks):
        col = kv_indices[b, h, row, i]     # 下一个 block 的列索引
        
        # 预取下一个 KV tile（HBM -> SRAM）
        if i + 1 < nz_blocks:
            next_col = kv_indices[b, h, row, i + 1]
            prefetch(K_tile[next_col], V_tile[next_col])
        
        # 加载当前 KV tile
        K_tile = load_K(col * KV_BLOCK_SIZE)
        V_tile = load_V(col * KV_BLOCK_SIZE)
        
        # 计算 score tile
        S_tile = Q_tile @ K_tile^T
        
        # 根据 block 类型选择性应用 mask_mod
        if is_partial_block(row, col):
            S_tile = apply_mask_mod(S_tile, row, col)
        
        # 所有 block 应用 score_mod
        S_tile = apply_score_mod(S_tile, row, col)
        
        # 在线 softmax + PV GEMM
        update_online_softmax(O, l, m, S_tile, V_tile)
```

BlockMask 将 attention 变体的稀疏模式与 kernel 调度解耦——同一 kernel 代码可通过不同的 BlockMask 支持 causal、sliding window、document mask 等多种稀疏模式，无需修改 kernel。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
BlockMask 通过 `create_block_mask()` 生成：
```python
from torch.nn.attention.flex_attention import create_block_mask

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

block_mask = create_block_mask(causal_mask, B=1, H=1, Q_LEN=8192, KV_LEN=8192)
# block_mask.kv_num_blocks: [1, 1, 64]  (8192/128 = 64 rows)
# block_mask.kv_indices: [1, 1, 64, 64]
```

BlockMask 的内部实现使用 `torch.vmap` 向量化评估 mask_mod 对所有 (q_block, kv_block) 组合的结果，将 block size 内所有元素的 mask_mod 结果聚合（AND reduction）判断 block 是否为 oblivious（全部 False）。

涉及论文标题：
- Flex Attention: A Programming Model for Generating Optimized Attention Kernels
