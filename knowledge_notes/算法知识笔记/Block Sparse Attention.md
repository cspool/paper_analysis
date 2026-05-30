## Block Sparse Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Block Sparse Attention（块稀疏注意力）是一种通过block粒度稀疏化attention score矩阵来降低$O(n^2)$计算复杂度的方法。与element-wise sparse attention不同，block sparse attention将Q和KV分别划分为blocks（如block size=128），根据先验知识定义哪些(Q-block, KV-block) pair需要计算（FB=Full）、哪些部分计算（CB=Causal，需逐元素mask）、哪些完全跳过（EB=Empty）。Block粒度保留了GPU Tensor Core友好的dense tile数据布局，避免了per-element sparse indexing的control divergence overhead。常见pattern：Causal Attention（下三角mask）、Strided Attention（对角线banded pattern）、Global+Local Attention（Longformer/BigBird，全局+局部窗口）、Star Attention（anchor block+其余causal）、Streaming Attention（attention sink + 最近token窗口）。

从算法pipeline角度拆解，block sparse attention在tiled attention kernel中的流程：
```
for each (q_block, kv_block) in attention mask:
    if (q_block, kv_block) in EB: continue  # 完全跳过
    Q_tile = load_to_sram(Q[q_block])
    K_tile = load_to_sram(K[kv_block])
    scores = Q_tile @ K_tile.T  # Tensor Core MMA
    if (q_block, kv_block) in CB:
        scores = mask_apply(scores, causal_mask)  # 仅CB blocks需要
    # online softmax with running (m, l, O)
    m_new = max(m_old, rowmax(scores))
    O_acc = O_acc * exp(m_old - m_new) + exp(scores - m_new) @ V_tile
    l = l * exp(m_old - m_new) + rowsum(exp(scores - m_new))
    m_old = m_new
O = O_acc / l  # final normalization
```
**Annotations**: EB blocks完全跳过（节省$block\_size^2$ FLOPs），CB blocks内部仍需逐元素mask（因下三角跨越block边界），FB blocks零mask overhead。

术语一般如何实现？如何使用？单GPU kernel：FlexAttention (PyTorch)通过create_block_mask生成BlockMask（kv_num_blocks + kv_indices紧凑编码），FlashInfer用BSR格式支持block sparse。分布式系统：UltraAttn首次提出通用CP系统支持block sparse attention，将FB/CB/EB集合编码为ILP约束自动解决分布式负载均衡。适用场景：long-context LLM training (Longformer, BigBird, SAMBA)、video generation (VideoGPT strided)、vision (Swin Transformer shifted window)、inference with attention sink (Star Attention, StreamingLLM)。

涉及论文标题：
- UltraAttn: Efficiently Parallelizing Attention through Hierarchical Context-Tiling
- FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness
