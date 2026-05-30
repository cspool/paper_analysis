## Hybrid Language Model (Linear RNN + Sparse Attention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
混合语言模型（Hybrid Language Model）是一种将不同类型的 token-mixing 机制（如 linear RNN/SSM + attention）在层级别组合的 LLM 架构。与纯 Transformer（全 attention, O(N²) 复杂度）和纯线性 RNN（全 recurrence, state capacity 有限）不同，混合模型利用不同层的互补优势：线性 RNN 层提供高效 short-range modeling（O(1) per-token），attention 层提供精确 long-range retrieval。早期混合模型（Jamba, Zamba, MiniMax）使用 full attention 层 + Mamba/SSM 层，保持了 O(N²) 瓶颈。RWKV-X 首次提出全线性复杂度混合架构——使用 Top-k Chunk Sparse Attention（O(N) training）替代 full attention 并结合 KV Cache Management（O(1) decoding），实现真正的线性复杂度混合模型。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
RWKV-X 混合架构的层级配置与数据流：
```
# 模型配置: 40 layers (32 original RWKV-7 + 8 Sparse Attention)
# Layer pattern: RWKV-7, RWKV-7, RWKV-7, SparseAttn, RWKV-7, ...
# 即 N:1 = 4:1 ratio (25% attention layers, 验证为最优)

Input: x ∈ R^{B×L×D}
h = embedding(x)

For layer l in 1..40:
    if l % 4 == 0:  # Sparse Attention block (每第4层)
        # O(kBN) training, O(1) decoding with compressed KV cache
        h_norm = RMSNorm(h)
        h_attn = TopKChunkSparseAttention(h_norm)
        h = h + h_attn  # residual
        h_norm2 = RMSNorm(h)
        h_ffn = SwiGLU_FFN(h_norm2)
        h = h + h_ffn  # residual
    else:  # RWKV-7 block
        # O(N) training (parallel scan), O(1) decoding (recurrent)
        h_norm = RMSNorm(h)
        h_time = TimeMixing_WKV(h_norm)  # Generalized Delta Rule
        h = h + h_time  # residual
        h_norm2 = RMSNorm(h)
        h_chan = ChannelMixing_FFN(h_norm2)
        h = h + h_chan  # residual

output = LM_head(RMSNorm(h))
```

混合架构的设计要点：(1) 注意力层比例：消融实验（Figure 5）表明 25% 注意力层比例在 126M 参数模型上实现最优 validation loss（纯 RWKV-7=0% 和纯 Sparse Attention Transformer=100% 均不如混合）；(2) 交错插入：注意力层均匀分布（而非集中），使每个抽象层级都有 long-range retrieval 能力；(3) 两阶段训练：alignment（仅训练新注意力层）+ long-context pretraining（全参数微调）确保混合架构的稳定收敛；(4) 无位置编码：RWKV-7 的递归已提供隐式位置信息，消融证明 No Pos 优于 Abs Pos/ROPE。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
RWKV-X 开源：https://github.com/howard-hou/RWKV-X。一般混合模型实现方式：(1) 选择主干线性 RNN/SSM 模型（RWKV、Mamba 等）的 checkpoint；(2) 使用 block expansion 方法插入 attention 层；(3) 分阶段训练（先冻结主干、后全参数）。混合比例通过小规模消融实验确定（论文中 12 层 126M 模型探索 0%-100% attention 比例）。适用于：需要同时满足短上下文 competitive performance 和长上下文 strong retrieval 的通用 LLM 训练场景。

涉及论文标题：
- RWKV-X__A_Linear_Complexity_Hybrid_Language_Model

---
