## Memory Cache with Retrieval (Video Context)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Memory Cache with Retrieval 是 VideoLLaMB 中解决递归记忆桥接层 BPTT（Backpropagation Through Time）梯度消失问题的机制。在每个 time step i，系统将所有历史 memory tokens 存储在 MemoryCache M_i = [m_1, ..., m_i] 中，使用当前 memory token m_i 作为 query、拼接的历史 cache M_i 作为 key 和 value，通过标准 multi-head cross-attention 检索历史信息并更新当前 memory：$m_{i+1} = \text{Softmax}(W_i^Q m_i (W_i^K M_i)^\top / \sqrt{d_k}) W_i^V M_i$。此机制的核心优势：(1) 提供直接的跨时间步信息通路，绕过 RNN 式的逐层梯度传播，缓解梯度消失；(2) 允许当前 memory 选择性地关注历史中相关的 memory 状态（而非简单平均或全量传递）；(3) memory cache 仅存储 32-dim memory tokens per step，额外存储开销极小（128KB per step @ fp32）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Memory Retrieval 的计算流程：

```
class MemoryRetrieval(nn.Module):
    """单层 Cross-Attention Retrieval"""
    def __init__(self):
        self.cross_attn = MultiheadAttention(embed_dim=1024, num_heads=8)
        self.norm = LayerNorm(1024)
    
    def forward(self, m_current, memory_cache):
        # m_current: [32, 1024]  ← 当前 Bridge Layer 输出的 memory
        # memory_cache: [32*K, 1024]  ← 所有历史 memory tokens 拼接
        
        # Cross-attention: query=当前memory, key/value=历史cache
        m_updated = m_current + self.cross_attn(
            query=self.norm(m_current),    # [32, 1024]
            key=memory_cache,               # [32*K, 1024]
            value=memory_cache              # [32*K, 1024]
        )
        # [32, 1024] × [32*K, 1024] → [32, 32*K] attention weights
        # → weighted sum over history → [32, 1024] updated memory
        
        return m_updated
```

检索过程的计算复杂度：O(32 * 32K * 1024) = O(M^2 * K * D) per step，其中 M=32 很小，K 随视频长度线性增长。总 Memory Retrieval 复杂度 O(M*K)。实践中 300s 视频（K≈75 at 4fps）的 retrieval 开销约 2ms，远小于 LLM 推理时间。

梯度流原理：
```
# 无 Retrieval (纯递归):
L → m_K → m_{K-1} → ... → m_1 (梯度逐层衰减，链长 K)
# 有 Retrieval (cross-attn shortcut):
L → m_K → [cross_attn: 直接读 m_1...m_{K-1}] (梯度有多条短路径)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VideoLLaMB 使用单层 Retrieval Layer（8 heads, hidden=1024），与 Bridge Layer 配置相同（Table 11）。实现上 memory cache 存储所有历史 memory tokens，但不需要反向传播通过整个 cache（仅通过 cross-attention 的 softmax 权重反向传播），因此避免了 BPTT 的长链问题。消融实验（Table 8）显示移除 retrieval 后 EgoSchema 性能下降 1.6 点（53.8→52.2），验证了 retrieval 对长视频理解的有效性。局限性：(1) memory cache 随视频长度线性增长（但 token 数极少），极长视频（>1000 segments）时自注意力计算可能成为瓶颈；(2) 论文未探索 retrieval 的 top-k 剪枝或稀疏注意力以进一步降低复杂度；(3) retrieval 机制引入了额外的训练参数（W_Q, W_K, W_V），增加了训练成本。

涉及论文标题：
- VideoLLaMB__Long-context_Video_Understanding_with_Recurrent_Memory_Bridges
