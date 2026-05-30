## Recurrent Memory Bridge Layers

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Recurrent Memory Bridge Layers（递归记忆桥接层）是 VideoLLaMB 中的核心模型组件，位于视觉编码器（ViT）和大语言模型（LLM）之间的 bridge 位置。它由单层 Transformer（8 attention heads, hidden size=1024）构成，在输入段前 prepend 固定数量的 learnable memory tokens（32 个），通过 self-attention 同时处理 memory tokens 和当前段视觉特征：`[m_{i+1}; o_i] = BridgeLayer([m_i; s_i])`，其中 m_i 是第 i 步输入的记忆 token，s_i 是第 i 个语义段的视觉特征，m_{i+1} 是更新后的记忆 token（输出给下一步），o_i 是当前段的视觉表示（输出给 LLM）。关键设计：(1) Bridge Layer 使用标准 self-attention，不修改 ViT 和 LLM 架构，保持 plug-and-play 特性；(2) 递归处理语义段，每个段仅与当前 memory tokens 交互，计算复杂度为 O((C+M)^2) per segment，其中 C 为段内帧数、M=32 为 memory token 数；(3) memory tokens 逐步累积全视频信息。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Memory Bridge Layer 的计算流程：

```
class MemoryBridgeLayer(nn.Module):
    """单层 Transformer Bridge with Self-Attention"""
    def __init__(self):
        self.self_attn = MultiheadAttention(embed_dim=1024, num_heads=8)
        self.norm1 = LayerNorm(1024)
        self.norm2 = LayerNorm(1024)
        self.ffn = FeedForward(1024, 4096)  # 标准 FFN
        # 注意：没有 cross-attention，仅 self-attention

    def forward(self, memory_tokens, segment_features):
        # memory_tokens: [32, 1024]  ← 来自前一步或初始化
        # segment_features: [C, 1024]  ← SceneTiling 分割的段
        
        # Step 1: Concat memory + segment
        x = torch.cat([memory_tokens, segment_features], dim=0)  # [32+C, 1024]
        
        # Step 2: Self-attention (所有 token 两两交互)
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        
        # Step 3: Split 回 memory 和 visual
        updated_memory = x[:32, :]        # [32, 1024] → m_{i+1}，传给下一步
        visual_output = x[32:, :]         # [C, 1024] → o_i，传给 LLM
        
        return updated_memory, visual_output
```

递归流程：
```
m_0 = nn.Parameter(torch.randn(32, 1024))  # 可学习初始化
MemoryCache = []

for i = 1 to K:  # K 个语义段
    m_i_raw, o_i = bridge_layer(m_{i-1}, s_i)
    
    # Memory Retrieval (cross-attn with cache)
    if MemoryCache:
        M_cache = torch.cat([m_0, m_1, ..., m_{i-1}], dim=0)  # [32*(i), 1024]
        m_i = cross_attn(query=m_i_raw, key=M_cache, value=M_cache)
    else:
        m_i = m_i_raw
    
    MemoryCache.append(m_i)
    llm_inputs.append(o_i)

# 最终: LLM 输入包含所有段的 o_i + 最终 memory m_K
```

训练时仅 Bridge Layer 和 LLM 的参数被更新（ViT 冻结），Bridge Layer 参数量约为 1 层 Transformer (~7M params for hidden=1024, heads=8)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VideoLLaMB 使用单层 Bridge Layer、32 个 memory tokens、hidden=1024、8 attention heads。训练时冻结 ViT-L/14 参数，仅训练 Bridge Layer 和 LLM（Vicuna-7B，使用 LoRA 或全参数微调，论文未明确说明 LLM 微调方式）。初始化来自 LLaVA-1.5 的权重。设计受 RMT (Recurrent Memory Transformer, Bulatov et al. 2022) 启发，但 VideoLLaMB 的关键区别在于：(1) Memory Bridge 在 vision-LLM 之间而非 LLM 内部，因此不影响 LLM 的推理能力；(2) 结合 SceneTiling 语义分割而非均匀分段；(3) 使用 cross-attention retrieval 而非简单传递。扩展性：Table 9 显示增加 Bridge Layer 层数（1→3）和 memory token 数（32→64）均可提升性能（53.8→54.6），但论文选择单层+32 tokens 的 lightweight 配置以平衡效率和性能。

涉及论文标题：
- VideoLLaMB__Long-context_Video_Understanding_with_Recurrent_Memory_Bridges
