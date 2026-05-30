## Short Window Attention for Video Generation（视频生成短窗口注意力）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Short Window Attention 是在长视频 AR 生成中将自注意力的感受野限制在固定时间窗口 W 内的注意力机制。传统 causal attention 复杂度 O(L²)（L = 总序列长度），随视频增长不可持续。利用视频生成中的时间局部性（temporal locality）——附近帧对预测下一帧更重要——将注意力限制在最近 W 个 latent frames 内，复杂度降为 O(W·L)，KV cache 需求从 O(L) 降为 O(W)。LongLive 将 window size 设为 W=9 latent frames（配合 S=3 sink tokens）。Window size 引入 quality-efficiency trade-off：大窗口高一致性但高延迟，小窗口快但一致性下降。Frame sink 机制（见下文）可恢复此 trade-off 中的一致性损失。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Short Window Attention (per self-attention layer)
# W=9, S=3 sink tokens, effective KV = W+S (不随序列增长)
def short_window_attn(Q, K, V, K_sink, V_sink, window=9):
    K_eff = concat([K_sink, K[-window:]], dim=-2)  # [S+W, d_head]
    V_eff = concat([V_sink, V[-window:]], dim=-2)
    return softmax(Q @ K_eff.T / sqrt(d)) @ V_eff
# KV cache eviction: oldest non-sink tokens evicted when count > W
```

Annotations: 对 Wan2.1-T2V-1.3B @ 832x480, W=9+3 sink: 端到端计算时间降低 28%，峰值显存降低 17%（vs full attention on H100）。训练时同步使用 short window（streaming long tuning alignment），resident KV per step = O(W+T+S)（T=5s clip length）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Short window attention 通过 attention mask 实现（-inf mask for out-of-window KV）。与 FlashAttention 兼容（causal mask 内嵌 window mask）。LongLive 在 self-attention 层应用，cross-attention 保持全注意力。适用于：(a) 长视频 AR 生成推理加速；(b) 配合 frame sink 恢复一致性；(c) 任意 causal transformer 的长序列低延迟推理。

涉及论文标题：
- LongLive__Real-time_Interactive_Long_Video_Generation
