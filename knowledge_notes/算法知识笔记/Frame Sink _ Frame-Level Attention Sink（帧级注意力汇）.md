## Frame Sink / Frame-Level Attention Sink（帧级注意力汇）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Frame Sink（帧级注意力汇）是 LongLive 在视频 AR 生成中提出的全局锚定机制，受 LLM 领域 attention sink 概念启发。在视频生成中，此前 Self-Forcing 报告 attention sink tokens 单独无法防止长 rollout collapse。LongLive 发现：一旦通过 streaming long tuning 解决了 long rollout collapse 问题，attention sink 即可生效。Frame Sink 将视频首帧 chunk（3 latent frames）固定为全局 sink token，永久保留在每层 self-attention 的 KV cache 中从不被驱逐，所有后续帧都能通过注意力访问它们。Sink tokens 作为"场景身份锚点"，缓存色调、风格、主体外观等持久视觉属性，补偿 short window attention 丢失的远距离时间上下文。实验中 W=9 local + S=3 sink 的一致性接近 W=21 full window，但保持 W=9 的速度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Frame Sink + Short Window Attention
sink_indices = [0,1,2]  # first chunk's 3 latent frames
def attn_with_sink(Q, K_all, V_all, cur_idx, window=9):
    K_sink = K_all[sink_indices]  # NEVER evicted
    V_sink = V_all[sink_indices]
    start = max(0, cur_idx - window)
    K_win = K_all[start:cur_idx]
    V_win = V_all[start:cur_idx]
    K_eff = cat([K_sink, K_win])  # sink always first
    V_eff = cat([V_sink, V_win])
    return sdpa(Q, K_eff, V_eff)
```

Annotations: S=3 sink tokens (first chunk of 3 latent frames)。20s 生成实验：Window 21 (no sink) 高一致性/慢；Window 12 (no sink) 一致性下降；Window 9 + Sink 3 一致性接近 Window 21。起效前提：streaming long tuning 必须先解决 long rollout collapse。KV recache 时不重算 sink。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
训练时 sink tokens 在 streaming long tuning 的 short window config 中常驻 KV；推理时在帧生成开始时确定。适用于：(a) 长视频 AR 高效推理；(b) 降低 attention 复杂度但保持长程一致性；(c) 多 prompt switch（sink 维持全局视觉身份，recache 更新 prompt 语义）。开源在 https://github.com/NVlabs/LongLive。

涉及论文标题：
- LongLive__Real-time_Interactive_Long_Video_Generation
