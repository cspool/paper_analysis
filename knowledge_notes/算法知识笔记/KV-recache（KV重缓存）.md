## KV-recache（KV重缓存）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
KV-recache 是 LongLive 提出的面向交互式长视频自回归（AR）生成的 KV cache 更新机制。在用户 prompt 切换时，传统的 KV cache 策略存在两种困境：(1) 丢弃全部 KV cache → 视觉断裂、时间不连续；(2) 保留全部 KV cache → 旧 prompt 语义残留在 cache 中，导致新 prompt 延迟响应或不跟随。KV-recache 通过在 prompt 切换边界重新计算 KV cache：将已生成视频前缀 x 作为视觉上下文，与新 prompt p_new 一起重新通过生成器的交叉注意力层（cross-attention: visual Q attend to text prompt K/V）和前向层计算新的 KV state。由于新 prompt 的 text embedding 替换了旧 prompt 的 text embedding，交叉注意力层中注入新的语义信号，清除旧 prompt 的残留语义；同时自注意力层的 causal attention 保留了已生成帧之间的视觉运动和外观连续性线索。每次 prompt switch 仅需一次 recache forward pass（过已生成前缀帧），随后步骤使用刷新后的 KV cache 正常进行。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# KV-recache: prompt switch 时的 KV cache 刷新
def kv_recache(G_theta, video_prefix_x, old_cache_C, p_new):
    C_new = []
    for step_i, f_i in enumerate(video_prefix_x):
        kv_self = G_theta.self_attn_cache(f_i, C_new)
        kv_cross = G_theta.cross_attn_cache(f_i, p_new)  # key: K/V from NEW prompt
        C_new.append((kv_self, kv_cross))
    return C_new
```

Annotations: recache 仅需单次 forward pass（过已生成前缀），对 10s video (single switch) 额外时间开销约 6%。训练时同步集成 recache（teacher 也接收新 prompt 输出 DMD 监督），消除 train-inference mismatch。推理时支持多次 prompt switch（n+1 个 prompt → n 个 switch 边界），每边界执行一次 recache。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
KV-recache 基于 Wan2.1-T2V-1.3B (DiT 架构) 实现。DiT 的 cross-attention + self-attention 交替结构使 recache 关键：self-attention 传播视觉连贯性（KV cache 中的自注意力状态），cross-attention 注入 prompt 语义（K/V 来自当前 prompt embedding）。训练时集成到 streaming long tuning loop。一般使用场景：交互式视频生成（用户逐步输入新 prompt）、叙事长视频（多场景切换）、实时内容创作（streaming prompt input）。开源在 https://github.com/NVlabs/LongLive。

涉及论文标题：
- LongLive__Real-time_Interactive_Long_Video_Generation
