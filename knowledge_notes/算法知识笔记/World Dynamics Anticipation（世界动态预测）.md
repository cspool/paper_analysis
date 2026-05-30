## World Dynamics Anticipation（世界动态预测）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
World Dynamics Anticipation 是 Owl-1 驱动内容多样性和可控性的机制。传统方法仅用 last frame 条件，缺乏未来预判，导致同质化内容。Owl-1 从 s_t（长期信息）和 o_t（短期参考）预测 d_t（文本）：d_t = f(s_t, o_t)。d_t 融入状态更新 s_{t+1} = g(s_t, d_t)，将未来预期编码进下一轮条件。d_t 还提供 controllability——用户可替换预测的 d_t 为自定义信号。在 LMM 中以自回归 next-token prediction 实现，训练时用 dense captions teacher-forcing。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Dynamics prediction (自回归)
d_t = ""
for k in range(max_tokens):
    logits = LMM.forward([...s_t, o_t_VQ, d_t_prev])
    next_token = sample(logits[-1])
    d_t += decode(next_token)
    if next_token == EOS: break

# State update (d_t融入s_t)
s_{t+1}_queries = LMM.causal_attention(
    queries=s_t_queries, keys_values=[text(d_t), o_t_VQ, history])
```
训练 teacher-forcing：L_dyn = -∑_i log P(d_t^(i) | s_t, o_t, d_t^(<i), θ)，d_t 由 dataset dense captions 提供。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
训练数据：ActivityNet Captions (100K captions, 平均 120s, 3+ events/video) + Vript (400K segments, 密集 script)。d_t 为自然语言（如 "man picks up pruning shears"）。推理：(1) 自主模式：LMM 预测 d_t；(2) 受控模式：用户指定 d_t。场景切换时丢弃 image condition，仅依赖 s_t——对 s_t 信息表达能力要求极高。Limitation：预测的 d_t 有重复性，Dynamic Degree 低于 DynamiCrafter baseline。

涉及论文标题：
- Owl-1__Omni_World_Model_for_Consistent_Long_Video_Generation
