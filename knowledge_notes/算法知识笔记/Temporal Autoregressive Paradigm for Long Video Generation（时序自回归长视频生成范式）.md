## Temporal Autoregressive Paradigm for Long Video Generation（时序自回归长视频生成范式）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Temporal Autoregressive Paradigm 是长视频生成的一类主流方法：将长视频分解为逐段生成短片段（2-4s），每轮以前序输出作为下一轮条件。代表方法：StreamingT2V（last frame + attention injection）、SEINE（transition prediction）、Phenaki（token-based autoregressive）、DynamiCrafter 迭代。优势：可任意扩展长度，复用预训练短视频模型。核心瓶颈：条件设计——大多仅用 last frame，时序感受野仅相邻 clip，导致 long-term inconsistency。Owl-1 的改进：条件从 "pixel-level last frame" 升级为 "latent state s_t（聚合所有历史的隐式表示）+ last frame（短期平滑）"。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
传统范式 vs Owl-1 改进：
```
# Traditional: 仅用last frame pixel
def baseline(I, prompt, N):
    clips = []; f = I
    for t in range(N):
        c = VideoDM(f, prompt); clips.append(c); f = c.last_frame()
    return concat(clips)  # f仅含2s pixel信息，远距离漂移

# Owl-1: last_frame + latent state
def owl1(I, d_0, N):
    s = LMM.encode(I, d_0)                 # world state
    o_0 = VideoDM(state=s, image=I)
    for t in range(N):
        d = LMM.predict(s, o_{t-1}); s = LMM.update(s, d)
        o_t = VideoDM(state=s, image=o_{t-1}.last_frame)
    return concat([o_0, ..., o_N])         # s通过causal attn承载全部历史
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) Clip 长度固定（如 4s），确保单次 VDM 推理在 GPU 内存内；(2) 条件类型多样：last frame latent (Phenaki)、last frame pixel (StreamingT2V)、attention features (SEINE)、latent state (Owl-1)；(3) Owl-1 的 latent state 创新在于用 LMM 的大感受野 causal attention 构建条件。适用场景：视频扩展、无限时长生成、电影生成、世界模拟。

涉及论文标题：
- Owl-1__Omni_World_Model_for_Consistent_Long_Video_Generation
