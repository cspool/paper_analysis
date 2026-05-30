## Omni World Model for Video Generation（视频生成全向世界模型）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Omni World Model (Owl-1) 是一种从世界模型视角解决长视频生成一致性问题的方法。核心思想：视频本质上是对底层演化世界的观测记录，因此长视频的一致性应从隐式世界的连贯性角度来保证，而非在像素空间进行帧间拼接。Owl-1 将世界建模为一组 latent state variables {s_t}，每个 s_t 编码当前时刻和历史所有信息。构建闭环 state-observation-dynamics 三元组模拟世界演化：(1) State Decoder D: o_t = D(s_t, o_{t-1})，将隐式状态解码为显式视频观测，s_t 负责长期一致性，o_{t-1} 负责短期平滑；(2) World Dynamics Prediction f: d_t = f(s_t, o_t)，从观测和状态预测未来世界动态（文本形式）；(3) State Update g: s_{t+1} = g(s_t, d_t)，用动态驱动状态更新。通过链式展开 s_{t+1} = h(s_0, o_0, ..., o_t)，证明 latent state 承载所有历史观测信息，解决了传统 last-frame 条件时序感受野有限的问题。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Owl-1 推理流程：
```
def owl1_generate(I, d_0, N_clips):
    first_frame = ImageDiffusion(I, d_0)           # SD2.1-v生成首帧
    s_0 = LMM.encode_state(I, d_0, 128 queries)    # 128 learnable queries
    o_0 = VideoDM.denoise(s_0, first_frame)        # s_0替代text condition
    for t in 1..N_clips:
        d_t = LMM.predict_dynamics(s_{t-1}, o_{t-1})    # Eq.2: next-token pred
        s_t = LMM.update_state(s_{t-1}, d_t)            # Eq.3: causal attn
        if scene_transition:
            o_t = VideoDM.denoise(s_t, prev_obs=o_{t-1})  # 仅用state条件
        else:
            o_t = VideoDM.denoise(s_t, o_{t-1}.last_frame) # state+last_frame
    return concat([o_0, ..., o_N])
```
具体计算过程：LMM (Chameleon) 以自回归序列 [I, d_0, s_0_queries, VQ(o_0_sampled), d_0_text, ...] 建模世界演化。VideoDM (DynamiCrafter-1024) 以 s_t (128×dim) 作为 cross-attention condition 替代原始 CLIP text embedding，通过 standard diffusion denoising 生成视频。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Chameleon LMM (LoRA rank=8, ~798M params) + DynamiCrafter-1024 (全参 ~1.2B params)，总可训练参数 ~2B。训练：8×NVIDIA A800 (80G)，三阶段（Alignment→Generative Pretraining→World Model Training）共约 7 天。数据：WebVid (400K) + Panda70m (2M) 用于前两阶段；ActivityNet Captions (20K) + Vript (12K) 用于第三阶段。推理：每 4s 一个 clip，可扩展到 24s+。开源：https://github.com/huang-yh/Owl。

涉及论文标题：
- Owl-1__Omni_World_Model_for_Consistent_Long_Video_Generation
