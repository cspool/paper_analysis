## Multi-Stage Training for Video World Models（视频世界模型多阶段训练）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-Stage Training 是 Owl-1 为解决 LMM 与 VideoDM 联合训练困难而设计的三阶段策略。挑战：(1) LMM 和 VideoDM 独立预训练，直接联合训练不稳定；(2) 长视频世界建模需长时长+dense caption 数据，此类数据稀缺不足以从零训练。三阶段从易到难：(1) Alignment: 冻结 VideoDM，MSE 对齐 s_t 与 VideoDM text encoder T(t)，仅训练 LMM；(2) Generative Pretraining: 联合微调，s_t 替代 text condition，用 diffusion denoising loss；(3) World Model Training: 引入 d_t prediction，next-token pred teacher-forcing + denoising，在少量高质量数据上微调。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Stage 1: Alignment
L_align = MSE(s_t, T(caption))           # 冻结 VideoDM, 训练 LMM(LoRA)

# Stage 2: Generative Pretraining
o_noisy = sqrt(α_m)*o_t + sqrt(1-α_m)*ε
ε_pred = VideoDM(o_noisy, m, cross_attn(s_t), concat(o_{t-1}))
L_pretrain = ||ε - ε_pred||²            # 联合训练 LMM+VideoDM

# Stage 3: World Model Training
d_pred = LMM.predict_next_tokens(s_t, o_t)
L_dyn = CrossEntropy(d_pred, d_gt)       # teacher-forcing
L_total = L_dyn + L_pretrain             # 少量高质量数据微调
```
关键设计：Stage 1→2 丢弃 MSE loss，仅用 denoising loss，使 s_t 从"模仿 text embedding"解放为"作为 diffusion optimal condition"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
训练平台：8×NVIDIA A800 (80G)。Stage 1: 1天 (2.4M videos, 10K iters)，Stage 2: 5天 (2.4M videos, 10K iters)，Stage 3: 1天 (20K videos, 1K steps)。该策略将昂贵的长视频 dense caption 数据用量最小化（仅 Stage 3），大量通用短视频数据用于基础能力建设。

涉及论文标题：
- Owl-1__Omni_World_Model_for_Consistent_Long_Video_Generation
