## Temporal Dynamic Context (TDC)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Temporal Dynamic Context (TDC) 是一种多模态长视频编码框架，由 Hao et al. 提出。核心思想：将视频表示分解为静态视觉特征（static visual features）和动态多模态上下文（dynamic multimodal context）。对于每个视频场景，首帧完整保留（144 visual + 50 audio tokens）作为静态参考帧；后续帧通过 Q-Former cross-attention 压缩为 K 个 context tokens（默认 K=16），这些 context tokens 聚合了帧间时序变化和视觉-音频跨模态信息。TDC 将每帧平均 token 数从 ~194 压缩至 16，使得 LLM 可在固定 context window 内处理更多帧。相比 prior work（VideoLLaMA2 仅采样 16 帧，token 简单拼接），TDC 以 1fps 密集采样所有帧，通过语义场景分割保证时序一致性，用 Q-Former 压缩替代简单采样/丢弃。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# TDC 完整 Pipeline
# 输入: video (T seconds), audio, question text F_s
# 输出: compressed video representation F_TDC

# Step 1: Scene Segmentation
frames = sample(video, fps=1)                # T frames, 1fps
emb = DINOv2(frames)                         # frame-level features
sim = cosine_sim(emb[i], emb[i+1])           # inter-frame similarity
split_points = top_k_lowest(sim, S-1)        # S≤24 semantic boundaries
scenes = segment(frames, split_points)        # S semantically consistent scenes

# Step 2: Per-Scene TDC Encoding
for scene in scenes:  # sliding window length N
    # Static: first frame fully retained
    F_x1 = SigLIP(scene[0])                  # (144, D)
    F_a1 = BEATs(audio[0])                   # (50, D)

    # Query generation via AvgPool
    Q = AvgPool(F_x1)                        # (K=16, D)

    # Dynamic: compress subsequent frames
    F_TDC = [F_x1, F_a1, <Sep>]
    for i in 2..N:
        F_xi = SigLIP(scene[i]); F_ai = BEATs(audio[i])
        F_Q_i = QFormer(Q, [F_xi·F_ai], F_s)  # cross-attn → (16, D)
        F_TDC.append(F_Q_i)

# Step 3: LLM Decoding
answer = LLM(F_TDC, F_s)
```
关键设计：(1) AvgPool queries 优于 learned queries；(2) instruction text F_s 注入 Q-Former 使压缩自适应于问题；(3) <Sep> token 区分静态和动态 token。消融：S=1 (不分割) MVBench 62.7→53.5 (-9.2)；text instruction 对长视频帮助最大 (MLVU Long -1.6)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TDC 在 PyTorch + HuggingFace Transformers 上实现。Visual encoder (SigLIP+DINOv2) 和 audio encoder (BEATs) 冻结；Q-Former (BERT-initialized) 和 LLM (Qwen2-7B / LLaMA3.2-3B) 可训练。三阶段训练：Stage 1 视觉-语言对齐 (LLaVA-OneVision 3.2M)，Stage 2 视频指令微调 (2M/540K)，Stage 3 音频-视频指令微调 (300K/120K + LoRA)。开源: github.com/Hoar012/TDC-Video。

涉及论文标题：
- Multimodal_Long_Video_Modeling_Based_on_Temporal_Dynamic_Context
