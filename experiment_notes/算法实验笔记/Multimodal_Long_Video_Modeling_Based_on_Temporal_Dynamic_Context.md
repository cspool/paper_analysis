## Multimodal_Long_Video_Modeling_Based_on_Temporal_Dynamic_Context

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 Temporal Dynamic Context (TDC) 视频编码方法：将视频分割为语义一致的场景，保留首帧静态特征，用 Q-Former 将后续帧的视觉+音频 token 压缩为动态上下文 token。同时提出 Long Video Chain-of-Thought (LVCoT) 训练无关策略，将超长视频分段处理，逐段推理后汇总得到最终答案。
  - 实验比较：在 MVBench、PerceptionTest、EgoSchema、MLVU、Video-MME 上与 vision-focused MLLMs（LLaVA-OneVision, InternVL2, LongVU, VideoChat2 等）和 audio-visual MLLMs（VideoLLaMA2, PandaGPT, NExT-GPT 等）对比；在 Music-AVQA、AVSD 上做 audio-visual 联合理解评测。Ablation 研究 segment 数量、query 类型（AvgPool vs Learned Query）、context token 数量、text instruction 作用、LVCoT 效果。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明训练所用 GPU 型号和数量。

- 模型是什么。数据集和bench分别是什么。
  - Backbone LLM: Qwen2-7B 和 LLaMA3.2-3B。
  - Visual Encoder: DINOv2 + SigLIP，每帧 144 tokens。
  - Audio Encoder: BEATs，16kHz 重采样，约 50 tokens/s。
  - Q-Former: 由预训练 BERT 初始化，默认 16 个 query tokens。
  - 训练数据 Stage 1（视觉-语言对齐）: LLaVA-OneVision (3.2M samples)。
  - 训练数据 Stage 2（视频指令微调）: LLaVA-Video, TextVR, YouCook2, EgoQA, Kinetics-710, NExTQA, CLEVRER, TGIF, WebVidQA, DiDeMo, ShareGPT4Video, MovieChat（Qwen2-7B: 2M, LLaMA3.2-3B: 540K samples）。
  - 训练数据 Stage 3（音频-视频指令微调）: AVQA, Music-AVQA, AVSD, LongVALE, AVInstruct + Stage 2 subset（Qwen2-7B: 300K, LLaMA3.2-3B: 120K）。
  - Benchmarks: MVBench (avg 16s), PerceptionTest (avg 23s), EgoSchema (avg 180s), MLVU (avg 651s), Video-MME (avg 1010s), Music-AVQA, AVSD。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/Hoar012/TDC-Video
  - 算法 Pipeline 伪代码：

```
# === Video Scene Segmentation ===
video_frames = sample_frames(video, fps=1)  # 每秒1帧
embeddings = DINOv2(video_frames)           # 高维特征
similarities = cosine_sim(embeddings[i], embeddings[i+1])  # 帧间相似度
split_points = top_k_lowest(similarities, S-1)  # S-1 个低相似度分割点
scenes = segment(video_frames, split_points)     # 分割为 S 个场景(S≤24)

# === Per-Scene Encoding (sliding window length N) ===
for scene in scenes:
    # 首帧静态特征（完整保留，144 visual + 50 audio tokens）
    F_x1 = SigLIP(scene[0]);  F_a1 = BEATs(audio[0])

    # AvgPool 首帧 visual tokens 得 K=16 个 query tokens
    Q = AvgPool(F_x1)  # shape: (16, D)

    # 后续帧动态压缩
    for i in 2..N:
        F_xi = SigLIP(scene[i]);  F_ai = BEATs(audio[i])
        # Q-Former cross-attention + instruction text F_s
        F_Q_i = QFormer(Q, [F_xi · F_ai], F_s)  # 压缩为 16 tokens

    # 场景最终表示: 静态 tokens + 动态上下文
    F_TDC = [F_x1 · F_a1 · <Sep> · F_Q_2 · F_Q_3 · ... · F_Q_N]

# === LVCoT for Extremely Long Videos ===
segments = divide_equally(video, M)   # M=3 by default
thoughts = []
for seg in segments:
    ans = LLM(F_TDC(seg), question)
    thoughts.append(f"From {seg.t_start}s to {seg.t_end}s: {ans}")

final = LLM(F_TDC(full_video), question, prev_thoughts=concat(thoughts))
```

  - 训练策略：三阶段训练，每阶段 1 epoch。Stage 1/2 全参数训练，Stage 3 用 LoRA 减少显存。优化器 AdamW，LR 1e-5 (stage 1/2) / 2e-5 (stage 3)，cosine decay，warmup ratio 0.03，max sequence length 8192。Visual/audio encoder 全程冻结，仅训练 temporal compressor 和 LLM。Q-Former 由 pre-trained BERT 初始化。
