## OmniVinci Enhancing Architecture and Data for Omni-Modal Understanding LLM

- 属于算法pipeline的实现是什么？实验比较什么？
  - 论文提出三个架构创新：(i) **OmniAlignNet**：通过CLIP-style对比学习在共享omni-modal潜在空间中强化视觉和音频嵌入的对齐；(ii) **Temporal Embedding Grouping (TEG)**：基于时间戳将视觉和音频嵌入按时间组重新排列，编码相对时序信息；(iii) **Constrained Rotary Time Embedding (CRTE)**：通过频率几何级数（$\omega_i = 2\pi / (T_{\max} \theta^{i/C})$）和元素级旋转变换编码绝对时间戳。此外提出**Omni-Modal Data Engine**：先用视觉/音频captioning模型独立生成标注，再用LLM进行跨模态纠错和总结生成omni-modal captions，最后用reasoning LLM合成QA对。训练策略包括Implicit Learning（利用视频自带的audio track进行隐式omni-modal监督）和Explicit Learning（通过data engine合成显式omni-modal标注数据）。最后应用GRPO post-training增强omni-modal reasoning。
  - 实验比较：(1) Ablation: Token Concatenation Baseline vs +TEG vs +Learned Time Embedding vs +RoTE vs +CRTE vs +OmniAlignNet，评估Worldsense/Dailyomni/Omnibench；(2) Implicit vs Explicit Learning消融，评估Video-MME;(3) 最终模型与Qwen2.5-Omni、Gemini、GPT-4o等对比omni/audio/video/image benchmarks；(4) GRPO消融；(5) downstream tasks（机器人导航、体育解说、语音翻译、医疗AI、半导体制造）。

- 硬件平台是什么，配置是什么。
  - 训练：NVIDIA DGX H100基础设施（论文致谢部分提及）。
  - 推理部署：NVIDIA A100（体育解说实验，AWQ量化后1.85s/clip）、NVIDIA L40s GPU、GeForce RTX 4090（24GB，评估latency：1.7x faster TTFT, 2.72x faster decoding vs Qwen2.5-Omni）。

- 模型是什么。数据集和bench分别是什么。
  - 模型：OmniVinci-9B，基于Qwen2.5-7B-Instruct LLM backbone，SigLip vision encoder (paligemma-siglip-so400m-patch14-448) + Dynamic S2 (2x2 spatial scale-then-compress)，AF-Whisper audio encoder (Audio Flamingo 3 backbone)，2-layer MLP projectors。
  - 训练数据：24M多模态对话样本（image 36%, non-speech sound 21%, speech 17%, omni 15%, video 11%），150+子数据集。Omni-modal conversations: 3.6M；Image-text: 8M；Video-text: 2.7M；Speech-text: 5.3M（ASR）；Sound-text: 4.3M（audio QA/captioning）。
  - Benchmarks：Omni: Worldsense, Dailyomni, Omnibench；Audio QA: MMAR, MMAU；Speech Recognition (WER): LibriSpeech clean/other, AMI, Tedlium, VoxPopuli；Video: LongVideoBench, MVBench, Video-MME；Image: AI2D, ChartQA, DocVQA, InfoVQA, MathVista, MMMU, RealWorldQA, SEED, TextVQA, VQAv2；Downstream: R2R-CE (robot nav), SPORTU-video, CoVoST2 (speech translation), WM-811K (wafer defect), UCR time-series。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：GitHub (NVlabs/OmniVinci)，HuggingFace (nvidia/omnivinci)，arXiv: 2510.15870。License: NVIDIA OneWay Noncommercial License。
  - 算法Pipeline（OmniAlignNet核心张量计算）：
    ```
    # 输入: 视频的视觉嵌入 E_v ∈ R^{Nv×C} 和音频嵌入 E_a ∈ R^{Na×C}
    # 可学习query: Q_v ∈ R^{1×C}, Q_a ∈ R^{1×C}
    
    # Step 1: Query-based projection (cross-attention)
    V_proj = CrossAttn(Q_v, E_v, E_v)  # → R^{1×C}
    A_proj = CrossAttn(Q_a, E_a, E_a)  # → R^{1×C}
    
    # Step 2: 3-layer self-attention + L2 normalize
    for batch with K videos:
        V = L2Norm(SelfAttn3(V_proj_batch))  # → R^{K×C}
        A = L2Norm(SelfAttn3(A_proj_batch))  # → R^{K×C}
    
    # Step 3: CLIP-style contrastive loss
    s_ij = dot(V_i, A_j)  # 相似度矩阵
    L_v→a = -1/K * Σ_i log(exp(s_ii) / Σ_j exp(s_ij))
    L_a→v = -1/K * Σ_i log(exp(s_ii) / Σ_j exp(s_ji))
    L_o-align = (L_v→a + L_a→v) / 2
    ```
  - CRTE核心计算：
    ```
    # 基础频率 (geometric progression)
    ω_i = 2π / (T_max * θ^{i/C}), for i = 0,...,C-1
    
    # 频率调制
    Ω_{i,j} = ω_i * t_j  # 维度i，时间戳t_j
    
    # Rotary Embedding (类似RoPE)
    CRTE(x, Ω) = x ⊙ cos(Ω) + RotateHalf(x) ⊙ sin(Ω)
    # RotateHalf(x) = [-x_2, x_1, -x_4, x_3, ..., -x_C, x_{C-1}]
    ```
  - 训练Pipeline（7阶段）：
    1. Vision Projector Alignment → 2. Vision Encoder Alignment → 3. Vision Pre-Training → 4. Image Instruction Tuning → 5. Video Instruction Tuning → 6. Audio Projector & Encoder Alignment + Audio Instruction Tuning → 7. Omni-Modal Joint Training (200B tokens, cosine LR schedule, base LR=2e-5, vision/audio encoders frozen)。GRPO post-training: 18K omni-modal MCQ, 64 frames, max prompt 1024 tokens, max response 2048 tokens, rollout=8, temperature=1.0, top-p=0.99。
