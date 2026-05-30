## Molmo2__Open_Weights_and_Data_for_Vision-Language_Models_with_Video_Understanding_and_Grounding

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：Molmo2 —— 一个全开源（权重、数据、代码）的视觉语言模型家族（4B/8B基于Qwen3，7B基于OLMo3），支持单图、多图、视频输入，具备像素级视觉grounding能力（点标注、计数、目标跟踪）。核心设计：
  (1) 架构：SigLIP 2 So400m/14 384px ViT → Connector (MLP + multi-head attention pooling, 2×2 for image / 3×3 for video) → LLM (Qwen3或OLMo3)，vision tokens之间启用双向注意力（bi-directional attention），视频帧以2 fps采样（max 128帧SFT / 384帧long-context），帧间插入文本时间戳标记，多图使用"Image N"标记。
  (2) 三阶段训练pipeline：① Pre-training（仅图像，32k steps, batch 128, seq 2560）：PixMo-Cap captioning (60%) + PixMo image pointing (30%) + NLP (10%)；② SFT（联合视频/图像/多图，30k steps, batch 128, seq 16384）：7大类数据混合（Table 1），人工定义采样率；③ Long-context SFT（同数据mix, seq 36864, 384 frames, 2k steps, context parallelism Ulysses attention 8 GPUs/example）。
  (3) 训练技术创新：token weighting（video caption weight=0.1, pointing weight=0.2, 其他 √(4/n)策略平衡长短输出）；序列packing（动态规划solver pool=48, 平均3.8 examples/packed sequence, ~15x训练效率）；message-tree编码（多annotations用custom attention mask防止跨分支attention）；pointing预训练（pre-training阶段引入pointing数据稳定效果）；point坐标用压缩HTML-like纯文本格式（<points coords="timestamp obj_id x y;...">），比JSON大幅减少token数。
  (4) SlowFast encoding（推理时）：default pooling 3×3 slow frames + 9×9 fast frames + query-based frame selection，在~43% fewer visual tokens下匹配224 frames性能。
  (5) 9个新数据集（7视频+2多图）：Molmo2-Cap (104k dense video captions, avg 924 words/video)、Molmo2-AskModelAnything (140k human QA)、Molmo2-CapQA (1M synthetic QA)、Molmo2-SubtitleQA (300k subtitle QA)、Molmo2-VideoPoint (650k space-time points)、Molmo2-VideoTrack (15k complex queries)、AcademicVideoPoint/AcademicVideoTrack (repurposed)、Molmo2-MultiImageQA (72k QA)、Molmo2-MultiImagePoint (470k points)。

  实验比较：
  (a) 视频理解（Table 2）—— 12视频benchmarks (NextQA, PerceptionTest, MVBench, Tomato, MotionBench, TempCompass, Video-MME, Video-MME-Sub, LongVideoBench, MLVU, LVBench, VideoEvalPro) + 自建Molmo2-CapTest caption F1 + Molmo2-VideoCount accuracy + human ELO。对比GPT-5/GPT-5 mini/Gemini 2.5/3 Pro/Claude Sonnet 4.5 (API)、InternVL3.5/Qwen3-VL/Keye-VL-1.5/GLM-4.1V/MiniCPM-V-4.5/Eagle2.5 (open-weight)、PLM/LLaVA-Video/VideoChat-Flash (open models)。
  (b) 视频Grounding（Table 3-5）—— 视频counting (BURST-VC, Molmo2-VideoCount)、视频pointing (Molmo2-VideoPointVal F1)、视频tracking (MeViS/Ref-YT-VOS/Ref-Davis/ReasonVOS/Molmo2-Track, J&F/F1/HOTA)。对比GPT-5/Gemini/Qwen3-VL/VideoLISA/VideoGLaMM/Sa2VA/VideoMolmo/SAM 3。
  (c) 图像Benchmark（Table 6）—— 11图像benchmarks (AI2D/ChartQA/DocVQA/InfoQA/TextVQA/VQA v2/RWQA/MMMU/MathVista/CountBench/PixMoCount) + MuirBench + MMIU + Blink。
  (d) 图像Pointing（Table 7）—— Point-Bench (Affordance/Spatial/Reasoning/Steerability/Counting)。
  (e) Ablations（Table 8-11, 18）—— 视频消融、counting/pointing消融、tracking消融、long-context SFT消融、pre-training pointing消融。

- 硬件平台是什么，配置是什么。
  训练：Nvidia H100 GPU。4B: pre-train 32 GPUs/15.2h, SFT 128 GPUs/58.8h (7.5k GPU hr), long-context 128 GPUs/25.3h。8B: pre-train 64 GPUs/12.1h, SFT 128 GPUs/63.0h (8.1k GPU hr), long-context 128 GPUs/26.0h。总计约11k GPU hours for 8B。
  框架：PyTorch + FSDP2 + SDPA (非FlashAttention, 因custom attention mask) + torch.compile (静态shape) + AMP bfloat16。
  数据加载：torchcodec抽帧。On-the-fly packing算法集成入PyTorch DataLoader。

- 模型是什么。数据集和bench分别是什么。
  模型：Vision Encoder —— SigLIP 2 So400m/14 384px (380M params, 27 layers, dim 1152, 16 heads)。Connector —— MH attentional pooling (image 2×2, video 3×3) + MLP SwiGLU projection → LLM dim。LLM —— Qwen3-4B (36 layers, dim 2560, 8 KV heads), Qwen3-8B (36 layers, dim 4096, 8 KV heads), OLMo3-7B (32 layers, dim 4096, 32 KV heads)。Optimizer: AdamW β=(0.9,0.95), separate LR: pre-train (ViT 6e-6/Connector 2e-4/LLM 2e-4), SFT (ViT 5e-6/Connector 5e-6/LLM 1e-5)。Max crops: train K=8 / inference K=24。
  训练数据：PixMo + Molmo2自建9数据集 + 开源image/video/NLP数据集(Table 13, 100+子数据集)，总training examples约8M+（图像QA 2.4M + 视频QA 2.4M + 图像Pointing 1.1M + 视频Pointing 0.37M + 视频Tracking 0.80M + Captions/LongQA 1.2M + NLP 0.99M）。
  Benchmarks：视频理解12项 + 视频grounding 6项 + 图像理解11项 + 图像pointing Point-Bench + 多图3项 + NLP 4项 + 人类偏好ELO。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  完全开源：Code https://github.com/allenai/molmo2，模型权重、训练数据、训练代码全部开源（不含closed VLMs蒸馏）。HuggingFace + vLLM集成。Demo: playground.allenai.org。

  算法pipeline伪代码（SFT训练核心流程）：
  ```
  # === 视频处理 ===
  frames = torchcodec.extract(video, fps=2, max_frames=128)
  # 每帧384×384 → ViT SigLIP 2 → 27 layers
  # → 取第3层和第9层hidden states concat
  # → MH pooling: 3×3 window, mean as query
  # → 每帧约81 visual tokens

  # === 模型前向 ===
  # Vision token序列: [video_start] frame1_tokens <t0.0> frame2_tokens <t0.5> ... [video_end]
  # Bidir attention: vision tokens互相attend (cross-frame/image)
  # Text tokens: causal attention + attend to all vision tokens

  # === Packing & Message Trees ===
  # DP solver选择最优packing组合, pool_size=48
  # Message tree: 同一visual input多个annotations → 分支
  # Custom attention mask防止cross-branch attention
  # 平均3.8 examples/packed sequence, 15x efficiency

  # === Token Weighted Loss ===
  if is_video_caption:
      weight = 0.1
  elif is_pointing:
      weight = 0.2
  else:
      weight = 4.0 / sqrt(n_answer_tokens)
  loss = weight * cross_entropy(logits, labels)

  # === Point Format (压缩版) ===
  # <points coords="ts obj_id x y;...">inline text</points>
  # <tracks coords="ts obj_id x y;...">inline text</tracks>
  # ts: seconds to 1 decimal, (x,y): 0-1000 normalized
  # obj_id: sequential starting at 1, used for tracking/counting
  ```

  张量计算示例（128 frame video, Molmo2-8B）：
  ```
  ViT per frame: 384×384 → 27×27=729 patches
  Pooling: 3×3 × 16 MH heads → 81 tokens/frame
  Total vision tokens = 128 × 81 ≈ 10,368
  LLM (Qwen3-8B): 36 layers, dim 4096, 8 KV heads
  Bidir vision attention block: [10368, 10368] → FLOPs ≈ O(10368²×4096)
  Causal text block: [1000, 11368] → standard causal attn
  Total FLOPs dominated by bidir vision block
  ```
