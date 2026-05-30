## Temporal Preference Optimization of Large Multimodal Models

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：**Temporal Preference Optimization (TPO)**，一种 video-LMM 的后训练（post-training）框架。核心 pipeline 三步：(1) **Temporal Preference Modeling** — 通过操纵视频输入生成对比响应对：给定从视频中采样的帧集合 F，先用 CogVLM2 生成每帧 caption，再由 GPT-4o-mini 生成相关问题 Q；preferred response 使用 Q + 相关帧 F 作为输入由 video-LMM 生成；dis-preferred response 使用 Q + 不相关帧（从剩余帧采样，Irrelevant）或部分相关帧（随机子采样，Incomplete）作为输入生成。(2) **LLM-based Post-Filtering** — 用 GPT-4o-mini 对偏好数据对进行规则化过滤，剔除 dis-preferred 优于 preferred、preferred 事实错误、或问题模糊的样本。(3) **DPO Training** — 使用 Direct Preference Optimization + SFT loss 联合训练，损失函数 L = L_DPO + α·L_SFT。

  实验比较：(1) 与 SFTSelf（自生成数据做 SFT）、SFTLLM（GPT-4o-mini 生成数据做 SFT）、Hound-DPO（DPO on video-LMM，使用 ChatGPT 评分生成偏好数据）三种训练策略对比；(2) 与 GPT-4o、Video-LLaVA、LLaVA-1.5、Qwen-VL-Max、ShareGPT4Video、InternVL-Chat-V1.5、VideoChat2、LongLLaVA、Video-CCAM、NVILA、Qwen2-VL、Apollo 等 SOTA 模型对比；(3) 消融实验：不同输入帧数、数据集规模（2k/5k/10k）、有无 post-filtering、不同 dis-preferred 数据混合比例（Incomplete:Irrelevant）、Needle-in-a-Haystack 任务。

- 硬件平台是什么，配置是什么。
  训练：8 × NVIDIA A100 80GB GPU，batch size 64，full fine-tuning（language model + multimodal projector），visual encoder frozen。LongVA-TPO 训练约 4 小时（lr=4×10⁻⁶），LLaVA-Video-TPO 训练约 4 小时（lr=3×10⁻⁷），cosine lr scheduler with warmup ratio 0.1，1 epoch。数据准备阶段使用 GPT-4o-mini（text-only input）进行 question curation 和 post-filtering。

- 模型是什么。数据集和bench分别是什么。
  模型：LongVA-7B（长上下文 video-LMM）和 LLaVA-Video-7B（SOTA 7B video-LMM），均基于此做 TPO 微调得到 LongVA-TPO 和 LLaVA-Video-TPO。
  训练数据：手动 curator 200 关键词，爬取 8000 个互联网视频，生成 10K 偏好数据对（LongVA-TPO）；对 LLaVA-Video-TPO，从 LLaVA-Video-178K 数据集中采样 10K QA 对，dis-preferred 仅用 Incomplete 方式生成。数据任务分布：Temporal Reasoning 8.7%、Action Reasoning 12.4%、Causal Reasoning 11.1%、Information Extraction 18.0%、Descriptive 12.8%、Summarization 7.5%、Object Reasoning 14.9%、Spatial Reasoning 13.5%。
  评测 benchmark：LongVideoBench（长视频上下文推理）、MLVU（多任务长视频理解）、Video-MME（多模态视频评测，含 Short/Medium/Long 三个子集）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/ruili33/TPO；数据集和 checkpoint：https://huggingface.co/collections/ruili0/temporal-preference-optimization-67874b451f65db189fa35e10。

  TPO 算法 pipeline 伪代码：
  ```
  输入: 视频 V, video-LMM π_θ, 参考模型 π_ref
  输出: 优化后的模型 π_θ*

  # === Phase 1: Temporal Preference Data Generation ===
  For each video V in crawled_videos:
      F = sample_frames(V)                    # 采样帧集合
      captions = CogVLM2.caption(F)           # 逐帧 caption
      Q = GPT4o-mini.generate_questions(captions)  # 问题生成

      # Preferred response: 使用相关帧
      r⁺ = π_θ(V[F], Q)

      # Dis-preferred response (两种策略):
      F_irrelevant = sample(V \ F)            # 策略(a): 不相关帧
      F_incomplete = random_subset(F)         # 策略(b): 不完整帧
      r⁻_irrelevant = π_θ(V[F_irrelevant], Q)
      r⁻_incomplete = π_θ(V[F_incomplete], Q)

      # Post-Filtering with GPT-4o-mini (3 条规则)
      keep = filter(captions, Q, r⁺, r⁻)
      if keep:
          D.add((V, Q, r⁺, r⁻))

  # === Phase 2: DPO Training ===
  For each (V, Q, r⁺, r⁻) in D:
      # DPO loss (公式 2)
      log_ratio⁺ = log π_θ(r⁺|V,Q) - log π_ref(r⁺|V,Q)
      log_ratio⁻ = log π_θ(r⁻|V,Q) - log π_ref(r⁻|V,Q)
      L_DPO = -log σ(β · (log_ratio⁺ - log_ratio⁻))

      # SFT loss (公式 3)
      L_SFT = -log π_θ(r⁺|V,Q)

      # Combined loss (公式 4)
      L = L_DPO + α · L_SFT

      θ ← θ - η · ∇_θ L
  ```
  关键超参：LongVA-TPO: β=0.3, α=0.5, lr=4×10⁻⁶；LLaVA-Video-TPO: β=0.2, α=1, lr=3×10⁻⁷。两者均 full fine-tuning（language model + multimodal projector），visual encoder frozen，1 epoch。
