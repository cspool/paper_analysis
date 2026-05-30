## VideoAuto-R1: Video Auto Reasoning via Thinking Once, Answering Twice

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：**VideoAuto-R1**，一个 video auto-reasoning 框架，包含两个核心组件：
  (1) **Thinking Once, Answering Twice 训练范式**：输出格式为 $\boxed{a_1} \rightarrow$ `<think>` $r$ `</think>` $\rightarrow \boxed{a_2}$。模型首先生成初始答案 $a_1$，然后进行显式推理 $r$，最后输出审查后答案 $a_2$。使用 GRPO（Group Relative Policy Optimization）进行 RL 训练，dual-answer reward 设计为 $R = w_1 R_{task}^{(1)}(a_1) + w_2 R_{task}^{(2)}(a_2) + \lambda R_{fmt} + \alpha R_{fallback}$，其中 $w_2 > w_1$（如 0.9:1.1）以鼓励最终答案准确。$R_{fallback}$ 允许模型在 $a_1$ 输出 fallback 字符串 "Let's analyze the problem step by step" 时获得奖励（当 $a_2$ 正确时），避免低置信度猜测。
  (2) **Confidence-based Early-Exit 推理策略**：推理时首先生成 $a_1$，计算其 length-normalized mean log probability $s(a_1) = \frac{1}{L} \sum_{\ell=1}^{L} \log p_{\theta}(t_{\ell} \mid t_{<\ell}, q)$ 作为置信度分数。若 $s(a_1) \geq \log \tau$（默认 $\tau = 0.97$），则早停直接返回 $a_1$；否则继续生成推理过程和 $a_2$。若 $a_1$ 为 fallback 字符串则 $s(a_1) = -\infty$ 强制继续推理。

  实验比较：
  (a) **Training Strategy Comparison（Table 6）**：对比 SFT、RL without thinking（仅 direct answer）、RL with thinking（标准 CoT GRPO）、VideoAuto-R1。VideoAuto-R1 在 VideoMME 67.3 vs RL with thinking 66.1，VideoMMMU 58.6 vs 56.4，同时平均响应长度从 149 tokens 降至 44 tokens。
  (b) **Adaptive Reasoning Strategies（Table 7）**：对比 training-based auto-thinking（AdaptThink 风格，按样本标注 think/no-think 标签）vs inference-based（VideoAuto-R1）。Training-based 方法存在 mode collapse，在 MVBench 上 auto 模式下甚至不如 no-think baseline（70.5 vs 71.1）。VideoAuto-R1 在 auto 模式下 71.0 vs no-think 70.9 vs always-think 71.0，且无需额外标签。
  (c) **Video QA Benchmarks（Table 3）**：与 10+ 个 thinking-only 视频推理模型（Video-R1, Time-R1, VideoChat-R1, Video-RTS, VITAL, LongVILA-R1, LOVE-R1, VideoChat-R1.5 等）在 VideoMME, MVBench, LongVideoBench, MMVU, VideoMMMU, MVP 上对比。VideoAuto-R1 (Qwen2.5-VL-7B) 在 VideoMME 67.3、VideoMMMU 58.6、MVP 39.4 均达到 SOTA。Qwen3-VL-8B 版本进一步提升至 VideoMMMU 65.0。
  (d) **Temporal Grounding Benchmarks（Table 4）**：Charades-STA、ActivityNet、NExT-GQA。VideoAuto-R1 (Qwen2.5-VL-7B) Charades-STA mIoU 60.0 vs Time-R1 58.8 vs VITAL 59.9。Qwen3-VL-8B 进一步提升。
  (e) **Image Benchmarks（Table 5）**：MathVista, MathVision, MathVerse, MMMU, MMMU-Pro, MM-Vet。
  (f) **Reward Design Ablation（Table 9）**：对比 $w^1:w^2$ 不同权重比（1:1, 0.9:1.1, 0.8:1.2）和 fallback reward $\alpha$ 有无。非对称权重 + fallback 最优。
  (g) **Early-Exit Threshold Analysis（Figure 3）**：$\tau$ 从 0.86 到 0.98 对 accuracy 和 think ratio 的影响。推理密集 benchmark 上提高 $\tau$ 持续改善准确率，感知 benchmark 上准确率几乎不变。
  (h) **Confidence-Task Correlation（Table 8）**：MVBench 平均置信度 0.948（think ratio 25%、gain +0.1），VideoMMMU 平均置信度 0.874（think ratio 51%、gain +4.0）。
  (i) **Data Filtering Ablation（Table 11）**：对比 Text/Image/Video 组合的过滤策略。Text+Image+Video filtered 83K 最优。
  (j) **Cold-Start SFT Ablation（Table 17）**：SFT with Video-R1-CoT data 导致性能下降（66.0→60.1）；SFT→RL 仍差于直接 RL（61.7 vs 66.1）。
  (k) **Frame Count Ablation（Table 15）**：64/128/256/2048 frames 下的性能变化。

- 硬件平台是什么，配置是什么。
  **32 NVIDIA H100 GPU**，训练约 35 小时。使用 DeepSpeed + vLLM 加速 GRPO rollout generation。GRPO rollout size G=16，全局 batch size 256，训练 1 epoch。测试使用 greedy decoding（temperature=0），最大 response length 4096 tokens。

- 模型是什么。数据集和bench分别是什么。
  模型：
  - **Qwen2.5-VL-7B-Instruct**（主 backbone）：最多 16K total video tokens，sweep over {64, 128, 256} frames
  - **Qwen3-VL-8B-Instruct**（扩展 backbone）：最多 128K total video tokens，sweep over {64, 256, 2048} frames
  - 训练时 visual encoder 冻结，仅微调 projector 和 LLM
  - 优化器：AdamW，lr=$1 \times 10^{-6}$，weight decay=0.01，max grad norm=1.0，constant lr schedule，KL penalty $\beta$=0.01

  训练数据（83K，从 137K 过滤）：
  - Text 6.4K：DAPO-Math（数学推理）
  - Image 27.5K：ViRL, ThinkLite-Hard（图像推理）
  - Video 49.4K：Video-R1, TVBench, STI-Bench, MMR-VBench, Charades-STA, ActivityNet, Time-R1, NExT-GQA（视频 QA + 时序定位）
  - 过滤策略：对每个样本生成 8 个 responses，用 Qwen3-30B-A3B-Instruct 评估 correctness。全对（too easy）或全错（too hard）的 QA 样本被丢弃。时序定位数据全保留。

  评估框架：lmms-eval（greedy decoding, temperature=0）

  Benchmarks：
  - **Video QA（Perception）**：VideoMME (w/o subtitles), MVBench, LongVideoBench, MMVU (multi-choice)
  - **Video QA（Reasoning）**：VideoMMMU, MVP (Minimal Video Pairs, pairwise accuracy on MVP-mini)
  - **Temporal Grounding**：Charades-STA (Recall@0.3/0.5/0.7, mIoU), ActivityNet (Recall@0.3/0.5/0.7, mIoU), NExT-GQA (Acc, mIoU)
  - **Image Reasoning**：MathVista (testmini), MathVision (testmini), MathVerse (testmini), MMMU (val), MMMU-Pro (overall), MM-Vet (test)

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  项目主页：https://ivul-kaust.github.io/projects/videoauto-r1。代码基于开源 Qwen2.5-VL / Qwen3-VL + DeepSpeed + vLLM + lmms-eval。

  训练算法 pipeline 伪代码：
  ```
  # === VideoAuto-R1 Training (GRPO with Dual-Answer Reward) ===
  # 输入: prompt q (video + question), base model π_{ref}
  # 输出格式要求: \boxed{a_1} <think> r </think> \boxed{a_2}

  for each training batch with prompts {q_i}:
      # Step 1: Rollout generation (vLLM-accelerated, temperature=1.0)
      for each q_i:
          {o_i^1, ..., o_i^G} = π_{θ_old}.sample(q_i, G=16)  # 每个 prompt 采样 16 个候选

      # Step 2: Reward computation
      for each output o_i^j:
          parse \boxed{a_1}, <think> r </think>, \boxed{a_2} from o_i^j

          # Task rewards
          if task_type == QA:
              R_task^{k} = exact_match_or_math_verify(normalize(a_k), GT) ∈ {0,1}  # k=1,2
          elif task_type == Temporal_Grounding:
              R_task^{k} = max_tIoU(pred_segments, GT_segments) ∈ [0,1]
          elif task_type == Grounding_QA:
              R_task^{k} = R_QA + R_TG ∈ [0,2]

          # Format reward: strict regex check for template compliance
          R_fmt = 1 if matches("\boxed{...}<think>...</think>\boxed{...}") else 0

          # Fallback reward
          R_fallback = 1 if (a_1 == "Let's analyze...") and (R_task^{2} > 0) else 0

          # Total reward (w1=0.9, w2=1.1, λ=1, α=0.3)
          R_i^j = 0.9 * R_task^{1} + 1.1 * R_task^{2} + 1 * R_fmt + 0.3 * R_fallback

      # Step 3: GRPO advantage normalization
      for each group i:
          μ_i = mean({R_i^j}_{j=1..G}), σ_i = std({R_i^j}_{j=1..G})
          A_i^j = (R_i^j - μ_i) / (σ_i + ε)

      # Step 4: Policy update
      for each (q_i, o_i^j, A_i^j):
          ρ_i^j = π_θ(o_i^j|q_i) / π_{θ_old}(o_i^j|q_i)
          L = -1/G * Σ min(ρ_i^j * A_i^j, clip(ρ_i^j, 1-ε, 1+ε) * A_i^j) + β * D_KL(π_θ || π_ref)

          θ ← AdamW(L, lr=1e-6, wd=0.01)
  ```

  推理算法伪代码（Algorithm 1）：
  ```
  # === VideoAuto-R1 Inference (Confidence-Based Early Exit) ===
  # 输入: trained model p_θ, video v, question q, threshold τ=0.97
  Require: p_θ, v, q, τ, fallback_string f = "Let's analyze the problem step by step."

  # Step 1: Generate until first <think> tag is detected
  tokens, logprobs = p_θ.greedy_decode(v, q, stop_token="<think>")

  # Step 2: Extract first boxed answer a_1
  a_1_tokens = extract_between(tokens, "\boxed{", "}")
  L = len(a_1_tokens)
  a_1 = detokenize(a_1_tokens)

  # Step 3: Compute confidence score
  if a_1 == f:
      s = -inf  # fallback forces continuation
  else:
      # Length-normalized mean log probability
      s = (1/L) * Σ_{ℓ=1..L} logprobs[ℓ]  # log p_θ(t_ℓ | t_{<ℓ}, v, q)

  # Step 4: Early-exit decision
  if s >= log(τ):  # e.g., τ=0.97 → log(0.97)≈-0.0305
      return a_1  # Early exit: direct answer
  else:
      # Continue generation: reasoning + reviewed answer
      tokens_rest = p_θ.continue_decode(max_tokens=4096)
      r = extract_between(tokens_rest, "<think>", "</think>")
      a_2 = extract_last_boxed(tokens_rest)
      return a_2  # Return reviewed answer
  ```

  张量计算层面：confidence score $s(a_1)$ 的分子是 LLM 自回归解码中已产生的 per-step log-probability 的均值，$a_1$ 通常仅包含不到 10 个 token，因此 confidence 计算开销极小。early-exit 时避免生成额外数百 tokens，显著降低延迟和推理成本。实现上通过检测第一个 `<think>` tag 出现来终止早期生成，无需外部校准器。
