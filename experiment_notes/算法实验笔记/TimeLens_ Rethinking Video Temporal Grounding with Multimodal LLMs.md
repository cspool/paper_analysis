## TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：**TimeLens**，一套构建强视频时间定位（VTG）能力的 MLLM 后训练 pipeline，核心包含四大组件：
  (1) **Data Curation** — 手动审查三大 VTG benchmark（Charades-STA、ActivityNet Captions、QVHighlights），按严格标准（query 清晰度、事件存在性、唯一性、标注精度、穷尽性）诊断错误并重新标注，产出 **TimeLens-Bench**（4279 视频、9404 标注）；同时对训练数据进行自动化重标注（使用 Gemini-2.5-Pro 重新描述事件并标注时间戳），产出 **TimeLens-100K**（约 20K 视频、100K 标注）。
  (2) **Interleaved Textual Timestamp Encoding** — 将每帧的原始时间戳（如 "10.2s"）通过 LLM text tokenizer 转为文本 token，交错插入到对应帧的 visual tokens 之前，形成 interleaved visual-text 序列。相比 position-embedding (MRoPE/3D RoPE)、visual overlay、non-interleaved textual 等方案更简单且效果最优。
  (3) **Thinking-free RLVR (GRPO)** — 使用 GRPO 作为 RL 算法，模型直接输出时间片段 `(t_start, t_end)` 而不生成思考过程。奖励函数简化为单一的 `r(y) = IoU(Ŝ, S*)`，无需格式奖励。训练效率比 thinking-based RLVR 更高（1.0× vs 1.9× 训练时间），且性能更优。
  (4) **RLVR Recipes** — (a) Early Stopping：当 temporal IoU reward 和 group 内 reward 标准差 plateau 时（约 310 steps / ~2.5K samples），停止训练以避免性能退化。(b) Difficulty-based Sampling：用待训练模型对训练数据进行离线推理计算 IoU，定义 difficulty d_i = 1 - IoU(Ŝ_i, S*_i)，以高斯分布 g(d; μ, σ²) 进行采样，μ=0.05, σ=0.2 时可获得最优性能。

  实验比较：(1) Timestamp Encoding 消融：Interleaved Textual vs Visual Overlay vs Non-Interleaved Textual vs Position Embedding (MRoPE)，每种用 raw timestamp 和 frame index 两种格式；(2) Training Paradigm 消融：SFT (32K/100K data) vs Thinking-based RLVR vs SFT+Thinking-free RLVR vs Thinking-free RLVR alone；(3) RLVR Recipes：Early stopping 有效性验证（追踪 reward 和 evaluation metrics 曲线）、Difficulty-based sampling 不同难度均值的影响；(4) 主表对比：TimeLens-7B 和 TimeLens-8B vs GPT-4o、GPT-5、Gemini-2.0/2.5-Flash/Pro、VideoChat-Flash-7B、VideoChat-R1-7B、Time-R1-7B、TRACE、TRACE-uni、TimeSuite、Grounded-VideoLLM、MiMo-VL-7B、Qwen2.5-VL-7B、Qwen3-VL-8B/235B；(5) 不同模型规模 (3B/7B) 验证；(6) 原始 noisy 训练数据 vs TimeLens-100K 消融。

- 硬件平台是什么，配置是什么。
  训练：8 × NVIDIA H20 GPU。RLVR 训练时间 1.0× ≈ 4h10m（约 310 steps、~2.5K 训练样本）。SFT 训练：batch size 128，lr=1×10⁻⁵，1 epoch。RLVR 训练：batch size 8，每 prompt 采样 8 roll-outs，lr=1×10⁻⁶，KL coefficient β=0。Vision encoder frozen，其余参数可训练。消融实验使用较低分辨率（min_tokens=16 per frame，total_tokens=3584），最终模型使用较高分辨率（min_tokens=64，total_tokens=14336）。视频采样 2 FPS；interleaved textual encoding 时额外用 1 FPS + frame duplication 绕过 MRoPE 机制做公平对比。

- 模型是什么。数据集和bench分别是什么。
  模型：基于 Qwen2.5-VL-7B 和 Qwen3-VL-8B 分别得到 TimeLens-7B 和 TimeLens-8B。也验证了 Qwen2.5-VL-3B → TimeLens-3B。
  训练数据：**TimeLens-100K**，从 CosMo-Cap、InternVid-VTime、DiDeMo、QuerYD、HiREST 等数据集中采样视频，使用 Gemini-2.5-Pro 自动化重标注，约 100K 条高质量 VTG 标注（~20K 视频，时长主要分布在 0-240s），按视频时长均匀采样。
  评测 benchmark：**TimeLens-Bench**，包含 Charades-TimeLens（Daily Life, 1313 视频）、ActivityNet-TimeLens（Activity, 1455 视频）、QVHighlights-TimeLens（Mixed, 1511 视频），四项指标：R1@0.3, R1@0.5, R1@0.7, mIoU。额外验证：VUE-TR（Vidi benchmark）、Video-MME（通用视频理解 benchmark）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文声明：代码、数据、模型将全部开源（项目页面 https://timelens-arc-lab.github.io/）。论文未提供已生效的 GitHub/模型下载链接，但给出了各 baseline 模型的来源引用。

  TimeLens 算法 pipeline 伪代码：
  ```
  # === Phase 0: Data Preparation (offline) ===
  # TimeLens-Bench: manual re-annotation
  For each (video, query, segment) in {Charades-STA, ActivityNet, QVHighlights}:
      # Diagnose-then-Refine
      errors = check_criteria(video, query)  # 检查 5 类错误
      if errors:
          revise_query_or_select_new_event(video)
          annotate_precise_segment(video, new_query)
      cross_validate(batch)  # 不同标注者交叉验证

  # TimeLens-100K: automated re-annotation
  For each video in training_corpus:
      prompt_events = "Identify distinct events distributed across time"
      events = Gemini-2.5-Pro(video, prompt_events)
      For each event in events:
          query = Gemini-2.5-Pro.describe(event)
          timestamp = Gemini-2.5-Pro.localize(event)  # ["MM:SS", "MM:SS"]
          D_train.add((video, query, timestamp))

  # === Phase 1: Timestamp Encoding ===
  def interleaved_textual_encode(video_frames, fps):
      """每个 frame 前插入文本时间戳 token"""
      tokens = []
      for i, frame in enumerate(video_frames):
          t = i / fps  # 当前帧时间（秒）
          timestamp_text = f"{t:.1f}s"  # e.g. "10.2s"
          text_tokens = tokenizer(timestamp_text)  # LLM text tokenizer
          frame_tokens = vision_encoder(frame)
          # frame 作为独立 image 处理（duplicate 为两份绕过 MRoPE）
          tokens.extend(text_tokens)      # 时间戳 token 在前
          tokens.extend(frame_tokens)      # 视觉 token 在后
      return tokens
  # 输入 prompt: "The numbers before each video frame indicate its
  #  sampling timestamp (in seconds). Please find the visual event
  #  described by the sentence '{query}', determining its starting
  #  and ending times. Format: 'The event happens in <start> - <end> seconds'."

  # === Phase 2: Difficulty-aware Sampling (offline, before RL) ===
  For each (v_i, q_i, S*_i) in D_train:
      Ŝ_i = π_θ(v_i, q_i)  # offline inference
      d_i = 1 - IoU(Ŝ_i, S*_i)  # difficulty: higher = harder
      # 按高斯分布采样权重
      w_i = g(d_i; μ=0.05, σ=0.2) / p̂(d_i)  # density-corrected
  D_sampled = weighted_sample(D_train, w_i, size=~12K)

  # === Phase 3: Thinking-free RLVR with GRPO ===
  π_θ = load(Qwen2.5-VL-7B)  # 或 Qwen3-VL-8B
  freeze(vision_encoder)  # vision encoder 冻结
  For step in 1..max_steps:
      For each (v, q) in D_sampled[batch]:
          # 对每个 (v,q) 采样 G=8 个 responses
          For g in 1..G:
              y^(g) = π_θ(v, q)  # 直接输出 "(t_start, t_end)"
              # Thinking-free: 无 thinking 过程
              r^(g) = IoU(Ŝ^(g), S*)  # 仅用 IoU 作为奖励
          # 计算 advantage
          r_mean = (1/G) * Σ r^(j)
          For each g:
              A^(g) = r^(g) - r_mean
          # GRPO loss
          L = -(1/G) * Σ A^(g) * log π_θ(y^(g) | v, q)
          # KL 正则项 β=0 在本工作中不使用
      θ = θ - lr * ∇L  # lr=1e-6

      # Early stopping check
      if reward_plateau(temporal_IoU_reward) and
         reward_plateau(group_stddev):
          break  # 约 310 steps 时触发
  # 输出: π_θ* (TimeLens model)
  ```

  张量计算示例（interleaved textual encoding 的 forward pass）：
  ```
  输入: video V ∈ R^{T×H×W×3}  # T frames
  # Step 1: 每帧复制为两个相同 copy（绕过 MRoPE 的 frame merge）
  V_expanded = duplicate(V)  # T × 2 个 frame copies

  # Step 2: Vision encoder 处理
  For each frame pair (f_i, f_i_copy):
      patch_i = patch_embed(concat(f_i, f_i_copy))  # 每两个 frame 合并
      visual_tokens_i = vision_transformer(patch_i)  # shape: [n_patches, d_model]

  # Step 3: Interleaved timestamp injection
  For each visual_tokens_i:
      t_i = i / fps  # 时间戳 (秒)
      text_token_i = text_embed(f"{t_i:.1f}s")  # [1, d_model]
      # 交错插入
      sequence_i = [text_token_i; visual_tokens_i]  # prefix: timestamp before visual

  # Step 4: Full sequence
  full_seq = concat([sequence_0, sequence_1, ..., sequence_{T-1}])
  # Also prepend system prompt tokens
  full_seq = [prompt_tokens; full_seq]

  # Step 5: LLM decode
  output = LLM(full_seq)  # autoregressive generate "(t_start, t_end)"
  ```

  训练数据质量对比消融（Tab. 5）：
  | Training Data | Charades-TimeLens mIoU | ActivityNet-TimeLens mIoU | QVHighlights-TimeLens mIoU |
  |---|---|---|---|
  | Original Noisy Data | 35.6 | 31.3 | 44.6 |
  | TimeLens-100K | 48.3 | 43.1 | 56.7 |

  核心发现总结：
  - Interleaved textual prefix + raw timestamps 是最优 timestamp encoding
  - Thinking-free RLVR 在效率和性能上均优于 SFT 和 thinking-based RLVR
  - Early stopping (reward plateau 时) 省计算且避免性能退化
  - Difficulty-based sampling (μ=0.05 → difficulty ~0.95) 对 RLVR 性能至关重要
