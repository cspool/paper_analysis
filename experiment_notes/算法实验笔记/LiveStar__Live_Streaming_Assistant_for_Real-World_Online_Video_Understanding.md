## LiveStar__Live_Streaming_Assistant_for_Real-World_Online_Video_Understanding

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：LiveStar —— 面向在线视频理解的直播流助手，核心包含三个算法创新：(1) **SCAM (Streaming Causal Attention Masks)** 训练策略 —— 构建交错帧-字幕序列（interleaved frame-caption sequences），通过 causal masked attention 约束进行增量式视频-语言对齐。训练目标为 `max P([Cap_i^k] | [Ctx^{< t_i} {Mask^{≤t_i}}], [Frm^{t_i}])`，Mask 矩阵阻止对当前语义片段中已生成字幕 token 的注意力，但保留前一语义片段的终端字幕以传递场景语义边界。从 M 个释义字幕池中随机采样字幕防止过拟合。(2) **SVeD (Streaming Verification Decoding)** 推理框架 —— 在每个 incoming frame 时刻 t_j，通过单次前向传播验证最新字幕 [Dec] 的 perplexity：`PPL^{t_j}([Dec]) = sqrt[N]{1/P([Dec] | [Ctx^{≤t_j}], [Frm^{t_j}])}`。若 `PPL^{t_j}([Dec]) > α · PPL^{t_i}([Dec])`（α=1.03 默认），激活解码 gate 生成新字幕；否则保持沉默并将 [Dec] 移至上下文末尾。(3) **Peak-End Memory Compression** —— 受认知科学 Peak-End 规则启发，对 10+ 分钟视频（3 fps）进行记忆压缩。利用预计算 PPL 检测关键帧，结合语义片段终端字幕，以概率剪枝超出窗口 W（默认 40 帧）的旧帧。

  实验比较：
  (a) OmniStar 在线评估 —— 5 tasks (RNG/OTG/FDQ/COQ/MIQ)，对比 VideoLLM-online, VideoLLM-MoD, MMDuet，指标含 SemCor, TimDiff, TimRedun, TimCover, SumFluen, FPS；(b) OmniStar 离线评估 —— 固定解码时间点，对比 GPT-4V/4o, LLaVA-Video, InternVideo2.5, InternVL2.5, MiniCPM-V 2.6, Qwen2.5-VL, VideoLLM-online, VideoLLM-MoD, MMDuet；(c) Ego4D Narration Stream —— 对比 VideoLLM-online, VideoLLM-MoD, LION-FS, MMDuet；(d) SVBench —— 对话和流式评估对比；(e) 消融 —— 响应-沉默阈值 α（1.0-1.1）、记忆压缩策略（Uniform/FIFO/Peak-End vs KV Cache）、释义字幕池大小 M（1/2/3）。

- 硬件平台是什么，配置是什么。
  训练：8× NVIDIA A800 GPU。Full fine-tuning（Vision Encoder 冻结，MLP projector + LLM 全更新）。AdamW optimizer (β1=0.9, β2=0.999, weight decay=0.05)，learning rate 4×10⁻⁵，per-device batch size=1，gradient accumulation 4 steps (effective bs=32)，cosine LR scheduling with warmup ratio=0.03，训练 1 epoch。每序列最多 8192 tokens（8K context window）。静态分辨率策略：输入帧 resize 至 448×448，patch downsampling ratio=0.5。

- 模型是什么。数据集和bench分别是什么。
  模型：Vision Encoder — InternViT（冻结），每帧提取 16 tokens，帧率 1-4 FPS。MLP Projector — 将帧嵌入映射到 LLM embedding 空间。LLM Backbone — InternLM2.5-7B（全微调）。序列长度最多 8192 tokens。
  训练数据：Phase I — 63K 视频片段（ActivityNet Captions 9K + Shot2Story 33K + Ego4D Narration Stream 20K + MVBench 1K）。Phase II — 20K OmniStar 训练样本。总计 83K。
  Benchmarks：(1) OmniStar — 20,137 视频，15 种真实场景（Travel & Events, Sports, Pets & Animals, Music, Autos & Vehicles, Film & Animation, Nonprofits & Activism, Science & Technology, Education, Howto & Style, News & Politics, Entertainment, Comedy, People & Blogs, Gaming），5 任务（RNG/OTG/FDQ/COQ/MIQ）；(2) Ego4D Narration Stream；(3) SVBench。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/yzy-bupt/LiveStar。

  算法 pipeline 伪代码：

  ```
  # ===== SCAM 训练: Streaming Causal Attention Masks =====
  # 输入: 视频帧序列 + 语义片段 
  # 对语义片段 C_k = {t_m,...,t_n}, 每帧 Frm^{t_i} 伴随字幕 Cap^k
  # 从 M 个 paraphrased captions 中随机采样

  def build_scam_mask(seq_len, clip_boundaries, caption_positions):
      mask = causal_mask(seq_len)  # 标准 causal mask
      for t_i in current_clip:
          # 掩蔽当前clip中已生成的字幕token
          for cap_pos in same_clip_captions_before(t_i):
              mask[t_i, cap_pos] = -inf
          # 掩蔽之前clips中非终端字幕
          for pos in non_terminal_captions(prev_clips):
              mask[t_i, pos] = -inf
      return mask

  # 训练目标: max P([Cap_i^k] | [Ctx^{<t_i} {Mask^{≤t_i}}], [Frm^{t_i}])
  # 仅对 assistant response tokens 计算 cross-entropy loss

  # ===== SVeD 推理: Streaming Verification Decoding =====
  def sved_inference(frame_stream, alpha=1.03):
      Dec = None; Ctx = []; t_i = 0
      for t_j, Frm in enumerate(frame_stream):
          Ctx.append(Frm)
          if Dec is not None:
              # 单次 forward pass 验证 perplexity
              PPL_new = forward_pass(Dec, Ctx).PPL
              if PPL_new > alpha * PPL_cache[t_i]:
                  Dec = generate(Ctx)       # 激活解码
                  Ctx.append(Dec); t_i = t_j
                  PPL_cache[t_j] = forward_pass(Dec, Ctx).PPL
              else:
                  swap_last_two(Ctx)        # 沉默: Dec移到末尾
          else:
              Dec = generate(Ctx)
              Ctx.append(Dec); t_i = t_j
              PPL_cache[t_j] = forward_pass(Dec, Ctx).PPL
      return all_captions

  # ===== Peak-End Memory Compression =====
  def peak_end_compression(frames, captions, window_W=40):
      # 关键帧: 低PPL=高重要性, PPL from SVeD
      for f in frames:
          f.score = 1.0 / f.precomputed_PPL
      terminal_caps = [clip.last_caption for clip in clips]
      # 概率剪枝: P(delete) ∝ relative_PPL × elapsed_time
      for f in frames_older_than(window_W):
          p = (f.PPL / max_PPL_in_clip) * \
              (f.time / total_duration)
          if random() < p: drop(f)
      return frames_after_prune, terminal_caps
  ```

  关键张量计算与设计要点：
  - Vision: 448×448 → InternViT → [16, D] per frame (16 visual tokens)
  - Perplexity: PPL = exp(-1/N Σ log P(token_i | context, past_tokens))
  - SCAM Mask: [seq_len, seq_len] 稀疏 causal mask，跨clip non-terminal captions 被masked
  - SVeD: 仅需单次 forward pass 算 PPL（非完整 decoding），比 EOS-based 方法更快
  - Peak-End: W=40 frames ≈ 13.3s @3fps, 剪枝概率正比于 PPL 和 elapsed time
  - KV Cache: 双级缓存 (intra-dialogue frame-level + inter-dialogue cross-conversation)
