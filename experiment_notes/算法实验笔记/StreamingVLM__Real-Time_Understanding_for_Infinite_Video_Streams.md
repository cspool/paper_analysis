## StreamingVLM__Real-Time_Understanding_for_Infinite_Video_Streams

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：(1) **Streaming-aware KV Cache**：推理时维持紧凑 KV cache，复用 attention sink tokens（Tsink=512）、近期 text 长窗口（Twindow=512）、近期 vision 短窗口（Vwindow=16s），旧 vision tokens 优先驱逐，旧 text 仅在超出 budget 时驱逐。(2) **Contiguous RoPE**：当旧 token 被驱逐后，后续 token 的 RoPE 位置索引左移保持连续，超出总窗口后位置索引不再增长，保持在训练长度范围内。(3) **Overlapped-Chunk Full-Attention Training**：训练时不复制推理的滑动窗口，而是将长视频切分为 W=24s 的 chunk（O=12s 重叠），每个 chunk 内做 full attention，让每个 chunk 内部的 attention pattern 近似推理时的 attention sink + sliding window。Vision/text tokens 以 1s 为间隔交错排列，仅在 text 位置计算 loss。
  
  实验比较：
  - Captioning：Inf-Streams-Eval（20 场完整比赛，平均 2.12 小时）上对比 GPT-4o mini（chunk 模式）、LiveCC-7B-Instruct（chunk/infinite 模式）、ReKV（训练无关 KV cache 驱逐方法），以 GPT-5 投票 win rate 为指标；LiveCC-Sports-3K CC 上对比 LLaVA-Video、GPT-4o、Gemini、LiveCC。
  - VQA：MVBench、VideoMME (w/o subs.)、LongVideoBench、OVOBench Realtime 上对比 Qwen2.5-VL-7B-Instruct（SFT 前的 base model），验证 SFT pipeline 提升通用视觉能力。
  - Efficiency：单卡 H100 上测试 per-token latency vs. 视频长度，对比 Full Attention、Sliding Window w/o Overlap、Sliding Window w/ Overlap。
  - Ablation：Contiguous RoPE vs. Native RoPE（infinite/chunk 模式）；Tsink/Twindow/Vwindow 大小；SFT strategy 和数据集消融（Live-WhisperX-526K → +Inf-Streams-Train → +High-Quality Annealing）。

- 硬件平台是什么，配置是什么。
  训练：NVIDIA H100 GPU，总计算量约 128 H100-days（两阶段：SFT 525K+526K 样本 → 高质量 annealing 14K 样本）。推理：单张 NVIDIA H100，bfloat16，维持 8 FPS 实时视频理解。

- 模型是什么。数据集和bench分别是什么。
  模型：Backbone 为 Qwen2.5-VL-7B-Instruct；对比模型包括 GPT-4o mini、LiveCC-7B-Instruct、LiveCC-7B-Instruct (infinite mode)。
  训练集：
  - Inf-Streams-Train：自行构建，5 种体育项目（篮球 712 场、足球 544 场、冰球 402 场、棒球 399 场、美式足球 392 场），总计 2,449 场比赛，使用 WhisperX 提取 ASR 实时解说 → GPT-5 清洗（keep 46.32%/edit 37.89%/delete 15.79%）→ overlapped chunking（W=24s, O=12s），最终 525K streaming samples。
  - LiveCC 的 Live-WhisperX-526K：526K streaming samples。
  - High-quality annealing data：14K samples（16-64s clips，GPT-5 筛选实时解说比例 > 80%）。
  评测 benchmark：
  - Captioning: Inf-Streams-Eval（20 场完整比赛，平均 2.12h，per-second 对齐）、LiveCC-Sports-3K CC（49 运动，416 clips, ≥10s）
  - VQA: MVBench（细粒度动作/物体/计数/时序）、VideoMME（多任务 QA/caption/grounding）、LongVideoBench（长视频 QA，需长期记忆和跨段推理）、OVOBench Realtime（流式感知理解）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/mit-han-lab/streaming-vlm。基于 Qwen2.5-VL-7B-Instruct（开源）训练。

  算法 pipeline 核心流程（推理时）：
  ```
  输入: 无限视频帧流，基础模型 Qwen2.5-VL
  参数: T_sink=512, T_window=512, V_window=16s

  初始化 KV cache = {}，position_offset = 0

  Loop over time steps:
      # Step 1: 新帧 tokenize
      V_new = vision_encoder(frames_new)   # 如 1s=24 frames → N_v tokens

      # Step 2: interleave vision/text tokens（1s 间隔）
      # 无解说词的秒插入占位符 "..."

      # Step 3: Contiguous RoPE 位置编码
      current_pos = KV_cache.length
      RoPE_indices = [position_offset, ..., position_offset + len(V_new) + len(T_new) - 1]
      # 但不超过训练最大长度 L_max
      RoPE_indices = RoPE_indices % L_max   # 简化，实际以 bounded range 保持连续

      # Step 4: Full attention 计算（仅对当前窗口内 tokens）
      Q, K, V = project(V_new + T_new)
      attend_to = KV_cache  # 复用历史 KV，不 recompute
      attention = softmax(Q @ K_attend.T / sqrt(d)) @ V_attend

      # Step 5: 更新 KV cache（eviction policy）
      KV_cache = KV_cache + new_KV
      # 保留: T_sink 个 attention sink tokens（system prompt + 早期 text）
      # 保留: T_window 个最近 text tokens  
      # 保留: V_window 秒的最近 vision tokens
      # 驱逐: 旧 vision tokens 优先，旧 text 仅在超 budget 时驱逐

      # Step 6: 自回归生成
      output = model.generate(..., past_key_values=KV_cache, position_ids=RoPE_indices)
  ```

  训练时（overlapped-chunk, full-attention）：
  ```
  输入: 完整体育比赛视频
  参数: W=24s, O=12s

  # 切分为 overlapped chunks
  For chunk_i in sliding_window(video, window=W, overlap=O):
      # chunk_i 包含 W 秒内以 1s 间隔 interleaved 的 vision+text tokens
      V_chunk = vision_encoder(chunk_i_frames)   # 24 frames@1fps
      T_chunk = tokenize(chunk_i_commentary)     # 对应解说词

      # Full attention within chunk（每个 token attend 到同 chunk 所有 token）
      # 不复制推理时的 sink/sliding window mask
      # 仅在 text position 计算 cross-entropy loss
      loss = CE(model(V_chunk ⊕ T_chunk).logits[text_positions], labels)

      # 前一段的 previous text 取 T_sink 开头 + T_window 结尾 tokens
  ```

  训练-推理一致性关键点：overlapped chunk 内 full attention 的 attention pattern 近似推理时 "sink tokens → 全可见 + 近期 text 窗口 + 近期 vision 窗口" 的有效注意力模式，teaching the model recency bias without training on prohibitively long contexts。

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：SAGE (Smart Any-horizon aGEnt) —— 一个面向长视频推理的Any-Horizon Agent系统。核心包含三部分：
  (1) **SAGE Agent System（System Design）**：两阶段多轮推理架构。Stage-1（Context VLM）接受128帧采样视频(F)、视频元数据(M)、工具定义(T)和用户查询(Q)，SAGE-MM输出 video-context + query-intent + final-answer 或 tool-call。Stage-2（Iterative Reasoner）迭代决定是否可回答或继续调用工具，最多11步。支持6种工具：web-search（Google Search API via Serper）、parse-website、transcribe-speech（Whisper-large-v3）、ground-event（Qwen3-VL-30B-A3B-Instruct）、extract-video-parts、analyze（Qwen3-VL-30B-A3B-Instruct）。关键创新：通过知识驱动（web search + speech transcription）而非纯时序定位来实现高效事件搜索；segment-level grounding（每次最多10分钟）而非whole-video grounding。
  (2) **Synthetic Data Generation Pipeline**：使用Gemini-2.5-Flash一次性处理完整长视频生成10-20个QnA pairs（覆盖全时间跨度，通过percent_video_parsed字段强制覆盖）。再使用SAGE系统（Gemini-2.5-Flash作为SAGE-MM）生成tool call trajectories用于cold-start SFT。相比人工标注节省~100倍成本、相比subclip pipeline节省~10倍时间。99.1k训练问题来自6659个YouTube视频，417.7k state-action pairs用于SFT。
  (3) **Multi-Reward RL Post-Training（GRPO）**：使用GRPO进行trajectory-level优化。Rollout 8条轨迹/sample。Reward由step-level rewards + accuracy reward组成：
    - s_format: JSON格式奖励（+0.05或-0.10）
    - s_reasonable-tool: GPT-4o判断当前tool call是否合理（+0.10或-0.10）
    - s_args-repeat: 惩罚重复tool call参数（-0.05 × sqrt(num-repetitions)）
    - s_args-valid: 惩罚无效参数（-0.1或0）
    - a_N (Accuracy Reward): LLM-as-Judge (GPT-4o) 判断最终答案语义正确性。错误回答:-0.5, 正确+visual tools:+1.25, 正确无tools:+1.0, JSON无效:-2.0
  KL-divergence loss coefficient=0.005。前100步Nmax=6稳定训练，之后Nmax=11。

  实验比较：
  (a) **SAGE-Bench主结果（Table 4）**：1744样本（802 MCQ + 942 open-ended），平均时长727秒。DIRECT baselines（Gemini-2.5-Flash, GPT-4o, Video-Thinker-7B, LongVILA-R1-7B, VideoRFT-7B, Video-R1-7B, Qwen3-VL系列）和AGENT baselines（VideoAgent, LVAgent, LongVT, VideoMind, VideoExplorer, VideoChat-R1.5）。SAGE-MM基于Qwen3-VL-8B-Instruct SFT+RL在SAGE-Bench达68.0 overall，SAGE-Flash（Gemini-2.5-Flash作为tool backend）达71.8。
  (b) **MINERVA（Table 5）**：SAGE在>600秒视频上improvement 2.6%。
  (c) **Video-MMMU & Video-MME（Table 6）**：SAGE-Flash在Video-MMMU达68.1，超Video-R1（61.5）。
  (d) **Duration-wise分析（Table 8）**：600-1200秒bucket改善8.2%（SAGE）、14.6%（SAGE-Flash）。
  (e) **Training Mode消融（Table 7）**：AGENT模式优于DIRECT模式训练。
  (f) **Any-Horizon Reasoning（Table 9）**：RL改善SFT模型的tool overcalling，使单轮/多轮分布更接近expert Gemini-2.5-Flash。
  (g) **Tool消融（Table 10）**：transcribe-speech和extract-video-parts最重要。
  (h) **Runtime（Table 11）**：SAGE 8.6s/sample，vs VideoMind 24.7s，VideoAgent 1445.0s。
  (i) **Appendices中的额外消融**：Video input重要性（Table 12）、Eval mode（Table 13）、SFT必要性（Table 14）、#Turns vs Duration（Table 15）、Nmax影响（Table 16）、Variance（Table 17）、Per-tool accuracy（Table 18）。

- 硬件平台是什么，配置是什么。
  训练：16×NVIDIA H100 GPUs，SFT和RL阶段均使用此配置。SFT: batch size=64, lr=1e-5（linear decay），1 epoch。RL: batch size=16, rollout 8条trajectories/sample, lr=1e-6（cosine decay），KL coeff=0.005，训练480 steps。RL前100步Nmax=6，之后Nmax=11。
  推理评估：使用vLLM serving所有模型，温度0.0。非确定性输出时temperature=0.7，最多4次重试。

- 模型是什么。数据集和bench分别是什么。
  模型：
  - SAGE-MM variants: Qwen3-VL-8B-Instruct（默认）、Qwen3-VL-4B-Instruct、Qwen2.5-VL-7B-Instruct、Molmo2-8B
  - Tool backends: Qwen3-VL-30B-A3B-Instruct（ground-event + analyze tools）、Whisper-large-v3（transcribe-speech）
  - Expert orchestrators（无training）: Gemini-2.5-Flash、GPT-4o
  - LLM-as-Judge: GPT-4o（accuracy reward + evaluation）
  数据集：
  - 训练：13个YouTube频道（Formula1, ZachChoi, TheDailyShow, MrBean, TheOffice, Friends, fluffyguy, trevornoah, Vox, kurzgesagt, veritasium, QuantaScienceChannel, WalkingAlice），6659视频 → 99.1k 训练问题 + 417.7k SFT state-action pairs。RL数据：7.68k样本（一半需tool calls，一半single-turn）。
  Benchmarks：
  - SAGE-Bench：1744样本（802 MCQ + 942 open-ended），平均727秒，来自娱乐YouTube视频
  - MINERVA：复杂视频推理benchmark（体育、短片、烹饪）
  - Video-MMMU：多学科专业视频知识获取
  - Video-MME：视频分析评估bechmark

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  开源: https://github.com/allenai/SAGE

  算法pipeline伪代码（SAGE推理 + GRPO训练核心流程）：

  ```
  # ========== SAGE推理 ==========
  # Input: 视频文件video_path, 用户查询Q
  # Stage-1: Context VLM
  F = sample_frames(video_path, n=128, fps=2)  # 128帧, 2 FPS采样
  M = {"path": video_path, "duration": duration_seconds}
  T = [web_search, parse_website, transcribe_speech, 
       ground_event, extract_video_parts, analyze]
  
  # SAGE-MM推理第一步
  prompt_1 = f"T|F|Q|M: {T}\n{M}\n{Q}"
  action_1 = SAGE-MM(prompt_1)  # JSON: video_context, query_intent, 
                                 #        recommended_tool | final_answer
  
  if action_1.final_answer is not None:
      return action_1.final_answer  # 单轮推理（any-horizon: 短视频直接回答）
  
  # Stage-2: Iterative Reasoner
  tool_results_1 = execute_tool(action_1.recommended_tool)
  C = action_1.video_context  # visual context cache
  
  for step in range(2, N_max+1):  # N_max=11
      # 累积历史中的所有action和tool结果
      history = [A_1, R_1, ..., A_{step-1}, R_{step-1}]
      prompt_k = f"T|Q|M|C|{history}"
      action_k = SAGE-MM(prompt_k)  # JSON: answerable, recommended_tool | final_answer
      
      if action_k.final_answer is not None:
          return action_k.final_answer
      
      tool_results_k = execute_tool(action_k.recommended_tool)
  
  return None  # 超时未回答

  # ========== GRPO RL训练 ==========
  # Input: S_i = {T, F, M, Q} for sample i
  # Rollout generation (batch_size=16, 8 trajectories each)
  for i in range(batch_size):
      for k in range(8):  # rollout 8 trajectories
          tau_{i,k} = []
          S_1 = {T, F, M, Q}
          for j in range(1, N_max+1):  # N_max=6 (前100步) or 11
              A_j = SAGE-MM(S_j)
              tau_{i,k}.append((S_j, A_j))
              if A_j.final_answer is not None:
                  break
              R_j = execute_tool(A_j.recommended_tool)
              S_{j+1} = {T, Q, M, C, A_1, R_1, ..., A_j, R_j}
  
  # Reward computation
  for each tau_i in rollout_trajectories:
      # Step-level rewards
      s_format    = +0.05 if valid_json else -0.10
      s_reasonable = +0.10 if GPT4o_judge_reasonable(tau_i, Q) else -0.10
      s_args_repeat = -0.05 * sqrt(count_repetitions(tau_i))
      s_args_valid  = -0.10 if invalid_args else 0
      
      # Accuracy reward (LLM-as-Judge: GPT-4o)
      if final_action_is_invalid_json:
          a_N = -2.0
      elif GPT4o_judge_correct(final_answer, ground_truth):
          a_N = +1.25 if used_visual_tools_in_tau_i else +1.0
      else:
          a_N = -0.5 if N >= 1 else ...
      
      # Uniform reward for all actions in trajectory
      R_i = sum(step_rewards) + a_N
      r(A_1) = r(A_2) = ... = r(A_N) = R_i
  
  # GRPO advantage computation and policy update
  for each sample i:
      advantages = compute_group_advantages([R_{i,1}, ..., R_{i,8}])
      # Update SAGE-MM using GRPO loss:
      # L = -E[min(r_t * A, clip(r_t, 1-eps, 1+eps) * A)] + beta * KL(pi||pi_ref)
      # KL coeff beta = 0.005
  ```
