## TimeSearch-R__Adaptive_Temporal_Search_for_Long-Form_Video_Understanding_via_Self-Verification_Reinforcement_Learning

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：**TimeSearch-R** 框架，包含两大核心创新：(1) **Interleaved Text-Video Thinking** —— 将时序搜索重新定义为文本-视频交错的思维过程。在每个推理步 k，policy model π_θ 生成文本推理 T_k；若 T_k 包含搜索指令，video environment 根据时间边界 [t_s^k, t_e^k] 和文本 query q_k 执行 search() 函数检索视频片段 V_k，将其追加到 CoT 中供后续步骤使用。搜索函数使用 SigLIP-400M 视觉编码器计算候选帧与 query 的相关性，再通过 DPP (Determinantal Point Process) 同时优化相关性和多样性，选出 F 帧最有信息量的帧。该过程循环直到模型输出最终答案或达到预算上限（最多 8 轮搜索，每轮最多 8 帧）。(2) **GRPO-CSV (Completeness Self-Verification)** —— 针对原始 GRPO 仅对最终答案给予奖励而忽略中间搜索决策导致的"搜索不充分"和"推理不一致"问题。CSV 在 GRPO rollout 阶段生成 text-video 交错 CoT C 和最终答案 A 后，提取 C 中的视频帧构成动态帧集 V_c，用同一 policy model 仅基于 V_c 重新回答（禁止新搜索）得到 CSV 答案 A_c。Completeness Reward: R_c = 1[Acc(A,A*)>0.5] · Acc(A_c,A*)，确保仅对正确轨迹施加 CSV 奖励。总奖励: R = R_c + R_fmt + R_acc。

  实验比较：(1) Temporal Search: 对比 Uniform (Qwen2.5-VL/GPT-4o)、VideoAgent (GPT-4)、Retrieval-based (GPT-4o)、T* (GPT-4o)、VideoTree (GPT-4) 等方法；指标包括 Temporal F1/Precision/Recall、Visual F1、QA Accuracy (Haystack-LVBench, Haystack-Ego4D)。(2) Long-Form Video Understanding: 对比 Qwen2.5-VL-7B、GPT-4o、Gemini-1.5-Pro、VideoAgent、VideoTree、T*、Video-R1-7B；指标包括 VideoMME (short/medium/long/overall)、MLVU (m-avg)、LongVideoBench。(3) Ablation: 训练阶段 (zero-shot CoT → SFT → GRPO → GRPO-CSV)、GRPO-CSV 组件消融 (w/o CSV vs w/ CSV vs w/ CSV+Acc)、数据组成消融 (有无 filtering、ego/exo domain diversity)。(4) Efficiency: end-to-end latency vs VideoAgent/T*/Retrieval-based on Haystack-Ego4D on A100。

- 硬件平台是什么，配置是什么。
  RL 训练：32 × NVIDIA A100 GPU。使用 DeepSpeed ZeRO-3 Offload 做内存优化，vLLM colocate mode 做 rollout 推理加速。Batch size per GPU = 1，Gradient Accumulation Steps = 2。Mixed precision bfloat16 + Flash Attention 2.0。推理效率评估：A100 GPU 上测 end-to-end latency。SFT 冷启动阶段使用 GPT-4o 生成 text-video 交错推理数据。

- 模型是什么。数据集和bench分别是什么。
  模型：基于 Qwen2.5-VL-7B-Instruct 做两阶段训练 (SFT → RL with GRPO-CSV)。基座模型使用 SigLIP-400M 作为搜索函数的视觉编码器（独立于 policy model）。训练数据经过两阶段过滤：(1) Stage 1: Visual Dependency Filtering —— 用 4 帧均匀采样过滤可被纯语言偏置解答的样本；(2) Stage 2: Search Usefulness Filtering —— 用最多 64 帧 + dynamic temporal search 过滤即使充分搜索也无法回答的样本。数据来源：Haystack-Ego4D (49.5%)、VideoMarathon/Panda-70M (35.6%)、CinePile (9.5%)、其他 (5.4%)；open-ended QA 占 60.3%，multiple-choice 占 39.7%；平均视频时长 1659s。

  评测 benchmark：(1) Temporal search: Haystack-LVBench (needle-in-a-haystack, 含 temporal/visual similarity + QA accuracy)、Haystack-Ego4D (test-tiny subset)。(2) Long-form video understanding: VideoMME (w/o sub, 分 short/medium/long)、MLVU (m-avg)、LongVideoBench (LVB)。另外自定义两个评估指标：completeness（搜索帧集是否足以回答正确）和 consistency（中间推理与最终答案是否一致）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/Time-Search/TimeSearch-R

  TimeSearch-R 算法 pipeline 伪代码：
  ```
  # ===== 初始化 =====
  # policy model π_θ: Qwen2.5-VL-7B-Instruct (Qwen2.5-VL backbone)
  # search function: SigLIP-400M encoder + DPP frame selection
  # 视频 V, 问题 Q, 初始预览 frames Ṽ (uniform sampling)

  # ===== Phase 0: 推理-搜索交错过程 =====
  def time_search_reasoning(V, Q):
      Ṽ = uniform_sample(V, fps=2, max_frames=768)  # 初始预览
      C = []  # text-video interleaved CoT
      for k in range(1, 9):  # max 8 search turns
          # Step 1: 文本推理 - policy model 生成 reasoning
          T_k = π_θ.think(Ṽ, Q, C)  # <think>...</think>
          
          # Step 2: 检查是否搜索或回答
          if contains_<tool_call>(T_k):
              t_s, t_e, q, F = parse_search_params(T_k)
              # Step 3: 视频环境执行搜索
              # DPP-based frame selection
              F_cand = subsample(V, t_s, t_e, N_cand)
              # 计算视觉嵌入
              h_i = SigLIP.encode(f_i) for f_i in F_cand  # [N, d]
              q_emb = SigLIP.encode_text(q)              # [d]
              # 相关性分数
              r_i = (q_emb^T · h_i)  # unnormalized relevance
              r_i = (r_i - min(r)) / (max(r) - min(r) + ε)  # [0,1]
              # DPP kernel: L̃ = diag(r) · S · diag(r)
              S_ij = h_i^T · h_j  # pairwise cosine similarity
              L̃_ij = r_i · r_j · S_ij
              # Fast greedy MAP inference 选 top-F 帧
              V_k = greedy_map_inference(L̃, F)  # |V_k| = F ≤ 8
              C.append((T_k, V_k))  # 追加到 CoT
          elif contains_<answer>(T_k):
              A = parse_answer(T_k)
              return C, A
      return C, None  # budget exhausted

  # ===== Phase 1: GRPO-CSV Training =====
  # 每步训练:
  for prompt (V, Q, A*) in batch:
      # --- GRPO Rollout ---
      # 生成 8 个 rollout trajectories
      for i in range(8):
          C_i, A_i = time_search_reasoning(V, Q)
          # 提取动态帧集 V_c
          V_c_i = extract_all_frames(C_i)
      
      # --- CSV Rollout ---
      for i in range(8):
          # 用同一 π_θ 基于 V_c_i 重新回答（禁止搜索）
          A_c_i = π_θ.answer_no_search(Q, V_c_i)
      
      # --- 奖励计算 ---
      for i in range(8):
          R_acc_i = 1 if A_i == A* else 0  # Accuracy reward
          R_fmt_i = 1 if format_valid(C_i) else 0  # Format reward
          if Acc(A_i, A*) > 0.5:  # 仅正确轨迹
              R_c_i = Acc(A_c_i, A*)  # Completeness reward
          else:
              R_c_i = 0
          R_i = R_c_i + R_fmt_i + R_acc_i
      
      # --- GRPO Policy Update ---
      # advantage = (R_i - mean(R)) / std(R)
      # L_GRPO = -E[min(r_i(θ)·A_i, clip(r_i(θ),1-ε,1+ε)·A_i)]
      # r_i(θ) = π_θ(C_i,A_i|Ṽ,Q) / π_old(C_i,A_i|Ṽ,Q)
      # plus KL penalty β · KL(π_θ || π_ref) with β=0.005
      θ = optimizer(AdamW, lr=1e-6).step(L_GRPO + β·KL)

  # ===== Phase 2: SFT Cold-Start (两阶段训练的前置步骤) =====
  # GPT-4o 生成 text-video interleaved CoT 训练数据
  # 训练时 mask 视频 token，只计算 reasoning token 的 cross-entropy loss
  # L_SFT = -Σ log π_θ(token_t | context, Ṽ, Q)  over reasoning tokens
  ```

  关键张量流：
  ```
  输入: 初始预览 Ṽ (uniform sampling, max 768 frames @ 2fps)
  每帧: Qwen2.5-VL 原生 dynamic-FPS + absolute time encoding
        frame → [12-256 visual tokens] concatenated with timestamps

  Step k 搜索:
    query q → SigLIP-400M text encoder → q_emb ∈ R^d (d=768)
    frames in [t_s,t_e] → SigLIP-400M vision encoder → h_i ∈ R^d
    DPP kernel: L̃ ∈ R^(N×N), 选 F=8 帧最大化 det(L̃_S)
    selected frames → V_k = {f_k^1, ..., f_k^F}

  Policy model forward:
    input: [Ṽ_tokens, T_1, V_1_tokens, T_2, V_2_tokens, ..., T_k, Q]
    model: Qwen2.5-VL-7B backbone (32 layers, GQA, RoPE)
    output: next reasoning text / search instruction / answer

  CSV phase:
    input: [V_c_tokens, Q]  (仅搜索到的帧 + 问题，禁止 tool_call)
    output: A_c (bare answer, "I don't know" if insufficient)
  ```

  训练两个阶段的作用：
  - **SFT (Cold-Start)**: 教模型正确的 reasoning format 和 <tool_call> 格式，从 zero-shot 无法搜索（Temporal F1=0.0）提升到 F1=7.8
  - **RL (GRPO-CSV)**: 进一步提升 reasoning consistency (+2.6%) 和 QA accuracy (59.2%→66.6%)，GRPO-CSV 防止训练崩塌（w/o CSV 约 300 步后模型停止搜索）
