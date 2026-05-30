## Divide__then_Ground__Adapting_Frame_Selection_to_Query_Types_for_Long-Form_Video_Understanding

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：DIG 是一个 training-free 的帧选择框架，按查询类型自适应选择采样策略。核心流程：(1) Query Identification —— 用 Qwen3-Next-80B-A3B 通过 Chain-of-Thought prompting 将查询分类为 global query（需要整体视频理解）或 localized query（针对特定时间片段）；(2) 对 global query 直接用 uniform sampling；对 localized query 启动三阶段 pipeline：(a) Content-Adaptive Frame Selection (CAFS) —— 以 2 fps 采样 M 帧，用 DINOv2 提取特征向量，计算相邻帧余弦距离 d_i = 1 - sim(V_I_i, V_I_{i+1})，检测距离峰值（prominence > 0.1）作为分割点，取相邻分割点间中间帧作为 r-frames（代表帧）；(b) Reward Assignment —— 用 LMM 对每个 r-frame 进行二维评分（0-100）：(1) 帧对问题的直接有用性，(2) 帧是否暗示相邻帧包含补充信息；(c) Video Refinement —— 迭代式 reward-guided selection：计算均值 R̄，每轮更新 R'_j = max(R_j - R̄, 0)，迭代至奖励集稳定，以最终正值 r-frames 为关键帧；对每个关键帧取其周围窗口 [K_{j-wlen}, K_{j+wlen+1}]（wlen=2）的视频段做 union，得到 refined video，最后从 refined video 中 uniform sampling 作为 LMM 输入。每帧用 56 tokens 表示。
  实验比较：(a) 与 uniform sampling (UNI)、AKS [18]、Q-Frame [64] 在 MLVU、LongVideoBench (LVB)、VideoMME (仅 medium+long splits) 上的 accuracy 对比，覆盖 Qwen2.5-VL-7B 和 Qwen2.5-VL-32B 两种 LMM，帧数从 8 到 256；(b) 模块分析 —— 6.1 帧选择策略 vs 查询类型：global vs localized 上的 uniform vs pipeline 性能对比；(c) 6.2 CAFS 有效性 —— LoC（Localized Coverage）和 GIC（Global Coverage）指标，与 UNI 和 FPS 对比，CAFS 替换为 uniform sampling 的消融；(d) 6.3 Reward Assignment —— LMM-based reward vs CLIPScore 对比（Qwen2.5-VL-7B/32B 作为 reward assigner）；(e) 6.4 窗口长度 wlen ∈ {0,2,4,8} 消融；(f) 6.5 效率分析 —— FLOPs vs accuracy 散点图；(g) 扩展实验 —— Qwen3-VL-8B 上 8 至 768 帧的 scalability 测试，对比 UNI 和 AKS；(h) 附录 F.3 逐任务 breakdown（MLVU: PQA/NQA/AC/AO/ER/AR/TR, VideoMME: ORA/ORC/ARA/INS/COP/TER/TEP/SPP/SPR/OCR/ATP/ACR, LVB: L1-Perception/L2-Relation 子任务）；(i) 附录 G 效率分析 —— runtime profiling（QI/CAFS/RA/VR 各阶段耗时）和 Query Identification 带来的效率增益。

- 硬件平台是什么，配置是什么。
  所有实验在 8 × NVIDIA A100 GPU 节点上执行。推理加速使用 vLLM backend（用于 query identification 和 reward assignment 阶段）。评估框架为 LMMs-Eval。

- 模型是什么。数据集和bench分别是什么。
  模型：(1) 主 LMM —— Qwen2.5-VL-7B 和 Qwen2.5-VL-32B（推理 backbone）；(2) Query Identification LLM —— Qwen3-Next-80B-A3B-Instruct；(3) 特征提取 —— DINOv2（CAFS 阶段）；(4) 扩展实验 —— Qwen3-VL-8B。所有模型参数冻结，不做训练。
  Benchmarks（3 个，均不使用字幕/音频，仅视频+问题）：
  (1) MLVU [54] —— 多任务长视频理解，~2600 QA pairs（dev set MC questions），平均时长 636.2s，含 462 global + 1708 localized queries，任务分 PlotQA、NeedleQA、Action Count、Action Order、Ego Reasoning、Anomaly Recognition、Topic Reasoning；
  (2) LongVideoBench [55] —— 3763 videos / 6678 QA pairs（val split 1337），平均时长 732.2s，全为 localized queries（referring reasoning），任务分 L1-Perception（S2E/S2A/O2E/T2O/S2O/T2E/E2O/T2A）和 L2-Relation（TOS/E3E/SAA/O3O/T3O/T3E/TAA/SSS/SOS）；
  (3) VideoMME [56] —— 900 videos / 2700 QA pairs，仅用 medium（516.8s）和 long（2466.3s）splits，含 479 global + 2221 localized queries，任务分 Object Reasoning、Object Recognition、Action Reasoning、Information Synopsis、Counting Problem、Temporal Reasoning、Temporal Perception、Spatial Perception、Spatial Reasoning、OCR、Attribute Perception、Action Recognition。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/Jialuo-Li/DIG
  
  算法 pipeline 伪代码：
  ```
  # === DIG 推理流程 ===
  # 输入: 视频 V (T frames, 原始帧率), 查询 Q
  # 参数: wlen=2, 每帧 56 tokens
  
  # Step 1: Query Identification
  is_global = LLM_classify(Q)  # Qwen3-Next-80B-A3B, CoT prompt
  if is_global:
      frames = uniform_sample(V, N)
      return LMM_inference(frames, Q)   # 直接推理
  
  # Step 2: Content-Adaptive Frame Selection (CAFS)
  # 以 2 fps 采样 M 帧
  F = sample_2fps(V)  # {f_{I_i}}_{i=1}^M
  # DINOv2 特征提取
  for i in 1..M:
      V_i = DINOv2(f_{I_i})   # shape: (d_dino,)
  # 逐帧距离计算
  for i in 1..M-1:
      d_i = 1 - cosine_sim(V_i, V_{i+1})  # scalar
  # 峰值检测 + prominence 过滤
  peaks = {i | d_{i-1} < d_i and d_{i+1} < d_i}
  valid_peaks = {j in peaks | prominence(d_j) > 0.1}
  # 选 r-frames (相邻峰值中间帧)
  R_idx = {(I_{p1} + I_{p2})/2 for consecutive p1, p2 in valid_peaks}
  r_frames = {f_I | I in R_idx}  # N_r 个代表帧
  
  # Step 3: Reward Assignment (LMM-based)
  for each r_frame in r_frames:
      score = LMM_reward(r_frame, Q)  # 0-100, 二维评分
      # prompt: 描述 + 打分 (直接有用性 + 相邻帧补充信息)
  rewards = [R_1, R_2, ..., R_{N_r}]
  
  # Step 4: Video Refinement (迭代式 reward-guided selection)
  while not converged:
      R_mean = mean(rewards)
      rewards = [max(R_i - R_mean, 0) for R_i in rewards]
      positives = {j | rewards[j] > 0}
      if positives == prev_positives: break
  
  # Step 5: Segment Combination
  refined_segments = []
  for each j where rewards[j] > 0:
      # 窗口合并: [K_{j-wlen}, K_{j+wlen+1}]
      segment = V[K_{j-wlen} : K_{j+wlen+1}]
      refined_segments.append(segment)
  refined_video = union(refined_segments)
  
  # Step 6: Final Sampling & Inference
  final_frames = uniform_sample(refined_video, N)
  answer = LMM_inference(final_frames, Q)
  ```
  
  张量计算细节：
  - DINOv2 特征：`V_i ∈ R^d, d=768`（ViT-B 或 ViT-L），为 frame-level global feature
  - 余弦距离：`d_i = 1 - (V_i · V_{i+1}) / (||V_i|| · ||V_{i+1}||)`，标量
  - Prominence 计算：左基线 l_min = min(d_k)，其中 k 从 j-1 向左搜索到 d_k > d_j 为止；右基线同理；prominence = d_j - max(l_min, r_min)
  - Reward 二维评分：prompt 驱动 LMM 输出 {"description": str, "reward": int}
  - 迭代收敛：reward 集不再变化时终止，无需预设 Top-K

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：D-CoDe 是一个 training-free 的视频理解框架，将图像预训练的 VLM（LLaVA-NeXT 7B）扩展到视频领域，由两个组件组成：(1) Dynamic Compression —— 时间维度：先从视频中均匀采样 ⌊α·N⌋ 帧（α=0.85），再从剩余帧中迭代选择与已选帧语义最不相似的 supplementary frame（基于 CLIP global feature 的余弦相似度），共选 N 帧；空间维度：对每帧的 M 个 visual token 按 ℓ2 norm 计算 salience，保留 top-⌊β·M⌋ 高激活 token（β=0.625），然后按余弦相似度（阈值 τ=0.9）贪婪合并冗余 token（anchor + cluster 取平均），输出压缩后的 token 集；(2) Question Decomposition —— 用 GPT-3.5-turbo-0125（temperature t=0.5）将原始问题分解为多个子问题，每个子问题独立用压缩后的 visual tokens 推理得到子答案，将子答案拼接后与原始问题、压缩 visual tokens 一起送入 LLM 生成最终答案。
  实验比较：(a) 与 training-required 方法（Video-LLaVA, Video-LLaMA2, MovieChat+, Vista-LLaMA）和 training-free 方法（DeepStack-L, M3, IG-VLM, SF-LLaVA, TS-LLaVA）在 Multiple Choice VideoQA（NExT-QA, EgoSchema, IntentQA）和 Open-Ended VideoQA（MSVD-QA, MSRVTT-QA, TGIF-QA, ANet-QA）上的 accuracy 对比；(b) 模块消融（EgoSchema）—— Baseline → +dynamic spatial token compression → +dynamic temporal frame selection → +question decomposition 的逐级增益；(c) 采样策略消融 —— Uniform vs Question-aware vs Supplementary Frame Selection；(d) 压缩范围消融 —— 空间 token 合并的距离约束的影响；(e) Decomposition Prompt 消融 —— 不同 prompt 变体的影响；(f) Decomposed Content 消融 —— 子问题 vs 子答案的有效性对比；(g) 超参数消融 —— α（uniform ratio）、β（retention ratio）、τ（similarity threshold）、t（temperature）在 EgoSchema 上的 sensitivity；(h) 错误分析 —— 频繁场景切换视频上的性能退化。

- 硬件平台是什么，配置是什么。
  所有实验在单卡 NVIDIA RTX A6000 GPU 上执行。基础模型 LLaVA-NeXT 7B（Vicuna-7B LLM），使用 RoPE 缩放因子 2 扩展 context length 到 8192 tokens。Question Decomposition 使用 OpenAI API 调用 gpt-3.5-turbo-0125。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaVA-NeXT 7B（基于 Vicuna-7B LLM + CLIP 视觉编码器），所有参数冻结不做微调。
  Benchmarks（7 个）：(1) Multiple Choice VideoQA —— NExT-QA（因果和时间理解）、EgoSchema（自我中心长视频模式级理解）、IntentQA（意图识别），metric 为 Accuracy；(2) Open-Ended VideoQA —— MSVD-QA（短视频文本描述）、MSRVTT-QA（多样化网络视频）、TGIF-QA（GIF 中的重复计数和状态转换）、ActivityNet-QA（长视频丰富活动语义），metric 为 GPT-Accuracy（事实正确性）和 GPT-Score 0-5（完整性和流畅性），统一使用 gpt-3.5-turbo-0125 评估。
  帧采样：每视频采样 N 帧，N 根据数据集平均视频长度经验性确定。所有帧统一 resize 到 336×336。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/hukcc/D-CoDe
  代码结构：`Dcode.py`（核心实现，含 `generate_subquestions()`、`supp_frame_selection()`、`token_select_and_merge()`）、`dataset.py`、`prompt.py`、`utils.py`、`run_inference_multiple_choice_qa.py`、`run_inference_video_qa.py`、`scripts/`（各 benchmark 评估脚本）、`eval/`（评估代码）。
  
  算法 pipeline 伪代码：
  ```
  # === D-CoDe 推理流程 ===
  # 输入: 视频 V (T frames), 问题 Q
  # 参数: N=15 (selected frames), α=0.85, β=0.625, τ=0.9, t=0.5

  # Step 1: 视频帧预编码
  for t in 1..T:
      g_t = CLIP_visual(I_t)         # 全局 CLIP 特征, shape: (d_clip,)
      F_t = VisualEnc(I_t)           # 逐帧 visual tokens, shape: (M, d)

  # Step 2: Dynamic Temporal Frame Selection
  # Stage 1: 均匀采样
  N_uniform = floor(α * N)           # e.g. 0.85 * 15 = 12
  V_uniform = uniform_sample({I_t}, N_uniform)
  V_selected = V_uniform

  # Stage 2: 补充帧选择 (基于语义多样性)
  for k in 1..(N - N_uniform):
      for each I_m in V \ V_selected:
          s_m = mean(cosine_sim(g_m, g_n) for I_n in V_selected)
      I* = argmin(s_m)               # 选与已选帧最不相似的帧
      V_selected = V_selected ∪ {I*}

  # Step 3: Dynamic Spatial Token Compression (每帧独立)
  F_compressed = []
  for each I_k in V_selected:
      F = VisualEnc(I_k)             # shape: (M, d)
      # 3a: Token Pruning (按 ℓ2 norm)
      a_i = ||f_i||_2 for i in 1..M
      K = floor(β * M)
      F_pruned = TopK(F, key=a_i, k=K)  # (K, d)
      
      # 3b: Greedy Token Merging (按余弦相似度)
      π = argsort({a_i}, descending)  # 按 salience 降序排列
      merged_set = []
      for i in π:
          if f_i is not merged:
              N_i = {j | cosine_sim(f_i, f_j) >= τ, f_j is unmerged}
              f_i_rep = (f_i + Σ_{j∈N_i} f_j) / (1 + |N_i|)
              mark all j in N_i as merged
              merged_set.append(f_i_rep)
      F_compressed.append(merged_set)

  # Step 4: 拼接压缩后 visual tokens
  F_final = concat(F_compressed)     # shape: (Σ|merged_set_k|, d)

  # Step 5: Question Decomposition
  prompt = "I am working on a video understanding task. ..."
  Q_1..Q_n = GPT3.5(Q, prompt, temperature=t)  # n 不限制

  # Step 6: 逐子问题推理
  A_sub = []
  for Q_i in Q_1..Q_n:
      A_i = LLaVA_NeXT(F_final, Q_i)
      A_sub.append(A_i)

  # Step 7: 最终答案生成
  A_final = LLaVA_NeXT(F_final, concat(A_sub), Q)
  ```

  关键张量维度：
  - 视频帧: T 帧(取决于视频长度), 采样 N 帧(N 经验性确定, EgoSchema 用 15)
  - 每帧 visual tokens M: LLaVA-NeXT 编码后 ~576 tokens (336×336, patch_size=14 或类似)
  - CLIP global feature d_clip: ~768 或 ~1024 (取决于 CLIP 变体)
  - Visual token hidden dim d: LLaVA-NeXT 7B 的 hidden size (~4096)
  - α=0.85: 85% 帧来自 uniform sampling, 15% 来自 supplementary selection
  - β=0.625: 保留 62.5% 高 ℓ2 norm tokens
  - τ=0.9: 余弦相似度 >= 0.9 的 token 被合并
  - RoPE 缩放因子 2, context length 8192
