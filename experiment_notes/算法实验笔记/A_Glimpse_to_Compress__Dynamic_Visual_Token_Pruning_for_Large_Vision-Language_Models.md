## A_Glimpse_to_Compress__Dynamic_Visual_Token_Pruning_for_Large_Vision-Language_Models

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：GlimpsePrune 是一个基于数据驱动的动态视觉 token 剪枝框架，受人类视觉认知的"一瞥"机制启发。核心组件：(1) Glimpse Token —— 在 LLM decoder 的 prefill 阶段，将可学习的 glimpse token 嵌入追加到指令 token 之后，利用因果注意力机制与所有 visual token 交互，在中间层 K 提取 glimpse token 对所有 visual token 的 cross-attention 分数；(2) Visual Importance Predictor (VIP) —— 由 M=4 层 Self-Attention Block 组成的轻量选择模块，以 glimpse attention A 和多层 visual features V 为输入，其中 visual features 通过 conditional self-attention（2D RoPE）注入到 query/key 中作为额外的视觉先验条件，输出每个 visual token 的重要性 map P；(3) One-shot Pruning —— 在 LLM decoder 第 K = ⌈2/3 × L⌉ 层进行一次性剪枝，移除不重要的 visual token 及其在所有前序层的 KV cache 条目，glimpse token 随即丢弃，后续 L-K 层 prefill 和全部 decoding 阶段均在缩减后的序列上运行。
  实验比较：(a) 与 SOTA 方法（PDrop[7], VisionZip[4], DivPrune[23], CDPruner[20], VScan[5]）在 free-form VQA（12个数据集）和 short-form VQA（10个 benchmark）上的 accuracy 对比，在三种 retention rate 约束下（≤11.1%, ≤22.2%, ≤33.3%）；(b) 不同架构上的泛化（Qwen2.5-VL-7B, LLaVA-1.5-7B, Qwen2.5-VL-3B, LLaVA-1.5-13B）；(c) 消融实验 —— glimpse token 的训练、语言 loss/定位 loss 的贡献、visual condition 的必要性、pruning layer K 的选择（Qwen2.5-VL-3B）；(d) 效率分析 —— prefill FLOPs、decoding 阶段初始 KV cache 长度、峰值内存、时间开销；(e) RL fine-tuning（GlimpsePrune+）—— 结合 GRPO 的强化学习微调在 free-form VQA 上达到 110% baseline 性能。

- 硬件平台是什么，配置是什么。
  训练：单卡 NVIDIA A100 GPU（训练 GlimpsePrune 约 0.5 小时）。RL fine-tuning：4× NVIDIA A100 GPU。
  推理评估：单卡 NVIDIA A100 GPU，100 个 DocVQA 样本测量 prefill/decoding 性能。

- 模型是什么。数据集和bench分别是什么。
  模型：Qwen2.5-VL-7B（主实验，28层 LLM decoder，支持 4~16384 visual tokens 动态高分辨率输入），Qwen2.5-VL-3B（消融实验，36层 LLM decoder），LLaVA-1.5-7B/13B（固定 576 visual tokens），评估模型用 Qwen2.5-32B-Instruct-GPTQ-Int8 作为 evaluator LLM。
  Free-form VQA 数据集（12个）：Flickr30k, V7W, GQA, OpenImages, VSR, CUB, DocVQA, TextCaps, TextVQA, DUDE, SROIE, InfoVQA —— 全部来自 VisCoT benchmark [36]。
  Short-form VQA 数据集（10个）：VQAv2, GQA, VizWiz, ScienceQA, POPE, MME, MMBench(en/cn), SEEDBench, V* bench —— 使用 LMMs-Eval 框架 [39] 评估。
  训练数据：GQA 数据集随机 20K 样本（训练 glimpse token + VIP，1 epoch）。RL fine-tuning：VisCoT 数据集随机 240K 样本（12个数据集各取 20K，10K 短回答 + 10K 自由回答）。
  评价指标：Free-form VQA 用 Qwen2.5-32B-Instruct-GPTQ-Int8 评分（0-1 分）；Short-form VQA 用 exact-match/rule-based string comparison。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/HVision-NKU/GlimpsePrune
  
  算法 pipeline 伪代码：
  ```
  # === Prefill 阶段前 K 层（含 Glimpse Token） ===
  # 输入: visual tokens V (shape: N_v × D)，text tokens T (shape: N_t × D)
  # 插入 glimpse token g (可学习嵌入矩阵 G ∈ R^{L × D})
  X_0 = concat([V, T, g[0]])  # shape: (N_v + N_t + 1) × D

  for l in 1..K:
      # 标准 causal self-attention + FFN
      X_l = DecoderLayer_l(X_{l-1})
      # 在 glimpse token 位置加上对应层的可学习嵌入
      X_l[-1] += g[l]

  # === 提取 Glimpse Attention（第 K 层） ===
  # 从第 K 层的 attention 计算中提取 glimpse token 对 visual tokens 的 cross-attention
  Q_g = W_Q @ X_K[-1]   # shape: H × D_head (multi-head)
  K_v = W_K @ X_K[:N_v] # shape: N_v × H × D_head
  A = Softmax(Q_g @ K_v^T / sqrt(D_head))  # glimpse attention, shape: N_v × H

  # === VIP: 视觉重要性预测 ===
  # 输入: A (glimpse attention, N_v × H), V_hier = {V_m | m ∈ selected_layers} (M=4层多尺度视觉特征)
  A' = Linear(A)  # projection: (N_v × H) → (N_v × E), E=256
  for m in 1..M:
      V_m' = Linear_v(V_m)  # projection: (N_v × C) → (N_v × F), F=512
      # Conditional Self-Attention with 2D RoPE
      Q_m = W_Q_vip @ A'     # shape: N_v × E
      K_m = W_K_vip @ A'
      # 将 visual feature 拼接到 Q/K 中作为额外条件
      Q_m' = Q_m + Linear_q_cond(V_m')  # visual conditioning
      K_m' = K_m + Linear_k_cond(V_m')
      V_m_val = W_V_vip @ A'
      A' = A' + SelfAttn_2DRoPE(Q_m', K_m', V_m_val)
  P = Sigmoid(Linear_out(A'))  # importance map, shape: N_v, per-token prob ∈ [0,1]

  # === One-shot Pruning（第 K 层） ===
  # 选 Top-N_v' 个最高 P 值的 visual token（N_v' = min(retention_rate × N_v, max_retain)）
  mask = TopK(P, N_v')
  X_K_pruned = X_K[mask]     # 移除不重要 visual token 的 hidden state
  # KV cache 中所有层（1..K）对应的不重要 visual token 条目同步移除
  for l in 1..K:
      KV_cache[l] = KV_cache[l][mask]  # 对应 visual token 位置的 K/V
  # 丢弃 glimpse token
  X_K_pruned = X_K_pruned[:-1]

  # === 剩余 Prefill 层（K+1..L） ===
  for l in K+1..L:
      X_l = DecoderLayer_l(X_{l-1})  # 序列长度 = N_v' + N_t

  # === Decoding 阶段 ===
  # 从 X_L 生成第一个 token，随后 autoregressively 生成，所有 token 在缩减后的 KV cache 上运行
  # 每个 decoding step 的 attention cost: O((N_v' + N_t + n_generated) × D)，而非 O((N_v + N_t + n_generated) × D)
  ```
  
  关键张量维度：
  - Visual tokens N_v: Qwen2.5-VL 支持 4~16384, LLaVA-1.5 固定 576
  - LLM hidden size D: 2048(Qwen2.5-VL-3B) / 3584(Qwen2.5-VL-7B) / 4096(LLaVA-1.5-7B) / 5120(LLaVA-1.5-13B)
  - VIP hidden size E=256, visual condition size F=512
  - M=4 个 self-attention block, 4 个 attention head
  - Pruning layer K: Qwen2.5-VL-7B: K=19, Qwen2.5-VL-3B: K=24, LLaVA-1.5-7B: K=22, LLaVA-1.5-13B: K=27
  - 训练目标: L_total = L_language + L_dice + 0.1 × L_bce，其中 L_dice 和 L_bce 基于 GQA 的 bounding box ground truth
  
  训练流程（GlimpsePrune+ RL fine-tuning）：
  ```
  # Phase 1: 训练 glimpse token + VIP（LVLM 参数冻结）
  # Phase 2: RL fine-tuning 循环
  for iteration in 1..N:
      # Step 1: 用当前 pruner 进行 token 剪枝
      P = VIP(glimpse_attn, visual_features)
      X_pruned = prune(X, P, retention_rate)
      
      # Step 2: GRPO 策略优化（group=4）
      for group in 1..4:
          response = PolicyModel.generate(X_pruned)  # LoRA rank=64
      reward = RewardModel(question, response, standard_answer)
      L_policy = GRPO_loss(responses, rewards)  # group-wise ranking
      L_KL = 0.04 × KL(PolicyModel || ReferenceModel)
      Update(PolicyModel, L_policy + L_KL)
      
      # Step 3: 更新 glimpse token + VIP
      Update(glimpse_token, VIP, L_language + L_loc)
  ```
