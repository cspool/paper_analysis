## DyCoke__Dynamic_Compression_of_Tokens_for_Fast_Video_Large_Language_Models

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：DyCoke 是一个 training-free 的两阶段视觉 token 动态压缩方法，专为 Video LLM 推理加速设计。核心流程：(1) Visual Token Temporal Merging (TTM) —— 在 prefilling 阶段，以滑动窗口（window length=4）对视频帧进行连续均匀采样，分为 O(Odd) 组和 E(Even) 组，计算相邻组对应位置 token 的余弦相似度 S = h_i·h_j / (||h_i|| ||h_j||)，剪枝 E 组中与 O 组高相似的 token，再计算 O 组内帧间相似度，保留窗口第一帧的完整 token，剪枝其余，使视觉 token 减少 k%（第一阶段约压缩 50-60%）；(2) KV Cache Dynamic Pruning —— 在 decoding 阶段，在第 L 层计算预测 token 与视觉 token 的 cross-attention 权重 A^(L) = Softmax(Q^(L) K^(L)^T / sqrt(D))，提取视觉 token 注意力分数子集 A_v^(L)，按 top p% 阈值 τ 保留高注意力 token 的 KV cache，将低注意力 token 存入 DP cache（Dynamic Pruning cache）；每隔 N 次迭代（当相邻迭代注意力分布余弦相似度低时），重新计算 cross-attention 矩阵，将 DP cache 中注意力回升的 token 动态加回 KV cache，同时将注意力下降的 token 移除至 DP cache。此阶段在 TTM 基础上进一步压缩 70-90% visual token。平均每帧保留约 15 tokens 参与注意力计算。该方法为 plug-and-play，三个超参数：K（第一阶剪枝率）、L（注意力评估层）、P（第二阶剪枝率），TTM 阶段处理时间 < 10^{-3} 秒（32 帧输入）。
  实验比较：(a) 在 ActivityNet-QA、NextQA、PerceptionTest、VideoDetailCaption、VideoMME、MVBench 六个 benchmark 上，与 FastV（基于 prefilling 阶段注意力分数的 one-shot 剪枝）和 LLaVA-PruMerge（基于 CLIP 视觉编码器注意力分数的 one-shot 剪枝）对比，覆盖 LLaVA-OV-0.5B/7B/72B 三种模型规模，使用 FLOPs 统一公平比较剪枝力度；(b) 效率分析 —— MVBench 上 16/32 帧的实际端到端推理 latency 和 GPU memory 对比，VideoDC 上 per-token decoding latency 对比；(c) Cost-Effectiveness —— 相同计算预算下增加输入帧数的性能收益（VideoMME: 16→32 frames）；(d) 消融实验 —— w/o DP cache（动态剪枝替换为 one-shot 剪枝）、Random Pruning（TTM 随机选择 token 替换相似度选择）、不同 L/P 超参数组合、TTM 过剪枝（K=0.9）的影响；(e) 附录 MVBench 32 帧输入下各子任务细粒度 accuracy；(f) 附录 K 值与输入帧数（8/16/32）的 joint ablation。

- 硬件平台是什么，配置是什么。
  LLaVA-OV-0.5B: NVIDIA RTX 4090 (24GB)；LLaVA-OV-7B: NVIDIA A6000 (48GB)；LLaVA-OV-72B: NVIDIA A100 (80GB)。框架：PyTorch。评估框架：LMMs-Eval（ActivityNet-QA, NextQA, PerceptionTest, VideoDetailCaption, VideoMME），MVBench 使用官方代码评估。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaVA-OneVision-0.5B (d=896, m=4864, T=24), LLaVA-OneVision-7B (d=3584, m=18944, T=28), LLaVA-OneVision-72B (d=8192, m=29568, T=80)。每帧 token 数 N_v=196。默认 32 帧输入（MVBench 额外测 16 帧）。所有模型参数冻结，不做训练。
  Benchmarks：(1) ActivityNet-QA —— 人工标注的动作相关问答，用 GPT-4o-mini 评分 Accuracy 和 Score (0-5)；(2) PerceptionTest —— 感知能力诊断；(3) VideoMME —— 涵盖短/中/长视频多领域，含 w/o subs 和 w-subs 两个子集；(4) NextQA —— 时序动作解释问答；(5) VideoDetailCaption —— 视频细节描述，GPT-4o-mini 评分；(6) MVBench —— 20 个视频理解子任务（Action Sequence, Action Prediction, Action Antonymy, Fine-grained Action, Unexpected Action, Object Existence, Object Interaction, Object Shuffle, Movement Direction, Action Localization, Scene Transition, Action Counting, Movement Counting, Movement Attributes, State Change, Fine-grained Pose, Character Order, Egocentric Navigation, Episodic Reasoning, Counterfactual Inference），每个子任务 200 样本，多选题格式。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/KD-TAO/DyCoke （Apache-2.0, CVPR 2025 accepted）。基于 LLaVA-NeXT 代码库，安装后通过 lmms-eval 调用，传入 dycoke=True, dycoke_k, dycoke_l, dycoke_p 参数启用。
  
  算法 pipeline 张量计算伪代码：
  ```
  # === DyCoke 两阶段压缩流程 ===
  # 输入: 视频 M_v 帧, 文本 prompt tokens
  # 超参数: K (TTM保留率), L (注意力评估层), P (DP保留率p%)

  # ---- Prefilling Stage ----
  # 视觉编码器: 每帧 → N_v 个 embedding tokens
  # H_v' shape: (M_v * N_v, D)  where D is hidden dim
  
  # Stage 1: Token Temporal Merging (TTM)
  # window_size = 4
  # 分组: O = frames[0::2] (odd indices), E = frames[1::2] (even indices)
  for i in range(0, M_v, window_size):
      window_tokens = H_v'[i * N_v : (i + window_size) * N_v]
      # 计算相邻组对应位置 token 余弦相似度
      S = cosine_similarity(O_tokens, E_tokens)  # (N_v,)
      # 剪枝 E 组中高相似 token
      mask_E = S < threshold  # top-K% by similarity pruned
      E_pruned = E_tokens[mask_E]
      # 组 O 内帧间相似度计算，保留第一帧完整
      O_first = O_tokens[0 * N_v : 1 * N_v]  # keep all
      for o_idx in range(1, len(O_tokens)//N_v):
          S_o = cosine_similarity(O_first, O_tokens[o_idx * N_v:...])
          mask_o = S_o < threshold_o
          O_pruned.append(O_tokens[o_idx][mask_o])
      merged = concat[O_first, O_pruned, E_pruned]

  H = concat[merged_tokens, text_tokens]  # LLM 输入

  # Standard prefilling: 计算 Q, K, V 并填充 KV cache
  # for each transformer layer l:
  #   Q^l = H W_Q^l, K^l = H W_K^l, V^l = H W_V^l
  #   KV_cache[l] = (K^l, V^l)

  # ---- Decoding Stage ----
  # Stage 2: KV Cache Dynamic Pruning
  # DP_cache = {}   (pruned tokens backup)
  
  for decoding_step t in range(1, max_new_tokens):
      # 每隔 N 次迭代 (当相邻迭代注意力分布余弦相似度低时)
      if t % N == 0 or similarity_low:
          # 在第 L 层计算 cross-attention
          A^(L) = Softmax(Q^(L)_pred K^(L)_visual^T / sqrt(D))
          # Q^(L)_pred shape: (1, D), K^(L)_visual shape: (N_v_remaining, D)
          # A^(L)_v shape: (N_v_remaining,)  — 预测 token 对各视觉 token 的 attention
          
          # 按 top P% 保留
          threshold = percentile(A^(L)_v, 100 - P)
          keep_idx = where(A^(L)_v >= threshold)
          prune_idx = where(A^(L)_v < threshold)
          
          # 更新 KV cache: 保留高注意力 token, 低注意力 → DP cache
          KV_cache_visual[L] = KV_cache_visual[L][keep_idx]
          DP_cache[L] = DP_cache[L] ∪ KV_cache_full[L][prune_idx]
          
          # 将 DP cache 中注意力回升的 token 加回
          A_dp = recompute_attention(Q^(L)_pred, DP_cache[L])
          recall_idx = where(A_dp >= threshold_new)
          KV_cache_visual[L] = KV_cache_visual[L] ∪ DP_cache[L][recall_idx]
          DP_cache[L] = DP_cache[L] - DP_cache[L][recall_idx]

      # 用压缩后的 KV cache 计算 attention 并自回归生成
      h_t = LLM_decode(KV_cache)  # 仅需计算当前 token 的 K/V
      KV_cache = concat[KV_cache, (h_t W_K, h_t W_V)]
  ```

  使用例子（基于开源仓库）：
  ```bash
  # 评估 DyCoke on MVBench (LLaVA-OV-7B)
  accelerate launch --num_processes=1 \
    -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained="lmms-lab/llava-onevision-qwen2-7b-ov",conv_template=qwen_1_5,model_name=llava_qwen,device_map=auto,dycoke=True,dycoke_k=0.5,dycoke_l=3,dycoke_p=0.7 \
    --tasks mvbench \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs
  ```
