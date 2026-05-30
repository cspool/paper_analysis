## Free-MoRef__Instantly_Multiplexing_Context_Perception_Capabilities_of_Video-MLLMs_within_Single_Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：Free-MoRef 是一个 training-free 的推理时方法，通过修改 Video-MLLM 中 LLM 的 self-attention 层来在单次推理中多路复用上下文感知能力。核心设计包含三个组件：(1) **Multi-Reference Partition** —— 将长 vision token 序列按时间关系划分为 M 个 temporal units × N 个 fragments，最终聚合为 N 个 reference chunks，每个 chunk 作为原始长视频的抽象摘要，且分配相同的 system prompt 和 question。(2) **MoRef Attention（Mixture-of-Reference Attention）** —— 替换 LLM 的标准 self-attention。对 N 个 parallel chunks 分别执行 FlashAttention 得到 O = [O^sys, O^vis, O^ques]，其中 O^sys 在各 chunk 间相同（因果注意力的单向性），O^vis 保持各 chunk 的差异性，O^ques 通过 gating 函数加权聚合：O^fusion = (Σ ω_i · O_i^ques).repeat(N)，其中 ω_i = max(A[i]) / Σ max(A[i])，A = softmax(Q^ques × (K^vis)^T) 为 query-vision 跨模态注意力图。最终输出为 O^MoRef = [O^sys, O^vis, O^fusion]。(3) **Reference Fusion** —— 在 decoder 的中间层 L，基于注意力图 A 计算重要性估计矩阵 E ∈ R^{N × l_vis}（沿 l_ques 维度平均），在每个 chunk 内剪枝 1-1/N 的不重要 vision tokens，然后将各 chunk 保留的 token 按时间关系聚合为 global reference，后续 decoder 层只用 global reference 推理。该方法以约 1/N 的计算复杂度实现对全部 vision tokens 的 completely-aware perception，支持 FlashAttention，且可与 streaming inference 或 token compression 结合。

  实验比较：(a) Multiplexed Context Understanding —— LLaVA-Video-7B 在 64/128/256/512 frames 下对比 Free-MoRef vs 直接扩展帧数（full attention），指标为 FLOPs 和 VideoMME/MLVU/LongVideoBench 准确率；(b) SOTA 对比 —— 在 VideoMME、MLVU、LongVideoBench 上对比 InternVL2、InternVL2.5、Qwen2-VL、LLaVA-OneVision、LLaVA-Video、Kangaroo、LongVILA、LongVA、Video-XL、RETAKE（均为 7B~8B 规模）；(c) 消融 —— 各组件贡献（Multi-Reference Partition、MoRef Attention、Reference Fusion）、Chunk 数 N（1/2/4/8）、Temporal Units M（1/4/32/64）、Reference Fusion 层 L（1/3/6）对性能的影响；(d) 任务类型分析 —— VideoMME 的 12 个子任务（TP/SP/AP/ARec/ORec/OCR/CP/TR/SR/AR/OR/IS）在不同 context 长度下的表现。

- 硬件平台是什么，配置是什么。
  单张 NVIDIA A100 GPU（论文实验部分明确说明 "all experiments are executed on a single A100 GPU"）。对超过 context length 限制的场景，使用 accelerate toolkit 辅助管理显存。256 frames baseline 在单卡 A100 上因超过 Qwen2 的 32768 token 限制直接 OOM，Free-MoRef 可在不使用 accelerate 的情况下直接推理 512 frames。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaVA-Video-7B（基于 Qwen2-7B 作为 LLM backbone，vision encoder 为 SigLIP，FPS=1，默认最大 64 frames，每帧 182 tokens 通过 spatial pooling 得到）。
  Benchmarks：(1) VideoMME —— 900 个视频、256 小时总时长、2700 个多选题 QA pair、分 short/medium/long 三个子集、覆盖 30 个子领域；(2) MLVU —— 多任务长视频理解 benchmark、7 种 QA 任务类型、视频长度 3分钟至 2小时+、平均 12 分钟；(3) LongVideoBench —— 强调细粒度推理问题、17 种问题类别、10 种视频类型、分 8-15s/15-60s/3-10min/15-60min 四组。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/wkfdb/Free-MoRef
  
  算法 pipeline 伪代码：

  ```
  # ==========================================
  # Free-MoRef 推理流程 (Training-Free)
  # ==========================================
  # 输入: N_frames 个视频帧, question Q_text
  # 超参数: M (temporal units), N (parallel chunks), L (fusion layer)

  # 步骤 1: Vision Encoding
  frames = load_video(video_path, fps=1, max_frames=N_frames)
  # 每帧: Vision Encoder → 182 tokens → spatial pooling → vision_tokens
  vision_tokens ∈ R^{N_frames × 182 × D}

  # 步骤 2: Multi-Reference Partition
  # 将 vision_tokens 按时间顺序划分为 M 个 units
  # 每个 unit 内按时间分解为 N 个 fragments
  # 聚合各 unit 的相同 index 的 fragment → N 个 reference chunks
  for i in range(N):
      chunk_i = []
      for unit_j in range(M):
          # 取 unit_j 的第 i 个 fragment
          chunk_i.append(units[unit_j].get_fragment(i))
      reference_chunks[i] = concat(chunk_i)  # ∈ R^{(total_tokens/N) × D}
  
  # 为每个 chunk 拼接相同的 system prompt + question
  for i in range(N):
      input_seq[i] = concat([system_prompt, vision_chunk[i], question])

  # 步骤 3: MoRef Attention (替换 LLM 的 self-attention, layers 0..L-1)
  for layer in range(L):  # shallow layers
      for each chunk i in range(N):
          Q_i, K_i, V_i = W_Q(input_seq[i]), W_K(input_seq[i]), W_V(input_seq[i])
          O_i = FlashAttention(Q_i, K_i, V_i)  # 标准 causal attention
          # O_i = [O_i^sys, O_i^vis, O_i^ques]
      
      # 计算 gating weights (跨模态 query-vision 注意力)
      for each chunk i:
          A_i = softmax(Q_i^ques @ K_i^vis.T)  # ∈ R^{l_ques × l_vis}
          w_i = max(A_i) / sum(max(A_j) for j in range(N))
      
      # 聚合 question tokens
      O_fusion = sum(w_i * O_i^ques for i in range(N))
      O_fusion = O_fusion.repeat(N)
      
      # 组装 MoRef 输出
      for each chunk i:
          O_i^MoRef = [O_i^sys, O_i^vis, O_fusion]
      
      # 残差连接 + FFN (标准 transformer)
      for each chunk i:
          input_seq[i] = input_seq[i] + O_i^MoRef
          input_seq[i] = input_seq[i] + FFN(LayerNorm(input_seq[i]))

  # 步骤 4: Reference Fusion (在 layer L)
  # 计算重要性估计矩阵
  for each chunk i:
      E_i = mean(A_i, dim=l_ques)  # ∈ R^{l_vis}
  # 剪枝: 每个 chunk 保留 top 1/N 的 vision tokens
  for each chunk i:
      topk_indices = topk(E_i, k=l_vis // N)
      kept_tokens[i] = vision_chunk[i][topk_indices]
  
  # 按时间关系聚合为 global reference
  global_vision = temporal_merge([kept_tokens[i] for i in range(N)])
  global_seq = concat([system_prompt, global_vision, question])
  
  # 步骤 5: Deep layers default inference (layers L..end)
  for layer in range(L, num_layers):
      Q, K, V = W_Q(global_seq), W_K(global_seq), W_V(global_seq)
      O = FlashAttention(Q, K, V)  # 标准 causal attention
      global_seq = global_seq + O
      global_seq = global_seq + FFN(LayerNorm(global_seq))
  
  # 步骤 6: 从 global_seq 的 question position 解码输出
  answer = decode(global_seq[ques_positions])
  ```

  关键张量形状：
  - vision_tokens: [N_frames, 182, D] → 展平后 [N_frames × 182, D]
  - 经 Multi-Reference Partition 后每个 chunk: [l_vis/N, D] + [l_sys, D] + [l_ques, D]
  - A_i ∈ [l_ques, l_vis/N], E_i ∈ [l_vis/N]
  - 复杂度对比：full attention O((N·l_vis + l_sys + l_ques)²) → MoRef O(N·(l_vis/N + l_sys + l_ques)²) ≈ O(1/N · full)，即约为 1/N
  - 128 frames@Free-MoRef: FLOPs = 110.4% of 64-frame baseline (vs 400% for full attention)
  - 256 frames@Free-MoRef: FLOPs = 163.2% of baseline (vs 1600% for full attention)
  - 512 frames@Free-MoRef: FLOPs = 400% of baseline (vs 6400% for full attention)
