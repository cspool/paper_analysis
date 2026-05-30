## RETAKE: Reducing Temporal and Knowledge Redundancy for Long Video Understanding

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：RETAKE 是一个 training-free 的即插即用方法，包含两个核心模块联合减少长视频中的时间冗余和知识冗余：
  (1) **DPSelect (Dist Peak Select)**：基于帧间 token-averaged cosine distance 计算相邻帧差异，通过 max pooling (window=3) 识别距离峰值帧作为 pivot frames，再按 top-k 补充剩余关键帧。该策略模仿人类视觉对运动峰值的感知机制，压缩比为 α_dp。
  (2) **PivotKV**：在 chunked prefilling 过程中，对每个 chunk 的 KV cache 进行压缩。对非 pivot 帧，基于层内 self-attention 权重（对 key 维度求和取 head-mean）计算 token 重要性分数。Pivot frames 的 token 被强制保留（通过将重要性分数加 ∞）。最终按 α_kv 比例选择 top-k token 保留到历史 KV cache 中。
  实验比较：(a) 与现有 VideoLLMs 和长视频理解方法（LongVA, LongVILA, Video-XL 等）的性能对比；(b) DPSelect 与 M2SM, A2Summ, MA-LLM 等 keyframe selection 方法的对比；(c) PivotKV 与 FastV, FitPrune, LOOK-M, SparseVLM, PyramidDrop, VL-Cache 等 token compression 方法的对比；(d) DPSelect 与 PivotKV 的消融实验及 trade-off 分析；(e) 细粒度时间感知能力（Needle QA, Action Order, Key Information Retrieval, Temporal Grounding）分析。

- 硬件平台是什么，配置是什么。
  硬件平台：NVIDIA A100 GPU（论文 Table 3 ablation 中提及直接增加帧数超过 A100 显存限制导致 OOM）。论文未明确说明具体 GPU 显存容量和 CPU 配置。

- 模型是什么。数据集和bench分别是什么。
  模型：QWen2VL-7B 和 LLaVA-Video-7B 作为基础 VideoLLM，RETAKE 以即插即用方式扩展。视觉编码器使用原始模型的 VFM（vision foundational model），输入帧较长边 resize 到 448px（QWen2VL）或较短边 336px（LLaVA-Video）。采样策略：2FPS 密集采样，最大 2048 帧（QWen2VL）或 1024 帧（LLaVA-Video）；通过调整压缩比确保 context length 不超过 32K。
  Benchmarks：VideoMME（900 视频/2700 MCQA，短/中/长三类）、MLVU（3 分钟-2 小时视频，9 任务）、LongVideoBench（3763 视频/6678 MCQA，最长 1 小时）、LVBench（平均 4101 秒/视频，1549 MCQA，6 任务）。所有数据集均为人工标注。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/SCZwangxiao/video-ReTaKe
  完整算法 pipeline（对应论文 Algorithm 1）：
  ```
  Input: 视频帧 F ∈ R^{T×3}（T 帧），prompt embeddings P ∈ R^{L×d}，
         chunk size τ，DPSelect 压缩比 α_dp ∈ (0,1]，PivotKV 压缩比 α_kv ∈ (0,1]
  Output: 生成文本 O

  Step 1 — DPSelect 帧压缩:
    M = VE(F)                    # M ∈ R^{T×N×d}，视觉编码
    M_hat, S = DPSelect(M)       # M_hat ∈ R^{α_dp·T·N×d}，S ∈ {0,1}^{α_dp·T·N}
    H = VTC(P, M_hat)            # 视觉-文本拼接，H ∈ R^{(α_dp·T·N+L)×d}

    DPSelect 内部:
      d_i = (1/N) Σ_j (1 - cos(M[i,:,j], M[i+1,:,j]))    # i ∈ [0,T-2]
      P = {i | d_i 是局部极大值 (max pooling window=3)}     # pivot frames
      K = P ∪ ArgTopK(d[非P], k=α_dp·T - |P|)            # 关键帧时间戳
      M_hat = Flatten(M[K, :, :])
      S = token 是否来自 P 中 pivot frame 的 binary mask

  Step 2 — Chunked Prefilling:
    将 H 划分为 chunks: H_1, H_2, ..., H_{α_dp·T/τ+1}
    KV = []  # 初始化空 KV cache

  Step 3 — 逐 chunk 处理 & PivotKV 压缩:
    for each chunk H_i:
      KV_i = LLM(H_i, KV)       # chunk 的 KV cache
      if H_i 是视频 chunk:
        # PivotKV 压缩
        A = Softmax(Q·K_i^T / √d_h)              # A ∈ R^{h×l_q×l_q}
        a_bar = Σ_{j=1}^{l_q} (1/h) Σ_{k=1}^h A_{k,:,j}   # ∈ R^{l_q}
        s = S[iτN : (i+1)τN]                      # pivot mask for chunk
        a_bar = a_bar + s · ∞                      # 强制保留 pivot tokens
        I = ArgTopK(a_bar, k=α_kv·l_q)
        K_hat_i = K_i[:, I, :], V_hat_i = V_i[:, I, :]
        KV = Concat(KV, K_hat_i, V_hat_i)
      else:  # 文本 chunk 不压缩
        KV = Concat(KV, KV_i)

  Step 4 — Decoding:
    O = LLM(Q, KV)              # 标准自回归解码
  ```

  效率优化：使用额外的 CUDA stream 将第 l 层的 PivotKV 压缩与第 l+1 层的 prefilling 重叠执行（见图 4），将 TTFT 开销从 +28%/62% 降低到 +8%/11%。TPOT 降低约 20%，FLOPs 降低 9-18%。

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：AdaRETAKE 是一个 training-free 的自适应视频冗余减少方法，通过动态分配视频帧（时间维度）和 LLM 层（模型维度）上的压缩比例来实现高效视频 token 压缩。包含三个模块：(1) Temporal-adaptive Allocation —— 将长视频等分为 chunk（chunk 内包含 τ 帧），计算每个 chunk 内相邻帧之间的余弦相似度距离 d_i，根据各 chunk 的平均距离按比例分配每个 chunk 的压缩比 α_i；(2) Layer-adaptive Allocation —— 在每个 chunk prefill 时，利用当前层的 video-prompt attention scores 计算每个视频 token 的重要性分数 a_i^(l)，基于全局 Top-K 阈值确定每层的显著性 token 数量 s_i^(l)，然后按比例分配层间压缩比 α_i^(l) = w_i^(l) × α_i（含最小权重 ε=0.01 的数值稳定化重归一化）；(3) Token Compression —— 在前两个模块确定的压缩比下，对每个 chunk 的 KV cache 进行压缩，仅保留 attention 分数最高的 Top-K 个 visual token，逐层更新 KV cache。理论分析证明该层间分配策略能近似最小化 L1 压缩损失的上界（基于 submodular function 的 greedy (1-1/e) 近似保证）。
  实验比较：(a) 与 SOTA MLLM（GLM-4V-Plus, GPT-4o, Gemini-1.5-Pro, VITA-1.5, mPLUG-Owl3, NVILA, ByteVideoLLM, TPO, VideoLLaMA3, LLaVA-Video, Qwen2-VL, Qwen2.5-VL, LLaVA-OneVision, Oryx-1.5, Aria, InternVL2.5）在 VideoMME (Long/Overall)、MLVU (dev)、LongVideoBench (val)、LVBench (val) 上的 accuracy 对比；(b) 与 token 压缩方法（FastV, FitPrune, LOOK-M, SparseVLM, PyramidDrop, VL-Cache）在 VideoMME Long、MLVU val、LVBench val 上的 accuracy 对比；(c) 消融实验（QWen2VL-7B）—— token compression 有无、scaling up frames、layer-wise allocation 有无、temporal allocation 有无、scaling up context length、scaling up max frames；(d) 细粒度感知能力消融——MLVU（Needle QA, Action Order, Action Count）和 LVBench（Key Information Retrieval, Temporal Grounding）；(e) 泛化性——集成到 LLaVA-Video-7B、QWen2VL-7B、QWen2.5VL-7B、QWen2.5VL-72B 多种 MLLM。

- 硬件平台是什么，配置是什么。
  论文未明确说明 GPU 型号。论文提到以 2 fps 采样视频，7B 模型最多 2048 帧、72B 模型最多 1024 帧，最大 context length C_max 为 16K（主实验）/ 1K（消融实验）。推理评估使用 LMMs-Eval 框架。论文引用 KV cache 占用 GPU 内存最多（Hooper et al., 2024）作为设计动机。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaVA-Video-7B (Zhang et al., 2024e)、QWen2-VL-7B (Wang et al., 2024a)、QWen2.5-VL-7B、QWen2.5-VL-72B。MLLM 视觉编码器 + LLM decoder 架构，LLM 层数 L 在模型间不同。
  数据集与 Benchmark：VideoMME（900 个视频，256 小时，2700 个多项选择题，30 个子领域，含 Short/Medium/Long 三个时长子集）；MLVU（视频时长 3 分钟到 2 小时，9 个评估任务包括 topic reasoning, anomaly recognition, video summarization, plot QA）；LongVideoBench（3763 个视频，最长 1 小时，6678 个多项选择题，17 个类别，聚焦 referring reasoning）；LVBench（平均视频时长 4101 秒，1549 个多项选择题，覆盖实体识别、事件理解、关键信息检索、时间定位和推理任务）。
  评价指标：Accuracy（多项选择题正确率）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/SCZwangxiao/video-FlexReduc.git
  
  算法 pipeline 伪代码：
  ```
  # === 参数定义 ===
  # T: 总帧数, N: 每帧 token 数, τ: 每 chunk 帧数, S: prompt 长度
  # L: LLM 层数, C_max: 最大 context length, d: hidden dimension
  # h: attention heads 数量

  # Step 1: 视频编码
  M = VisualEncoder(video_frames)        # shape: (T, N, d)
  P = WordEmbedding(text_prompt)         # shape: (S, d)
  M_chunks = split(M, chunk_size=τ)     # [M_1, M_2, ..., M_{T/τ}], each (τ, N, d)

  # Step 2: Temporal-adaptive Allocation
  for i in 1..T/τ:
      # 计算 chunk i 内相邻帧的余弦距离
      d_i = zeros(τ-1)
      for t in 1..τ-1:
          sim_sum = 0
          for j in 1..N:
              sim_sum += CosineSim(M_i[t,j], M_i[t+1,j])
          d_i[t] = 1 - sim_sum / N
      d_bar_i = mean(d_i)  # chunk i 的平均帧间距离

  # 按距离比例分配压缩比
  total_budget = (C_max - S) / (T * N)
  for i in 1..T/τ:
      α_i = total_budget * d_bar_i / sum(d_bar_1 ... d_bar_{T/τ})
  # α_i: chunk i 的压缩比（压缩后/压缩前 token 数）

  # Step 3: Layer-adaptive Allocation + Token Compression（逐 chunk 处理）
  for i in 1..T/τ:
      # Chunked Prefilling（等价于标准 prefill 的 autoregressive 行为）
      for l in 1..L:
          # 常规 attention 计算
          Q_i^(l) = X_i^(l) @ W_Q^(l)      # 含 prompt tokens
          K_i^(l) = X_i^(l) @ W_K^(l)
          V_i^(l) = X_i^(l) @ W_V^(l)
          A_i^(l) = Softmax(Q_i^(l) @ K_i^(l)^T / sqrt(d_head))  # (h, S+τN, S+τN)

          # Layer-adaptive: 计算 video-prompt 显著性
          # 提取 video token 对 prompt token 的 attention
          A_video_prompt = A_i^(l)[:, :S, S:]  # (h, S, τN)
          a_i^(l) = sum_{j=1..S} (1/h) * sum_{heads} A_video_prompt[:, j, :]  # (τN,)

      # 全局阈值 K
      a_concat = concat([a_i^(1), a_i^(2), ..., a_i^(L)])  # (τN × L,)
      a_hat = Kth_largest(a_concat, K=α_i × τN × L)

      # 计算每层显著性 token 数
      for l in 1..L:
          s_i^(l) = sum_{j=1..τN} 1(a_i^(l)[j] > a_hat)

      # 计算每层压缩比权重
      for l in 1..L:
          w_i^(l) = s_i^(l) / sum_{k=1..L} s_i^(k)
          # 数值稳定化（ε=0.01）
          w_hat_i^(l) = max(w_i^(l) - ε, 0) / sum_k(max(w_i^(k) - ε, 0)) × (1 - Lε) + ε

      # 最终层压缩比
      for l in 1..L:
          α_i^(l) = w_hat_i^(l) × α_i

      # Token Compression（每层独立）
      for l in 1..L:
          K_keep = α_i^(l) × τN
          I = ArgTopK(a_i^(l), K_keep)   # Top-K 重要性 token 索引
          K^(l) = concat([K^(l), K_i^(l)[:, I]])  # 仅保留重要 token 的 KV cache
          V^(l) = concat([V^(l), V_i^(l)[:, I]])

      # 最后一个 chunk 保留 prompt tokens，其余丢弃
      if i < T/τ:
          drop prompt KV cache entries

  # Step 4: Decoding
  # 在所有 chunk 处理完后，使用压缩后的 KV cache 进行 autoregressive 生成
  answer = LLM.generate(X_L, KV_cache_compressed)
  ```

  关键张量维度：
  - 输入帧: T ∈ [128, 2048]（7B: max 2048, 72B: max 1024），fps=2
  - 每帧 token 数 N: 论文未明确说明每帧 token 数，取决于 MLLM 的 visual encoder/projector
  - Chunk 大小 τ: 10 秒（帧数 = τ × fps）
  - Context length C_max: 16K（主实验）/ 1K（消融）
  - LLM hidden dim d: ~3584 (7B) / ~8192 (72B)，取决于具体 MLLM
  - KV cache 内存: 对长序列推理 KV cache 占 GPU 内存最多（如 16K context 下 ~GB 级别）
  - 压缩比 α_i 全局满足 Σ α_i = (C_max - S) / (TN)，将 T×N 个 video token 压缩到 C_max - S 长度
  - ε=0.01: 最小层压缩权重，防止某层完全不保留 token 导致数值不稳定

  理论保证（Theorem 4.1 / A.4-A.5）：
  - Token 压缩损失的 L1 上界: ε^L = 2C^(L) - 2C^(L) ∏_{l=1}^{L} Σ_i I_i^(l) A_i^(l)
  - 通过选择全局 Top-K attention score 的 token 保留，可达到 (1-1/e) 近似的 submodular 最优
  - 贪婪算法（逐 token 选最大边际增益）在 cardinality constraint K 下达到最优
