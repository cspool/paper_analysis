## TriAttention: Efficient Long Reasoning with Trigonometric KV Compression

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 TriAttention —— 一种基于 pre-RoPE Q/K 浓度的 KV 缓存压缩方法。核心实现：
  (1) 发现 Q/K 浓度现象（Q/K concentration）：pre-RoPE 空间中，大量 attention head 的 Q 和 K 向量围绕固定的非零中心高度聚集（Mean Resultant Length R → 1），且跨位置、跨上下文稳定。
  (2) 三角函数级数（Trigonometric Series）：当 Q/K 聚集时，attention logit 退化为仅依赖 Q-K 距离 Δ 的三角函数级数：logit(Δ) ≈ Σ_f ‖q̄_f‖ · ‖k̄_f‖ · cos(ω_f·Δ + φ̄_f)，其中频段 f 的系数由 Q/K 中心决定。这个距离偏好曲线可仅通过中心预测。
  (3) 评分函数 S(k, Δ) = S_trig + S_norm，其中 S_trig 利用三角函数级数和 Q 中心 E[q_f] 对 key 按距离打分（捕获距离偏好），S_norm = Σ_f (1-R_f) · E[‖q_f‖] · ‖k_f‖ 补充范数信号（捕获低范数 key）。自适应加权：R_f 高时（浓度强）S_trig 主导，R_f 低时 S_norm 贡献更大。
  (4) 推理时每 128 tokens 触发一次剪枝（window-based pruning），使用多 future offset（几何间隔 D = {1,2,4,...,2^16}）平均化评分，GQA 场景下 z-score normalize 后 max 聚合。
  
  实验比较：(1) 与 Full Attention（无剪枝上限）、SnapKV（基于历史 attention score 选择 token）、R-KV（attention + redundancy detection）对比，在 AIME24、AIME25、MATH 500 上评估数学推理准确率；
  (2) KV budget sweep (512-4096) 与 R-KV 对比准确率-效率 trade-off；
  (3) 记忆保留 benchmark：基于 DFS 的 Recursive State Query，评估剪枝对中间状态保留的影响；
  (4) 消融实验：去掉 S_trig、去掉 R 加权、跨域校准（coding vs reasoning）、future offset 范围和间距策略、校准数据量和质量敏感性；
  (5) 吞吐量对比：在 A100 80GB 上测量 16K 解码长度的 tokens/s，与 Full Attention 和 R-KV 比较；
  (6) 额外 baseline：LazyEviction、H2O、TOVA、RaaS、StreamingLLM、PyramidKV、KnormPress、Ada-KV+SnapKV；
  (7) 通用 benchmark：LongBench（16 个子任务）、RULER（检索任务）；
  (8) 实际部署：RTX 4090 上运行 OpenClaw（Qwen3-32B INT4），Full Attention OOM 而 TriAttention 成功完成多轮 agent 任务；
  (9) MLA 架构验证：GLM-4.7-Flash (MLA) 上验证 Q/K 浓度具有架构泛化性，96.6% heads 的 R > 0.95。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 80GB GPU（bfloat16 + FlashAttention-2，默认设置）；GPT-OSS-20B 使用 NVIDIA H100 GPU（FlashAttention-3，需 Hopper 架构）；OpenClaw 部署使用单张 RTX 4090 24GB（Qwen3-32B AWQ INT4 量化）。

- 模型是什么。数据集和bench分别是什么。
  模型：Qwen3-8B（32 heads × 36 layers = 1152 heads, GQA）、DeepSeek-R1-Distill-Llama-8B、DeepSeek-R1-Distill-Qwen-7B、GPT-OSS-20B，共四个 reasoning-capable LLM，覆盖 GQA 多种架构。额外验证：GLM-4.7-Flash（MLA architecture, 940 heads）。
  数据集/Benchmark：(1) AIME 2024（30 道竞赛数学题）; (2) AIME 2025（30 道）; (3) MATH 500（500 道，多样数学推理）; (4) LongBench（16 个子任务：QA, summarization, few-shot, retrieval, counting, code）; (5) RULER（检索任务，4K context）; (6) Recursive State Query（自建 DFS 记忆保留 benchmark）。
  生成设置：max length 32K tokens, temperature 0.6, top-p 0.95。AIME 每题采样 8 次取平均，MATH 500 每题采样 1 次。
  校准数据：使用 LiveCodeBench（coding）或 ShareGPT（chat），校准数据量 50K-960K tokens 均稳定。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源仓库：https://github.com/WeianMao/triattention（Apache-2.0 license, 761 stars），包含完整 TriAttention 实现、vLLM plugin、SGLang 集成、MLX 支持、预计算频率统计。

  算法 Pipeline 伪代码级解释：
  ```
  # 1. 离线校准阶段 —— 计算 Q/K 中心和聚集度量
  for each head h, frequency_band f:
      E_q[h][f] = mean(calibration_query_vectors[h, f])     # Q 中心（复数）
      E_k[h][f] = mean(calibration_key_vectors[h, f])       # K 中心（复数）
      E_norm_q[h][f] = mean(norm(calibration_query_vectors[h, f]))  # 期望 Q 范数
      R_f[h][f] = norm(E_q[h][f]) / E_norm_q[h][f]         # Mean Resultant Length

  # 2. 推理时 —— 每 128 tokens 触发一次 KV 剪枝
  def score_key(k, p_k, current_pos_p):
      delta = current_pos_p - p_k                             # Q-K 距离
      S_trig = 0
      S_norm = 0
      for f in frequency_bands:
          phi_f = angle(E_q[f]) - angle(k[f])                 # k[f] 是 cache 中的 key（复数表示）
          S_trig += norm(E_q[f]) * norm(k[f]) * cos(omega_f * delta + phi_f)
          S_norm += (1 - R_f[f]) * E_norm_q[f] * norm(k[f])
      return S_trig + S_norm

  # 3. 多 future offset 平均化
  def final_score(k):
      D = {1, 2, 4, 8, ..., 2^16}                            # 几何间隔
      scores = [score_key(k, p_k, p_k + delta) for delta in D]
      return mean(scores)

  # 4. GQA 聚合：每个 KV head 分享给 G 个 query head
  def gqa_aggregate(k):
      scores = [final_score_for_head_g(k, g) for g in range(G)]
      z_scores = [(s - mean(scores)) / std(scores) for s in scores]  # per-head z-score normalize
      return max(z_scores)                                                # max 聚合

  # 5. 剪枝：保留 top-B keys，其余 evict
  scores = [gqa_aggregate(k) for k in cached_keys]
  retained = topk(scores, B)
  evict_others()
  ```

  为什么 S_trig 能捕获距离偏好：当 Q/K 高度聚集（R → 1）时，可近似为 q ≈ E[q], k ≈ E[k]。代入 RoPE attention 的复形式后，logit = Re(q · k̄ · e^{iωΔ})，展开为三角函数级数。该级数的峰值位置由 Q/K 中心的相位差 ϕ̄_f 决定，对应实际注意力最强的 Q-K 距离。
  
  与 post-RoPE 方法的根本区别：post-RoPE 方法（如 SnapKV, R-KV）使用最近几个 query 的 attention scores 估计 key 重要性，但 query 经 RoPE 旋转后仅最近几个 query 的朝向是"当前"的——观察窗口极小（约 25 个 query 最优）。TriAttention 回到 pre-RoPE 空间，利用 Q/K 浓度（跨位置稳定）来预测注意力模式，不受位置旋转限制。
