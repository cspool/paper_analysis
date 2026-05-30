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

## Why Attend to Everything? Focus is the Key (Composing Sparse Attention via Learned Grouping)

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 Focus —— 一种可组合（composable）的稀疏注意力方法。核心实现：在每个注意力层添加少量可学习的 centroid 向量（C ∈ R^{K×d_g}，dg=16，仅 148K 参数），通过投影 W_g ∈ R^{d×d_g} 将 token 映射到 centroid 空间，使用 Sinkhorn 归一化（N=10 次迭代）强制均衡分组，阻止单个 group 吸收所有 token（group dominance）。注意力计算为：s_ij = q_i^T k_j · (1_local(i,j) + (1 - 1_local(i,j)) · σ(λ · g_i^T g_j))，即局部窗口内的 token 总有完整注意力，远距离 token 仅在同组内（g_i^T g_j ≈ 1）才参与注意力。所有原始权重冻结，仅训练 centroid 参数。

  实验比较：(1) 与 efficient attention retrofitting 方法对比（Table 1）：Longformer、Performer、Routing Transformer、Full attention FT，在 GPT-2 124M + PG-19 上，Focus 是唯一同时改善 PPL 且不降 benchmark 的方法；(2) 扩展到 124M→70B 七种模型五种 attention 架构（Table 2）：GPT-2 124M/774M、Mistral 7B、Qwen2.5 7B、OLMo-27B、LLaMA-2 13B/70B，验证零 benchmark 退化；(3) 与 LoRA 对比（Table 3-4）：LoRA 在任意学习率下均退化 benchmark，Focus 零退化（Table 3: GPT-2 124M），在 Mistral-7B-Instruct 上 Focus 保持对齐（TruthfulQA +0.3），LoRA 跨全部 benchmark 退化（Table 4）；(4) Full fine-tuning（Table 5）：GPT-2 124M/774M/1.5B 上 Focus FT vs Full attention FT，124M 上 Focus 超越 full attention（30.3 vs 31.4 PPL），774M/1.5B 匹配（差异 0.3-0.4 PPL）；(5) 多域验证：WikiText-103、OpenWebText 上 Focus FT 均匹配或超越 full attention；(6) 从零训练 7B：Mistral 7B + 2B token OpenWebText，Focus 匹配 full attention（13.82 vs 13.89 PPL）；(7) 长上下文（3.5）：Mistral 7B 上 centroids 从 T=1024 零额外训练 transfer 到 T=8192，PPL 差距稳定（+0.26-0.47）；(8) 速度-质量 tradeoff（Table 7）：sweep top-k (1/2/3/4)，top-k=2 在 GPT-2 124M 上 2× 加速 + 质量超越 pretrained（41.3 vs 42.8 PPL），Mistral 7B 上 +0.7 PPL；(9) SparQ 和 MagicPIG token-selection 方法对比（Table 10）：Focus 提升 PPL 6.6 点，token-selection 方法退化 5-10 点；(10) Sinkhorn 稳定性消融（Table 8）：对比 entropy+balance loss / stop-gradient / EMA / reclustering / balance weight×5 / Sinkhorn，仅 Sinkhorn 同时阻止三条 escape pathway；(11) 超参消融（Table 9）：K/w/τ/Sinkhorn iters 四个维度 16 种配置，fine-tuned PPL 波动仅 0.6。

- 硬件平台是什么，配置是什么。
  NVIDIA H100-80GB（用于 wall-clock speedup 测量 Table 6）；训练硬件论文未明确说明具体 GPU 型号和数量。

- 模型是什么。数据集和bench分别是什么。
  模型：GPT-2 124M/774M/1.5B (MHA)、Mistral 7B (GQA)、Qwen2.5 7B (GQA+bias)、OLMo-27B (MHA+QK-norm)、LLaMA-2 13B/70B (GQA)、Mistral-7B-Instruct (interleaved+softcap)，共计七种模型、五种 attention 架构。
  数据集：PG-19（主要语言建模）、WikiText-103（维基百科）、OpenWebText（网页文本）、GSM8K（数学推理，8-shot CoT）。
  Benchmark：HellaSwag、ARC-Easy、PIQA、LAMBADA、TruthfulQA MC1。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文未明确提供开源链接（arXiv 2604.03260 无代码仓库链接）。

  算法 Pipeline 伪代码级解释：
  ```
  # 初始化（仅添加，不影响原模型）
  C = randn(K, d_g)          # K 个 centroid，d_g=16
  W_g = randn(d, d_g)        # 投影矩阵

  # 每个 attention 层的前向传播
  def focus_attention(h, Q, K, V, w):
      # h: [T, d] hidden states
      # 1. 分组分配 (Group Assignment)
      S = W_g @ h.T           # [d_g, T]
      scores = C @ S          # [K, T]，centroid-token 亲和度
      g = sinkhorn(scores/τ, N=10)  # [K, T]，Sinkhorn 双随机归一化

      # 2. 标准 QKV projection（原模型权重，冻结）
      q, k, v = Q(h), K(h), V(h)   # [T, d_head]

      # 3. 门控注意力分数
      for i in 0..T:
          for j in 0..i:
              local_mask = (i - j <= w)
              group_affinity = g[:,i]^T @ g[:,j]  # 同组≈1，异组≈0
              gate = local_mask or σ(λ * group_affinity)
              s_ij = q_i^T k_j * gate

      # 4. Softmax + V 加权（同组+local 标准 softmax）
      attn = softmax(s, dim=-1)  # 仅有效 pair 非零
      output = attn @ V
  ```

  推理时 hard assignment：每个 token 取 top-k 个 group，仅同组 token 间计算注意力。注意力 mask: M(i,j) = 1[j≤i] ∧ (1[g(i)=g(j)] ∨ 1[i-j≤w])，分解为两个不相交的 FlashAttention 调用：A = {(i,j): g(i)=g(j), causal}（按 group sort 后 reshape 为 K 个序列） + B = {(i,j): i-j≤w, g(i)≠g(j)}（cross-group local），通过 logsumexp merge 数学精确合并。复杂度：O(n²/K) + O(nw)。

## TransPrune: Token Transition Pruning for Efficient Large Vision-Language Model

- 属于算法pipeline的实现是什么？实验比较什么？
  TransPrune 是一种训练无关（training-free）的 LVLM 视觉 token 剪枝方法，属于 within-LLM pruning。核心实现包括两个互补的 token 重要性评估准则：
  (1) **Token Transition Variation (TTV)**：测量每个 token 在 self-attention 和 FFN 模块中表征的变化。定义幅度变化 m(F, T_in) = ||T_out||₂ / ||T_in||₂，方向变化 d(F, T_in) = cosine_similarity(T_out, T_in)。TTV = Softmax(1 - |d|) · m，对 self-attention 和 FFN 的 TTV 求和得到每层总 TTV。
  (2) **Instruction-Guided Attention (IGA)**：计算 instruction token 对 image token 的 attention 权重平均值，引入任务相关的语义监督。
  (3) **Accumulation Mechanism**：由于 TTV 模式在各层间不稳定，对中间层（7-12）的 TTV 进行跨层累积，在每个 pruning layer 做出更精确的剪枝决策。最终 Score = α·TTV + (1-α)·IGA，α=0.5，pruning 在 layers 7, 9, 12 执行。

  实验比较：(1) 与 within-LLM 方法（FastV、TopV、PDrop、ShortV、SparseVLM）在 LLaVA-v1.5-7B 上的 8 个 benchmark 对比（MME^P、VQA^v2、Seed^I、TextVQA、SQA^I、POPE、GQA、MMB^en），分为 ~40-50% TFLOPs（TransPrune-High: 1.56 TFLOPs / 40.8%）和 ~25-35% TFLOPs（TransPrune-Low: 1.19 TFLOPs / 31.2%）两档（Table 1）；(2) LLaVA-NeXT-7B 上的同类对比，TransPrune-High 达 8.33 TFLOPs (40.0%)（Table 2）；(3) Qwen2.5-VL-7B 上与 FastV 的对比（Table 3）；(4) 与 projector-based 方法（VisionZip、CDPruner）的联合剪枝效果（Table 4-5）；(5) Video-LLaVA 上视频 benchmark（TGIF、MSVD）的泛化实验（Table 6）；(6) 消融实验：TTV-only 有效性（Table 7）、不同 layer 选择（Table 9）、浅层 vs 深层 accumulation（Table 10）、accumulation 机制有无（Table 11）、magnitude vs direction 贡献（Table 12）、α 参数影响（Table 13）；(7) 延迟（ms）和显存（GB）的实际测量对比（Table 8）。

- 硬件平台是什么，配置是什么。
  所有实验在 **A100 GPU (40GB)** 上进行。推理时使用 **FlashAttention** 进行高效 attention 计算。TransPrune 的 TTV 计算仅需模块输入/输出，IGA 仅计算 instruction→image token 的 attention（非完整 attention map），因此与 FlashAttention 兼容。

- 模型是什么。数据集和bench分别是什么。
  模型：**LLaVA-v1.5-7B**、**LLaVA-NeXT-7B**、**Qwen2.5-VL-7B**（不同架构验证泛化性）。视频模型：**Video-LLaVA**。
  数据集/Benchmark：**MME、MMBench(MMB^en)、SEED(Seed^I)、ScienceQA(SQA^I)、VQA-v2、POPE、GQA、TextVQA**（共 8 个），覆盖 perception、reasoning、VQA 任务。视频 benchmark：**TGIF、MSVD**。

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  论文声明代码将在接收后开源于 https://github.com/liaolea/TransPrune（截至分析时尚未完全公开）。
  
  算法 Pipeline 伪代码：

  ```
  # 输入: 视觉 tokens T_I (shape: [N, d]), instruction tokens T_inst (shape: [L, d])
  # 超参: accumulation_layers A = {7,8,9,10,11,12}, pruning_layers P = {7,9,12}, α=0.5

  retained_indices = range(N)  # 初始保留所有 token

  for each transformer layer l = 1 to max_layer:
      # 前向传播（FlashAttention）
      T_out = TransformerLayer_l(T_in)
      
      if l in A:  # accumulation layer
          # 计算 TTV
          T_attn_out = SelfAttention(T_in[retained_indices])
          T_ffn_out = FFN(T_attn_out)
          
          d_attn = cosine_similarity(T_attn_out, T_in[retained_indices])  # [N_retained]
          m_attn = ||T_attn_out||_2 / ||T_in[retained_indices]||_2       # [N_retained]
          TTV_attn = Softmax(1 - |d_attn|) * m_attn
          
          d_ffn = cosine_similarity(T_ffn_out, T_attn_out)
          m_ffn = ||T_ffn_out||_2 / ||T_attn_out||_2
          TTV_ffn = Softmax(1 - |d_ffn|) * m_ffn
          
          TTV[l] = TTV_attn + TTV_ffn  # 存储当前层 TTV
      
      if l in P:  # pruning layer (e.g., 7, 9, 12)
          # 累积 TTV
          TTV_acc = sum(TTV[j] for j in A where j <= l)   # Equation (4)
          
          # 计算 IGA（用下一层 l+1 的 attention）
          A_inst2img = softmax(Q_inst[l+1] @ K_img[l+1].T / sqrt(d))
          IGA = mean(A_inst2img, dim=instruction)          # Equation (5), [N_retained]
          
          # 组合得分
          Score = α * TTV_acc + (1-α) * IGA                # Equation (6)
          
          # 剪枝：保留得分最高的 K 个 token
          keep_count = schedule[l]  # TransPrune-High/Low 预设保留数
          retained_indices = topk(Score, keep_count)
  ```

  TTV 关键张量计算（Equation 1-3）：
  - m(F, T_in) = ||T_out||₂ / ||T_in||₂ （幅度变化率）
  - d(F, T_in) = (T_out · T_in) / (||T_out||₂ · ||T_in||₂) （方向余弦相似度）
  - TTV(F, T_I) = Softmax(1 - |d(F, T_I)|) · m(F, T_I) （Equation 2）
  - TTV_l(T_I) = TTV(Attention, T_I) + TTV(FFN, T_I) （Equation 3，每层汇总）

  额外 FLOPs 开销（Equation 7）：主要由 TTV 的 L2 norm 和 cosine similarity 计算、IGA 的 instruction-visual attention 组成，开销与 stage 数 s 和 token 维度 d 线性相关（O(sd)），对比 baseline 总计算量可忽略（Table 8 显示 TransPrune 延迟最低 111.4ms，显存 14.82GB）。

## TransMLA: Multi-Head Latent Attention Is All You Need

- 属于算法pipeline的实现是什么？实验比较什么？
  TransMLA 是一种将 GQA（Group-Query Attention）预训练模型（LLaMA、Qwen、Gemma、Mistral/Mixtral）无缝转换为 MLA（Multi-Head Latent Attention）模型的框架。核心实现包括三个技术：(1) **RoRoPE**——对合并后的多 head key 向量按 RoPE 频率维度分组，在每个 2D 子空间内用正交矩阵 U_l 旋转 Q 和 K，通过 PCA 将位置信息集中到第一个 attention head 中（等价变换，不改变 attention 输出）；(2) **FreqFold**——利用相邻 RoPE 频率的相似性，将多个频率维度的 key 段拼接后做联合 PCA，使 K_rope 能占用更多维度以保留更丰富的位置信息；(3) **BKV-PCA**——先计算 α = E[||K_NoPE||₂]/E[||V||₂]，将 K_NoPE 缩放 1/α 使 norm 与 V 对齐后，对 [K_NoPE; V] 联合做 PCA 低秩分解，避免 key 主导主成分方向导致 value 信息丢失。转换后的模型兼容 DeepSeek 代码库，可直接使用 DeepSeek 的 Absorb 操作进行推理加速。

  实验比较：(1) SmolLM-1.7B 和 LLaMA-2-7B 在三种 KV cache 压缩率（-68.75%、-81.25%、-87.50%，LLaMA 额外 -92.97%）下与 MHA2MLA 方法在 6 个 benchmark（MMLU、ARC、PIQA、HellaSwag、OpenBookQA、Winogrande）上的准确率对比（Table 1）；(2) 不同 fine-tuning token 量（0 / 300M-1B / 500M-6B）下的性能恢复曲线；(3) LLaMA-3-8B 上 RoRoPE + FreqFold 的 key norm 分布可视化和 RoPE 去除比例 vs log-perplexity 消融实验（Figure 3）；(4) KV balancing 前后的 key/value norm 对比和 weight-based vs activation-based PCA 消融（Figure 4）；(5) 三款消费级 GPU（165.2 TFLOPS 24GB、312 TFLOPS 40GB、320 TFLOPS 64GB）上 vLLM 推理吞吐量对比（Table 4），输入/输出等长设置，1K-32K 总 context length。

- 硬件平台是什么，配置是什么。
  **训练**：8-GPU 机器，每 GPU 40GB 显存，312 TFLOPS FP16 算力。**推理 benchmark**：三款消费级 AI 加速器——165.2 TFLOPS / 24GB、312 TFLOPS / 40GB、320 TFLOPS / 64GB。使用 vLLM 推理框架。

- 模型是什么。数据集和bench分别是什么。
  **模型**：SmolLM-1.7B（1T tokens 预训练）、LLaMA-2-7B（2T tokens 预训练）、LLaMA-3-8B（仅用于分析实验）。**训练数据集**：SmolLM pretraining corpus，组成——FineWeb-Edu-Dedup (70%)、Cosmopedia-v2 (15%)、Python-Edu (6%)、Open-Web-Math (8%)、StackOverflow (1%)。**分析/校准数据集**：WikiText-2（用于 RoRoPE PCA 校准和 perplexity 评估）。**Benchmark**：MMLU、ARC (easy + challenge)、PIQA、HellaSwag、OpenBookQA、Winogrande（6 个常识推理任务）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  **开源**：https://github.com/MuLabPKU/TransMLA（NeurIPS 2025 Spotlight）。支持 LLaMA-2/3、Qwen2.5、Gemma2、Mistral、Mixtral 模型转换，兼容 DeepSeek 代码库（vLLM、SGlang、FlashMLA）。

  **算法 Pipeline 伪代码（以 LLaMA-2-7B GQA → MLA 转换为例）**：

  ```
  Input: GQA model with h query heads, g KV groups, d per-head dim, D hidden dim
         Calibration dataset (WikiText-2 subset)

  Step 1 — Merge all KV heads:
    W^{DKV} = [W^K; W^V] ∈ R^{2gd × D}   // 合并所有 KV head 投影矩阵
    初始化 W_i^{UK} ∈ R^{d × gd}：对 head i 所属 group j，W_i^{UK}[:, jd:(j+1)d] = I_d
    初始化 W_i^{UV} 同理（identity selector）
    // 此时 K cache = c_t^{KV} = W^{DKV} x_t ∈ R^{2gd}，与原始 GQA 相同

  Step 2 — RoRoPE: 解耦 RoPE 位置信息:
    For each RoPE frequency l ∈ {1, ..., d/2}:
      // 收集所有 g 个 head 中第 l 个 RoPE 子空间
      K_x_real = concat across heads: k_j^{[2l-1::d]} for j=0..g-1  // g-dim vectors
      K_y_imag = concat across heads: k_j^{[2l::d]} for j=0..g-1    // g-dim vectors
      // 构建联合协方差矩阵
      Σ_l = K_x_real^T K_x_real + K_y_imag^T K_y_imag  ∈ R^{g×g}
      // 特征分解得到最优正交旋转矩阵 U_l
      U_l = eigendecomposition(Σ_l).eigenvectors  // 按特征值降序排列
      // 旋转 W^K 和 W_i^{UK}（等价变换，不改变 attention 输出）
      W^K[l-related dims] ← U_l applied to corresponding dimensions
      W_i^{UK}[l-related dims] ← U_l applied to corresponding dimensions
    // 旋转后第一个 head 的 key 集中了主要位置分量 → K_rope
    // 其余 head 的 key 位置信息可忽略 → K_nope，移除其 RoPE

  Step 3 — FreqFold（可选）:
    // 将频率相近的 M 个 RoPE 频率索引合并
    For each merged group of M frequencies:
      Concat the M×2g dimensional segments across heads and frequencies
      Perform joint PCA on the concatenated vectors
      Keep top M principal components for K_rope
    // 使 K_rope 占用 M×d（而非仅 d）维度，保留更多位置信息

  Step 4 — BKV-PCA: 联合低秩压缩 K_nope 和 V:
    α = E[||W_NoPE^{DK} x_t||₂] / E[||W^{DV} x_t||₂]  // norm 平衡因子
    // 缩放后拼接
    c_NoPE,t = [1/α · W_NoPE^{DK} x_t; W^{DV} x_t] ∈ R^{(2g-1)d}
    // 在 calibration set 上 PCA
    R_KV = top-r_kv eigenvectors of Cov(c_NoPE)  ∈ R^{(2g-1)d × r_kv}
    // 低秩分解
    W^{DKV'} = R_KV^T · [W_NoPE^{DK}; W^{DV}]  ∈ R^{r_kv × D}
    W^{UKV'} = [W_NoPE^{UK} 0; 0 W^{UV}] · R_KV  ∈ R^{2hd × r_kv}
    // 推理时仅缓存 c_t^{KV'} = W^{DKV'} x_t ∈ R^{r_kv}（压缩后 KV cache）

  Step 5 — Fine-tuning（可选，恢复性能）:
    batch_size=64/256, lr=1e-4 或 2e-5, warmup 0-3%, constant/cosine scheduler
    seq_len=2048(SmolLM) 或 4096(LLaMA), tokens=300M-6B
  ```

  **推理时 Absorb 操作**（MLA 推理范式，Equation 10）：
  ```
  // 将 W_i^{UK} 吸收到 query projection 中，避免先投影再计算
  q̂_{t,i} = [(W_i^{UK})^T q_{t,i}^C; q_{t,i}^R]  // 变换后的 query
  k̂_t = [c_t^{KV}; k_t^R]                          // 共享 latent key
  // 所有 head 共享一个 KV head（类似 MQA），仅需缓存 c_t^{KV}
  ô_{t,i} = Σ_j softmax(q̂_{t,i}^T k̂_j / √(d+d^R)) · c_j^{KV}
  y_t = W^O [W_1^{UV} ô_{t,1}; ...; W_h^{UV} ô_{t,h}]
  ```

## The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  对 training-free 稀疏注意力方法进行最大规模的实证分析。基于四个设计轴（sparsification unit、importance estimation、budget allocation、KV cache management）建立分类体系，选取六种代表性方法并统一实现：**prefilling 阶段**——Vertical-Slash（全局垂直列+对角线斜杠，基于近端 query token 近似估计重要性，均匀 budget）、FlexPrefill（Vertical-Slash 增强版，threshold-based 动态 budget 分配，由 coverage 参数 α 和 min_budget 控制）、Block-Sparse（固定 block 大小 16×16，按 query block 选 top-k key block，均匀 budget）；**decoding 阶段**——SnapKV（token 级重要性估计，1D average pooling kernel=21 平滑，均匀 budget，KV cache eviction）、Ada-SnapKV（SnapKV 增强版，max-aggregation 替代 mean-aggregation 用于跨 head 动态 budget 分配，每 head 最低 budget 20%）、Quest（page 级选择，page size=16，使用 page 的 min/max key 值近似 query-page 相似度，均匀 budget，全 KV cache 保留）。所有方法保留 attention sinks（前 4 个 token）和局部上下文。通过 vLLM 的 FlashAttention 层级拦截实现。

  实验比较：(1) isoCost 分析——相同计算成本下密集小模型 vs 稀疏大模型的 Pareto 前沿对比（Figure 1）；(2) 9 个长期上下文任务（SQuAD/QuALITY/TOEFL QA、RULER NIAH/VT/CWE、Story Retrieval/Multi-hop/Filtering）上的 per-task 性能（Figure 2），分四个信息检索特征组；(3) 不同序列长度 (16k/32k/64k/128k) 下的稀疏容忍度（Figure 3）；(4) 模型大小效应——7B 到 72B 参数量的稀疏容忍度缩放分析（Figure 20/21）；(5) 六种方法之间的横向对比（Vertical-Slash vs FlexPrefill vs Block-Sparse for prefill; SnapKV vs Ada-SnapKV vs Quest for decode）；(6) 多种消融实验——block size (Block-Sparse)、page size (Quest)、近似窗口大小 (Vertical-Slash/FlexPrefill)、kernel size 和近似窗口 (SnapKV/Ada-SnapKV)、min budget (FlexPrefill/Ada-SnapKV)。

- 硬件平台是什么，配置是什么。
  4 个计算节点，每个节点 8 块 NVIDIA H100 GPU，共 ~32 块 H100，运行 21 天。使用 vLLM 推理引擎，全 bf16 精度。总共评估 7065 个配置，每个配置 100 个样本（Qwen）或 50 个样本（Llama/Gemma）。

- 模型是什么。数据集和bench分别是什么。
  **模型**：Qwen 2.5 (7B/14B/32B/72B)，Llama 3.1 (8B/70B)，Gemma 3 (4B/12B/27B)。所有使用 instruction-tuned 变体以支持 chain-of-thought 评估。Gemma 3 采用混合注意力——5/6 层使用 sliding window (1024 tokens)，仅在密集（global attention）层应用稀疏注意力方法。**数据集/Benchmark**：9 个任务——QA 类 (SQuAD/QuALITY/TOEFL)、RULER 合成任务 (NIAH/VT/CWE)、新增 Story 任务 (Story Retrieval/Multi-hop/Filtering) 基于程序化生成的多章叙述。**指标**：Exact Match Accuracy、IoU、F1（范围 0-1），计算成本使用 FLOPs (prefilling) 和 memory transfers (decoding)。稀疏度 0 到 0.95（对应 attention budget 1/1.5 到 1/20）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  **开源**：https://github.com/PiotrNawrot/sparse-frontier（MIT 许可证）。代码在 vLLM 的 FlashAttention 层级拦截 attention 计算，提供 `AbstractAttention` 基类，用户继承并注册即可实现自定义稀疏注意力。

  **算法 Pipeline 伪代码（以 Vertical-Slash prefill 为例）**：
  ```
  Input: Q, K, V ∈ R^{n×d}, sparsity_level, window_size=256
  1. 选择近端 query token: Q_recent = Q[-window_size:, :]  # shape: [w, d]
  2. 近似 attention 分数: S_approx = Q_recent @ K^T / sqrt(d)  # shape: [w, n]
  3. 沿近端 query token 聚合: S_agg = mean(S_approx, dim=0)  # shape: [n]
  4. 分出 prefix + local tokens（始终保留）: prefix=[0:4], local=[n-64:n]
  5. 剩余 token: S_remain = S_agg[4:n-64]
  6. 选择 top-(k_v + k_s) 个最大 S_remain 对应的 token 索引
  7. 将这些索引映射为 vertical columns（全局可见）和 slashes（对角线偏移）
  8. 仅对所选 QK pairs 计算 attention，使用 FlashAttention 的块稀疏模式
  Output: O = attention(Q, K, V) with sparsity = 1 - (selected_pairs / n^2)
  ```

  **Quest decoding 伪代码**：
  ```
  Input: q ∈ R^{d}, KV cache with page_size=16
  1. 将全 KV cache 分为 pages: pages = chunk(KV_cache, page_size)
  2. 对每个 page p 计算: K_min_p = min(p.keys, dim=0), K_max_p = max(p.keys, dim=0)
  3. 计算 page 级近似分数:
     S_approx[p] = max(|q·K_min_p|, |q·K_max_p|) / sqrt(d)
  4. 选择 top-k pages: selected_pages = topk(S_approx, k=token_budget/page_size)
  5. 对 selected_pages 内的所有 token 计算精确 attention
  6. 保留全部 KV cache（不 eviction）
  Output: O with sparsity = 1 - (selected_tokens / total_tokens)
  ```



- 属于算法pipeline的实现是什么？实验比较什么？
  提出 TailorKV，一个混合 KV Cache 压缩框架，核心算法包括：(1) **Offline Identification（离线层分类）**：定义 dense preference score $P = n_q - \sum_{(i,j) \in \hat{\mathcal{I}}} \hat{\mathbf{A}}_{i,j}$，其中 $\hat{\mathbf{A}} = \operatorname{Softmax}(\mathbf{Q}_{\operatorname{last\_q}} \mathbf{K}^{\top} / \sqrt{d_h})$ 使用最近 $n_q$ 个 query 和全部 key 计算 attention，$\hat{\mathcal{I}}$ 为 Top-k attention score 位置集合。若 $P_l > \tau$ 则层 l 为 quantization-friendly（密集注意力分布，适合量化），否则为 sparsity-friendly（稀疏注意力分布，适合动态检索 Top-K tokens）。(2) **Static Quantization（静态量化）**：对 quantization-friendly 层使用 per-channel key 量化 + per-token value 量化，支持极低 1-bit/2-bit 精度，group size=64。(3) **Dynamic Retrieval（动态检索）**：对 sparsity-friendly 层，利用 inter-layer similarity 在当前层 l-1 预估层 l 的 query $\hat{\mathbf{q}}^{(l)} = \mathbf{W}_q^{(l)}(\mathbf{h}^{(l-1)})$，计算 channel 重要性 $s_i = |\hat{\mathbf{q}}_i| \cdot \max(|\mathbf{K}_i|)$，选择 critical channels 对应的 critical key cache 从 CPU 预取到 GPU，在 GPU 上近似 attention scores 后选出 Top-K tokens 从 CPU 获取。

  实验比较：(1) LongBench (13 子任务聚合为 6 类)、InfiniteBench (9 子任务聚合为 5 类)、RULER (13 任务，4K-128K 长度) 上的 task accuracy，与 StreamingLLM、SnapKV、Quest、PQCache 对比；(2) 不同 layer 量化策略的消融（Figure 9a）——仅量化 layer 0（dense）性能最佳，量化 sparsity-friendly 层性能骤降；(3) 动态 vs 静态 channel selection 对比（Figure 9b）——动态选 critical channels 优于离线静态校准；(4) 不同 critical channel 数量 (2/4/8/12) 对性能和延迟的消融（Figure 9c）；(5) 与 SimLayerKV（另一混合方法）的对比（Table 7），TailorKV 在 34.2× 压缩率下与 SimLayerKV 1.53× 压缩率性能相当；(6) Peak GPU memory usage 对比（Figure 7）——128k 序列下相比 Full Cache 降低约 73.8%；(7) 端到端 decoding latency 对比（Table 4, Table 13）。

- 硬件平台是什么，配置是什么。
  两个配置：(1) NVIDIA RTX 3090 (24GB 显存, PCIe 1.0 ×16, 4GB/s) + Intel Xeon Gold 6240 CPU (64GB RAM)。(2) NVIDIA A100 (80GB 显存, PCIe 4.0 ×16, 32GB/s) + Intel Xeon Platinum 8369B CPU。推理精度 FP16/BF16，prefill 阶段结合 4-bit AWQ weight-only 量化。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-3.1-8B-Instruct (128K context, GQA, 32 layers, Q={0})、Llama-2-7B-32K-Instruct (32K context, MHA, 32 layers, Q={0,1})、Yi-6B-200K (200K context, GQA, 32 layers, Q={0,1})、Yi-9B-200K (200K context, GQA, 48 layers, Q={0,1})。Dataset/Benchmark：LongBench（Qspr/MulFi/HQA/WMQA/GRpt/MulN/TREC/SMSM/TriQA/Repo/LCC/PsgC/PsgR）、InfiniteBench（R.PK/R.Num/En.Dia/Sum/En.MC/En.QA/Zh.QA/Math.F/Code.D）、RULER（N-S1~N-S3/N-MK1~N-MK3/N-MV/N-MQ/VT/CWE/FWE/QA-1/QA-2）。Synthetic Longbench 用于离线确定 τ=0.2。超参：LongBench 用 8 critical channels, 64 local + 128 topk tokens；InfiniteBench/RULER 用 12 critical channels, 128 local + 896 topk tokens。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/ydyhello/TailorKV（ACL 2025 Findings，代码已发布）。依赖：torch==2.4.0, flash-attn==2.6.3, dgl, transformers==4.46.1。MIT License。

  算法流程（伪代码）：
  ```
  # === Offline Identification（离线阶段）===
  # 输入：校准数据集 prompt，模型权重
  # 输出：每层类型 C(l) ∈ {Quantization-Friendly, Sparsity-Friendly}

  for layer l in 0..L-1:
      Q_last_q = recent n_q query vectors at layer l  # shape: (n_q, d_h)
      K = all key vectors at layer l                   # shape: (n, d_h)
      A_hat = Softmax(Q_last_q @ K.T / sqrt(d_h))     # shape: (n_q, n)
      I_hat = {(i, Top_k(A_hat[i,:], k)) for i in 1..n_q}
      P_l = n_q - sum(A_hat[i,j] for (i,j) in I_hat)   # Eq.(8), dense preference score
      if P_l > τ:  # τ=0.2
          C(l) = Quantization-Friendly
      else:
          C(l) = Sparsity-Friendly

  # === 推理阶段 decode step ===

  # Quantization-Friendly 层：静态 1-bit KV cache 量化
  # Key: per-channel quantization, Value: per-token quantization
  # 量化公式 (Eq.4): X_Q = clamp(round((X - z) / s), 0, 2^b - 1)
  # b=1 → 1-bit, group_size=64, zero_point z 和 scaler s 存 FP16

  # Sparsity-Friendly 层：动态检索
  # Stage 1 (at layer l-1): 预估 critical channels → prefetch critical key cache
  q_hat = W_q[l] @ h[l-1]               # inter-layer 预估 query (Eq.13)
  s_i = |q_hat_i| * max(|K_i|)          # channel 重要性 (Eq.10), i=1..d_h
  critical_channels = Top_d_s(s)        # 选 d_s=8 或 12 个 critical channels
  # 从 CPU 异步 prefetch critical key cache (double buffering)

  # Stage 2 (at layer l): 近似 attention → 选择 Top-K tokens
  q = W_q[l] @ h[l]                     # 当前层真实 query
  q_critical = q[critical_channels]     
  K_critical = K[critical_channels]     # 已预取到 GPU
  a_approx = q_critical @ K_critical.T  # 近似 attention scores
  topk_indices = Top_K(a_approx, k=n_topk)
  # 从 CPU 异步 fetch Top-K tokens 的完整 key/value (唯一不可 overlap 的操作)
  
  # 完整 attention: 使用 n_local GPU tokens + n_topk CPU tokens
  ```

## StreamKV: Streaming Video Question-Answering with Segment-based KV Cache Retrieval and Compression

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 StreamKV，一个 training-free 框架，为 Video-LLMs 提供 KV cache 检索与压缩能力。核心算法包括：(1) **语义分段划分**：基于相邻帧 ViT embedding 的 cosine similarity 检测语义边界，配合 exclusion window（最小段长 m）和 segment merging（最大段长 M），动态划分视频流为语义段；(2) **Summary Vector**：每段内逐 spatial location 平均 frame-level features 得到 summary vector，其对应的 KV block 保留不做压缩，用于保留段级语义信息；(3) **Guidance Prompt 驱动的 KV 压缩**：引入 guidance prompt 捕获段内关键语义元素（salient entities、key events、temporal/causal relationships、contextual cues、factual details），用 guidance prompt 的 query vector 作为 selection criterion 选出每段中最 informative 的 KV blocks 保留；(4) **Unified Layer-Adaptive KV Selection Module**：将压缩和检索统一为 per-layer cosine similarity 排序 + 跨层 adaptive budget allocation 问题。每层计算候选 representative key vectors 与 selection criterion 的 softmax-normalized 相似度并按降序排列；通过 binary search 确定全局 cumulative score threshold p，使得跨层累积达到 total budget N 时自适应分配每层选中数量 K_l。

  实验比较：(1) StreamingBench 上 18 个子任务的 VideoQA 准确率，与 ReKV、Dispider、Flash-VStream、VideoLLM-online 等 Online Video-LLMs 及离线 Video-LLMs、闭源 MLLMs (Gemini1.5/ GPT-4o/ Claude3.5) 对比；(2) 不同压缩率下 (0%-90%) 语义分段 vs 均匀分段的性能对比；(3) 有无 summary vector 的性能对比；(4) 压缩和检索分别使用 uniform/adaptive 策略的四象限消融实验；(5) 检索帧数 (0-32 frames) 对准确率的影响对比 (vs ReKV)；(6) 内存使用和推理延迟对比。

- 硬件平台是什么，配置是什么。
  NVIDIA H20 GPU (96GB 显存)，FP16 精度。处理帧率 0.5 FPS，local window size = 15K tokens。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaVA-OneVision-Qwen2-7B-OV 作为基座模型（视觉编码器 + MLP projector + Qwen2-7B LLM）。数据集/Benchmark：StreamingBench，覆盖 18 个子任务分为三大类——Real-Time Visual Understanding (OP, CR, CS, ATP, EU, TR, PR, SU)、Omni-Source Understanding (ACP, CT, All, ER, SCU, SD, MA)、Contextual Understanding (ACU, MCU, SQA, PO)。评测指标：各类子任务准确率及 Overall 准确率。动态分段参数：m=4, M=64 frames，partitioning threshold=0.99。检索帧数 N_r=8。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/sou1p0wer/StreamKV（AAAI-26 接收，当前为 placeholder 状态，代码尚未发布）。MIT License。

  算法流程：
  ```
  # === 语义分段划分 ===
  # 输入：视频帧序列，ViT 编码器提取每帧 embedding f_t ∈ R^{P²×D}
  
  for each adjacent pair (f_{t-1}, f_t):
      s_t = cosine_similarity(f_{t-1}, f_t)  # Eq.(1)
      if s_t < threshold:  # threshold=0.99, 标记为语义边界
          boundaries.append(t)
  
  # 应用 exclusion window (size m=4) 避免过短段
  # 若段长超过 M=64，合并最相似的相邻帧对 (segment merging)
  # 输出：语义段序列 [S^i], S^i = [f_t^i]_{t=1}^{T_i}, T_i ∈ [m, M]
  
  # 每个段的 summary vector: f_s^i = mean(f_t^i) over t  # 逐空间位置平均
  ```

  ```
  # === Segment-based Sliding-window Encoding ===
  # X^i = concat(S^i, f_s^i)  # 段帧 + summary vector
  # 使用 local window L (过去 KV pairs, size=15K) + 当前段 KV 计算 attention
  O = Attn(W_Q X^i, [L_k, W_K X^i], [L_v, W_V X^i])  # Eq.(2)
  
  # 每帧 m 的 KV block: b_m^i = [(k_{m,p}^i, v_{m,p}^i)]_{p=1}^{P²}
  # representative key: r_m^i = (1/P²) Σ_p k_{m,p}^i ∈ R^{D'}  # Eq.(3)
  # 不区分 attention heads，拼接为 D' 维向量
  ```

  ```
  # === Unified Layer-Adaptive KV Selection Module ===
  # 输入: {R_l, c^l}_{l=1}^L (每层候选 representative keys + selection criterion), 总预算 N

  # Step 1: 计算每层每个候选的 cosine similarity
  Sim_l(j) = cos_sim(r_j^l, c^l)  # j ∈ idx(R_l)

  # Step 2: Softmax 归一化 + 降序排序
  ~Sim_l(j) = exp(Sim_l(j)) / Σ_k exp(Sim_l(k))  # Eq.(6)
  priority_l = sort_descending(~Sim_l)

  # Step 3: Binary Search 确定全局阈值 p（Algorithm 1）
  p_1=0, p_2=1
  while p_2 - p_1 > ε:
      p = (p_1 + p_2) / 2
      for each layer l:
          K_l^p = min{k | Σ_{j=1}^k ~Sim_l(s_l(j)) ≥ p}  # Eq.(7)
      if Σ_l K_l^p == N: return p
      elif Σ_l K_l^p < N: p_1 = p
      else: p_2 = p
  # 输出：自适应分配的 {K_l}_{l=1}^L，逐层取 top-K_l 候选为 I_l
  ```

  ```
  # === KV 压缩 (per segment) ===
  # selection criterion: guidance prompt vector g^l = (1/N_g) Σ_k g_k^l
  # 总预算: N = ⌈(1-θ) × T_i⌉ × L  (θ = compression ratio)

  {I_l^i}_{l=1}^L = SelectKV({R_l^i, g^l}_{l=1}^L, N)  # Eq.(9)
  ~B_l^i = [b_m^{i,l} | m ∈ I_l^i]  # 压缩后 frame-level KV blocks
  ~R_l^i = [r_m^{i,l} | m ∈ I_l^i]  # 对应的 representative keys

  # 更新 KV Bank (含 summary KV block b_s^{i,l}, 不参与压缩)
  B_l ← [B_l, ~B_l^i, b_s^{i,l}]  # Eq.(10)
  R_l ← [R_l, ~R_l^i, r_s^{i,l}]
  ```

  ```
  # === KV 检索 (回答问题) ===
  # selection criterion: question vector q^l = (1/N_q) Σ_k q_k^l
  # 总预算: N = N_r × L  (N_r = 期望每层检索帧数, 论文设为 8)

  {I_l}_{l=1}^L = SelectKV({R_l, q^l}_{l=1}^L, N)  # Eq.(11)
  P_l = [B_l[j] | j ∈ I_l]  # Eq.(12) 检索到的 KV blocks

  # 使用检索到的 KV blocks 作为 context 进行 QA
  O = Attn(W_Q X, [C_k, W_K X], [C_v, W_V X])  # Eq.(13)
  # C_k, C_v 包含: 检索到的 KV caches + question + 已生成 tokens
  # RoPE 策略: encoding 阶段仅应用于 local window; QA 阶段基于 relative positions
  ```

## SageAttention2++: A More Efficient Implementation of SageAttention2

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 SageAttention2++，在 SageAttention2 的量化 attention 基础上进一步加速。核心算法改动：对 attention 中 P×V 矩阵乘法，SageAttention2 使用 FP8 量化搭配 mma.f32.f8.f8.f32 指令（FP32 累加器，相对 FP16 仅 2× 加速）。SageAttention2++ 改用更快的 mma.f16.f8.f8.f16 指令（FP16 累加器，相对 FP16 达 4× 加速），同时通过两步保证精度：(1) **Narrowing FP8 Quantization Range**：将 P 和 V 的 FP8 E4M3 量化范围从 max(|x|)/448 缩小到满足 $P_r \times V_r \le 2047$ 的约束（如 $P_r=224, V_r=4.5$），确保 mma.m16n8k32 的 32 次乘积累加后不超出 FP16 表示范围（±65504）；(2) **Delayed FP32 Buffering**：连续两次 mma.m16n8k32 结果在 FP16 中累加后再转换到 FP32，将数据类型转换 PTX 指令开销减半，对应更严格的约束 $P_r \times V_r \le 1023.5$。

  实验比较：(1) Kernel 速度：RTX4090/RTX5090 上对比 FlashAttention2、SageAttention、SageAttention2，在 headdim=64/128 且带/不带 Causal Mask 下的速度；(2) 端到端模型指标：LLaMA3.1-8B（text）、CogvideoX-2B/HunyuanVideo/Wan（video）、Flux/Stable-Diffusion3.5（image）上对比 Full-Precision、SageAttn2(4+8)、SageAttn2(8+8) 的 perplexity、CLIPSIM、FID 等指标。

- 硬件平台是什么，配置是什么。
  NVIDIA RTX 4090 和 NVIDIA RTX 5090 GPU。这两代 Ada/Blackwell 架构 GPU 均支持 mma.f16.f8.f8.f16 指令（FP8 Matmul with FP16 accumulator，4× speedup over FP16）。FlashAttention3 仅支持 Hopper GPU，因此 FlashAttention2 是 RTX4090/5090 上最快的 baseline。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama3.1 (8B) — text2text；CogvideoX (2B)、HunyuanVideo、Wan — text2video；Flux (schnell)、Stable-Diffusion3.5 (turbo) — text2image。
  数据集：WikiText（perplexity）、LAMBADA（accuracy）、Needle-in-a-Haystack (NIAH) — 语言模型评估；Open-Sora prompt sets — 视频生成评估；COCO annotations — 图像生成评估。
  指标：Ppl.（WikiText）、Acc.（LAMBADA, NIAH）— 文本；CLIPSIM, CLIP-T, VQA-a, VQA-t, FScore — 视频；FID, sFID, CLIP, ImageReward — 图像；CosSim, L1, RMSE — attention 精度。
  Baselines：FlashAttention2、SageAttention、SageAttention2（两种变体：(4+8) INT4 for Q,K + FP8 for P,V；(8+8) INT8 for Q,K + FP8 for P,V）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/thu-ml/SageAttention（SageAttention2++ 将合入同一仓库）。
  实现语言：CUDA。

  算法流程：
  ```
  # 输入：Q, K, V ∈ R^{N×d}，Q,K 量化同 SageAttention2
  # 以下仅描述 P×V 的改进部分

  # Step 1: 计算 P = softmax(QK^T/√d) 同 SageAttention2
  #   Q,K 使用 INT4/INT8 per-block 量化
  #   P̃ 使用 FP8 E4M3 per-block 量化（SageAttention2 原有）

  # Step 2: Narrowing FP8 Quantization Range for P and V
  #   原 SageAttention2: δ_P = max(|P̃|)/448, δ_V = colmax(|V|)/448
  #   SageAttention2++:
  P_r = 224       # 缩小 P 量化范围
  V_r = 4.5       # 缩小 V 量化范围
  δ_P = max(|P̃|) / P_r      # 约束: P_r × V_r ≤ 2047/2 = 1023.5
  δ_V = colmax(|V|) / V_r

  # Step 3: 量化
  P̂ = round(P̃ / δ_P)    # 取值范围 [-224, 224]
  V̂ = round(V / δ_V)    # 取值范围 [-4.5, 4.5]

  # Step 4: FP8 Matmul with FP16 Accumulator
  #   使用 mma.m16n8k32 指令 (mma.f16.f8.f8.f16)
  #   每 32 个 pv 乘积在 FP16 中累加
  #   |32 × P̂ × V̂| ≤ 32 × 224 × 4.5 = 32256 ≤ 65504 ✓

  # Step 5: Delayed FP32 Buffering
  #   连续两次 mma.m16n8k32 结果在 FP16 累加后再转 FP32
  #   acc_fp16 += mma_result_1    # 第一次 MMA 结果在 FP16 中
  #   acc_fp16 += mma_result_2    # 第二次 MMA 结果继续累加
  #   acc_fp32 = convert(acc_fp16) # 两轮后才转 FP32，转换开销减半

  # Step 6: 反量化
  O = P̂V̂ * δ_P * δ_V    # 恢复到原始数值范围
  ```

  关键约束推导：
  - mma.m16n8k32 一条指令处理 32 个 p×v 乘积
  - FP16 最大可表示值 = 65504
  - 需要 |32 × p_max × v_max| ≤ 65504
  - 即 P_r × V_r ≤ 65504/32 = 2047
  - 使用 Delayed FP32 Buffering 后需满足 P_r × V_r ≤ 2047/2 = 1023.5
  - 选择 (P_r=224, V_r=4.5): 224×4.5 = 1008 ≤ 1023.5 ✓

## Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 SPECPREFILL，一种训练无关（training-free）的 token 重要性估计框架。核心算法：利用一个较小的"推测器"模型（speculator，如 Llama-3.1-8B-Instruct BF16）计算 prompt 中每个 token 的注意力分数，通过以下步骤筛选出局部重要 token：1) Look-ahead decoding（向前解码 N=8 步以缓解 attention sink 和 proximity bias）；2) Max-mean aggregation（对 [N, L, S, H] 注意力张量，在 H 和 L 维度取 max 以突出显著 token，在 N 维度取 mean 以公平贡献）；3) Chunk selection + 1D average pooling（将 context 分块，块内平均 token 分数后选 Top-K 块，利用邻近 token 相似性降低方差）；4) Position ID restoration（保持原始非连续 position IDs 送入主模型）。仅将筛选出的 token 子集送入主模型（Llama-3.1-70B-Instruct BF16 / 405B-Instruct FP8）进行 prefill，跳过其余 token 的 attention + MLP 计算和 all-reduce 通信，TTFT 加速正比于 token 丢弃率。

  实验比较：(1) Long context 质量：LongBench 六类任务（Single-Doc QA, Multi-Doc QA, Summarization, Few-Shot, Code, Synthetic），对比 Baseline（Llama 原始模型）、RAG-LLAMA（sentence-level RAG）、LLMLingua（文本级压缩）、MInference（sparse attention）；(2) Synthetic context probing：RULER suite（4K-128K），10% 保持率；(3) Standard short tasks：MMLU, IFEval, GSM8K 8-shot, HumanEval, MBPP, Arc Challenge, GPQA 8-shot；(4) Efficiency：端到端 QPS 实验（LongBench 数据集、vLLM server + OpenAI API client）、合成数据 TTFT 测量（不同 batch size × sequence length）。
  实验变体：SPECPREFILL（仅原始 attention 分数）、SPECPREFILL Full（所有技术，无 look-ahead）、SPECPREFILL Full LAH（所有技术 + 8 步 look-ahead）。

## SpindleKV: A Novel KV Cache Reduction Method Balancing Both Shallow and Deep Layers

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 SpindleKV，一种平衡浅层和深层 KV cache 压缩的方法。核心算法分为两部分：(1) 深层（deep layers）：基于注意力权重的 token eviction，使用金字塔形（pyramid-shaped）层间 KV cache 分配策略（随层数加深，保留比例线性递减）；(2) 浅层（shallow layers）：基于余弦相似度的 codebook token replacement，利用 KV cache 中 token 向量之间的高余弦相似性（constituent redundancy），对 KV cache 构建 codebook，仅存储 codebook 条目、每个 token 的索引（int 类型）和 L2 magnitude（float 类型），推理时通过 $Γ_r = C_Γ[r_Γ] \otimes m_Γ$ 重建。GQA 处理：将 KV head 展开（repeat $h_n$ 次），展开后使用 eviction + codebook 压缩。计算实际 KV cache 保留率 $r$ 时综合考虑 eviction 保留率 $r_1$、codebook 替换保留率 $r_2$ 和 dtype 转换比率 $r_3$。

  实验比较：(1) LongBench 16 个子任务（Single-Doc QA, Multi-Doc QA, Summarization, Few-shot Learning, Synthetic, Code）在多个 KV cache 保留率（~40%, ~30%, ~25%, ~20%, ~15%）下对比 PyramidInfer 和 PyramidKV；(2) Needle-in-a-Haystack 长上下文检索任务，15% KV cache 下对比 PyramidInfer 和 PyramidKV；(3) 额外 baseline 对比：H2O, SnapKV, StreamingLLM（LongBench on LLaMA3-8B-Instruct）；(4) 消融实验：GQA 集成策略（with/without repeat）、纯 codebook 压缩（无 eviction）、magnitude reconstruction 有效性。

- 硬件平台是什么，配置是什么。
  单卡 NVIDIA RTX 3090 GPU（推理速度测试使用，context length 4096, generation length 1000）。主要实验计算平台论文未明确说明具体配置。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA2-7b-chat（MHA，h=32），LLaMA3-8b-instruct（GQA，h=32, $h_n$=8, $h_g$=4），Mistral-7b-instruct-v0.2（GQA，h=32, $h_n$=8, $h_g$=4）。最大 context length 4K-32K。
  数据集/Benchmark：LongBench（16 个子集：narrativeqa, qasper, multifieldqa_en, hotpotqa, 2wikimqa, musique, gov_report, qmsum, multi_news, trec, triviaqa, samsum, passage_count, passage_retrieval_en, lcc, repobench-p），Needle-in-a-Haystack 检索任务。
  Baselines：PyramidInfer, PyramidKV, H2O, SnapKV, StreamingLLM。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/tyxqc/SpindleKV
  超参数：Key Threshold $\theta_K=0.98$, Value Threshold $\theta_V=0.95$, $\beta=0.05$, $\alpha=0.525$。

  算法流程（伪代码）：
  ```
  # Prefilling 阶段
  # 1. 对所有 prefill token 计算全量 attention
  # 2. 深层 eviction（Section 3.3）
  #    对每层 λ，计算保留率 r_c(λ) = r_c(0) + (r_c(m-1)-r_c(0))/(m-1)·λ
  #    按 accumulated attention score ac_i 选 Top-K token 保留
  #    KV_r = KV[argTopK(ac, k=⌊r_c(λ)×l_c⌋)]
  #
  # 3. GQA 展开（对 GQA 模型）
  #    将 KV head repeat h_n 次：K_expanded = K.repeat(h_n, ...)
  #
  # 4. 浅层 codebook 构建（Section 3.4, Algorithm 1）
  m_Γ = L2_Norm(Γ, dim=-1)           # 记录 magnitude
  Γ_r = Γ_r / m_Γ                    # 归一化
  S_Γ = cos_sim(Γ_r, Γ_r)            # 余弦相似度矩阵
  G_Γ = where(S_Γ > θ_Γ, 1, 0)      # 邻接矩阵
  C_Γ = []                           # CodeBook
  r_Γ = [-1, -1, ..., -1]            # 每个 token 的 codebook 引用
  while G_Γ != 0:
      s_Γ = sum(G_Γ, dim=1)          # 每个节点的度数
      ι = argmax(s_Γ)                # 选度数最高的 token
      C_Γ.append(Γ_r[ι])             # 加入 CodeBook
      η_ι = argwhere(G_Γ[ι] == 1)    # 找到该 token 可合并的邻居
      r_Γ[η_ι] = len(C_Γ) - 1        # 记录引用索引
      mask_Γ = matmul(¬G_Γ[ι]^T, ¬G_Γ[ι])
      G_Γ = G_Γ & mask_Γ             # 从图中移除已处理节点

  # 5. 推理时重建
  Γ_reconstructed = C_Γ[r_Γ] ⊗ m_Γ  # 从 CodeBook + magnitude 恢复
  # 对重建后的 K 重新应用 RoPE
  ```

  推理速度：FullKV 22.16 token/s vs SpindleKV 40% cache 18.39 token/s (LLaMA3-8B)，约 18% 额外开销。

- 硬件平台是什么，配置是什么。
  8 × NVIDIA H200（Tensor Parallelism = 8），CUDA 12.7，总 GPU TFLOPS 428.2，总 RAM 1123.2 GB，单 GPU 内存带宽 4052.8 GB/s，NVLink 带宽 478.1 GB/s，PCIe 5.0 x16。部分 MInference 对比实验在 8 × NVIDIA H100 上进行。vLLM 0.6.3.post1，enforce_eager=True，chunked_prefill=False。

- 模型是什么。数据集和bench分别是什么。
  主模型（base model）：Llama-3.1-70B-Instruct (BF16)、Llama-3.1-405B-Instruct-FP8（neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8）。
  推测器（speculator）：Llama-3.1-8B-Instruct (BF16)。
  数据集：LongBench（含 Single-Doc QA, Multi-Doc QA, Summarization, Few-Shot Learning, Code Completion, Synthetic 六个类别），RULER（含 NIAH variants, Multi-hop Tracking, SQuAD & HotpotQA, CWE & FWE），MMLU（Generative），IFEval，GSM8K（8-shot），HumanEval，MBPP，Arc Challenge，GPQA（8-shot）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/anonymous/speculative_prefill（匿名仓库，ICML 2025 发表时公开）。
  评估框架：LM-EVAL-HARNESS（标准短任务）、EVAL-PLUS（代码任务）。

  算法伪代码（Algorithm 1 from paper）：
  ```
  Require: Base model M, speculator S, look-ahead steps N,
           batch of mixed requests B, base model QKV cache C_b,
           speculator KV cache C_s
  1: B_p, B_d ← split_prefill_decode_requests(B)
  2: for i = 1 to N do                    # Sec 3.2.1 Look-ahead
  3:     B'_p ← model_forward(S, B_p, C_s, store_q=True)
  4:     B_p ← update_requests(B_p, B'_p)
  5:     B_p ← check_for_eos(B_p)
  6: end for
  7: if is_tensor_paralleled() then
  8:     tp_gather_qk(C_s)                # 收集 TP 分片的 Q,K
  9: end if
  10: Q, K ← retrieve_qk(B_p, C_s)
  11: A ← compute_attention_score(Q, K)    # shape: [N, L, S, H]
  12: A ← aggregate_attention_score(A)     # Sec 3.2.2: max over H,L, mean over N → [S]
  13: T ← chunk_select_from_smoothed_attention(A)  # Sec 3.2.3: 1D avg pool + chunk + Top-K
  14: P ← restore_pos_ids(T, B_n)         # Sec 3.2.4: 非连续 position IDs
  15: B ← merge_requests(T, P, B_p, B_d)
  16: Return model_forward(M, B, C_b)
  ```

  张量计算过程：对于 prompt 长度 M、look-ahead N 步、L 层、S 序列长、H 头数，注意力分数 $a_{ij} = \text{Softmax}(Q_{M+j} K^T)_i$（对第 j 个解码 token 的第 i 个 prompt token 的注意力）。聚合策略：$\text{score}(i) = \frac{1}{N}\sum_{j=0}^{N-1} \max_{l \in [0,L), h \in [0,H)} a_{ij}^{lh}$。然后对 score 序列做 1D average pooling 平滑，分 chunk 后取每个 chunk 内平均分数的 Top-K chunks，选中的 token 连同其原始 position IDs 送入主模型 forward。

## LightTransfer: Your Long-Context LLM is Secretly a Hybrid Model with Effortless Adaptation

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 LightTransfer，一种将标准 Transformer 无损转换为 Hybrid 模型的轻量框架。核心实现：通过分析每层注意力分布定义"懒惰比例"（lazy ratio）$r_i = \frac{1}{w_{\text{last}}} \sum_{\hat{x} \in X_{\text{last}}} \sum_{x \in \{X_{\text{initial}}, X_{\text{recent}}\}} A_i(\hat{x}, x)$，识别将大部分注意力集中在初始 sink token 和最近 token 上的"懒惰层"（lazy layers），将其 full attention 替换为 streaming attention（仅保留 $w_{\text{sink}}=4$ 个 sink token 和 $w_{\text{recent}}=1020$ 个最近 token 的 KV cache）。分两种模式：LightTransfer-TEST（test-time 在线识别，无需训练，适用于长上下文理解）和 LightTransfer-TRAIN（基于训练集预选懒惰层后 SFT 微调 ~5K 样本，适用于 o1-like 长推理）。提供理论保证：网络输出误差被移除 KV 对的注意力分数之和上界约束（Theorem 5.1）。

  实验比较：(1) Long-context understanding：在 LongBench（16/21 任务）和 NIAH（Ruler benchmark, 4K-32K）上对比 Standard transformer、StreamingLLM、MiniCache、SqueezeAttention；(2) o1-like long reasoning：在 MATH-OAI、AIME24、GSM8K 上对比 QwQ-STILL、LongGen、DuoAttention；(3) Ablation：不同标准层保留比例（0.25-0.75）、不同层替换策略（Pyramid/Random/Shapley/BERTology）、与 SnapKV 组合、MoE 架构（Qwen1.5-MoE-14.3B）、head-wise 对比。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 GPU（论文未明确说明具体显存和数量，但 Appendix B.2 提到使用 8×A100 40G 节点进行 TP vs DP+TP 实验）。PyTorch + HuggingFace Transformers，使用 flash_attention_with_kvcache 加速。所有模型权重、激活、KV cache 使用 BF16 精度，无量化。LightTransfer-TRAIN 使用 Flex Attention 优化训练。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA2-7B-chat（上下文窗口 4K）、Mistral-7B-Instruct（8K）、LLaMA3-8B-Instruct（32K）、LLaMA3-70B-Instruct（32K）、QwQ-32B-STILL（基于 Qwen2.5-32B-Instruct 蒸馏，用于 o1-like 推理）、Qwen1.5-MoE-14.3B-A2.7B（MoE 验证）、Qwen2.5-3B-chat-32K（SnapKV 组合实验）、LLaMA3-8B-Instruct-Gradient-1048K（head-wise 对比）。
  数据集/Bench：LongBench（多任务长上下文理解，16/21 子任务）、NIAH/Ruler（单 key 和多 key needle-in-a-haystack 检索，4K-32K）、MATH-OAI、AIME24、GSM8K（数学推理）。训练数据：QwQ-STILL 公开训练集 ~5K 样本（用于 long-reasoning 蒸馏）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源：GitHub https://github.com/sail-sg/LightTrans，HuggingFace 模型 cxdu/QwQ-32B-LightTransfer。算法核心伪代码（基于论文 Table 1）：
  ```
  # 懒惰比例计算（利用 FlashAttention 的 LSE 值避免完整重计算注意力矩阵）
  def lazy_ratio_calculation(q, k, v, w_last, w_sink, w_recent):
      # q, k, v: [bs, num_heads, seq_len, head_dim]
      attn_out, lse = flash_attn(q, k, v, causal=True, return_lse=True)
      # lse: [bs, num_heads, seq_len] -- log-sum-exp of attention scores
      
      q_last = q[:, :, -w_last:, :]              # 最后 w_last 个 query token
      k_comb = torch.cat([k[:, :, :w_sink, :],    # 前 w_sink 个 sink token
                           k[:, :, -w_recent:, :]], dim=2)  # 后 w_recent 个 recent token
      
      # 计算 q_last 对 k_comb 的 log attention score (O(w_last * (w_sink+w_recent)))
      log_lazy_ratio = torch.matmul(q_last, k_comb.transpose(-1, -2)).logsumexp(dim=-1) - lse
      
      return log_lazy_ratio  # 高值 → layer "懒惰"，attention 集中在 sink+recent tokens
  ```
  
  **完整算法流程 (LightTransfer-TEST)**:
  1. Prefilling 阶段逐层处理输入，对每层 i 计算 lazy ratio r_i
  2. 使用大小为 P 的最大堆优先队列维护 lazy ratio：超过容量时弹出 ratio 最高的层，标记为 lazy layer，将其 KV cache 缩减为仅保留 {X_initial, X_recent}
  3. Non-lazy 层保留完整 full attention KV cache
  4. Decoding 阶段直接使用 prefilling 后已缩减的 KV cache
  5. 复杂度：识别过程 O(1) 相对于序列长度（仅需一次小矩阵乘法），超长序列下开销可忽略
  
  LightTransfer-TRAIN：在训练集上喂入 question+answer 以充分暴露各层的 lazy 行为，统计各层被识别为 lazy 的频率，选频率最高的层预选为 lazy layer，然后在新 hybrid 架构下 SFT 微调。

## Scale-invariant Attention

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 Scale-invariant Attention：一种对 attention logits 施加位置依赖的乘性缩放和加性偏置的算法，使注意力机制满足两个性质——scale-invariant total attention（每个 token range 内的总注意力渐进恒定）和 scale-invariant attention sparsity（注意力稀疏性随上下文变长而增长）。变换形式为 $L_t = a_t \bar{L}_t + m_t$，其中 $a_t = \sqrt{2[\log(t/\tau+1) - \log\alpha + \beta/\alpha]}$，$m_t = -a_t^2 + \beta/\alpha$，施加边界条件后 $\alpha = \beta = e^{0.5}$，唯一超参数 $\tau=10$。该变换与 p-RoPE 结合使用（称为 scale-invariant p-RoPE），实现从短上下文（4k）零样本泛化到长上下文（64k）而无需额外长上下文训练。
  
  实验比较：在 GPT-2-style 162M/304M 模型和 Llama 2 7B 上，对比 RoPE、p-RoPE、NoPE、RoPE+NTK、YaRN、LogN+RoPE、LogN+p-RoPE、LogN+NTK、ALiBi、Infini-attention 等基线方法，评估验证 loss（in-distribution 和 zero-shot length generalization）以及 needle-in-a-haystack 长上下文检索准确率。

- 硬件平台是什么，配置是什么。
  162M 模型：单卡 A100 80GB GPU；304M 模型：4×H100 Grace Hopper 节点（DDP）；Llama 2 7B continual pretraining：论文未明确说明 GPU，使用 Torchtune 库训练。

- 模型是什么。数据集和bench分别是什么。
  模型：GPT-2-style（modded-nanogpt 变体，使用 RMSNorm、ReLU² 激活、QK-Norm），162M 参数（12 layers, hidden 768, 6 heads）和 304M 参数（16 layers, hidden 1024, 8 heads）；Llama 2 7B（continual pretraining）。数据集：FineWeb（10B token subset 用于 162M，100B subset 用于 304M，实际使用 ~10B tokens）。Bench：语言建模验证 loss（4k/16k/64k context lengths），needle-in-a-haystack 检索任务（使用 C4 数据集构造）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源：论文提供补充材料中的代码，基于 modded-nanogpt（MIT 协议）实现。使用 FlexAttention 实现 scale-invariant attention 和 ALiBi。算法核心：
  ```
  # 给定查询 q (dim d)，键矩阵 K (seq_len T × d)，位置距离 t ∈ [1, T]
  # 标准 attention score: S_t = (1/√d) * Σ_λ q_λ * K_{t,λ}
  
  # 超参数: τ = 10, α = β = e^{0.5}
  # 计算位置依赖的 a_t 和 m_t:
  f_t = log(t/τ + 1) - log(α)
  a_t = sqrt(2 * (f_t + β/α))
  m_t = -a_t^2 + β/α
  
  # 变换 logits:
  L_t = a_t * S_t + m_t      # S_t 是已应用 p-RoPE 后的 score
  
  # 标准 softmax attention:
  A_t = softmax(L_t)         # 在 t=1..T 上归一化
  output = Σ_t A_t * V_t     # 加权和值向量
  ```
  关键特性：当 t 较小时（局部上下文），$a_t^2 \approx 1$, $m_t \approx 0$，近似标准 attention；当 t 增大时，$a_t^2$ 对数增长（使分布更尖锐/sparse），$m_t$ 对数下降（压低远距离 token 的总体权重），实现局部稠密、全局稀疏的 attention 模式。

- 属于算法pipeline的实现是什么？实验比较什么？
  本论文是一篇 revisit/survey 型工作，非提出新算法，而是从实际生产部署角度全面评估四类代表性 KV cache 压缩算法的吞吐、响应长度分布和 negative sample 表现。被评估的实现包括：
  - **KIVI**（量化类）：per-channel key quantization + per-token value quantization。关键参数 group_size=G=32, residual_length=R=128（保留最近 128 tokens 为全精度）。开源：https://github.com/jy-yuan/KIVI
  - **GEAR**（量化类）：用 low-rank matrix 近似量化误差 + sparse matrix 处理 outlier。关键参数 sparsity_ratio=s=2%, rank=r=2%。开源：https://github.com/opengear-project/GEAR
  - **StreamingLLM**（稀疏类）：仅保留 initial tokens (64) + recent tokens (448)，总计 KV cache 大小 = 512。无动态 eviction 计算，结构化计算模式。
  - **H2O**（稀疏类）：基于 accumulated attention scores 动态 evict KV cache。heavy hitter oracle token size=64 + recent size=448，总计 cache size=512。

  实验比较：
  (a) Prefill/Decoding 吞吐 vs FP16 Baseline，在 TRL、TRL+FlashAttention、LMDeploy（含 PagedAttention+FlashAttention）三种框架下，batch size 1~32，prompt length 512~8192。
  (b) 不同 tensor parallelism (TP=1/2/4) 下的相对加速比。
  (c) 响应长度分布差异：比较压缩算法 vs Temperature=0.9/1.1 对输出长度的影响，评估 verbose output 现象。
  (d) Negative sample 分析：使用 LongBench 在 LLaMA-3.1-8B-instruct 和 Mistral-7B 上分析个体样本的精度退化，按 task type（Summarization/QA/Code）分类统计。
  (e) 吞吐预测器（Throughput Predictor）精度：基于 Vidur 框架 profiled attention operator runtime，预测不同 batch×seqlen 的组合。
  (f) 请求路由器（Request Router）：结合吞吐+长度预测器路由请求以最小化端到端延迟。

- 硬件平台是什么，配置是什么。
  主要：4× NVIDIA A6000 (48GB) 通过 NVLink 互联，Intel Xeon Gold 6326 CPU @ 2.90GHz。
  部分实验扩展至：NVIDIA H800 GPU（LLaMA-70B 实验，Figure 2）。
  框架：PyTorch 2.1.2, Transformers 4.43.1, FlashAttention 2.5.6, LMDeploy v6.0.1。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA-2-7B, LLaMA-2-13B, LLaMA-2-70B, LLaMA-3.1-8B-instruct, Mistral-7B-v0.1。
  数据集/Benchmark：ShareGPT（吞吐分析和长度分布实验，1000 样本子集，max generation tokens=1024），LongBench（negative sample 分析，覆盖 multi-document QA、single-document QA、summarization、few-shot learning、code completion、synthetic tasks）。

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  论文开源：https://github.com/LLMkvsys/rethink-kv-compression（含 throughput/length predictor、negative sample benchmark、LMDeploy 集成代码）。

  以 KIVI (per-channel key quantization) 为例说明算法 pipeline：
  ```
  # Prefill 阶段：正常计算 KV cache，不量化
  X_K = X @ W_K      # [b, l, d]，全精度存储
  X_V = X @ W_V      # [b, l, d]，全精度存储

  # Decoding 阶段：每步新 token 的 K/V 量化后追加
  for each decode step t:
      x_k = x @ W_K   # [b, 1, d]
      x_v = x @ W_V   # [b, 1, d]
      # Key: per-channel 量化，group_size=G=32
      for c in range(0, d, G):
          x_k_quant[c:c+G] = quantize_per_channel(x_k[c:c+G])  # → INT4
      # Value: per-token 量化
      x_v_quant = quantize_per_token(x_v)                      # → INT4
      # 保留最近 R=128 tokens 为全精度
      append_to_kv_cache(x_k_quant, x_v_quant)
      # Attention 计算时 dequantize
      scores = Q @ dequantize(K_quant)^T / sqrt(d_head)
      output = softmax(scores) @ dequantize(V_quant)
  ```

  以 H2O (accumulated attention score eviction) 为例说明算法 pipeline：
  ```
  # Prefill 阶段：正常计算 attention，累积 attention scores
  scores = Q @ K^T / sqrt(d_head)     # [b, heads, l, l]
  attn_scores_sum = scores.sum(dim=-2) # [b, heads, l]，累积每 token 被 attend 的分数

  # Decoding 阶段：每步动态 evict
  for each decode step:
      # 计算当前 attention scores 并累积
      scores = Q @ K^T / sqrt(d_head)
      attn_scores_sum += scores.sum(dim=-2)

      # 保留 heavy hitter (top 64) + recent (448)
      important_idx = topk(attn_scores_sum, k=64)
      recent_idx = last_n_tokens(448)
      keep_idx = union(important_idx, recent_idx)

      # Evict: 删除不在 keep_idx 中的 KV cache entries
      K = K[keep_idx]
      V = V[keep_idx]
  ```
  H2O 的 eviction 计算需要 multi-pass attention（为计算 importance metric），与 FlashAttention 的单 pass 设计不兼容，导致额外内存访问开销。


## Rectified Sparse Attention

- 属于算法pipeline的实现是什么？实验比较什么？
  ReSA 提出了一种结合 block-sparse attention 和周期性 dense rectification 的稀疏解码方法，通过定期刷新 KV cache 来限制稀疏注意力的误差累积，在不牺牲生成质量的前提下显著加速长序列推理。实现包括三个核心组件：
  (1) **Group Block Sparse Attention (GBSA)**：基于 Quest 算法的 query-aware block-sparse attention，使用 block-wise min/max 描述符进行块级近似匹配，结合 GQA group 内共享注意力模式（来自 NSA 的 shared grouping）以进一步提升效率。Block 大小 b=16，使用 dynamic top-n 策略：永远保留最近 n_local=1 个 block，强制最少 n_min=16 个 block，其余根据活跃 ratio p 动态选择。
  (2) **Dense Rectification**：每 f=32 个 token 后，将最近生成的 token 批量用 dense attention 并行重编码，刷新 KV cache 和 block key cache。这保证稀疏注意力的 KV cache 误差被限制在常数窗口内。
  (3) **Memory Access 模型**：平均每步 memory access 为 mem(KV cache) × (1/b + p + 1/f)，相对于 dense decoding 的理论加速因子由 b、p、f 控制。

  实验比较：
  (a) Math reasoning (test-time scaling)：DeepSeek-R1-Qwen-Distill 1.5B/7B 在 Minerva Math、Gaokao2023En、OlympiadBench、AIME24、AMC23 共 5 个基准上的准确率，对比 Dense、Sparse、Sparse_dense2（前两层 dense）和 ReSA，均在 4K–12K token 平均推理长度。
  (b) Language modeling：Qwen2.5 模型在长序列 book data 下，模拟 sparse decoding pattern，评测不同 rectification frequency x 和 sparsity ratio p 下的 top-3 next-token prediction accuracy。对比 Decode Only（upper bound）和 sparse baseline。
  (c) Retrieval (RULER benchmark)：Qwen2.5 7B 在 RULER 的 8 个子任务（QA、MultiQuery、FWE、VT、MultiKey、MultiValue、CWE、Single）上评测不同 sparsity ratio 下的准确率。
  (d) Inference efficiency：Qwen-2.5 7B 在 NVIDIA A100-80G 上的 kernel-level latency breakdown（16K/64K/256K）和 end-to-end throughput（FP16 和 INT4，4K/16K/64K/256K context）。
  (e) Ablation：f ∈ {16,32,64,128} 和 p ∈ {0.9,0.95,0.98} 网格搜索。
  (f) 与 sparse KV-based self-speculation 的 decoding 速度对比（Table 3），ReSA 平均 1.92× speedup。

- 硬件平台是什么，配置是什么。
  NVIDIA A100-80G GPU。所有实验基于 PyTorch 实现。INT4 实验使用 Marlin kernel 进行 low-bit matmul，group-wise scaling（group size=128）。Custom kernel 参考 Flash Decoding 的 split-execution 策略，使用 TileLang 库实现 group block sparse attention。评测 latency 时仅报告 CUDA kernel 执行时间，排除 CPU-side scheduling overhead。

- 模型是什么。数据集和bench分别是什么。
  模型：Qwen2.5 系列（Qwen2.5 7B 为主），DeepSeek-R1-Qwen-Distill 1.5B/7B。DeepSeek-R1-Qwen-Distill 7B 配置：28 层，28 attention heads，4 KV heads (GQA)，hidden size 3584。
  数据集/Benchmark：
  - Math reasoning: Minerva Math, Gaokao2023En, OlympiadBench, AIME24, AMC23
  - Language modeling: long-sequence book data
  - Retrieval: RULER benchmark (8 子任务)
  - Efficiency: 无特定数据集，测量 kernel latency 和 throughput

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源在 https://aka.ms/ReSA-LM。核心算法流程见论文 Algorithm 1（Rectified Sparse Decoding），张量计算如下：

  **Step 1 - Prefill（dense attention）**：
  ```python
  # 标准 dense prefill，构建完整 KV cache K 和 block key cache B
  K, B = DensePrefill(input_ids, model)
  # K: KV cache, B: block key descriptors (min/max per block)
  ```

  **Step 2 - Group Block Sparse Attention (GBSA)**：
  ```python
  # Block representation: 将 key 矩阵按 block size b 分区
  # k_block_min_i = min(k[i*b:(i+1)*b]), k_block_max_i = max(k[i*b:(i+1)*b])
  
  # Block selection per GQA group:
  q_pool = avg_pool(Q_group)  # 组内 query heads 平均池化
  for each block i:
      score_i = sum_j(max(q_j * k_block_max_i[j], q_j * k_block_min_i[j]))
  # 选择 top n blocks (n = max(n_min, ceil(M * p))), n_local 个最近 block 强制保留
  
  # Sparse attention:
  M = create_sparse_mask(selected_blocks)  # M ∈ {0,1}^{h × n × n/b}
  O = softmax(Q @ K^T * extended_mask(M) / sqrt(d)) @ V
  ```

  **Step 3 - Dense Rectification（每 f=32 tokens）**：
  ```python
  if step % f == 0:
      # 将最近 f 个 token batch 用 dense attention 并行重编码
      K, B = DenseForward(tokens[-f:], K, B)
      # 刷新 block key cache B 以匹配更新后的 KV cache K
  ```

  **Step 4 - 循环生成**：交替 sparse decoding → rectification → sparse decoding，直到生成完毕。

  Decode Only 设置（upper bound）：KV cache 全部由 dense attention 构建，仅新 token 使用 sparse attention 解码——代表 ReSA 理论上界。

## ReCalKV: Low-Rank KV Cache Compression via Head Reordering and Offline Calibration

- 属于算法pipeline的实现是什么？实验比较什么？
  ReCalKV 是一种后训练（post-training）低秩 KV Cache 压缩方法，通过分别针对 Key 和 Value 的不同特性设计差异化压缩策略来减少 KV Cache 的 hidden dimension 大小。具体实现包括三个核心组件：
  (1) **Fisher Information 引导的压缩率分配**：使用标定数据计算每层的 Fisher Information 分数（继承自 Palu），按层重要性分配不同的压缩率，重要层保留更多 rank。
  (2) **Head-wise Similarity-aware Reordering (HSR) for Keys**：先计算每层所有 attention head 之间的 CKA（Centered Kernel Alignment）相似度矩阵 S ∈ R^{h×h}，贪心地将相似度最高的 head 分组（每组 s=4 heads），然后在每组内分别进行 SVD 低秩分解。Key projection matrix W_k ∈ R^{m×n}（n = h·d_h）被按列分成 h 个 head-wise 子矩阵，每组 s 个 head 拼接后进行 group SVD：W_{g_j} ≈ L_{g_j} R_{g_j}，其中 L_{g_j} ∈ R^{d×r_g}，R_{g_j} ∈ R^{r_g×(d_h·s)}。推理时先计算共享 latent z_{g_j} = x L_{g_j}，再逐 head 重建 [y_{j,1},...,y_{j,s}] = z_{g_j} R_{g_j}。HSR 之后需要 inverse reordering 恢复原始 head 顺序以保证解码等价性。
  (3) **Offline Value Calibration (OVC) for Values**：对 Value projection matrix W_v ∈ R^{m×n} 直接进行全矩阵 SVD 分解：W_v ≈ L_v R_v，然后用标定数据 X 对 L_v 和 R_v 进行闭式校准，最小化近似误差 E = ||L_v R_v X - W_v X||_F^2。校准后通过 Matrix Fusion 将 R_v 融合进 output projection W_o：W̃_o = R_v·W_o，推理时无需显式重建 Value cache。

  实验比较：(a) 与 Palu (G-LRD, group size=4) 对比，在 LLaMA-7B、LLaMA-2-7B、Mistral-7B-Instruct-v0.2、LongChat-7B-v1.5-32k、LLaMA-2-13B-Chat 上评测 50%/60%/70% 三种压缩率下的语言建模困惑度（WikiText2, PTB, C4）和 6 项零样本 QA 准确率（OBQA, HellaSwag, PIQA, ARC-e, ARC-c, Winogrande）；(b) LongBench 长文本 benchmark（8 项任务）下的平均准确率；(c) 集成 KV Cache 量化（4-bit/3-bit per-token quantization + Hadamard transform）后的组合压缩效果；(d) 消融实验：HSR 和 OVC 各自的贡献（80% 压缩率）；(e) 推理效率：Triton 自定义 fused attention kernel 在 4K/16K/65K prompt 下的延迟加速比。

- 硬件平台是什么，配置是什么。
  单张 NVIDIA A800 GPU（80GB 显存）。推理效率评测同样使用 A800。所有实验基于 PyTorch 和 HuggingFace Transformers 实现。标定数据集：从 WikiText-2 随机选取 256 个样本。SVD 前应用 whitening 变换（参考 SVD-LLM 的设置）。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA-7B、LLaMA-2-7B、LLaMA-2-13B-Chat（MHA）、Mistral-7B-Instruct-v0.2（GQA）、LongChat-7B-v1.5-32k（MHA, 32K context length）。LLaMA-3.1 结果在补充材料中。
  数据集/Benchmark：(a) 语言建模困惑度：WikiText2、Penn Treebank (PTB)、C4 子集；(b) 零样本 QA 准确率：OBQA、HellaSwag、PIQA、ARC-e、ARC-c、Winogrande；(c) 长文本理解：LongBench（Qasper, QMSum, MultiNews, TREC, TriviaQA, SAMSum, LCC, RepoBench-P 共 8 项任务）；(d) 标定数据：WikiText-2 中 256 个随机样本。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码和模型将在 https://github.com/XIANGLONGYAN/ReCalKV 发布（论文标注 "will be available"）。完整算法伪代码见论文 Algorithm 1，核心流程：

  **Step 1 - Fisher Information 计算与压缩率分配**：
  ```python
  # 使用标定数据 X_calib 计算每层的 Fisher Information
  F = calculate_fisher_info(model, X_calib)  # {layer_idx: fisher_score}
  R = allocate_compression_ratio(model, target_ratio, F)
  # 重要层（高 Fisher）分配更多 rank，不重要层分配更少 rank
  ```

  **Step 2 - HSR: Key Cache 压缩**（per Key projection layer）：
  ```python
  # W_k: Key projection matrix, shape [d, h*d_h], h=32 heads, d_h=head_dim
  # Step 2a: 计算 head 间 CKA 相似度矩阵
  S = zeros(h, h)
  for i, j in range(h), range(h):
      H_i = W_k[:, i*d_h:(i+1)*d_h]  # head i 的投影子矩阵
      H_j = W_k[:, j*d_h:(j+1)*d_h]  # head j 的投影子矩阵
      G_i = H_i @ H_i.T  # Gram matrix
      G_j = H_j @ H_j.T
      # Centering: H_c = I - (1/d)11^T
      G_i_centered = H_c @ G_i @ H_c
      G_j_centered = H_c @ G_j @ H_c
      S[i,j] = Tr(G_i_centered @ G_j_centered) / sqrt(
               Tr(G_i_centered @ G_i_centered) * Tr(G_j_centered @ G_j_centered))

  # Step 2b: Greedy head reordering (group_size=4, 32 heads -> 8 groups)
  groups = [[] for _ in range(8)]
  remaining = set(range(32))
  while remaining:
      i, j = argmax_{i,j in remaining, i<j} S[i,j]  # 最高相似度对
      assign i, j to same group (greedy, 填满 group_size)
      remaining.remove(i); remaining.remove(j)
  # 剩余未分配 head 填入有空位的组

  # Step 2c: Group-wise SVD with whitening
  for group_j in groups:
      W_gj = concat([W_k[:, head*d_h:(head+1)*d_h] for head in group_j])  # [d, s*d_h]
      W_gj_whitened = apply_whitening(W_gj, X_calib)  # 参考 SVD-LLM
      U, Sigma, Vt = SVD(W_gj_whitened)
      r = R[layer_idx]  # 该层分配的 rank (按 Fisher 信息比例)
      L_gj = U[:, :r] @ sqrt(Sigma[:r, :r])  # [d, r]
      R_gj = sqrt(Sigma[:r, :r]) @ Vt[:r, :]  # [r, s*d_h]
  ```

  **Step 3 - OVC: Value Cache 压缩**（per Value projection layer）：
  ```python
  # W_v: Value projection matrix, shape [d, h*d_h]
  # Step 3a: SVD
  U, Sigma, Vt = SVD(W_v_whitened)
  r = R_v[layer_idx]
  L_v = U[:, :r] @ sqrt(Sigma[:r, :r])  # [d, r]
  R_v = sqrt(Sigma[:r, :r]) @ Vt[:r, :]  # [r, h*d_h]

  # Step 3b: Offline calibration on L_v
  # minimize E = ||L_v R_v X - W_v X||_F^2
  # Closed-form: dE/dL_v = 0 =>
  L_v = W_v @ X @ X.T @ R_v.T @ inv(R_v @ X @ X.T @ R_v.T)

  # Step 3c: Offline calibration on R_v
  # dE/dR_v = 0 =>
  R_v = inv(L_v.T @ L_v) @ L_v.T @ W_v

  # Step 3d: Matrix Fusion (offline, no inference overhead)
  W_o_new = R_v @ W_o  # fuse R_v into output projection
  # 推理时: Output = Attention(Q, K, X @ L_v) @ W_o_new
  # 无需重建 X @ L_v @ R_v，直接使用 fused output projection
  ```

  **Step 4 - 推理时**：
  ```python
  # Key 路径（每 token, 有 HSR 在线重排开销）:
  z_gj = x @ L_k_gj              # 共享 latent, [1, r_k]
  # 每 head 独立重建:
  [y_j1, y_j2, y_j3, y_j4] = z_gj @ R_k_gj  # [1, 4*d_h]
  # inverse reordering 恢复原始 head 顺序
  # 应用 RoPE 位置编码

  # Value 路径（每 token, 无重建开销 — Matrix Fusion 已消除）:
  z_v = x @ L_v                 # [1, r_v], 存入 KV cache
  # ... attention computation ...
  output = softmax(QK^T/sqrt(d)) @ (z_v) @ W_o_fused  # W_o_fused = R_v @ W_o
  ```
  压缩比 = r/n（Key）或 r/n（Value），50% KV cache 压缩比意味着 KV cache 总大小减半。

## PowerAttention: Exponentially Scaling of Receptive Fields for Effective Sparse Attention

- 属于算法pipeline的实现是什么？实验比较什么？
  POWERATTENTION 提出一种新型静态稀疏注意力模式，核心思想是让每个 token 仅关注距离为 2 的幂次的位置（power-of-2 distances），配合局部滑动窗口和 sink tokens。具体实现：(1) Power-of-2 Mask：`mask_power = (blk_qk & (blk_qk - 1)) == 0`，即只保留 query 与 key 的 block 索引差值为 2 的幂次的注意力连接；(2) Sliding Window：5-block 局部窗口，保留局部上下文信息；(3) Sink Tokens：1 block 的初始 token 作为 attention sink；(4) Causal Mask：保证自回归因果性。最终 mask = causal & (mask_window | mask_power | mask_sink)。所有模式使用 256-token blocks 对齐 GPU 内存访问。稀疏度约 94%（所有 pattern 保持一致）。理论保证：在 d 层 LLM 中，每个 token 可以访问到距离 ≤ 2^d 的所有 token（指数级感受野增长），且每个 token 的出度不超过 log n。

  实验比较：(a) POWERATTENTION vs Sliding Window / Stride Slash / Dilated Attention / LongNet / Full Attention on PG19 语言建模困惑度（4k-32k context）；(b) Passkey Retrieval 检索任务（32k 和 64k 扩展 context）；(c) RULER benchmark 13 项子任务（NIAH, Variable Tracing, Aggregation, QA）在 4k/8k/16k/32k context 下的平均分；(d) 端到端延迟对比 Full Attention 和 MInference（128K context, 1024 decode steps）；(e) 信息流探针实验（probe analysis）：在每层每位置训练 logistic classifier 检测 passkey 信息传播。

- 硬件平台是什么，配置是什么。
  NVIDIA A800 GPU。基础模型 Qwen2-7B（28 layers, 32K 原生 context length）。训练配置：SlimPajama 1B tokens continued pre-training，ChatQA 2 fine-tuning for long context tasks。POWERATTENTION 超参：5-block local window (5×256=1280 tokens)，1 block sink tokens (256 tokens)，4 个 power-of-2 slash tokens（总计每 token 最多关注 10 blocks = 2560 tokens，即 ~94% 稀疏度 @32K）。

- 模型是什么。数据集和bench分别是什么。
  模型：Qwen2-7B（Yang et al., 2024a），原生 32K context length，28 layers。数据集：(a) 预训练：SlimPajama（Soboleva et al., 2023），1B tokens continued pre-training；(b) 微调：ChatQA 2（Xu et al., 2024），含 long-range dependencies 的自然监督信号；(c) passkey retrieval 合成数据（课程学习：4K→32K，每阶段 200 steps）。Benchmarks：(a) PG19 test set（语言建模困惑度，4k/8k/16k/32k）；(b) Passkey Retrieval（32k/64k）；(c) RULER（Hsieh et al., 2024），14 子任务四类：Needle-in-a-Haystack (NIAH), Variable Tracing (VT), Aggregation (Agg.), Question Answering (QA)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文未明确提供开源 GitHub 链接。算法使用 PyTorch FlexAttention（Dong et al., 2024）实现 mask 定义，Triton（Tillet et al., 2019）结合 RingAttention（Liu et al., 2024）用于序列并行训练以扩展到更长的序列。核心张量计算 pipeline：

  **Step 1 - 构建 Attention Mask（POWERATTENTION 核心）**：
  ```python
  # q_idx [M, 1], kv_idx [1, N] 为 token 索引
  block_size = 256  # CUDA block size
  # Sink token mask: 前 block_size 个 token 全局可见
  mask_sink = kv_idx < block_size  # [1, N]
  # Sliding window mask: 5-block 局部窗口
  blk_qk = q_idx // block_size - kv_idx // block_size  # [M, N]
  mask_window = blk_qk < 5  # [M, N]
  # PowerAttention mask: 仅 block 距离为 2 的幂次
  mask_power = (blk_qk & (blk_qk - 1)) == 0  # [M, N]
  # 因果性 + 组合
  causal = q_idx >= kv_idx  # [M, N]
  mask = causal & (mask_window | mask_power | mask_sink)  # [M, N]
  ```

  **Step 2 - 感受野指数扩展原理**：
  对于距离 d（用二进制表示），d 中为 1 的 bit 最多有 log n 个。设 k₁, k₂, ..., k_m 为 d 的二进制中 1 的位置，则路径为：
  ```
  i → (i - 2^{k₁}) → (i - 2^{k₁} - 2^{k₂}) → ... → j
  ```
  路径长度 = d 的二进制表示中 1 的个数 ≤ log n。因此 d 层内可到达距离 ≤ 2^d 的所有 token。

  **Step 3 - 训练流程**：
  1. Continued pre-training: SlimPajama 1B tokens, Qwen2-7B base model
  2. Fine-tuning: ChatQA 2 data（含跨窗口的 long-range dependencies）
  3. 对于 RULER 评估，采用 hybrid architecture：每 7 层中保留 2 层 Full Attention，其余 5 层使用 POWERATTENTION

## Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention

- 属于算法pipeline的实现是什么？实验比较什么？
  NSA（Native Sparse Attention）提出一种原生可训练的稀疏注意力机制，通过三条并行注意力路径（compression、selection、sliding window）替代 Full Attention 的密集计算。核心实现：(1) Token Compression：将 key/value 序列按块（block length l=32, stride d=16）通过可学习 MLP φ（含 intra-block position encoding）压缩为块级表示 $\tilde{K}_t^{\text{cmp}}, \tilde{V}_t^{\text{cmp}}$，捕获粗粒度全局语义；(2) Token Selection：利用压缩注意力的中间 attention score $\mathbf{p}_t^{\text{cmp}}$ 推导 selection block 的重要性分数，Top-n（n=16，含 1 个初始块和 2 个局部块）保留精细 token 块（block size l'=64），实现 blockwise 连续内存访问以利用 Tensor Core；(3) Sliding Window：独立窗口分支（w=512）显式处理局部上下文，三条分支输出通过可学习门控 $g_t^c = \text{Sigmoid}(\text{MLP}(\mathbf{q}_t))$ 融合；(4) 三路 attention 使用独立的 key/value 投影矩阵，防止局部短路学习。

  实验比较：(a) NSA vs Full Attention baseline on 通用 benchmarks（MMLU, MMLU-PRO, CMMLU, BBH, GSM8K, MATH, DROP, MBPP, HumanEval — 27B 模型, 270B tokens 预训练）；(b) NSA vs H2O/InfLLM/Quest/Exact-Top on LongBench（SQA, MQA, Synthetic, Code 子类）；(c) NSA vs Full Attention on Needle-in-a-Haystack（64k context）；(d) NSA-R vs Full Attention-R on AIME 数学推理（8k/16k 生成 token 限制，DeepSeek-R1 蒸馏 SFT）；(e) 预训练 loss 曲线对比（NSA 始终低于 Full Attention）；(f) 消融：token selection 策略对比（Key-Clustering, auxiliary loss-based, heuristic parameter-free vs NSA 的 compression-based 重要性分数推导）。

- 硬件平台是什么，配置是什么。
  8-GPU NVIDIA A100 系统。预训练：27B 总参数（3B active），GQA+MoE backbone，30 layers，hidden dimension 2560，64 attention heads，GQA groups=4（每 group 16 heads），d_q=d_k=192，d_v=128，MoE 72 routed experts + 2 shared experts，top-k=6。训练数据：270B tokens of 8k-length texts，后续用 YaRN 在 32k-length texts 上 continued training + SFT。NSA 超参：compression block size=32，sliding stride=16，selected block size=64，selected block count=16，sliding window=512。

- 模型是什么。数据集和bench分别是什么。
  模型：27B 参数 Transformer（GQA + DeepSeekMoE），3B active parameters。数据集：270B tokens 预训练语料（8k 长度），10B tokens 32k-length 数学推理链用于 SFT（蒸馏自 DeepSeek-R1）。Benchmarks：(a) 通用：MMLU (5-shot), MMLU-PRO (5-shot), CMMLU (5-shot), BBH (3-shot), GSM8K (8-shot), MATH (4-shot), DROP (1-shot F1), MBPP (3-shot Pass@1), HumanEval (0-shot Pass@1)；(b) 长上下文：LongBench（MFQA-en, MFQA-zh, Qasper, HPQ, 2Wiki, GovRpt, Dur, PassR-en, PassR-zh, LCC）；(c) Needle-in-a-Haystack (64k)；(d) 推理：AIME 24（16 samples, temperature 0.7, top-p 0.95）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文为 DeepSeek-AI 出品，论文中未提供显式 GitHub 链接但建议关注 DeepSeek 官方仓库。算法 pipeline 的核心张量计算如下：

  **Step 1 - Token Compression（粗粒度压缩）**：
  输入序列 X ∈ R^{t×d}，对 key 序列 k_{:t}，每 l=32 个连续 token 为一个 block，stride d=16：
  $\tilde{K}_t^{\text{cmp}} = \{\varphi(\mathbf{k}_{id+1:id+l}) \mid 0 \le i \le \lfloor\frac{t-l}{d}\rfloor\}$
  其中 φ 为含 intra-block position encoding 的 MLP，将每个 l×d_k block 映射为 1×d_k 压缩 key。同理得到 $\tilde{V}_t^{\text{cmp}}$。Block 数 ≈ t/16。

  **Step 2 - Blockwise Selection（基于压缩注意力的细粒度选择）**：
  先计算压缩注意力分数：$\mathbf{p}_t^{\text{cmp}} = \text{Softmax}(\mathbf{q}_t^T \tilde{K}_t^{\text{cmp}} / \sqrt{d_k}) \in \mathbb{R}^{\lfloor(t-l)/d\rfloor + 1}$
  将压缩分数按空间对应关系聚合为 selection block（l'=64）的重要性分数：
  若 d|l 且 d|l'：$\mathbf{p}_t^{\text{slc}}[j] = \sum_{m=0}^{l'/d-1}\sum_{n=0}^{l-1} \mathbf{p}_t^{\text{cmp}}[\frac{l'}{d}j - m - n]$
  GQA 场景下跨 head 聚合：$\mathbf{p}_t^{\text{slc'}} = \sum_{h=1}^{H} \mathbf{p}_t^{\text{slc},(h)}$
  取 Top-n（n=16）blocks：$\mathcal{I}_t = \{i \mid \operatorname{rank}(\mathbf{p}_t^{\text{slc'}}[i]) \le n\}$
  拼接选中 block 的原始 K, V：$\tilde{K}_t^{\text{slc}} = \text{Cat}[\{\mathbf{k}_{il'+1:(i+1)l'} \mid i \in \mathcal{I}_t\}] \in \mathbb{R}^{d_k \times nl'}$

  **Step 3 - Sliding Window（局部窗口）**：
  $\tilde{K}_t^{\text{win}} = \mathbf{k}_{t-w:t}$，$\tilde{V}_t^{\text{win}} = \mathbf{v}_{t-w:t}$，w=512。

  **Step 4 - Gated Fusion（门控融合）**：
  三条路径分别计算 attention 后加权融合：
  $\mathbf{o}_t^* = \sum_{c \in \{\text{cmp},\text{slc},\text{win}\}} g_t^c \cdot \text{Attn}(\mathbf{q}_t, \tilde{K}_t^c, \tilde{V}_t^c)$
  其中 $g_t^c = \text{Sigmoid}(\text{MLP}_g(\mathbf{q}_t))$，三路使用独立 K, V 投影。

  总稀疏度：$N_t = |\tilde{K}_t^{\text{cmp}}| + |\tilde{K}_t^{\text{slc}}| + |\tilde{K}_t^{\text{win}}| \approx t/16 + 1024 + 512 \ll t$（长序列下）。

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  见 kernel调度 分层。

## Multi-head Temporal Latent Attention

- 属于算法pipeline的实现是什么？实验比较什么？
  MTLA（Multi-head Temporal Latent Attention）在 MLA 的 latent 维度压缩基础上，进一步在 temporal 维度压缩 KV cache。核心实现：(1) 用共享低秩 latent 向量 C ∈ R^{T×r}（r=256 ≪ d=512）压缩跨 head 的 key/value 信息（继承 MLA）；(2) 用 hyper-network 动态生成 temporal merging weights w_i = Sigmoid(Linear(c_i) · Linear(pe_j))，将每 s 个 temporally adjacent latent vector 加权合并为一个 compressed vector ĉ_j，使 KV cache 序列长度从 T 降至 t = ⌈T/s⌉；(3) 用 stride-aware causal mask 解决 training 时 compressed KV cache 长度与 sequence length 不匹配的问题，确保 parallel training 与 incremental inference 的 attention pattern 一致；(4) decoupled RoPE 的 key 同样沿 temporal 维压缩，仅保留 position-specific RoPE key 用于 attention score 增强。

  实验比较：(a) MTLA vs MHA vs MLA on ST（MuST-C En-De, BLEU）、text summarisation（XSum, ROUGE）、ASR（AMI, WER）、SLU（SLURP, Accuracy）；(b) MTLA vs MQA/GQA/MLA w/ SnapKV/Mamba-2 on ST；(c) MTLA 不同 temporal compression ratio s=2/3/4 的 tradeoff；(d) MTLA + FlashAttention-2 vs MHA + FlashAttention-2；(e) MTLA on LRA benchmark vs 11 种 efficient attention；(f) MTLA vs MLA on MT（WMT14 En-De）。

- 硬件平台是什么，配置是什么。
  单张 NVIDIA RTX 6000 Ada GPU（48GB 显存）。训练：ST 每 epoch ~13 分钟（78M params），text summarisation 每 epoch ~20 分钟（79M params），ASR 每 epoch ~50 分钟（67M params），SLU 每 epoch ~15 分钟（103M params）。推理：所有模型使用相同的 batch size 和 beam size，报告 inference time 和 average GPU memory usage。

- 模型是什么。数据集和bench分别是什么。
  模型：Decoder-only Transformer（encoder 输出 prepend 到 self-attention 输入，移除 cross-attention），9 层 decoder + 12 层 encoder，d=512，n_h=8，d_h=64，FFN d_ff=2048。MTLA/MLA 使用 r=256，d_h^R=32。默认 s=2。MHA/MLA/MTLA 参数量：ST 78M，summarisation 79M，ASR 67M，SLU 103M。MQA/GQA 约 74M。

  数据集：MuST-C v1.0 En-De（400h TED Talks speech translation），XSum（226K BBC news articles summarisation），AMI Meeting Corpus（100h meetings ASR），SLURP（120K utterances, 18 domains SLU）。LRA benchmark 和 WMT14 En-De 用于补充评估。

  Benchmarks：BLEU（ST）、ROUGE-1/2/L（summarisation）、WER（ASR）、Accuracy（SLU intent classification）。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/D-Keqi/mtla。基于 Fairseq 实现。

  **MTLA 单层前向—推理 pipeline（张量级，s=2, r=256, d=512, n_h=8）**：

  ```
  Step 1 - 输入: x_i ∈ R^{1×d}（单 token, incremental decoding）

  Step 2 - 多头 Query（标准 MHA）:
    q_i = x_i @ W_Q           # W_Q ∈ R^{d×(n_h·d_h)}, q_i ∈ R^{1×512}

  Step 3 - 低秩 Latent Vector（继承 MLA）:
    c_i = LayerNorm(x_i @ W_r)  # W_r ∈ R^{d×r}, c_i ∈ R^{1×256}

  Step 4 - Hyper-network 动态生成 merge weight:
    j = ceil(i / s)           # 当前所属的 compressed slot
    pe_j: positional embedding at step j
    w_i = Sigmoid(Linear(c_i) · Linear(pe_j))  # 两个 Linear 各映射 256→64, · 逐元素乘
    w_i ∈ R^{64}（实际实现中可能标量）

  Step 5 - 更新压缩 temporal-latent KV cache:
    if i % s == 1:  # 新 slot
      Ĉ = Concat(Ĉ, w_i ⊙ c_i)    # 添加新 compressed vector
    else:            # 合并到当前 slot
      Ĉ_j = Ĉ_j + w_i ⊙ c_i      # 动态融合相邻 token 信息

  Step 6 - Absorbed Attention（不显式计算 K, V）:
    # W_Q·W_K^T 预计算为吸收矩阵（训练时）
    # W_V·W_O 预计算为吸收矩阵
    scores = (x_i @ W_Q_absorbed) @ Ĉ^T / sqrt(d_h)
    # W_Q_absorbed = W_Q @ W_K^T, 形状 ∈ R^{d×(r)} → 但实际等价于 query-key 直接算
    attn_out = softmax(scores) @ (Ĉ @ W_V_absorbed)
    # Ĉ @ W_V_absorbed = Ĉ @ (W_V @ W_O), 直接输出无需中间 K, V
  ```

  **MTLA 训练—stride-aware causal mask（并行，s=2）**：

  ```
  Step 1 - Full sequence 输入: X ∈ R^{T×d}

  Step 2 - 低秩 latent: C = LayerNorm(X @ W_r) ∈ R^{T×256}

  Step 3 - Hyper-network 批量生成权重矩阵:
    PE = replicated positional embeddings: (pe_1,...,pe_1,...,pe_t,...,pe_t)
    每个 pe_j 重复 s 次，总长 T
    W = Sigmoid(Linear(PE) × Linear(C)) ∈ R^{T×T}
    chunk_mask(W)  # 仅保留对角线附近 chunk，去除跨 chunk 连接
    Ĉ' = W @ C     # 生成 extended compressed sequence ∈ R^{T×256}
    # Ĉ' 长度仍为 T（仅训练时），每 s 个 position 有相同的 compressed vector

  Step 4 - Absorbed attention with stride-aware causal mask:
    scores = (X @ W_Q_absorbed) @ Ĉ'^T / sqrt(d_h)  # ∈ R^{T×T}
    mask[n, m] = 0 if n==m or (m < n and m % s == 0) else -∞
    attn = softmax(scores + mask) @ (Ĉ' @ W_V_absorbed) ∈ R^{T×d}
  ```

  **KV cache 大小对比**（per token, n_h=8, d_h=64, s=2）：
  - MHA: 2 · d_h · n_h · l = 2 · 64 · 8 · l = 1024l elements
  - MQA: 2 · d_h · 1 · l = 128l elements
  - MLA: 9d_h · l / 2 = 288l elements
  - MTLA (s=2): 9d_h · l / (2s) = 2.25d_h · l = 144l elements（接近 MQA 水平）
  - MTLA (s=4): 9d_h · l / 8 = 72l elements

- 属于算法pipeline的实现是什么？实验比较什么？
  MoM（Mixture-of-Memories）是一种新的线性序列建模架构，核心创新是用多个独立的 memory state 替代传统线性模型中单一固定大小的 memory state，并通过 router 网络将每个 token 路由到 top-k 个 memory state 进行选择性更新，最后将多个 memory 加权混合后得到输出。该方法受生物神经元 theta-gamma 振荡机制和 MoE 思想的启发。MoM 支持多种 memory update 机制（Linear Attn、RetNet、GLA、DeltaNet、G-DeltaNet、TTT、Titans、Mamba2、HGRN2、RWKV6/7，详见 Table 1）。论文选择 Gated DeltaNet 作为主要的 memory update 方法。默认配置：4 个 memory state + 1 个 shared memory，top-2 激活（activation ratio=0.5）。Router 使用线性层计算 importance scores，经 softmax+TopK 后归一化得到 routing weights。Shared memory 始终被激活，用于捕获全局长期信息。硬件高效实现：将 tokens 按 routing 结果重排序分组 → concat 为 varlen 输入 → Triton kernel 计算 → 输出拆分回原始顺序 → 加权求和。训练使用 auxiliary loss（类似 Switch Transformer 的 load balancing loss）。

  实验比较：MoM vs Transformer++、RetNet、HGRN2、GLA、GSA、Gated DeltaNet。实验维度包括：(1) Recall-intensive tasks（FDA, SWDE, SQuAD, NQ, TriviaQA, Drop）；(2) Commonsense reasoning（WikiText ppl, LAMBADA ppl/acc, ARC-e, ARC-c, HellaSwag, PIQA, WinoGrande）；(3) LongBench long-context benchmark（单文档QA、多文档QA、摘要、少样本学习、合成任务、代码补全）；(4) 推理效率对比（inference time, GPU memory vs sequence length）；(5) Length extrapolation（2K → 32K perplexity）；(6) Memory scaling（memory 数量 1-8，activation ratio 0.25/0.5）；(7) Ablation on auxiliary loss scale 和 shared memory；(8) MoM-Transformer hybrid（每 7 层 MoM + 1 层 Transformer）；(9) Fairness comparison（等 activated params 和等 memory capacity）。两个模型规模：380M（24 layers, d=1024, 15B tokens）和 1.3B（24 layers, d=2048, 100B tokens）。

- 硬件平台是什么，配置是什么。
  32 张 NVIDIA A800 GPU。380M 模型训练约 10 小时，1.3B 模型训练约 6 天。使用 AdamW optimizer（lr=3e-4, cosine schedule, weight decay=0.01, gradient clipping=1.0）。380M 模型 batch size=0.5M tokens, warmup=0.25M tokens；1.3B 模型 batch size=2M tokens, warmup=1B tokens。硬件高效实现基于 Triton kernel（varlen operations）。

- 模型是什么。数据集和bench分别是什么。
  模型：MoM 380M（24 layers, hidden size=1024, hidden ratio=3, 4 memory states + shared memory, top-2 activation）和 MoM 1.3B（24 layers, hidden size=2048, 同样 memory 配置）。Baseline 模型包括：Transformer++（RoPE + GLU）、RetNet、HGRN2、GLA、GSA、Gated DeltaNet。

  数据集：训练使用 SlimPajama 数据集（627B tokens 的清洗版本），采样 100B tokens。Tokenizer: Mistral tokenizer。Length extrapolation 使用 Fineweb 数据集。

  Benchmarks：(1) Recall-intensive: FDA, SWDE, SQuAD, NQ, TriviaQA, Drop（max length 2K tokens）；(2) Commonsense reasoning: WikiText ppl, LAMBADA ppl/acc, ARC-easy, ARC-challenge, HellaSwag, PIQA, WinoGrande（使用 lm-evaluation-harness 评估）；(3) LongBench: 中英双语长文本理解 benchmark。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/OpenSparseLLMs/MoM 和 https://github.com/OpenSparseLLMs/Linear-MoE。

  **MoM 单层前向 pipeline（张量级，top-k=2, M=4 memories + shared）**：

  ```
  输入: X ∈ R^{T×d}  (T=sequence length, d=hidden dim)
  参数: Router W_g ∈ R^{d×M}, memory-specific W_k^m, W_v^m ∈ R^{d×d}, W_q ∈ R^{d×d}
       M: total memory count (默认4), k: activated count (默认2)

  Step 1 - Router:
    scores = softmax(X @ W_g)        # scores ∈ R^{T×M}
    topk_scores, topk_indices = TopK(scores, k)  # 选择 top-k 个 memory
    g_t = topk_scores / sum(topk_scores)  # 归一化 importance weights ∈ R^{T×k}

  Step 2 - Hardware-Efficient Reordering (varlen):
    按 routing 结果将 tokens 分组到各 memory bucket:
    I_{b,m} = {t : token t routed to memory m for batch b}
    将同 bucket tokens concat 为 varlen sequence:
    X̃ = concat([X[I_{1,1}], ..., X[I_{B,M}]])

  Step 3 - Per-Memory QKV Projection & Update:
    对于每个 memory m:
      q̃_u = X̃_u @ W_q             # shared query projection
      k̃_u^m = X̃_u @ W_k^m         # memory-specific key
      ṽ_u^m = X̃_u @ W_v^m         # memory-specific value
      用 Gated DeltaNet update:
        M_t^m = a_t (I - (k_t^m)^T k_t^m) M_{t-1}^m + b_t (k_t^m)^T v_t^m
      或用其他 update rule（Table 1）

  Step 4 - Output Computation:
    对每个 token 的每个激活 memory:
      o_t^m = q_t @ M_t^m          # per-memory output
    加权混合:
      ỹ_t = Σ_{m in activated} g_t^{(m)} · o_t^m
    最终输出:
      o_t = activation(norm(ỹ_t)) @ W_o
  ```

  关键设计点：(1) Memory states 是 R^{d×d} 矩阵，类似 linear attention 的 KV 外积累积；(2) 非激活 memory 保持上一时刻状态不变，避免当前输入干扰；(3) Shared memory 始终被所有 token 激活，通过全局信息提升长程依赖；(4) 稀疏激活仅作用于 K/V projection（占参数量小），MLP 部分保持不变；(5) 计算复杂度：training O(n)（每 memory 线性），inference O(1)（constant memory state per step）。

## Mixture of Sparse Attention: Content-Based Learnable Sparse Attention via Expert-Choice Routing

- 属于算法pipeline的实现是什么？实验比较什么？
  MoSA（Mixture of Sparse Attention）是一种基于 Expert-Choice Routing 的内容感知可学习稀疏注意力算法。核心创新：每个 attention head 配有一个可学习的 router（权重矩阵 W^r ∈ R^h），通过 sigmoid 计算每个 token 的 selection score，再用 TopK 选出每个 head 专属的 k 个 token，仅对这些 token 计算 Q、K、V 投影和 attention 矩阵。复杂度从 dense attention 的 O(T²) 降至 O(k²+T)。节省的 FLOPs 用于增加注意力头数，提升 head 专业化程度。关键细节：(1) router 输出 r_topk 通过 diag(r_topk)·A 乘到 attention 输出上，使 router 可通过梯度下降学习；(2) casual mask 适配 token 原始位置索引 I；(3) RoPE 旋转角度也基于原始位置而非子集位置；(4) 混合模型：4 个 dense head 保持训练稳定性，其余 head 用 MoSA 替换；(5) IsoFLOP 实验中每个 head 始终包含序列的第一个 token（attention sink）。

  实验比较 MoSA vs Dense baseline、Fixed Sparse Attention（基于位置的固定稀疏，stride=ρ，k=T/ρ 个 token）、Routing Transformer（online K-means 聚类，ρ 个簇各含 k 个 token）。共四个模型规模：Tiny(28M)、Small(113M)、Medium(210M)、Large(516M)。在 IsoFLOP 设定（逐步增加 sparsity ρ=T/k，用节省 FLOPs 增加 head 数）下评估 perplexity；在 perplexity-matched 设定下评估 wall-clock time、GPU memory、KV-cache 大小。还测试了长序列（T=8192，结合 local attention）和 6 个下游 zero-shot 任务。

- 硬件平台是什么，配置是什么。
  Tiny/Small/Medium 模型：单张 NVIDIA A100 GPU。Large 模型：两张 A100 GPU。纯 PyTorch 实现（无专用 CUDA kernel），使用 einsum、scatter、gather 操作。训练使用 Adam optimizer，lr=0.00025，gradient clipping norm=0.25，linear warmup 4k steps。

- 模型是什么。数据集和bench分别是什么。
  模型：Tiny（6 layers, h=512, FFN=2048, 9 heads, 28M params）、Small（9 layers, h=1024, FFN=4096, 9 heads, 113M）、Medium（18 layers, h=1024, FFN=4096, 9 heads, 210M）、Large（27 layers, h=1280, FFN=5120, 16 heads, 516M）。所有模型 head hidden size=64，基于 Pre-LN Transformer + RoPE。

  数据集：C4 训练集，100k batches，batch size=64，sequence length T=1024（约 6.5B tokens）。Tokenizer: SentencePiece，vocab size=8000，基于 sub-word units。

  Benchmark：C4 测试集 perplexity（主指标）；下游 zero-shot：LAMBADA、WinoGrande、BLiMP、HellaSwag、PIQA、AI2ARC（6 个任务）。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/piotrpiekos/MoSA。基于 PyTorch 实现，使用 einsum/scatter/gather。

  **MoSA 单层前向 pipeline（张量级）**：

  ```
  输入: X ∈ R^{T×h}  (T=sequence length, h=hidden dim)
  参数: router W^r_i ∈ R^h, Q/K/V projections W^Q_i/W^K_i/W^V_i ∈ R^{h×h'}, W^O_i ∈ R^{h'×h}
        h' = head dim (默认 64), k = 每个 head 选择的 token 数

  对于每个 head i (i=1..H):
    Step 1 - Token Selection:
      r = σ(X @ W^r_i)              # r ∈ R^T, sigmoid 逐元素
      r_topk, I = TopK(r, k)        # r_topk ∈ R^k, I ∈ {0..T-1}^k

    Step 2 - Gather selected tokens:
      X^s = X[I]                     # X^s ∈ R^{k×h}, 按索引 gather

    Step 3 - Q/K/V projections (仅对 k 个 token):
      Q = X^s @ W^Q_i               # Q ∈ R^{k×h'}
      K = X^s @ W^K_i               # K ∈ R^{k×h'}
      V = X^s @ W^V_i               # V ∈ R^{k×h'}

    Step 4 - Causal mask (基于原始位置索引 I):
      M_{a,b} = 0 if I_a ≥ I_b else -∞    # M ∈ R^{k×k}

    Step 5 - Sparse Attention:
      A = softmax(Q @ K^T / √h' + M) @ V  # A ∈ R^{k×h'}

    Step 6 - Router gate and output projection:
      X^o = diag(r_topk) @ A @ W^O_i       # X^o ∈ R^{k×h}, router gradient 通过乘法传递

    Step 7 - Scatter back to full sequence:
      Y_j = X^o_{idx} if j = I_{idx}, else 0   # Y ∈ R^{T×h}

  最终输出: Y = Σ_{i=1..H} Y_i
  ```

  **FLOPs 对比**：
  - Dense head: FLOP = 8hh'T + 4h'T²
  - MoSA head: FLOP = 8hh'k + 4h'k² + 2hT + h'k  (routing overhead: 2hT from scoring + h'k from gating)
  - Fixed sparse head: FLOP = 8hh'k + 4h'k²
  - Routing Transformer head: FLOP = ρ(6hh'k + 4h'k²) + 2h'T  (must compute all Q/K/V/O for all T tokens, Q=K in auto-regressive)

  **混合模型构建**：保持 4 个 dense head，其余 head 用 MoSA 替换。MoSA head 数量 = max H 使得 total FLOPs ≤ baseline FLOPs。sparsity ρ = T/k。例如 Tiny 模型 ρ=64 时：4 dense heads + 505 MoSA heads, 总参数 423M, perplexity 16.39（vs dense 22.46, -27%）。

  **Perplexity-matched 资源优化**：固定 ρ=32（Large: ρ=16），逐步增加 MoSA head 数直到 perplexity 匹配 dense baseline。结果：wall-clock time -7.3%~-12.9%，memory -1.6%~-10.0%，KV-cache -51.1%~-69.5%。

## MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention

- 属于算法pipeline的实现是什么？实验比较什么？
  MInference 是一种免训练的稀疏计算算法，通过动态稀疏注意力加速长上下文 LLM 的 pre-filling 阶段。核心方法分为三步：(1) 离线识别每个 attention head 的最优稀疏模式（A-shape、Vertical-Slash、Block-Sparse 三种之一）；(2) 推理时根据分配的稀疏模式和具体输入，在线估计并动态构建稀疏索引（dynamic sparse mask）；(3) 仅对稀疏索引内的区域执行注意力计算，其余位置置零（通过 c(1-M) 大常数掩码）。目标是 $\min |A(M) - A_{\text{dense}}|$ 且 $\min t_{\text{sparse}}(M) + t_{\text{overhead}}(M)$。

  实验比较 MInference vs 五种免训练稀疏注意力 baseline：StreamingLLM（对应 A-shape 模式，1K global + 4K local window）、StreamingLLM w/ dilated（1K global + 8K dilated, interval=1）、StreamingLLM w/ strided（1K global + 2K local + 4K dilated）、InfLLM（128 global + 8K local window）、Ours w/ static（Vertical-Slash 和 Block-Sparse 头使用静态稀疏索引）。所有 baseline 仅在 pre-filling 阶段执行稀疏计算，decoding 阶段保持 dense 计算。评估在 InfiniteBench（10 任务，平均 214K context）、RULER（13 任务，4K-128K）、Needle In A Haystack（1K-1M）、PG-19（语言建模，100K tokens）上进行。

- 硬件平台是什么，配置是什么。
  单张 NVIDIA A100 80GB GPU（bfloat16 格式）。分布式实验使用 8x A100 GPU（tensor parallel + context parallel 可进一步将 1M pre-filling 延迟降至 22 秒）。Kernel 基于 Triton 语言实现，可轻松移植到 H100 或 MI300X。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA-3-8B-Instruct-262K（gradientai/Llama-3-8B-Instruct-Gradient-262k）、LLaMA-3-8B-Instruct-1048k（gradientai/Llama-3-8B-Instruct-Gradient-1048k）、GLM-4-9B-1M、Yi-9B-200K、Phi-3-Mini-128K、Qwen2-7B-128K、LLaMA-3-70B-Instruct-262K。
  
  Benchmark：InfiniteBench（En.Sum/En.QA/En.MC/En.Dia/Zh.QA/Code.Debug/Math.Find/Retr.PassKey/Retr.Num/Retr.KV 共 10 任务，~214K tokens 平均，3992 样本）、RULER（Retrieval/Multi-hop Tracing/Aggregation/QA 四类 13 任务，4K-128K 六档 context 长度，每档 2600 样本）、Needle In A Haystack（scaled to 1M context，750 样本）、PG-19（1000 个 >100K tokens 的随机样本，perplexity 评估）。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  开源：https://aka.ms/MInference（GitHub）。基于 PyTorch + FlashAttention + Triton + PIT（动态稀疏编译器）实现。

  **算法 pipeline（三步流程）**：

  **Step 1 — 离线 Kernel-Aware Sparse Pattern Search（Algorithm 1）**：
  对每个 attention head，在 kernel-aware search space 中搜索最优稀疏模式及其参数：
  ```
  输入: Q, K, V ∈ R^{S×d_h}, patterns p, search space σ, target FLOPs t
  # Step 1a: 构建 kernel-aware search space
  for i ← 1 to |σ|:
      t_i ← FLOPs_in_kernel(σ_i)       # 真实 GPU kernel FLOPs
      while |t_i - t| > ε:
          σ_i ← ChangeSpace(σ_i, p_i)   # 调整参数逼近 target FLOPs
          t_i ← FLOPs_in_kernel(σ_i)
      ρ ← ρ ∪ σ_i                      # 加入搜索空间
  # Step 1b: 基于 reference example 选择最优模式
  y ← Softmax(QK^T/√d)                # Dense attention 作为 ground truth
  for i ← 1 to |ρ|:
      y_i ← SparseAttention(QK^T/√d, ρ_i)
  p_best ← argmin(|y_i - y|, p_best)  # 最小化 attention output 误差
  ```
  Search space 设置：A-shape → {(1024, 4096)}（1K global + 4K local）；Vertical-Slash → {(30, 2048), (100, 1800), (500, 1500), (3000, 200)}；Block-Sparse → {100}（top-100 blocks）。搜索使用一条 30K KV retrieval 合成样本，约 15 分钟/A100。同一模型的不同 context 版本（262K vs 1M）复用相同最优配置。

  **Step 2 — 在线动态稀疏索引近似（Algorithm 2/3）**：

  *Vertical-Slash Head*（Algorithm 2）：
  ```
  输入: Q, K, V ∈ R^{S×d_h}, k_v, k_s
  # 使用最后 last_q=64 个 query 估计注意力分布
  Â ← softmax(Q_{[-last_q:]} K^T / √d + m_causal)
  # 提取 top-k_v 垂直列索引（沿垂直方向求和）
  i_v ← argtopk(sum_v(Â), k_v)
  # 提取 top-k_s 斜线索引（沿斜线方向求和）
  i_s ← argtopk(sum_s(Â), k_s)
  # 构建稀疏索引
  i_vs ← sparseformat(i_v, i_s)
  # 最终稀疏注意力
  A ← softmax(sparse(QK^T, i_vs) / √d)
  y ← sparse(AV, i_vs)
  ```

  *Block-Sparse Head*（Algorithm 3）：
  ```
  输入: Q, K, V ∈ R^{S×d_h}, k_b
  # Mean pooling 降采样 Q, K (block_size=64)
  Q̂ ← MeanPooling(Q, 64)
  K̂ ← MeanPooling(K, 64)
  # 块级注意力近似
  Â ← softmax(Q̂K̂^T / √d + m_causal)
  # 提取 top-k_b 块
  i_b ← argtopk(Â, k_b)
  i_b ← sparseformat(i_b)
  # 最终稀疏注意力
  A ← softmax(sparse(QK^T, i_b) / √d)
  y ← sparse(AV, i_b)
  ```

  *A-shape Head*：静态稀疏掩码——始终保留初始 global tokens（1K）+ 局部 window tokens（4K），无需在线估计开销。

  **Step 3 — 稀疏注意力计算**：
  使用针对三种模式优化的 GPU kernel 执行稀疏注意力。详见 kernel调度 条目。

  **张量计算示例（LLaMA-3-8B, 128K context, Vertical-Slash head）**：
  ```
  Q, K, V ∈ R^{131072×128}（S=128K, d_h=128）
  # 估计阶段：仅使用最后 64 个 query
  Q_est = Q[-64:]                                   # [64, 128]
  Â = softmax(Q_est @ K^T / √128)                   # [64, 131072]
  i_v = argtopk(Â.sum(dim=0), k_v=30)               # 30 条垂直列
  i_s = argtopk(Â 沿斜线求和, k_s=2000)              # 2000 条斜线
  # 稀疏计算：仅计算 i_vs 索引内的 QK^T 和 AV
  A_sparse = softmax(Q @ K[i_vs]^T / √128)          # [131072, |i_vs|]
  y = A_sparse @ V[i_vs]                            # [131072, 128]
  ```
  稀疏度（sparsity）：128K context 下约 96.8%，1M context 下 >95%。理论加速比 $s_p = S / (2B × k_b)$（Block-Sparse），实际端到端 speedup：100K → 1.8×, 300K → 4.1×, 500K → 6.8×, 1M → 10×。

## Less Is More: Fast and Accurate Reasoning with Cross-Head Unified Sparse Attention

- 属于算法pipeline的实现是什么？实验比较什么？
  LessIsMore 是一种免训练的稀疏注意力机制，专为长程推理设计。核心发现：推理模型中 token 重要性是全局属性而非 head-local 属性——(1) 跨 head 空间局部性：不同 attention head 的 token 重要性高度重叠；(2) 时间近邻局部性：最近 token 的高 attention 比例在 decoding 全程稳定。基于此提出 Cross-Head Unified Sparse Attention (CUSA)：各 head 独立提案 top-k token，通过 UnionFlatten 聚合为统一候选集后全局排名，保留 top K·(1-r) token 并固定比例 r=0.25 分配给近邻窗口。Token 选择仅在一层执行（token selection layer），产生的统一索引 ρ 跨后续所有层复用，摊销选择开销。实验比较 LessIsMore vs 免训练方法（TidalDecode、Quest、StreamingLLM）和需训练方法（SeerAttention-r）在 AIME-24/25、GPQA-Diamond、MATH500 上的推理准确率与生成长度，以及 LongBench 和 Needle-in-the-Haystack 上的长上下文能力。

- 硬件平台是什么，配置是什么。
  准确率评估：NVIDIA RTX A5000 GPU（HuggingFace 实现，32K token 生成需 >20 分钟）。效率评估：单张 NVIDIA A100 80GB GPU（DeepSeek-R1-Distill-Llama-8B，端到端 TBT 和 kernel 级延迟）。Serving 集成：单张 NVIDIA A5000 GPU（SGLang + FlashInfer）。

- 模型是什么。数据集和bench分别是什么。
  模型：DeepSeek-R1-Distill-Llama-8B（GQA）、Qwen3-4B/8B/14B（GQA）——四个推理模型；LongChat-7B-v1.5-32k（MHA）——长上下文模型；Llama-3-8B-Instruct-Gradient-1048k、Llama-3.1-8B-Instruct（non-reasoning 模型）。Benchmark：AIME-24/25（64 traces/problem）、MATH500（8 traces）、GPQA-Diamond（16 traces）、Needle-in-the-Haystack（10K/32K/100K）、LongBench（MultiFieldQA/Qasper/HotpotQA/TriviaQA/PassageRetrieval-en）。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/DerrickYLJ/LessIsMore

  **三种层类型**（Algorithm 1）：
  1. Full Attention Layers（Layer 0-1）：o = FullAttention(q, C[:])，确保早期上下文建模准确。
  2. Token Selection Layer（如 Layer 12）：计算 P = q·C.K^T，各 head 独立 TopK → 跨 head 统一聚合：
     ρ_head = TopKIndices(P[:, :-(K·r)], k=K·(1-r))
     ρ_unified = UnionFlatten(ρ_head)
     ρ = ρ_unified[:K·(1-r)] ∪ Recent(K·r)
  3. Sparse Attention Layers（其余层）：o = SparseAttention(q, C[ρ])，复用 token selection layer 生成的统一索引 ρ。

  **GQA 下的 CUSA 张量计算**（以 DeepSeek-R1-8B 为例，hq=32, hkv=8, r=4）：
  1. Q = hW_Q (32 query heads), K = hW_K (8 KV heads), V = hW_V (8 KV heads)
  2. 对每个 KV group g（4 query heads 共享 1 KV head）：
     P_g = Q[4g:4(g+1)] @ K_g^T / √d     # [4, 1, L_kv]
     按 query 维度 max pooling：P_g_agg = max(P_g, dim=0)
     每个 query head 独立 TopK：idx_h = TopKIndices(P_g_agg[h], k=K·(1-r))  for h=1..4
     (GQA 下 query head 独立选择但共享 KV head，这些 head 选择通过后续 UnionFlatten 跨 head 统一)
  3. 全局统一：idx_all = unique(flatten([idx_h for all 32 heads]))
     按 attention score 排序取 top：idx_hist = sort_by_score(idx_all)[:K·(1-r)]
  4. 近邻窗口：idx_recent = [L_kv-K·r, ..., L_kv-1]
  5. ρ = idx_hist ∪ idx_recent（所有 32 query heads 共享同一 ρ）

  **低频重选验证**（图 4）：LessIsMore 仅 Layer 2 选择 vs 每层都选，attention recall 几乎相同（~95% vs ~96%），而 head-to-head / randomized top-k 的方法从 ~96% 降至 ~65%/72%。因为 CUSA 的全局 token 重要性是跨层稳定的。

  **关键效果**：AIME-24 上 Qwen3-8B 以 2K token budget（87.5% sparsity）达 73.8% 准确率 vs Full Attention 74.5%，生成长度 15.8K vs Full Attention 14.8K——几乎无长度膨胀。SeerAttention-r 需 2K budget 仅达 58.2% 且生成 19.8K token。

## LASP-2: Rethinking Sequence Parallelism for Linear Attention and Its Hybrid

- 属于算法pipeline的实现是什么？实验比较什么？
  LASP-2 是一种针对线性注意力（Linear Attention）的序列并行（Sequence Parallelism, SP）算法。核心实现：将长序列切分为 T 个 chunk 分布到 W 个设备上，各设备并行计算 Q_t, K_t, V_t 及 local memory state M_t = K_t^T V_t（形状 d×d，与序列长度无关），然后通过**单次 AllGather 集合通信**将所有设备的 M_t 汇聚到所有设备，各设备本地累加得到全局 M_{1:T} = Sum([M_t]_1^T)，最后本地计算 O_t = Q_t M_{1:T}。对有 causal mask 的自回归任务，LASP-2 采用计算分解（computation decomposition）：intra-chunk 部分保持 quadratic 左乘计算 O_{t,intra} = [(Q_t K_t^T) ⊙ Ψ] V_t，inter-chunk 部分用线性右乘 O_{t,inter} = Q_t M_{1:t-1}，且 AllGather 通信可与 intra-chunk 计算 overlap。LASP-2H 将同样的 AllGather 通信范式扩展到标准 softmax attention 的 Context Parallelism（AllGather K_t, V_t 后本地计算 attention）。实验比较 LASP-2 vs Megatron-SP、Ring Attention、LASP-1 在 throughput (tokens/s)、scalability（序列长度 2K-2048K，GPU 数 16-128）、convergence performance（多种 linear attention 变体）等维度。

- 硬件平台是什么，配置是什么。
  最多 16 台 DGX-A100 服务器，每台 8 张 A100 GPU（共 128 卡），NVSwitch 互联提供 600 GB/s GPU 间带宽。PyTorch 2.3.1 + CUDA 12.1 + cuDNN 8.9.2 + NCCL 2.20.5。基于 Megatron-Core 0.9.0 开发，Triton 2.3.1 加速 GPU 上的线性注意力计算，FlashAttention-2 作为标准 attention 实现。

- 模型是什么。数据集和bench分别是什么。
  模型：Linear-Llama3-1B（16 层，将 Llama3 的标准 softmax attention 替换为多种线性注意力模块：Basic Linear Attention、Lightning Attention、Retention、GLA、Based、Rebased），hidden dim d=2048，16 heads。Hybrid 模型：每第 4 层保留标准 softmax attention（1/4 hybrid）。额外评估 RoBERTa + Basic Linear Attention 在双向语言建模任务上。
  数据集：SlimPajama（627B tokens 全量），实验使用训练集第一个 chunk 的 50B tokens 子集。Llama3 tokenizer。GPT-style 自回归语言建模（带 causal mask）。评估指标为 training loss 和 validation loss。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/OpenSparseLLMs/Linear-MoE（LASP-2 作为 Linear-MoE 系统的 SP 子模块发布）

  **算法 Pipeline（LASP-2 without Masking，基于 Algorithm 1）**：

  **输入**：序列 X ∈ R^{N×d}，分布式 world size W，SP size T=W，将 X 切分为 T 个 chunk [X_t]_1^T。

  **Step 1 - 并行计算 local Q/K/V/M**（每个设备并行）：
  ```
  Q_t = X_t W_Q    # shape: [C, d]
  K_t = X_t W_K    # shape: [C, d]
  V_t = X_t W_V    # shape: [C, d]
  M_t = K_t^T V_t  # shape: [d, d]  ← 与 chunk 长度 C 无关！
  ```

  **Step 2 - AllGather 通信**：
  ```
  [M_1, M_2, ..., M_T] = AllGather([M_1, M_2, ..., M_T])
  # 每个设备获得全部 T 个 memory states，通信量 = T × d × d × BH
  ```

  **Step 3 - 本地累加**（所有设备并行，递归执行）：
  ```
  M_{1:T} = M_1 + M_2 + ... + M_T   # 使用递归：M_{1:t} = M_{1:t-1} + M_t
  # 缓存 M_{1:T} 到 HBM 用于 backward pass
  ```

  **Step 4 - 本地输出计算**：
  ```
  O_t = Q_t M_{1:T}   # shape: [C, d]
  ```

  **有 Mask 版本（LASP-2 with Masking, Algorithm 2）**：
  - **Intra-chunk**：O_{t,intra} = [(Q_t K_t^T) ⊙ Ψ] V_t（quadratic 左乘，但仅限 chunk 内，可并行）
  - **Inter-chunk**：O_{t,inter} = Q_t M_{1:t-1}（线性右乘，其中 M_{1:t-1} = PrefixSum([M_1, ..., M_{t-1}])）
  - **关键优化**：AllGather（line 7）与 intra-chunk 计算（line 8）可在不同 CUDA stream 上 overlap
  - **最终输出**：O_t = O_{t,intra} + O_{t,inter}

  **Backward Pass（Algorithm 3, 4）**：
  - 计算 dM_t = Q_t^T dO_t
  - AllGather([dM_t]_1^T) 汇聚梯度
  - 无 mask：dM_{1:T} = Sum([dM_{t+1}]_T)，推导 dQ_t, dK_t, dV_t
  - 有 mask：intra/inter 分别计算后合并 dQ_t = dQ_{t,intra} + dQ_{t,inter}（类似 forward）

  **与 LASP-1 的关键差异**：
  - LASP-1：ring-style P2P 逐设备顺序收发 M_t，共 2(W-1) 个通信步骤，每次传 BHd^2 数据
  - LASP-2：单次 AllGather，仅 2 个通信步骤（forward + backward），通信量同为 BHd^2 但并行度大幅提升

  **LASP-2H 混合模型扩展（Algorithm 7）**：
  - Linear Attention 层：同上，AllGather M_t（d×d 大小）
  - Standard Attention 层：AllGather K_t, V_t（C×d 大小），本地计算 Softmax(Q_t K^T / √d) V

- 属于算法pipeline的实现是什么？实验比较什么？
  KVzip 是一种 query-agnostic（查询无关）的 KV cache 淘汰算法。核心实现是基于上下文重建（context reconstruction）的 KV pair 重要性评分：将 "Repeat the previous context:" prompt + 原始 context chunk 拼接后通过 LLM forward pass，利用 teacher-forced decoding 模拟上下文重建过程，对每个 KV pair 取其在重建过程中收到的最大 cross-attention score 作为重要性分数 S ∈ R^{L×H×n_c}，随后按 non-uniform head-budget allocation 保留 top r% 高分的 KV pairs，淘汰低分 pairs。支持 context-dependent eviction（per-context 压缩，更高压缩比）和 context-independent eviction（预计算 head-level score，部署时零开销）两种模式。实验比较 KVzip 与 query-aware 方法（H2O、SnapKV、PyramidKV）以及 head-level 淘汰方法（DuoAttention）在 KV cache budget ratio 0.1-1.0 下的多查询/单查询场景性能，涵盖 12 个 benchmark 数据集。

- 硬件平台是什么，配置是什么。
  单卡 NVIDIA A100 80GB GPU，Bfloat16 精度。FlashAttention-2 加速注意力计算。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA3.1-8B（GQA group=4）、Qwen2.5-7B-1M（GQA group=7）、Qwen2.5-14B-1M、Gemma3-12B（hybrid attention: global + sliding window 1:5）、LLaMA3.1-3B、LLaMA3-8B-W8A8KV4（QServe W8A8KV4 量化）。
  数据集/Benchmark：SQuAD、GSM8K（数学推理）、Needle-in-a-Haystack / NIAH（检索）、SCBench 9 个任务（En.QA、En.MultiChoice、Retr.KV、Retr.Prefix-Suffix、Math.Find、En.Summary 等，含 retrieval-intensive、contextual understanding、high context redundancy 三类）、RULER benchmark、SCBench multi-task datasets（Mix.Sum+NIAH、Mix.RepoQA+KV）。上下文长度 100 到 170K tokens（Qwen2.5 tokenizer），评估主要在多查询 query-agnostic 框架下（Figure 1c）。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/snu-mllab/KVzip
  
  **算法 Pipeline（基于论文 Algorithm 1）**：
  
  **Step 1 - Prefill**：将输入 context c（n_c tokens）通过 f_LM 前向传播，生成完整 KV cache KV_c，共 L×H×n_c 个 KV pairs（L 层，H 个 KV head，使用 GQA）。
  
  **Step 2 - Chunking**：将 c 划分为 T = ⌈n_c/m⌉ 个 chunk，每 chunk 固定大小 m=2K（与上下文长度、模型、任务无关，Section C.1 验证影响 <2%）。
  
  **Step 3 - 逐 Chunk 重要性评分**（for t = 1,...,T）：
  - 构造 input：
    - t=1: `"Repeat the previous context:" + c_1`
    - t≥2: `"Repeat the previous context starting with <c_{t-1} last 8 tokens>:" + c_t`
  - 将 input（长度 n_in = n_prompt + m）通过 f_LM 前向，使用 KV_c 作为 KV cache。
  - 对每层 l、每 KV head h：
    - 获取 Query: Q_{l,h} ∈ R^{G×n_in×d}（G 为 grouped-query size）
    - Subsample Key: K̄_{l,h} ∈ R^{(m+n_in)×d}（从 KV_c 中取出当前 chunk 对应部分 + input 自身的 keys）
    - 计算注意力: A_{l,h} = Softmax(Q_{l,h} K̄_{l,h}^T) ∈ R^{G×n_in×(m+n_in)}
    - 切片 KV_c part: Ā_{l,h} = A_{l,h}[:,:,:m] ∈ R^{G×n_in×m}
    - 沿 query 维度取 max: S_{l,h,t} = max_{g=1..G, i=1..n_in} Ā_{l,h}[g,:,i] ∈ R^{H×m}
  
  **Step 4 - 聚合得分**：将所有 chunk 得分拼接为完整得分 S ∈ R^{L×H×n_c}。
  
  **Step 5 - 淘汰（Non-uniform head-budget）**：保留所有 KV pairs 中 top r% 最高 S 值的 pairs。system prompt 的 KV pairs 始终保留。r=1.0 为全量 cache。
  
  **Step 6 - Decoding**：使用压缩后的 KV_{c,evicted} 进行 FlashAttention 解码，享受降低后的内存占用和注意力延迟。
  
  **复杂度**：O(m·n_c) 线性于上下文长度（vs 标准 prefill 的 O(n_c²/2)），压缩开销约 2× prefill（Figure 8b）。峰值内存 O(m²) 恒定。chunked scoring 的 FlashAttention 总 FLOPs 为 O(n_c² + n_c·m/2)。
  
  **上下文无关变体（Context-independent eviction）**：对每 head 取 S_head[l,h] = max_i S[l,h,i]，使用单个 88K-token 英文书样本预计算 head-level 重要性分数。部署后无需任何压缩开销，直接应用 DuoAttention 的 head-level KV eviction 策略。性能略低于 context-dependent mode 但显著优于 DuoAttention 的原版 head-score 优化（后者需数小时 8-GPU 优化，KVzip 仅需数次 forward pass 一分钟内完成）。

## NACL: A General and Effective KV Cache Eviction Framework for LLMs at Inference Time

- 属于算法pipeline的实现是什么？实验比较什么？
  NACL 提出一种混合 KV cache 淘汰框架，在 encoding 阶段一次性完成全局最优淘汰（而非逐 token step-by-step 贪心淘汰），包含两个核心策略：(1) **PROXY-TOKENS EVICTION**：选取输入末尾的任务相关 token（如用户问题）作为 proxy tokens，利用 proxy tokens 对所有 prefix token 的全局 attention score 求和作为 token 重要性评分（F_score = Σ_{x_p∈P} Softmax(A(x_p, *))），保留 top-C_p 高分 token。相比 H2O 的全体 token 累加 attention score（引入冗余信息）和 MSRNN 的单当前 token attention（信息不足），proxy tokens 提供更精准的任务相关重要性估计。(2) **RANDOM EVICTION**：将 PROXY-TOKENS EVICTION 的评分经 Softmax 归一化后作为概率分布 P_prompt，从该分布中采样 C_r 个 token 保留，每个 attention head 和每层使用不同随机种子。这种 per-head per-layer 的多样化采样使信息在更多维度上被保留（LLaMA-7B 32层×32头，budget=20% 时 token 在至少一个 head 中保留概率达 99.92%）。最终 KV cache budget C = C_p + C_r，proxy tokens 默认约 10% budget。NACL 将淘汰建模为一次 encoding phase 全局操作，时间复杂度从 O(p+T) 降至 O(1)（long-context 下 T ≪ p）。

  实验比较：(a) short-text tasks (5-shot/25-shot)：NACL vs Attention Sink、H2O、MSRNN 在 lm-eval-harness 七个任务上的 accuracy；(b) long-text tasks (LongBench)：NACL vs Scissorhands、H2O、MSRNN 在 budget 10%/20%/30% 下的七个任务 accuracy；(c) KV cache 内存使用 vs sequence length（NACL 20% vs H2O 20%）；(d) 消融实验：移除 PROXY-TOKENS EVICTION（-28.1 short-text）、移除 RANDOM EVICTION（-1.2 short-text / -9.2 long-text）、uniform 采样替代 attention-score 采样（-0.8 short-text / -1.1 long-text）、step-by-step global eviction 替代 one-eviction（-1.3 short-text）、per-layer 替代 head-wise eviction（-2.1 short-text / -2.7 long-text）。

- 硬件平台是什么，配置是什么。
  单张 NVIDIA A100 80GB GPU（bfloat16 精度）。FlashAttention-2 用于加速注意力计算。Reduce Attention Scores CUDA kernel 实现兼容 FlashAttention-2 的 128K long-text 推理。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA2-7B-base、LLaMA2-7B-Chat（decoder-only Transformer，32 layers，32 heads per layer，d=4096）。
  
  数据集/Benchmark：
  - 短文本（lm-eval-harness）：PiQA（5-shot: 319 tokens）、COPA（118 tokens）、OpenBookQA（97 tokens）、Winogrande（160 tokens）、SciQA（508 tokens）、ARC-Easy（296 tokens）、ARC-Challenge（239 tokens）。25-shot 设置下 token 数约 5×。
  - 长文本（LongBench，4K context）：PassageRetrieval-Zh、PassageRetrieval-En、RepoBench-P、HotpotQA、NarrativeQA、TriviaQA、QMSum。
  - 辅助：perplexity 在 OpenBookQA 上计算。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2024-NACL

  **NACL 单层 encoding phase 前向 pipeline（张量级，LLaMA2-7B, 32 heads, d=4096, C=20%, C_p=6%, C_r=12%, proxy_budget≈10%）**：

  ```
  Step 1 - Prefill Q/K/V:
    x_prompt ∈ R^{p×4096}（p=4K tokens）
    for each head h (0..31):
      Q_h = x_prompt @ W_Q^h     # W_Q^h ∈ R^{4096×128}, Q_h ∈ R^{4096×128}
      K_h = x_prompt @ W_K^h     # K_h ∈ R^{4096×128}
      V_h = x_prompt @ W_V^h     # V_h ∈ R^{4096×128}

  Step 2 - Attention Scores（全量）:
    A_h = Q_h @ K_h^T / sqrt(128)     # A_h ∈ R^{4096×4096}, causal masked

  Step 3 - Proxy Tokens Selection:
    # 默认取输入末尾 ~10% 的 token 作为 proxy tokens
    # 实际应用中 proxy tokens = 用户问题部分（位于 prompt 末尾）
    P = {p * 0.9, ..., p-1}           # |P| ≈ 410 tokens (10% of 4K)
    P_h = P                           # head h 的 proxy token 索引

  Step 4 - PROXY-TOKENS EVICTION (F_score):
    # 仅用 proxy tokens 行求和作为 token importance
    F_score = Σ_{x_p ∈ P_h} Softmax(A_h[x_p, :])    # ∈ R^{4096}
    # F_score[j] 表示 token j 对所有 proxy tokens 的综合重要性

  Step 5 - Top-K selection for proxy eviction:
    R = x_prompt \ P_h                              # 非 proxy token 集合
    C_p = 0.06 * p ≈ 246 tokens                     # proxy eviction budget
    u_score = TopK(F_score[R], C_p)                 # 选非 proxy 中最高分 C_p 个
    u_score = u_score ∪ P_h                         # proxy tokens 默认保留

  Step 6 - RANDOM EVICTION:
    # 从 F_score 构建概率分布
    P_prompt_h = Softmax(F_score)                   # ∈ R^{4096}, 概率分布
    C_r = 0.12 * p ≈ 492 tokens                     # random eviction budget
    u_random_h ~ Multinomial(P_prompt_h, C_r, seed=h)  # per-head 不同 seed

  Step 7 - 合并保留集:
    S_encoding^h = u_score ∪ u_random_h             # 共 C = 738 tokens (20%)
    K_cache^h = K_h[S_encoding^h]                   # [738, 128]
    V_cache^h = V_h[S_encoding^h]                   # [738, 128]
    # 淘汰率 80% (4K → 738)

  Step 8 - Generation Phase（逐 token decoding）:
    for each new token z_t:
      for each head h:
        K_cache^h = [K_cache^h, z_t @ W_K^h]       # 追加新 token KV
        V_cache^h = [V_cache^h, z_t @ W_V^h]
        if t % m == 0:                              # 每 m 步淘汰一次
          A_t = (z_t @ W_Q^h) @ K_cache^h^T / sqrt(128)
          S_t^h = Eviction(A_t, C)                  # 重复 Step3-7
          K_cache^h, V_cache^h = K_cache^h[S_t^h], V_cache^h[S_t^h]
  ```

  **与 baselines 的关键差异**：
  - H2O: F_score = Σ_{all tokens} Softmax(A[i, :])（全量累加 attention → 冗余信息 + attention bias）
  - MSRNN: F_score = Softmax(A[current_token, :])（仅当前 token → 信息不足）
  - NACL: F_score = Σ_{x_p∈P} Softmax(A[x_p, :])（proxy tokens 子集 → 精准 + 抗 bias）+
    head-wise RANDOM EVICTION（概率采样 → 增加信息多样性）

  **KV cache 压缩效果**：
  - LLaMA2-7B, batch=4, seq_len=32K, bf16: 64GB → NACL 20% ≈ 12.8GB (5× reduction)
  - NACL 20% short-text avg: 63.8 vs Full 64.6 (-0.8), H2O 20%: 60.3 (-4.3) — 80% improvement
  - NACL 20% long-text avg: 30.8 vs Full 31.5 (-0.7), H2O 20%: 28.6 (-2.9) — 76% improvement
  - NACL 30% long-text: PR-Zh=6.8 (H2O=3.7), PR-En=9.0 (H2O=5.0) — NACL 在 passkey retrieval 上显著优于 H2O

## DAM: Dynamic Attention Mask for Long-Context Large Language Model Inference Acceleration

- 属于算法pipeline的实现是什么？实验比较什么？
  DAM 提出一种免微调的动态稀疏注意力机制，通过两阶段流程为每个 attention map 生成自适应稀疏 mask，保留跨层、跨 head 的异构 attention 模式。Stage 1：冻结的预训练模型在 Pattern Capture Length (PCL) 范围内处理输入序列，提取完整 attention map → 对累积的注意力分数计算均值 → Box-Cox 变换（λ=0.5）放大中小注意力值 → 归一化（减去全局最小值） → 以阈值 τ=0.3 进行二值化生成 "true mask" → 通过结构模式匹配（匹配 score 阈值 μ=0.8）识别对角线模式 P_diag,r 和垂直模式 P_vert,c → 对超过 PCL 的长序列，将匹配到的结构模式外推生成 "extended mask"。Stage 2：将生成的动态稀疏 mask 在 softmax 之前应用于 attention score，mask 位置设为 -∞ 使注意力概率为 0，将 FLOPs 复杂度从 O(L²) 降至 O(sL)（s 为每个 query 保留的平均 key 数，s ≪ L）。实验比较 LongEval（长度 3K-104K tokens）和 LV-Eval（16K-256K tokens，含单跳 QA 如 cmrc-mixup 和多跳 QA 如 dureader-mixup）上的检索准确率/评分，以及在 LLaMA 3.2 1B/3B 和 Vicuna 7B 上与 Full Attention（Original）、FlashAttention、MoA、StreamingLLM、H2O 的 GPU 内存、吞吐量、平均时延对比。

- 硬件平台是什么，配置是什么。
  4× NVIDIA A100 40GB（LongEval 评测）；2× NVIDIA H100 80GB（LV-Eval 评测）；1× NVIDIA A100 40GB（效率评测，即表 1 的 Memory/Throughput/Latency）。原始 LLaMA 3.2 3B 在单卡 A100 40GB 上处理超 4K 序列即 OOM。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA-3.2-1B-Instruct、LLaMA-3.2-3B-Instruct（用于可扩展性分析）；Vicuna-7B（效率评测）。Attention map 捕获数据集：Multi-News（大规模多文档摘要数据集）。Benchmark：LongEval（100 data items per length level，以行级 key-value 检索精度衡量）、LV-Eval（含 single-hop: cmrc-mixup, multifieldqa-en-mixup, multifieldqa-zh-mixup；multi-hop: dureader-mixup, loogle-CR-mixup, loogle-MR-mixup, hotpotwikiqa-mixup, lic-mixup）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/HanzhiZhang-Ulrica/DAM。算法 pipeline 如下：

  **Stage 1 — Mask 生成（离线，基于冻结模型）**：
  1. 在 Multi-News 数据集上运行冻结的 LLaMA 模型，对长度不超过 PCL（L=512）的序列提取各层各 head 的完整 attention map A_{ℓ,h,i,j}。
  2. 计算跨 batch 的累积注意力与 token 位置计数，求均值：\bar{A}_{ℓ,h,i,j} = A_{ℓ,h,i,j} / (C_{ℓ,h,i,j} + ε)。
  3. Box-Cox 变换（λ=0.5）：B_{ℓ,h,i,j} = (X^{0.5} - 1) / 0.5，其中 X = max(\bar{A}, ε)。
  4. 归一化：B^*_{ℓ,h,i,j} = B_{ℓ,h,i,j} - min_{all}(B)，得到 \tilde{A}_{ℓ,h,i,j}。
  5. True mask 生成（阈值 τ=0.3）：m_{i,j} = 1 if \tilde{A}_{ℓ,h,i,j} ≥ τ else 0。
  6. 结构模式匹配（阈值 μ=0.8）：对每个 true mask M_{ℓ,h}，与模式池 P = {P_diag,r} ∪ {P_vert,c} 中的每个模式 P_k 计算匹配分数 γ_k = Σ_{i,j} M·P_k / Σ_{i,j} P_k。若 γ_k ≥ μ，该模式被匹配。
  7. 扩展 mask：\tilde{M}_{ℓ,h} = Σ_{P_k: γ_k≥μ} P_k，二值化：\tilde{M}^{(i,j)} = 1 if 该位置任一匹配模式为 1 else 0。

  **Stage 2 — 推理时 mask 应用**：
  若 S ≤ L：直接使用 true mask M_{ℓ,h}。
  若 S > L：将 PCL 范围内的 M_{ℓ,h} 与外推的扩展 mask 组合为 S×S 的完整 mask \tilde{M}_{ℓ,h}。
  在 softmax 前应用：A'_{ℓ,h} = (QK^T / √d_k) ⊙ \tilde{M}_{ℓ,h}，mask 位置设为 -∞，最终 O' = softmax(A')V。

  DAM 不修改模型权重，不引入额外训练，与基于 tile 的 GPU 执行兼容（未来可与 FlashAttention 等 memory-efficient kernel 融合）。

## InfiniteHiP: Extending Language Model Context Up to 3 Million Tokens on a Single GPU

- 属于算法pipeline的实现是什么？实验比较什么？
  提出一种免训练的模块化层次化 token 剪枝算法（Modular Hierarchical Token Pruning），通过多阶段剪枝动态消除不相关上下文 token，结合动态 RoPE 调整实现超长上下文外推（OOL generalization）。算法核心：(1) 将输入序列划分为固定大小的 chunk（l_c，默认 Stage 1: 256, Stage 2: 32, Stage 3: 8），在每个 chunk 内通过层次化 top-1 选择（SelectRep，O(log₂ l_c) 时间）选出代表性 token；(2) 利用代表性 token 估计每个 chunk 的注意力分数（跨 head max-pooling），保留 top-K 个 chunk（K = k/l_c），其余丢弃；(3) 通过堆叠 3 个剪枝 stage（N=3），逐步将候选 key 从全量缩减到 2K-4K tokens（3K/5K window preset），最终输出 block sparse attention mask；(4) 动态 RoPE 调整：前 3 层使用 Chunk-indexed RoPE（每 chunk 一个 position ID），后续层使用 Relative-style RoPE（层次化选择中左右分支获得不同偏移），Block Sparse Attention 阶段使用 StreamingLLM-style RoPE；(5) 稀疏注意力 mask 缓存：利用 mask 的时序局部性，以 refresh interval（默认 16/8/4 step）周期性更新各 stage mask，大幅降低 decoding 开销。实验比较 LongBench（平均 32K tokens）、∞Bench（>100K tokens）、RULER 上的 NLU 性能，在 Llama 3 8B、Mistral 0.2 7B、Gemma2 9B、EXAONE 3/3.5 7.8B 上与 FA2（truncated）、Dynamic-NTK、SelfExtend、LM-Infinite、StreamingLLM、H2O、InfLLM、HiP Attention 对比。

- 硬件平台是什么，配置是什么。
  评测使用单卡：(1) NVIDIA RTX 4090 24GB（PCIe 4.0 x8），搭配 AMD Ryzen 7950X 16 核 CPU、128GB DDR5 5600MHz RAM、Ubuntu 22.04.4 LTS、GPU Driver 535.171.04；(2) NVIDIA L40S 48GB（AWS g6e.48xlarge 节点）。长上下文吞吐量测试因显存限制采用估计值（SRT 基线在 1M 上下文需约 64GB KV cache，3M 需约 192GB KV cache，超出单卡容量）。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama 3 8B Instruct、Llama 3.1 8B（AWQ 量化 + FP8 KV cache）、Mistral 0.2 7B Instruct、Gemma2 9B、EXAONE 3 7.8B、EXAONE 3.5 7.8B、DeepSeek R1 Distilled Qwen2 14B（Passkey 测试）。Benchmark：(1) LongBench（含 NQA、Qasper、MFQA、HQA、2WMQ、MSQ、GR、QMS、MN、TREC、TQA、SAMS、PC、PR、RBP、LCC 共 16 子集，平均长度 ~32K）；(2) ∞Bench（含 RPK、RN、RKV、MF、MC、QA、SUM 子集，平均长度 >100K，额外含 En.MC/En.QA 用于 OOL 评测）；(3) RULER（含 NIAH 1-3 SK/MK/MV/MQ、VR、CWE、FWE、QA1/2）；(4) Passkey Retrieval（评估 KV cache offloading 延迟）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：(1) hip-attention 核心库：https://github.com/DeepAuto-AI/hip-attention/；(2) SGLang 集成：https://github.com/DeepAuto-AI/sglang/。算法 pipeline 如下：

  **阶段 1 — Multi-stage Context Pruning（生成稀疏 attention mask）**：
  
  给定 query Q ∈ R^(H×T_q×d)，key K ∈ R^(H×T_kv×d)，设 n_sink=256, n_stream=1024, N=3 stages：
  
  1. 初始化 I_m^(0) = [n_sink, ..., b_q^(1)*m - n_stream]（排除 sink/streaming 保持因果性）
  
  2. 对每个 pruning stage i=1..N，参数 S^(i) = (b_q^(i), l_c^(i), k^(i))：
     - 将 query 分为 b_q^(i) 大小的 block：q_{h,m}^(i) = Q_{h, m*b_q : (m+1)*b_q-1}
     - 将上阶段 key indices I_m^(i-1) 分为 l_c^(i) 大小的 chunk：C_{m,j}
     - 对每个 chunk j，每个 head h，执行 SelectRep(q_{h,m}, C_j)：
       * 层次化二分搜索，log₂(l_c) 次迭代
       * 每次迭代：取左右两个分支的第一个 token，ApplyRopeK后用与 q 的点积计算分支分数
       * 选择高分分支继续，最终收敛到单个代表 token 索引 r_{h,m,j}
     - 估计 chunk 分数：s_{m,j} = max_{h=1..H, t=1..b_q} [(q̃_{h,m})_t^T · k̂_{h, r_{h,m,j}}]
     - 保留 top K = k^(i)/l_c^(i) 个 chunk：I_m'^(i) = ∪_{j∈T_m} C_{m,j}（T_m = argtop_K(s_{m,j})）
  
  3. 最终输出稀疏 key indices I_m^(N)，用于 Block Sparse Attention

  **阶段 2 — Block Sparse Attention（基于 mask 的稀疏注意力计算）**：
  - 使用 Triton kernel，Combine PagedAttention + FlashAttention (prefill) / FlashDecoding (decoding)
  - 仅对 I_m^(N) 中的选中 key token 计算完整注意力
  - 注意力 mask 缓存：每 n_refresh^(i) 步更新一次第 i stage 的 mask（默认 16/8/4）
  
  **阶段 3 — Dynamic RoPE for OOL Generalization**：
  - 前 3 层 (l ≤ 3)：ApplyRopeQ_l(q) = ApplyRope(q, p[min(i_orig, l_c + n_stream)])（Chunk-indexed RoPE）；ApplyRopeK_{l,j}(k) = ApplyRope(k, p[c_orig])（chunk 索引作为 position）
  - 第 4 层及以上 (l > 3)：ApplyRopeQ_l(q) = ApplyRope(q, p[n_stream+1])（Relative-style RoPE）；ApplyRopeK_{l,j}(k) = ApplyRope(k, p[j-1])（分支索引作为 position，j ∈ {1,2}）
  - BSA 阶段：选中 key（含 sink+streaming）按原始顺序排列，尾部 token 获得与当前 query 相同的 position ID（StreamingLLM-style）

  **复杂度分析**：
  - 初始 pruning stage: O(T_q * T_kv)（分 chunk+SelectRep 每个 chunk 仅 O(log l_c) 次点积）
  - 后续 pruning stages: O(T_q)（候选 key 数已缩减至常数 k^(i)）
  - BSA: O(T_q * k^(N))，其中 k^(3) = 2K-4K，远小于 T_kv

  **关键超参数**（默认 3K preset）：
  - n_sink=256, n_stream=1024, N=3
  - Stage 1: b_q=64, l_c=256, k=32K
  - Stage 2: b_q=64, l_c=32, k=8K
  - Stage 3: b_q=64, l_c=8, k=2048 (4096 for l≤3)
  - refresh interval: (16, 8, 4)

## Dynamic-LLaVA: Efficient Multimodal Large Language Models via Dynamic Vision-language Context Sparsification

- 属于算法pipeline的实现是什么？实验比较什么？
  Dynamic-LLaVA 提出一种动态视觉-语言上下文稀疏化框架，通过两个可学习的轻量预测器（image predictor 和 output text predictor）在 MLLM 推理的 prefill 和 decoding 阶段分别稀疏化视觉 token 和语言 token。具体实现分为三部分：
  1. **稀疏化推理**（Sec 3.3.2）：在 prefill 阶段，image predictor 基于第 l 层（l=2）解码器输出的图像 token 特征，通过 `argmax(P^I(S_l^I))` 生成二值 mask M^I，丢弃不重要的图像 token（保留比例 r^I ≈ 20%）。在 decoding without KV cache 模式下，output predictor 类似地稀疏化输出文本 token 集合（保留比例 r^OT ≈ 50%）。在 decoding with KV cache 模式下，output predictor 对每个输出 token 生成一个二值决策 M^{OT}_{N^{OT}_l} ∈ {0,1}，决定是否将其 KV activations 加入 KV cache，实现在线 KV cache 压缩。
  2. **端到端稀疏化训练**（Sec 3.3.3）：训练时使用 MaskedSoftmax（Eq. 7）替代标准 Softmax，通过二值 mask 矩阵 G 隔离非必要 token 对重要 token 的影响，同时保持自回归并行训练。使用 Gumbel-Softmax + Straight-Through Estimator（STE）解决 argmax 不可微问题（Gumbel temperature τ 从 1 指数衰减至 0.1）。加入约束正则项 R（Eq. 10）约束 mask 的保留比例接近预定义的 r^I 和 r^OT，仅对输出长度 ≥ LEN^{OT}=50 的样本进行语言稀疏化训练。
  3. **批量并行稀疏化推理**（Appendix A.1）：通过 Left Padding + TopkArgmax 策略实现 mini-batch 内的并行预测和变长 token 集合的 GPU 批量计算。
  实验比较：(a) 视觉理解 benchmark 上与 SoTA 视觉上下文稀疏化方法（FastV、LLaVA-PruMerge+、VoCo-LLaMA、LLaVA-HiRED、IVTP、TRIM、SparseVLM）和高效视觉投影方法（TokenPacker、LLaVA-Resampler、C-Abstractor、Pixel-Shuffle、LDP-v2）比较准确率；(b) 生成能力 benchmark（LVIS-VQA single/multi-round、ShareGPT4V-VQA）上与 Random/Structure 静态丢弃、H2O KV cache 压缩、FastV+H2O 组合比较 PPL 和 METEOR；(c) 实际推理效率比较 prefill 时间、decoding 时间、GPU 内存（batch size=8）。

- 硬件平台是什么，配置是什么。
  训练：8× NVIDIA A100 (80G)。推理效率测量：1× NVIDIA A100 (80G)，batch size=8。延迟测量：1× A100 (80G)，batch size=1。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaVA-1.5-7B（LLM: Vicuna-7B, 32 decoder layers, d=4096）、LLaVA-1.5-13B（LLM: Vicuna-13B, 40 decoder layers, d=5120）。Vision encoder: CLIP ViT-L/14@336px，生成 576 个 image token。
  训练数据集：656K Mixture Dataset（与 LLaVA-1.5 一致），仅使用含图像的数据训练 predictor。
  视觉理解 Benchmark：VQAv2、GQA、VizWiz、SciQA、TextVQA、POPE、MME、MMBench (en)、SEED (image)、MM-Vet、MMVP、RealWorldQA、CVBench-2D。
  生成能力 Benchmark：LVIS-VQA single-round（1000 样本，答案长度 >100 words）、LVIS-VQA multi-round（1000 样本，平均答案 >300 words，>7 轮交互）、ShareGPT4V-VQA single-round（178 样本，caption ≥300 words，平均输出 >1000 tokens）。
  评估指标：准确率（vision understanding）、PPL（生成流畅度）、METEOR（生成相似度）、TFLOPs（计算量）、GPU Memory（KV cache 开销）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/Osilly/dynamic_llava。算法 pipeline 如下：

  **预训练阶段（基于 LLaVA-1.5 开源权重，冻结 vision encoder 和 projector，更新 LLM + predictors）**：

  ```
  # 超参数: l=2, r^I=0.2, r^OT=0.5, LEN^{OT}=50, λ=100
  # Gumbel temperature τ: 1.0 → 0.1 (指数衰减)
  # LLM lr=5e-6, Predictors lr=2e-4, batch_size=64

  for each training step:
      # Forward: 前 l 层 full computation
      S_l^I, S_l^T, S_l^OT = LLM_layers_1_to_l(image_tokens, text_tokens, output_tokens)

      # Image predictor (Vision Transformer blocks + MLP 512→256→128→2)
      D^I = P^I(S_l^I)                           # [N_l^I, 2]
      D^{I†} = GumbelSoftmax(D^I, τ)             # 松弛决策
      M^I = argmax(D^{I†})                        # [N_l^I], 二值mask

      # Output text predictor (MLP only)
      D^{OT} = P^{OT}(S_l^{OT})                   # [N_l^{OT}, 2]
      D^{OT†} = GumbelSoftmax(D^{OT}, τ)
      M^{OT} = argmax(D^{OT†})                    # [N_l^{OT}], 二值mask

      # 合并 mask: M = M^I ∪ {1}^{N^T} ∪ M^{OT}
      # 生成 mask 矩阵: G = {M}^{N_l} ∈ R^{N_l×N_l}, diag(G)=1

      # 修改 attention: 替换 Softmax 为 MaskedSoftmax
      # MaskedSoftmax(X_{i,j}, G) = exp(X_{i,j})*G_{i,j} / Σ_k exp(X_{i,k})*G_{i,k}

      # 后续层用 full token set 参与计算但 attention 被 mask 隔离
      loss = LM_loss + λ * R
      # R = ||sum(M^I)/|S_l^I| - r^I|| + ||sum(M^{OT})/|S_l^{OT}| - r^{OT}|| (if |S_l^{OT}| ≥ LEN^{OT})

      # Backward: STE 梯度 ∂L/∂D^{I†} → ∂L/∂M^I 直接传至 D^{I†}
  ```

  **推理阶段（prefill，Eq. 5）**：
  ```
  S_l^I, S_l^T = LLM_layers_1_to_l(image_tokens, text_tokens)
  D^I = P^I(S_l^I)                          # [N_l^I, 2]
  M^I = argmax_j(D^I)                       # j=0 discard, j=1 keep
  S_l^{I*} = {S_{l,i}^I | M_i^I == 1}      # 保留 ~20% image tokens
  S_l^{P*} = S_l^{I*} ∪ S_l^T               # 后续层仅处理缩减后的 token 集
  ```

  **推理阶段（decoding w/ KV cache，Eq. 6）**：
  ```
  Q, K, V = W^Q S_{l,N^{OT}}^{OT}, W^K S_{l,N^{OT}}^{OT}, W^V S_{l,N^{OT}}^{OT}
  O = W^O Attention(Q, S_l^K ∪ K, S_l^V ∪ V)
  M^{OT}_{N^{OT}} = argmax(P^{OT}(S_{l,N^{OT}}^{OT}))
  if M^{OT}_{N^{OT}} == 1: S_l^K ∪= K, S_l^V ∪= V   # 保留 KV
  else:                    S_l^K ∪= ∅, S_l^V ∪= ∅     # 丢弃 KV
  S_{l+1,N^{OT}}^{OT} = FFN(O)
  ```

  **实际效果（LLaVA-1.5-13B，1×A100 80G，batch=8，生成 2K tokens）**：
  Prefill Time: 0.83s (baseline) → 0.37s (Dynamic-LLaVA)，Decoding Time: 4117s → 2382s，GPU Memory (decode 2K): 58G → 42G。Image token 减少约 80%，Decoding TFLOPs 减少约 50%，GPU memory 减少约 50%。

- 属于算法pipeline的实现是什么？实验比较什么？
  提出一种基于加性向量量化（Additive Quantization）的 KV cache 压缩方法 CommVQ，将每个 token 的 key/value 向量作为整体进行量化，而非逐标量量化。核心设计包括三部分：(1) **编码器**：一个轻量级神经网络（线性层 + 激活函数 + 线性层），使用 Gumbel-Softmax 保证端到端可微，将 d 维 key/value 向量编码为长度为 N_c 的二进制序列 s_i ∈ {0,1}^{N_c}；(2) **码本解码**：通过 s_i × C 的矩阵乘法从码本 C ∈ R^{N_c×d} 中重建原始向量 t̂_i = s_i C，编码器和码本通过最小化 MSE loss 联合训练；(3) **RoPE-可交换码本**：利用 RoPE 矩阵的 2×2 块对角结构，设计满足 C = [[x, y], [-y, x]] 形式的 2×2 子码本 C_K^{jl}，使其与 RoPE 旋转矩阵 R_i^j 满足交换律 R_i^j C_K^{jl} = C_K^{jl} R_i^j，从而将 self-attention 中的 key-query 计算改写为 α_i = Σ_j,l (q^j R_t^j) C_K^{jlT} R_i^{jT} [s_i^j = l]^T，使得 (q^j R_t^j) C_K^{jlT} 可跨所有 token 复用，大幅降低解码开销。Key 码本通过 EM 算法在 FineWeb-Edu 校准集上优化（含 soft clustering center assignment 和 temperature annealing 技术）。Value 量化沿用原加法量化方法，但重排矩阵乘法为 Softmax(A) S_V C_V 以降低计算量。实验比较 LongBench、InfiniteBench、GSM8K、Needle-in-a-Haystack benchmark 上与 KIVI、KVQuant、VQLLM 在 2-bit 和 1-bit 量化下的准确率，以及量化误差 MSE。
- 硬件平台是什么，配置是什么。
  NVIDIA H100-80GB GPU（主要实验平台）；NVIDIA RTX 4090（验证单卡推理可行性）。LLaMA-3.1 8B 模型在 RTX 4090 上以 128K context length 运行。
- 模型是什么。数据集和bench分别是什么。
  主要模型：LLaMA-3.1-8B-Instruct（128K context）。额外模型：LLaMA-2-7B（32K context, Together.ai 版本）、Mistral-7B-v0.3（32K context）。校准/训练集：FineWeb-Edu 子集。Benchmark：LongBench（8 个子任务：Qasper, QMSum, MultiNews, TREC, TriviaQA, SAMSum, LCC, RepoBench-P）、InfiniteBench（10 个子任务：R.PK, R.Num, R.KV, En.Sum, En.QA, En.MC, En.Dia, Code.D, Math.F）、GSM8K、Needle-in-a-Haystack。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/UMass-Embodied-AGI/CommVQ。算法 pipeline 如下：

  1. **离线训练阶段**：在 FineWeb-Edu 校准集上运行 LLaMA 模型，收集各层的 K/V cache 向量。对 Key cache 使用 EM 算法按 2D 子空间训练 RoPE-可交换码本（Algorithm 1），对 Value cache 使用梯度下降训练编码器和码本（MSE loss + Gumbel-Softmax）。
  
  2. **Prefill 阶段**：正常计算 QKV 投影，将生成的 K/V 向量输入编码器 E，得到量化表示 s_i（Key 的 s_i^j ∈ {0,...,N_c'-1}^2，Value 的 s_i ∈ {0,1}^{N_c}），存储量化后的 KV cache S 替代原始 FP16 KV cache。
  
  3. **Decoding 阶段（Key 解码与 attention 融合）**：
     伪代码：
     ```
     q = x @ W_Q                              # [1, d]
     q_rope = apply_rope(q)                   # [1, d]
     # 预计算，可跨所有 token 复用
     q_pre = (q_rope @ C_K^T)                 # [1, N_c] 等价于 (q^j R_t^j) C_K^{jlT} 对所有 j,l
     # 对每个已缓存的 token i
     for i in range(num_cached_tokens):
         # R_i^T s_i^T 的计算，利用 RoPE 旋转的稀疏性
         alpha[i] = dot(q_pre, rope_rotate(s_i))
     # Softmax 后的 attention reordering
     attn_weights = softmax(alpha / sqrt(d))   # [1, N]
     # Value 解码重排
     output = (attn_weights @ S_V) @ C_V       # [1, N_c] @ [N_c, d] -> [1, d]
     ```
     关键优化：(qR_t) C_K^T 仅计算一次，后续每个 token i 仅需 R_i^T s_i^T 的轻量旋转操作。Value 解码由 O(d N_c N + dN) 降至 O(N_c N + d N_c)。
  
  4. **压缩率计算**：Avg. bit = N_c/d（Value），Avg. bit = R·log₂(N_c')/g（Key），总 KV cache 由 B×N×d×2×16 bits 降至 B×N×N_c×2×1 bits。LLaMA-3.1-8B（d=1024）的配置：
     - 2-bit: N_c=2048, R=21, N_c'=64, g=64, Avg. bit=2.00
     - 1-bit: N_c=1024, R=11, N_c'=64, g=64, Avg. bit=1.03

## CompressKV: Semantic Retrieval Heads Know What Tokens are Not Important Before Generation

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 CompressKV，一种针对 GQA-based LLM 的 KV cache 压缩框架，包含三个核心组件：(1) **Semantic Retrieval Head (SRH) 识别**：不依赖传统的 top-1/top-k 精确命中标准，而是通过聚合 attention head 在整个 answer span 上的 attention scores 来评估 head 的语义检索能力，公式为 SemanticRetrievalScore(h) = Σ_{t=1}^N I[y_t ∈ A] Σ_{j∈A} a_{t,j}^h，其中 y_t 是生成 token，A 是 answer span，a_{t,j}^h 是 head h 在 token j 上的 attention weight。得分越高说明 head 越能捕捉语义信息而非仅 copy-paste 行为。(2) **SRH 驱动的 Token 选择**：每层选取 top-k SRH（默认为 top-4），将这些 head 的 attention score 矩阵在 observation window 上求和并在 token 维度上做 1D average pooling（kernel size=5），取平均后选出 top-N 高 attention 的 token 保留其 KV cache，其余 token 的 KV cache 被 evict。同一层内所有 head 共享统一的 token 索引集。(3) **Error-Aware 层级自适应 Cache 分配**：离线阶段在 LongBench 上模拟极端压缩（每层仅 32 tokens，约 0.3% 容量），计算每层的压缩误差 e^(l) = Σ_t ||O_comp,t^l - O_full,t^l||_F / (||O_full,t^l||_F + ε)，跨数据集归一化平均后得到最终重要性分数 ẽ^(l)。在线推理时按 ẽ^(l) 比例分配 cache budget，并设置 per-layer 上下界 [m=32, M=3×B_per-layer] 防止极端分配。实验比较 LongBench（16 个数据集）和 Needle-in-a-Haystack 上不同 KV cache budget（128/256/512/1024/2048）下与 StreamingLLM、SnapKV、PyramidKV、CAKE 的准确率，以及 masking-based ablation 比较 SRH 与传统 Retrieval Head 的重要性，和端到端延迟/峰值内存对比。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 GPU（单卡），用于延迟和峰值内存评估。FlashAttention-2 默认启用。SRH 识别和误差分数计算可离线完成。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-3.1-8B-Instruct（GQA，128K context）和 Mistral-7B-Instruct-v0.3（GQA，32K context）。Benchmark：(1) LongBench——16 个数据集，分 6 类：Single-Doc QA (NarrativeQA, Qasper, MultiFieldQA-en)、Multi-Doc QA (HotpotQA, 2WikiMultihopQA, MuSiQue)、Summarization (GovReport, QMSum, MultiNews)、Few-shot Learning (TREC, TriviaQA, SAMSum)、Synthetic (PassageCount, PassageRetrieval-en)、Code (LCC, RepoBench-P)；(2) Needle-in-a-Haystack (NIAH)——评估长上下文中检索隐藏答案的能力，测试 1K-128K 长度。评估使用 greedy decoding 保证公平比较。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/TUDa-HWAI/CompressKV.git。算法 pipeline 如下：

  1. **离线阶段 — SRH 识别**：在 LongBench 验证集上运行完整模型，对每层每个 head h 计算 SemanticRetrievalScore(h)。取每层 top-4 SRH 作为 token 选择的依据。
  
  2. **离线阶段 — 误差分数计算**：
     伪代码（Algorithm 1 核心逻辑）：
     ```
     for each layer l:
         # 模拟极端压缩：每层仅保留 32 tokens 的 KV cache
         O_full^l = Attention(Q^l, K_full^l, V_full^l) @ W_O^l
         O_comp^l = Attention(Q^l, K_comp^l, V_comp^l) @ W_O^l
         e^l = Σ_t ||O_comp,t^l - O_full,t^l||_F / (||O_full,t^l||_F + ε)
         ẽ^l = normalize(average_over_datasets(e^l))
     
     # 分配算法
     B_i = m  for all layers i
     R = B_total - Σ B_i
     B_i = clip(B_i + round(ẽ_i * R), m, M)
     # 贪心调整剩余/超出预算至满足 Σ B_i = B_total
     ```
  
  3. **在线 Prefill 阶段**：
     伪代码（Token 选择）：
     ```
     for each layer l:
         select top-k SRH heads (e.g., k=4)
         for each selected head h:
             # observation window 内的 attention scores
             A_h = attention_scores[h, :, -W:]  # [seq_len, W]
             # sum over observation window
             S_h = sum(A_h, dim=-1)  # [seq_len]
             # 1D average pooling (kernel=5)
             S_h = avg_pool1d(S_h, kernel_size=5)
         # average across selected SRH
         S = mean([S_h for h in selected_heads])
         # select top-N highest-scoring tokens
         keep_indices = topk(S, N)
         # retain KV pairs for keep_indices only
         K_cache = K[keep_indices]
         V_cache = V[keep_indices]
     ```
     参数：observation window = 8 tokens，pooling kernel size = 5，average pooling。
  
  4. **在线 Decoding 阶段**：使用压缩后的 KV cache 进行 attention 计算，新生成 token 的 KV pair 追加到 cache 中。cache 大小受 per-layer budget B_i 限制。

  5. **压缩率计算**：在 128K context 下，256 KV cache entries 仅占全量 KV cache 的 0.07%，仍能达到 NIAH 上 90% 的 full-cache 准确率。LongBench 上 19% KV entries 保持 >99% full-cache 性能。

## KV-Compress: Paged KV-Cache Compression with Variable Compression Rates per Attention Head

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 KV-Compress，一种基于 PagedAttention 的 KV cache 压缩方法，支持可变压缩率（variable-head-rate compression）和分块级 eviction。核心算法设计：(1) Query-Group Compression：针对 GQA 模型，将每个 key 的 eviction 指标在所有属于该 key 的 query group 的 queries 上聚合（Sum_{h in H_k}），而非先 repeat KV cache 到 query head 数量再压缩，使得同样 max-cache-size C 下 KV-Compress 实际持有 1/r 的 KVs（r 为 query head 与 KV head 之比，Llama-3/Mistral 中 r=4），实现 4x 额外压缩；(2) Paged KV Block Eviction（MoveCache 算法）：在 block size b 的 paged KV cache 中，先确定每 head 的 (be)^{th} 最小 metric 值 m(h,e)，再跨 head 排序候选 block eviction，按 budget E_s 选出总 metric 最低的 blocks，通过 MoveCache 重排物理 cache 使 evicted blocks 可被释放；(3) Squared Attention Metric (L2)：使用 Σ(A_hij)² 替代标准 ΣA_hij 作为 eviction metric，在 LongBench 上各变体一致优于 L1 聚合；(4) 两个变体——KVC-full：聚合全部过去 queries 的 squared attention（排除 local window v=10）；KVC-w：仅聚 observation window w=8 的 squared attention + max-pooling p=7；(5) Continual Compression：在 decoding 阶段持续累积新生成 token 的 squared attention 到已有 metrics 中；(6) Variable per-head and per-layer compression rates。实验比较 LongBench 16 子集上与 H2O、SnapKV、PyramidKV、Ada-SnapKV、Ada-PyramidKV 的性能（Mistral-7B-Instruct-v0.2, Llama-3.1-8B-Instruct），以及不同压缩率 (1x-64x) 下 continual compression 的 LongBench 性能百分比（Llama-3.1-8B 和 Llama-3.1-70B-FP8）。

- 硬件平台是什么，配置是什么。
  NVIDIA L4 GPU（Llama-3.1-8B throughput 和 LongBench 实验），gpu_memory_utilization=0.9，max-model-length=19,000；NVIDIA H100 GPU（Llama-3.1-70B-FP8 throughput 和 LongBench 实验），gpu_memory_utilization=0.96，max-model-length=33,000。KVC-full 在 H100 上需 gpu_memory_utilization=0.6 以预留 metric 计算空间，逐 query block 计算（block size=1024）。默认 block size b=16，vLLM v0.6.0 eager mode。

- 模型是什么。数据集和bench分别是什么。
  模型：Mistral-7B-Instruct-v0.2（32 层，GQA r=32/8=4）、Llama-3.1-8B-Instruct（GQA r=32/8=4）、Llama-3.1-70B-Instruct-FP8（FP8 量化，GQA r=64/8=8）。Benchmark：LongBench（16 子集，6 类别——Single-Doc QA: NarrativeQA, Qasper, MultiFieldQA-en；Multi-Doc QA: HotpotQA, 2WikiMultihopQA, MuSiQue；Summarization: GovReport, QMSum, MultiNews；Few-shot Learning: TREC, TriviaQA, SAMSum；Synthetic: PassageCount, PassageRetrieval-en；Code: LCC, RepoBench-P）。评测 max-cache-size C={128, 256, 512, 1024}，baseline 方法 C 定义保留 C×H×H 个 KVs（repeat 后），KV-Compress 保留 C×H×H/r 个 KVs（非 repeat cache）。Throughput benchmark：256 prompts，fixed output=500 tokens，varied input lengths {500, 1000, 2000, 4000, 6000, 8000, 10000, 12000}。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/IsaacRe/vllm-kvcompress/tree/main（vLLM 集成 fork from v0.6.0）。PyramidKV baseline 实现：https://github.com/IsaacRe/PyramidKV。算法 pipeline 如下：

  **Prefill 阶段（Eviction Metric 计算）**：
  ```
  输入：input sequence length L, model with l layers, H kv heads (GQA group count), r query heads per group
  超参：observation window w=8, pooling p=7, excluded local query window v=10

  for each layer m in 1..l and each KV head h_k in 1..H:
      # 计算完整 attention 矩阵 A ∈ R^{r × L × L}（causal masked）
      # GQA: 该 key head 的 query group H_k = {h: r·h_k ≤ h < r·(h_k+1)}

      # KVC-w: 有限 observation window + squared attention + max-pooling
      for each query head h in H_k:
          for query i = L-w to L:           # 仅最后 w 个 queries
              for key j = 1 to i:            # causal range
                  M_{h_k,j} += (A_{h,i,j})^2
      for each key j:                         # max-pooling over keys
          M_{h_k,j} = max(M_{h_k, t}) for t in [j-p/2, j+p/2]

      # KVC-full: 全部 queries（排除 v 个 local queries）
      for each h in H_k:
          for key j = 1 to L:
              for query i = j+v to L:         # 排除 key j 之后的 v 个 local queries
                  M_{h_k,j} += (A_{h,i,j})^2
  ```

  **Block-level Eviction（MoveCache, Algorithm 1）**：
  ```
  输入：physical K_u, V_u ∈ R^{N×b×d}, metrics M ∈ R^{N×b}, logical indices P, evict count E_s blocks

  # Step 1: 排序 metrics 以获得 per-head per-eviction-block 的最大 metric
  M1 = M.view(-1)  # [Nb], 每个元素对应一个 KV
  M2 = sort(M1, by=(head_id, metric))  # 按 head 分组
  M3 = reshape(M2, [N, b])  # 每行: 该 head 的 b 个最低 metric
  # m(h,e) = M3[o_h+e-1, b-1]  # head h 的 e-th eviction block 的最大 metric

  # Step 2: 跨 head 排序候选 block evictions
  M4 = sort(M3, by=M3[:, b-1])  # 按每 block 的最大 metric 排序

  # Step 3: 为序列 s 选择 E_s 个 block eviction
  W = zeros(N, b)
  for the first E_s blocks (offset O_s for sequence s):
      mark all KVs in these blocks for eviction in W

  # Step 4: 重排物理 cache（MoveCache）
  eviction_range = [end - E_s*b, end]
  i = end, j = end - 1
  while i > eviction_range_start:
      while W[i] == 0: i -= 1        # 找到 eviction range 内的非 evicted KV
      while W[j] == 1: j -= 1        # 找 eviction range 外的 evicted KV
      # swap: 将 range 外的 evicted KV 移到 range 内
      swap(K_u[P[i]], K_u[P[j]]), swap(V_u[P[i]], V_u[P[j]]), swap(M[P[i]], M[P[j]])
      i -= 1, j -= 1
  # 释放 eviction range 内连续的 E_s 个 block
  ```

  **压缩调度策略**（Section 4.2.3）：
  - 方案 1: 每 c 次 model iteration 压缩一次
  - 方案 2: total uncompressed tokens 超过阈值时压缩
  - 方案 3: 当有序列被新 prefill 时压缩（选中）
  - 方案 4: 当 preemption 即将发生时压缩（选中）
  - 最终使用方案 3+4 组合。
  - Sort 操作 overhead：额外内存约 8× sorted tensor 大小，runtime 在 1.7e8 元素后线性增长。限制每次压缩 iteration 中总 KV 数不超过阈值。

  **Continual Compression（Equation 20）**：
  M_{h_k,j}^{(cc)} = M_{h_k,j}^{(pool)} + Σ_{i=L_c}^{L_c+t} Σ_{h∈H_k} (A_{h,i,j})^2
  其中 M^{(pool)} 为 prefill 阶段 metric，L_c 为 input context 长度，t 为当前 decoding step。

  **关键性能**：
  - Llama-3.1-8B, LongBench C=128: KVC avg 46.26 vs PyramidKV 45.97, SnapKV 45.93（同时 KV-Compress 仅使用 1/4 KVs）
  - Mistral-7B, LongBench C=128: KVC-w8-L2 avg 37.64 vs Ada-SnapKV 36.71（同时仅使用 1/4 KVs）
  - 8B@6000 tokens, compression rate 32x: 4.93x throughput over vanilla vLLM on L4
  - 70B-FP8@6000 tokens, compression rate 64x: 2.14x throughput over vanilla vLLM on H100
  - 8B compression rates 8x-64x 保持 negligible impact，Summarization 任务最敏感
  - 70B 模型对压缩更不敏感，多数 non-summarization 任务 64x 压缩保持 >90% 性能

## CoKV: Optimizing KV Cache Allocation via Cooperative Game

- 属于算法pipeline的实现是什么？实验比较什么？
  提出一种基于合作博弈论中 Shapley Value 的 attention head 重要性评估方法，称为 Sliced Shapley Value (SSV)，用于评估每个 attention head 在模型推理中的合作贡献，并据此动态分配 KV cache 预算。方法分为两阶段：(1) 预计算阶段：在验证集上采样不同 coalition size 的 head 子集（H={32,64,96,128}），计算 complementary contribution (U(S) - U(N\S))，通过多次采样逼近每个 head 的 SSV 分数；(2) 推理阶段：根据归一化的 SSV 分数按比例分配 cache budget（含 local window 固定部分和 shared budget 按分数分配部分），每个 head 内部使用 SnapKV 的 attention pooling 机制选择保留的 token。实验在 LongBench 16 个数据集上与 SnapKV、PyramidKV、Ada-SnapKV、HeadKV-R2 比较不同 KV cache 大小（64/128/256/512/1024）下的准确率，同时与 Full Cache 对比。还进行了 head masking 实验（按重要性分数 mask top/low groups）和 Needle-in-a-Haystack 检索测试。

- 硬件平台是什么，配置是什么。
  推理实验：NVIDIA A100 40GB GPU；SSV 预计算：8× NVIDIA RTX 3090 GPU 服务器。FlashAttention 默认启用。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-3-8B-Instruct（32层×8 KV groups via GQA，共256 groups）、Mistral-7B-Instruct-v0.2（32层×8 KV groups，共256 groups）。数据集：LongBench 16 个数据集，覆盖6类任务——Single-Doc QA (NarrativeQA, Qasper, MultiFieldQA-en)、Multi-Doc QA (HotpotQA, 2WikiMQA, Musique)、Summarization (GovReport, QMSum, MultiNews)、Few-shot Learning (TREC, TriviaQA, SAMSum)、Synthetic (PassageCount, PassageRetrieval-en)、Code (LCC, RepoBench-P)。额外使用 Needle-in-a-Haystack 测试长上下文检索能力（1k-31k tokens）。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/nawei1010/CoKV。
  
  **算法 Pipeline 详解（两阶段）：**
  
  **阶段一：Head Importance Evaluation（预计算 SSV）**
  
  输入：Heads N = {h_1,...,h_n}，采样次数 M，coalition size 集合 H={32,64,96,128}
  
  1. 初始化累计矩阵 SV_{i,j} = 0, 计数矩阵 m_{i,j} = 0
  2. 对 k=1 到 M 次迭代：
     a. 随机排列 heads 得到 π^k
     b. 从 H 中随机选 coalition size j
     c. 构造 coalition S = {π^k(1),...,π^k(j)}
     d. 计算 U(S)：mask N\S 中的 heads（仅保留 local window 内的 KV），在验证集上推理得准确率
     e. 计算 U(N\S)：mask S 中的 heads，在验证集上推理得准确率
     f. 计算 complementary contribution u = U(S) - U(N\S)
     g. 对 S 中所有 head π^k(j)，更新 SV_{π^k(j),|S|} += u, m_{π^k(j),|S|} += 1
  3. 对每个 head h_i，SSV_i^H = (1/|H|) * Σ_{j∈H} (SV_{i,j} / m_{i,j})
  
  复杂度：O(M·|H|·T)，T 为单次验证集推理时间。250 samples/coalition size 时 MAE < 1/256，耗时约 20.93 小时（8×3090）。
  
  **阶段二：KV Cache Compression（推理时动态分配）**
  
  输入：shared budget B, local window size s, attention heads, SSV 分数
  
  1. Budget Allocation:
     - 归一化 SSV：NSV_i = (SSV_i - min^α(SSV)) / (max(SSV) - min^α(SSV))，α 为最低分 head 数量超参
     - α 个最低分 head 的 NSV 置 0（不分配除 local window 外的额外 cache）
     - head h_i 的 cache size: c_i = B · (NSV_i / Σ_j NSV_j) + s
  
  2. Token Selection（per head, 基于 SnapKV 机制）：
     - 计算 local window 内 tokens 的 Query: Q_i^{win} = X^{win} · W_i^Q
     - 计算 local window 对所有前缀 KV 的 attention: Ā_i = softmax(Q_i^{win} · K_i^T / √d_h)
     - 对 attention weights 做 max pooling (dim=1) 后取 mean (dim=0)，得到每个非 local window token 的重要性分数
     - 保留 top-c_i 个最高分 token 及其 KV pairs
     - 将保留的 KV 与 local window KV 拼接：{K̂_i, V̂_i} = Cat({selected KV}, {K_i^{win}, V_i^{win}})
  
  张量维度：X^{win} ∈ R^{s×d_model}, W_i^Q/K/V ∈ R^{d_model×d_h}, Ā_i ∈ R^{s×m}, d_h = d_model/num_heads, s=8 (local window), m 为前缀 token 数。

## AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning

- 属于算法pipeline的实现是什么？实验比较什么？
  提出一种 training-free 的自适应多模态 LLM 推理方法，由两个阶段组成：(1) LLM 前基于 Token Embedding 余弦相似度的迭代 Token 合并（Token Merging），将输入 LLM 的冗余视觉 token 两两配对合并，每次迭代最多减半；(2) LLM 内部基于 PageRank 的渐进式 Token 剪枝（Token Pruning），在每层 Transformer 对 Self-Attention 权重矩阵应用 PageRank 算法计算每个视觉 token 的重要性分数，保留高分 token 并剪除低分 token，同时设计分段线性 Scheduler 控制各层的 retention ratio。实验比较与 Base 模型（LLaVA-OV-7B / LLaVA-1.5-7B）和 training-free baseline 方法（FastV、VTW、PDrop、LLaVA-Prumerge）在不同 FLOPs/prefill time 下的准确率 trade-off。

- 硬件平台是什么，配置是什么。
  FLOPs 和 prefill time 使用 LLM-Viewer 库计算，假设 video LLM 有 100 个 text token、image LLM 有 40 个 text token。GPU 硬件论文未明确说明具体型号。

- 模型是什么。数据集和bench分别是什么。
  Video LLM 模型：LLaVA-OneVision-7B（LLM backbone 为 Qwen2-7B，28 layers），采样 32 frames/video。
  Image LLM 模型：LLaVA-1.5-7B（LLM backbone 为 Vicuna-v1.5-7B，32 layers）。
  Video Benchmarks：VideoMME、MVBench、MLVU、EgoSchema、NextQA、PerceptionTest。
  Image Benchmarks：GQA（12,578 samples）、VQAv2（107,394 samples）、MME（2,374 samples）、TextVQA（5,000 samples）、SQA-IMG（2,017 samples）、MMB（4,377 samples）、POPE（8,910 samples）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/LaVi-Lab/AIM（ICCV 2025），基于 PyTorch 2.3.1 + CUDA 12.1。

  **Stage 1: Token Merging (LLM 前)**：
  输入：视觉 token embedding v⁰ ∈ R^{N⁰×D}
  流程：
  ```
  // 单次迭代（每次减半）
  将 N 个 visual tokens 分为 A（偶数位置）和 B（奇数位置）
  对 A 中每个 token i，在 B 中找余弦相似度最高的 token j：
      j* = argmax_j  cos_sim(v_A[i], v_B[j])
  对最高相似度的 token pair 取均值合并：v_merged = (v_A[i] + v_B[j*]) / 2
  合并后保留 N/2 个 token
  // 重复 I 次，保留率为 r_merge = (1/2)^I
  ```
  视频任务中仅在单帧内（spatial）合并，不跨帧（temporal）合并。

  **Stage 2: Token Pruning (LLM 内部)**：
  对第 l 层，输入 token x^l = [v^l; t^l]（视觉 + 文本），用 Attention 权重矩阵 A^l ∈ R^{(N^l+M^l)×(N^l+M^l)} 计算 PageRank 重要性分数：
  $$s_i^l = \frac{1}{N^l + M^l} \sum_{j=1}^{N^l + M^l} \mathbf{A}_{i,j}^l \cdot s_j^l$$
  初始 s_j 均匀分布。仅对视觉 token 按 s_i 排序剪枝，文本 token 保持完整。

  **Scheduler（分段线性保留率）**：
  $$r^l = \begin{cases} 1, & l < l_1 \\ 1 - k(l - l_1), & l_1 \leq l \leq l_2 \\ 0, & l > l_2 \end{cases}$$
  其中 k = 1/(l₂-l₁)。l₁ 控制开始剪枝层，l₂ 控制完全移除层。

  **默认配置**：
  - Video：r_merge=25%, l₁=14, l₂=22（共 28 层 Qwen2）
  - Image：r_merge=12.5%, l₁=13, l₂=21（共 32 层 Vicuna）

  **执行流程**：
  1. Visual Encoder 产生 N⁰ 个 visual tokens
  2. 通过 I 次迭代 Token Merging（基于余弦相似度），保留 N⁰ × r_merge 个 token
  3. Merged visual tokens + text tokens → LLM Layer 1
  4. 每层：计算 Self-Attention → 用 A^l 计算 PageRank 分数 → 按 Scheduler 的 r^l 保留 top-K visual tokens（K = N^{l-1} × r^l）→ 被剪枝 token 从后续层 KV Cache 中移除
  5. l > l₂ 后所有视觉 token 已移除，仅剩 text token 继续推理

  额外开销极小：Token Merging 88.25 GFLOPs + Token Pruning 4.18 GFLOPs（合计 92.43 GFLOPs），仅占 Qwen2-7B LLM 推理 FLOPs（14757 GFLOPs）的 0.6%。

## KVSharer: Efficient Inference via Layer-Wise Dissimilar KV Cache Sharing

- 属于算法pipeline的实现是什么？实验比较什么？
  KVSharer 提出一种 plug-and-play 的层间 KV cache 共享方法，无需额外训练。核心实现基于一个反直觉的发现：共享不相似的 KV cache（而非相似）可更好地保持模型性能。方法分为两个阶段：(1) Strategy Searching——在校准数据集上执行推理，计算任意两层 KV cache（分别 flatten keys 和 values 为 1D 向量后平均）之间的欧氏距离，按降序排列（距离大=相似度低），依次尝试替换 KV cache 对（将靠近输出端的层用靠近输入端的层替换），若替换后最后一层 hidden state 与原始模型的余弦相似度超过阈值 T=0.5 则保留该替换，直到达到预定的压缩层数 C；(2) Inference with KV Cache Sharing——获得共享策略 Z 后，在 prefill 和 generation 过程中直接将被替换层的 KV cache 从前层拷贝过来，跳过本层的 KV cache 计算。实验比较在 Llama2-7B/13B/70B（Chat 和 Base 版本）、InternLM2-7B/20B、Mistral-7B-Instruct-v0.3 上，不同压缩率（12.5%/25%/37.5%）下与 Full KV Cache 的性能对比（perplexity + OpenCompass 多 benchmark 评分），以及与 H2O、PyramidInfer 等 intra-layer 压缩方法的组合效果，以及与相似度共享（+Sim.）和随机共享的消融对比。

- 硬件平台是什么，配置是什么。
  4× NVIDIA A100 80GB GPU 服务器。PPL 评估使用 Wikipedia 数据集 200 句，每句 2048 tokens。策略搜索使用 30 句随机 Wikipedia 句子（每句 64 tokens）作为校准数据集。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama2-7B-Chat、Llama2-13B-Chat、Llama2-70B（部分实验）、InternLM2-7B-Chat、InternLM2-20B-Chat、Mistral-7B-Instruct-v0.3，以及各模型的 Base 版本。数据集/Calibration：Wikipedia（30 句随机采样的 64-token 句子）、BookCorpus（同等大小子集）。Benchmark（通过 OpenCompass 评估框架）：Reasoning（CMNLI, HellaSwag, PIQA）、Language（CHID, WSC）、Knowledge（CommonSenseQA, BoolQ）、Examination（MMLU, CMMLU）、Understanding（Race-High/Middle, XSum, C3）。PPL 在 Wikipedia 200 句、每句 2048 tokens 上评估。评估模式包括 PPL 和 GEN 两类。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/yangyifei729/KVSharer。算法 pipeline 如下：

  **阶段一：Strategy Searching（Algorithm 1）**

  输入：LLM M、目标共享层数 C、校准数据集 D、相似度阈值 T
  输出：共享策略 Z（包含哪些层用哪些层的 KV cache 替换）

  ```
  1. S ← Euclidean_KV_Distance(M, D)
     // 在 D 上执行前向传播，保存每层 KV cache
     // 将每层的 keys 和 values 分别 flatten 为 1D 向量，取平均作为该层 KV cache 表示
     // 计算任意两层之间的欧氏距离
  2. R ← Descend_Rank(S)
     // 按欧氏距离降序排列（距离大 = 不相似度高，优先尝试）
  3. Z ← ∅, P ← 0
  4. for each r in R:  // r = (layer_src, layer_dst)
       Z ← Z ∪ {r}
       // 替换时：layer_dst（靠近输出端）的 KV cache ← layer_src（靠近输入端）的 KV cache
       // 不可逆方向：靠近输入的层更敏感，不做替换
       M_tmp ← Sharing_KV(M, Z)
       // 在 D 上计算 M_tmp 和 M 的最后一层 hidden state 的余弦相似度 s
       s ← Avg_CosSim(M_tmp, M, D)
       if s ≤ T:
           Z ← Z \ {r}   // 丢弃该替换对
       else:
           P ← P + 1
           if P == C: return Z
  5. return Z
  ```

  **阶段二：Inference with KV Cache Sharing**

  在 prefill 和 generation 阶段，根据 Z 中记录的替换关系：
  ```
  for each layer l in model.layers:
      if l in Z.dst_layers:
          src_layer = Z.get_src(l)
          K_cache[l], V_cache[l] = K_cache[src_layer], V_cache[src_layer]  // 直接拷贝
      else:
          K_cache[l], V_cache[l] = compute_KV(l, input_hidden)
  // 后续 attention 和 FFN 正常进行
  ```

  **关键设计**：替换只发生在靠近输出的层用靠近输入的层替换，因为靠近输入的层更敏感（修改会导致更大性能退化）。一次搜索可通用于所有下游任务（非 task-specific）。

  **组合 intra-layer 压缩**：将 H2O 或 PyramidInfer 应用于各层 KV cache 的稀疏化，然后 KVSharer 共享的层直接拷贝已稀疏化的 KV cache。Hyperparameters 先在全注意力模型上调至约 20% 压缩率，然后直接应用于与 KVSharer 的组合。

## A2SF: Accumulative Attention Scoring with Forgetting Factor for Token Pruning in Transformer Decoder

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 A2SF（Accumulative Attention Score with Forgetting Factor）算法，在 Attention Score 累积过程中引入 Forgetting Factor α（0 < α < 1），对过去的 Attention Score 施加指数衰减惩罚，解决 Transformer Decoder 中因 Causal Mask 导致早期 token 累积次数过多而产生的不公平比较问题。实验比较 A2SF 与 Full cache（无剪枝上限）、Local Attention（仅保留最近 token）和 H2O（基于 A2S 的 token 剪枝）在不同 cache ratio [0.1, 0.8] 下的准确率和与 Ideal Mask 的 cosine similarity。

- 硬件平台是什么，配置是什么。
  RTX 3090 GPU，FP16 精度。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA-2-7B、LLaMA-7B、OPT-6.7B、OPT-2.7B。
  数据集/Benchmark：OpenbookQA、Winogrande、PiQA、COPA、MathQA、ARC-easy、ARC-challenge，使用 lm-eval-harness (v0.4.0) 在 0-shot 和 1-shot 设置下评估 Commonsense-reasoning 性能。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/Dirac-Notation/A2SF

  **算法核心（伪代码）**：
  输入：每层每头的 Attention Score 矩阵 S ∈ R^{N×N}，Forgetting Factor α
  输出：每 token 的重要性分数 A_k，用于决定保留/剪枝

  ```
  // H2O baseline: 直接累积 Attention Score（考虑 causal mask）
  for n in 1..N (generation step):
      for k in 1..n (key token, k <= n due to causal mask):
          A_k += S[n][k]  // 早期 token (k 小) 被累加更多次 → 不公平

  // A2SF: 引入 Forgetting Factor 的累积
  for n in 1..N (generation step):
      for k in 1..n:
          A_k += α^(n - generation_step_of_score) × S[n][k]
  ```

  **张量计算形式**：
  公式 (5-6)：
  $$A_{n,k}^h = \sum_{q=1}^n \alpha^{n-q} \times S_{q,k}^h$$
  $$A_{n,k}^{h} = S_{n,k}^{h} + \alpha \cdot S_{n-1,k}^{h} + \alpha^{2} \cdot S_{n-2,k}^{h} + \dots + \alpha^{N-k} \cdot S_{k,k}^{h}$$

  其中 α ∈ (0, 1)。每次生成新 token 时，所有历史 Attention Score 乘以 α 后再加入新的 Score。越早的 Score 经历越多次 α 乘法，趋近于 0。这使得近期 Attention Score 权重更大，消除 token 生成顺序造成的累积次数不平衡。

  **执行流程**：
  1. 每层每头计算 Attention Score 矩阵（带 Causal Mask）
  2. 按 A2SF 公式累积带遗忘因子的重要性分数
  3. 在下一 Generation Step 前，按 A_k 排序，保留前 K 个 token（K = cache_ratio × N），剪枝其余 token 的 KV Cache
  4. A2SF 不分配 local cache，全部 cache budget 用于 selective cache（与 H2O 各半分配不同）

  **关键超参数**：
  - Forgetting Factor α：实验表明最优范围为 [0.1, 0.3]
  - Cache Ratio：在 [0.1, 0.8] 范围内评估
  - α = 0.0 等价于完全不用历史，仅用当前步 Attention Score
  - α = 1.0 等价于 H2O 的原始 A2S（无衰减）

## AdaSkip: Adaptive Sublayer Skipping for Accelerating Long-Context LLM Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 AdaSkip，一种 training-free、自适应的 sublayer-wise skipping 策略，专为长上下文 LLM 推理设计。核心思想：(1) 利用 IO Similarity（输入输出向量的余弦相似度）评估 sublayer 重要性——相似度高表示 sublayer 输出接近输入，该 sublayer 对前向传播贡献小，可以被跳过；(2) Offline Importance Learning：从历史推理任务中学习各 sublayer 的平均 IO Similarity 和 Scale Factor，用于 Prefilling 阶段的 sublayer-wise skipping；(3) Online Importance Learning：利用 Decoding 阶段前 P 个 token（online learning window）计算当前上下文的 IO Similarity，额外再跳过部分 FFN sublayer（因为 Observation 3 发现 FFN 在 decoding 阶段相似度更高）。实验比较 AdaSkip 与三种 layer-wise skipping baseline——Early Exit（跳过后几层）、SkipDecode（跳过前几层）、Unified Skipping（均匀跳过中间层）——在 Prefilling 任务（Doc QA + Few-shot Learning）、Decoding 任务（Text Summarization）和 End-to-End（prefill+decode 同时跳过）下的生成质量（F1/ACC/Rouge-L）和加速比（SU, SpeedUp）。

- 硬件平台是什么，配置是什么。
  单张 NVIDIA L20 GPU，CUDA 12.1。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA3.1-8B-128k、InternLM-7B-8k、Vicuna-v1.5-7B-16k。
  Prefilling 任务（输出长度 cap 32）：MultiFieldQA-en（F1，avg input 6493）、TriviaQA（F1，avg input 8677）、TREC（ACC，avg input 8208）。
  Decoding 任务（输出长度 limit 512）：GovReport（Rouge-L，avg input 9214）、MultiNews（Rouge-L，avg input 8265）。
  End-to-End 任务：prefill 和 decode 均跳过 sublayer。
  Offline Importance Learning 所用数据：2WikiMQA、MultiFieldQA-en、TriviaQA（来自 Stanford Alpaca 数据集）。
  Online Importance Learning 所用数据：TREC、GovReport。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/ASISys/AdaSkip

  **核心概念 — IO Similarity 作为重要性度量**：
  给定两个 n 维向量 a⃗ 和 b⃗，余弦相似度定义：
  $$Similarity(\vec{a}, \vec{b}) = \frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\| \|\vec{b}\|} = \frac{\sum_{i=1}^{n} a_i b_i}{\sqrt{\sum_{i=1}^{n} a_i^2} \sqrt{\sum_{i=1}^{n} b_i^2}}$$

  IO Similarity 越高 → sublayer 输出越接近输入 → sublayer 越不重要 → 应被跳过。论文通过 LeastSkip vs MostSkip 实验验证：跳过 IO Similarity 最高的 sublayer 的 GPT 评分（跳 1/3/5 层：8.9/6.1/4.2）远优于跳过最低的（跳 1 层即低于 1.0）。

  **Phase 1: Offline Importance Learning（Prefilling 阶段）**：

  伪代码：
  ```
  // 输入：N 个历史推理样本（sample i 含 |T_i| 个 token）
  // 模型：M 个 Transformer Layer，含 M 个 Attention sublayer + M 个 FFN sublayer（共 2M 个 sublayer）

  // Step 1: 对每个 sublayer j，累积 IO Similarity 和 Scale Factor
  for each sample i in 1..N:
      for each token t in 1..|T_i|:
          for each sublayer j in 1..2M:
              Simi_j += cosine_similarity(a_it^j, b_it^j)  // 公式(2)
              Scale_j += ||b_it^j|| / ||a_it^j||            // 公式(3)

  // Step 2: 取平均
  Simi_j = Simi_j / sum(|T_i|)
  Scale_j = Scale_j / sum(|T_i|)

  // Step 3: 按 Simi_j 对所有 2M 个 sublayer 降序排序，得 sorted list
  sorted = argsort(Simi_j, descending=True)

  // Step 4: 根据加速比 α 确定跳过的 sublayer 数量
  m = M - M/α  // 跳过的 layer 数
  // 跳过前 2m 个 sublayer（IO Similarity 最高的）
  skipped = sorted[0:2m]

  // Step 5: 用 Scale_j 补偿被跳过 sublayer 的信息损失
  b_approx_it^j = Scale_j * a_it^j  // 公式(4)
  ```

  **Phase 2: Online Importance Learning（Decoding 阶段额外 FFN 跳过）**：

  伪代码：
  ```
  // 输入：当前新上下文的解码 token、Phase 1 的 skipped set
  // P = online learning window size（前 P 个 decoded token）

  // Step 1: 用前 P 个 token 计算每个 FFN sublayer 的当前上下文 IO Similarity
  for token t in 1..P:
      for each FFN sublayer j in index:  // index = 所有 FFN sublayer 索引
          Simi_j^P += cosine_similarity(a_t^j, b_t^j)

  Simi_j^P = Simi_j^P / P  // 公式(5)

  // Step 2: 计算阈值 β（skipped set 中最低的 Similarity）
  β = min{Simi_j | j in skipped}

  // Step 3: 找出当前上下文中 Similarity 高于 β 的额外 FFN sublayer
  for each FFN sublayer j in index:
      if Simi_j^P > β:
          skipped_extra.append(j)

  // Step 4: 合并
  skipped^P = skipped ∪ skipped_extra  // 最终的 sublayer-wise skipping set

  // Step 5: 同样用 Scale_j 补偿
  ```

  **数据复用交叉验证（Table 1）**：
  Offline 学习的 IO Similarity 特征在不同数据集间有高 hit rate：
  - Src=TriviaQA, Dest=MFieldQA: 3.76/4（top-4 hit）, 4.86/6, 9.31/10
  - Src=MFieldQA, Dest=Wiki: 3.80/4, 5.54/6, 9.90/10
  - FFN 跨数据集 hit rate 略低于 ATTN 但仍在 9.38-9.56/10 水平

  **Online Window Size 消融（Table 2）**：
  - TREC: size=5 → 0.84/2, size=20 → 1.08/2, size=40 → 1.07/2（20 起趋于饱和）
  - GovReport: size=5 → 1.01/2, size=20 → 1.14/2, size=40 → 1.19/2

  **执行流程全貌**：
  1. Offline：在历史数据集上跑推理，累积各 sublayer 的 Simi_j 和 Scale_j
  2. Prefilling：用 sorted + α 确定 skipped set，跳过高 Similarity sublayer，用 Scale 补偿
  3. Decoding：前 P 个 token 正常执行所有 sublayer → 计算当前上下文的 Simi_j^P → 用 β 阈值筛选额外 FFN → 后续 token 用 skipped^P 跳过；用 Scale 补偿
  4. 支持 Prefilling-only、Decoding-only、End-to-End（两阶段同时 skip）三种模式

  **关键性能数据（Table 3）**：
  - Prefilling task（LLaMA3.1-8B-128k, skip 8 sublayers）：AdaSkip TREC ACC 72.8%（Full: 75.0%），远超 SkipDecode 0.0% / Unified Skipping 2.2%
  - Decoding task（LLaMA3.1-8B-128k, skip 8 sublayers）：AdaSkip GovReport Rouge-L 30.9（Full: 34.2），实测加速 SU=1.15；SkipDecode 19.3（SU=1.07）
  - End-to-End（LLaMA3.1-8B-128k, skip 16 sublayers）：AdaSkip GovReport/MultiNews 18.9/17.8，baselines <5
  - Decoding 加速比：最高达 17% acceleration improvement over baselines（跳过更多 attention sublayer + 额外 FFN）
  - Prefilling 加速比：InternLM 上 >10% speedup advantage，LLaMA 上 attention 已有优化故略低于 baseline

## APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 APB，一种结合序列并行与近似注意力的分布式长上下文推理框架。核心设计：(1) Context Splitting：将长文档按 host 数量均分，每 host 持有 local context block + anchor block（包含 query + 文档开头 token）；(2) Block Compression：用 LOCRET 的 retaining heads（训练的小型 MLP）在每个 host 上独立压缩 KV cache，提取 top-l_p 个最重要的 KV pair，无需全局序列视图；(3) Communication：通过 AllGather 在各 host 间共享压缩后的 KV cache，构造 passing block；(4) Computation：用修改后 attention mask 的定制 FLASHATTN kernel 执行 [anchor, passing, local] 三部分联合注意力计算。实验比较与 FLASHATTN、ULYSSES、RINGATTN、MINFERENCE、STARATTN 在 ∞Bench 和 RULER 基准上的任务性能与推理速度。APB 实现最高 9.2×（vs FLASHATTN）、4.2×（vs RINGATTN）、1.6×（vs STARATTN）加速，无观测性能退化。

- 硬件平台是什么，配置是什么。
  8× NVIDIA A800-80GB GPU（NVLink 3.0 互联），搭配 104 核 Intel Xeon Platinum 8470 CPU，跨机通信使用 HDR InfiniBand，运行 CentOS Linux 7 (Core)。FLASHATTN 和 MINFERENCE 实验在单 GPU 上进行，其余方法在 8 GPU 上进行。Yi-34B 因模型较大使用两台机器（layer 均分）。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-3.1-8B-instruct、Qwen-2.5-14B-instruct、Yi-34B-200K、Llama-3-8B-Instruct-Gradient-1048k（支持 1M 上下文）。
  数据集/Benchmark：
  - ∞Bench（10 个任务，平均上下文 >100K tokens）：Retrieve.PassKey、Retrieve.Number、Retrieve.KV、En.Sum、En.QA、En.MC、En.Dia、Zh.QA、Code.Debug、Math.Find
  - RULER（13 个任务，可控上下文长度）：Single NIAH 1/2/3、Multi-keys NIAH 1/2/3、Multi-values NIAH、Multi-queries NIAH、Variable Tracking、Common Words Extraction、Frequent Words Extraction、Question Answering 1/2

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/thunlp/APB

  **算法 Pipeline 详解（4 阶段）**：

  **Stage 1: Context Splitting（上下文切分）**：
  输入序列 t = {d, q}（文档 d 和查询 q），d 按 H 个 host 均分，每 host 持有 block B_h（长度 l_b = l_d / H）。每 host（除 host 1）的 B_h 前 prepend anchor block A = {q_1, ..., q_{l_q}, d_1, ..., d_{l_a}}（包含 query + 文档首部 l_a 个 token）。APB 使用远小于 STARATTN 的 anchor：l_a = l_b/4 或 l_b/8（STARATTN 为 l_a = l_b）。

  **Stage 2: Block Compression（块压缩）**：
  在每层每 host 上，用 retaining heads R（小型 MLP，中间维度 1024）对 local KV cache 打分：
  $$s_1, \cdots, s_{l_b} = \mathcal{R}([\mathbf{Q}_h, \mathbf{K}_h, \mathbf{V}_h])$$
  取 Top-l_p 个分数最高的 KV pair 作为压缩块：
  $$\{s_{i_1}, \cdots, s_{i_{l_p}}\} = \text{Top-}l_p(s_1, \cdots, s_{l_b})$$
  $$\mathbf{B}_h^C = (\mathbf{K}_h^C, \mathbf{V}_h^C) = (\{\mathbf{k}_h[i_1], \cdots, \mathbf{k}_h[i_{l_p}]\}, \{\mathbf{v}_h[i_1], \cdots, \mathbf{v}_h[i_{l_p}]\})$$
  Retaining heads 基于 LongAlign 数据集前 3000 样本训练 3000 steps（lr=5e-4, AdamW, batch_size=1, 最大输入长度 10240）。

  **Stage 3: Communication（通信）**：
  对压缩后的 KV cache 执行 AllGather（两次，分别对 K^C 和 V^C）：
  $$(\mathbf{K}_{[1:H]}^{C}, \mathbf{V}_{[1:H]}^{C}) = \text{AllGather}(\mathbf{K}_{h}^{C}, \mathbf{V}_{h}^{C})$$
  构造 passing block P_h = (K_p^C, V_p^C) = 前一 host 的压缩块拼接（忽略后续 host）。

  **Stage 4: Computation（计算）**：
  每 host 的 context layout = [A, P_h, B_h]，用修改的 attention mask M' 计算：
  $$\mathbf{Q}^{(i)} = [\mathbf{Q}_a^{(i)}, \mathbf{Q}_h^{(i)}], \quad \mathbf{K}^{(i)} = [\mathbf{K}_a^{(i)}, \mathbf{K}_p^C, \mathbf{K}_h^{(i)}], \quad \mathbf{V}^{(i)} = [\mathbf{V}_a^{(i)}, \mathbf{V}_p^C, \mathbf{V}_h^{(i)}]$$
  $$[\mathbf{A}_a^{(i)}, \mathbf{A}_h^{(i)}] = \text{softmax}\left(\mathbf{M}' \odot \frac{\mathbf{Q}^{(i)} \mathbf{K}^{(i)\top}}{\sqrt{d_m}}\right) \cdot \mathbf{V}^{(i)}$$
  Passing block 在 Attention 计算后丢弃，不参与 FFN。

  **FLOPs 公式**：
  $$\text{APB FLOPs/forward} = L \times [4(1 + \frac{1}{g} + \frac{0.5n}{Hd} + \frac{1.5I}{d})\frac{n}{H}d^2 + 4(H-1)(1 + \frac{1}{g} + \frac{0.5(n/H + l_a)}{d} + \frac{1.5I}{d})(\frac{n}{H} + l_a)d^2 + l_p H(H-1)(\frac{n}{H} + l_a)d]$$

  **Decoding 阶段**：使用 STARATTN 的 stage-2 精确注意力，各 host 独立计算 partial attention，通过 Gather + MergeScore（利用 online softmax 的 lse）合并为全局 attention score。

  **超参数配置（默认 128K 输入，H=8）**：
  - l_b = 16K, l_a = 4K, l_p = 2K
  - 不同输入长度配置见 Table 8

  **Retaining Head 训练**：
  - 数据集：LongAlign（前 3000 样本）
  - 优化器：AdamW（lr=5e-4，β1=0.9，β2=0.95，linear scheduler，warmup=300 steps）
  - Loss：regression loss + smoothing loss（α=0.0025）
  - Gradient clipping：0.5

## BitDecoding: Unlocking Tensor Cores for Long-Context LLMs Decoding with Low-Bit KV Cache

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 BitDecoding 系统，通过 cooperative use of Tensor Cores + CUDA Cores 实现低比特 KV cache 的高效解码。核心算法 pipeline 包括：(1) **Layout Induction 方法**：利用 ldmatrix 硬件指令的 thread-to-register 映射自动 induce Tensor Core compatible 低比特 packing layout——各线程在寄存器内量化和打包，保持 FP16 interleaved fragment layout，解量化后直接匹配 Tensor Core 寄存器，无需全局 reshape 或离线 layout transformation；(2) **Residual Block Size 对齐**：定义 residual block size N_r = P_n × W_n × R（R = ω/β 为 packing ratio），使低比特 KV cache fragment 精确对齐 Tensor Core warp-level tiling，饱和计算单元；(3) **75316420 Pattern Remapping**：基于 lop3 指令的 bitwise 操作，将 INT4/INT2 数据高效转换为 FP16，对齐 Tensor Core 的 interleaved 计算模式；(4) **Query Transformation**：将 [1, (gq, hkv)] reshape 为 [gq, hkv]，在 GQA/MQA 下饱满 Tensor Core tile；(5) **Warp Parallelism Strategy**：W_m=1（decode Q length 小），增加 W_n 提高 warps 并行度，SM warp scheduler 交替调度多个 warp 的 dequantization，避免 stall；(6) **Multi-level Memory Hierarchy Cooperative Softmax**：register→shared memory→register 的跨 warp reduction 和同步。实验比较：(1) kernel-level speedup vs FP16 FlashDecoding-v2、Kivi、QServe、Atom；(2) end-to-end 吞吐和延迟 vs Kivi、QServe；(3) 精度 trade-off（LongBench accuracy vs throughput）；(4) 各组件 ablation（layout induction、warp parallelism、pipeline optimization 的 speedup breakdown）；(5) 量化+打包延迟 overhead vs Marlin/Ladder。

- 硬件平台是什么，配置是什么。
  Blackwell (RTX 5090, RTX PRO 6000)：原生 MXFP4/NVFP4 低精度 Tensor Core。Hopper (H100 80GB)：WGMMA 指令 + warp-specialized pipeline + TMA。Ada (RTX 4090)：带宽受限。Ampere (A100 80GB)：高带宽。多 GPU：8×A100 for LLaMA-3.1-70B。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA-2-7B (MHA)、LLaMA-3.1-8B (GQA)、LLaMA-3.1-70B (GQA)、Qwen3-8B (GQA)、Qwen3-14B (GQA)。精度 benchmark：LongBench（bilingual multi-task long-context understanding benchmark），评估平均准确率。Kernel benchmark：synthetic workloads 下不同 seq_len（最高 128K）、不同 batch_size、不同 attention variant（MHA/MQA/GQA）下的 latency 和 speedup。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/OpenBitSys/BitDecoding

  **算法 Pipeline 核心流程**：

  **Phase 1: KV Cache Partitioning（prefill 后）**
  ```
  输入：Prefill 后的 FP16 KV cache X ∈ R^{L×d}
  参数：bit_width β ∈ {2,4}, 量化粒度 granularity ∈ {tensor-wise, channel-wise}

  // 计算 packing ratio 和 residual block size
  R = 16 / β                    // ω=16 (INT16 pack), e.g., R=8 for 2-bit
  N_r = P_n × W_n × R           // P_n=8 (mma.m16n8k16), W_n: warp count along N

  // 分区
  N_p = L - (L mod N_r)          // 对齐 packed 部分长度
  res_len = L mod N_r            // residual 部分长度

  X_pack = X[:N_p]               // 将被量化+打包为低比特
  X_res  = X[N_p:]               // 保留 FP16 residual KV cache
  ```

  **Phase 2: Quantization & Packing（Residual Kernel）**
  ```
  // 对 X_pack 中每个 N_r 大小的 block 执行
  for each block of size N_r:
      // Step 1: ldmatrix 加载 FP16 KV tile（建立 interleaved layout）
      registers = ldmatrix(FP16_KV_block)

      // Step 2: 线程级 reduction（warp-level via __shfl_xor_sync）
      local_min, local_max = thread_level_reduction(registers)
      warp_min = __shfl_xor_sync(local_min)
      warp_max = __shfl_xor_sync(local_max)

      // Step 3: 计算 scale 和 zero-point
      if granularity == tensor-wise:
          scale = (warp_max - warp_min) / (2^β - 1)
          zero_point = round(-warp_min / scale)
      elif granularity == channel-wise:
          // 沿 seq_len 维（在 residual block 内）channel-wise 量化
          scale[d] = (max_d - min_d) / (2^β - 1)  // per channel
          zero_point[d] = round(-min_d / scale[d])

      // Step 4: 各线程在寄存器内量化+pack（保持 interleaved layout）
      for each thread's register values:
          quantized = clamp(round(fp16_val / scale) + zero_point, 0, 2^β-1)
          packed = pack 16/β 个 quantized values → INT16

      // Step 5: 写出到 low-bit KV cache
      K_pack[N_p:N_p+N_r] = packed_K
      V_pack[N_p:N_p+N_r] = packed_V
      K_params += {scale, zero_point}  // half2 格式存储
  ```

  **Phase 3: Autoregressive Decoding（Packing Kernel, 每生成 token 执行）**
  ```
  输入：Q ∈ R^{1×hq×d}, K_pack, V_pack, K_params, V_params, K_res, V_res

  // Step 1: Query Transformation（GQA/MQA）
  Q_reshaped = reshape(Q, [gq, hkv, d])  // gq = hq/hkv

  // Step 2: Packed KV Cache Attention（main body）
  for c in 0..ceil(N_p / T_n):           // T_n: KV tile size along N
      // 异步 Global→Shared Memory
      Q_tile = cp.async.cg(Q_reshaped) → SMEM
      K_tile_pack = cp.async.cg(K_pack[c*T_n:(c+1)*T_n]) → SMEM
      V_tile_pack = cp.async.cg(V_pack[c*T_n:(c+1)*T_n]) → SMEM
      K_tile_param = cp.async.ca(K_params[c*T_n:(c+1)*T_n]) → SMEM
      V_tile_param = cp.async.ca(V_params[c*T_n:(c+1)*T_n]) → SMEM

      // Pipeline: ldmatrix + dequant (CUDA Cores) overlap with mma (Tensor Cores)
      for each warp tile in K_tile_pack:
          // Stage A (CUDA Cores): Load & Dequant
          reg_K = ldmatrix(K_tile_pack[tile])     // 加载 packed INT16
          reg_Kp = ldmatrix(K_tile_param[tile])   // 加载 scale/zp (half2)
          reg_Kfp16 = lop3_75316420_remap(reg_K)  // bitwise remapping
          reg_Kfp16 = reg_Kfp16 * reg_Kp.scale + reg_Kp.zp  // dequant

          // Stage B (Tensor Cores): Matmul（与下一个 tile 的 Stage A 重叠）
          S = mma(Q_tile, reg_Kfp16)              // T_m × T_n

          // Cooperative Softmax (cross-warp)
          m_new = max(m_old, rowmax_warp_reduce(S, sTMP))
          P = exp(S - m_new)                      // element-wise (CUDA Cores)
          sAcc[tile] = P                           // store to SMEM
          P_aligned = ldmatrix(sAcc[tile])         // reload for MMA alignment

          reg_Vfp16 = ldmatrix(V_tile_pack[tile]) → dequant
          O_new = mma(P_aligned, reg_Vfp16) + diag(exp(m_old - m_new)) @ O_old
          O_old, m_old = O_new, m_new              // online update

  // Step 3: Residual KV Cache Attention（标准 FlashAttention）
  O += FlashAttention(Q_reshaped, K_res, V_res)

  // Step 4: 更新 Residual KV Cache
  K_res = concat(K_res, new_K_token)
  V_res = concat(V_res, new_V_token)
  if len(K_res) == N_r:
      // 触发 Residual Kernel：量化满的 residual block → packed cache
      quantize_and_pack(K_res, V_res) → append to K_pack, V_pack
      K_res, V_res = [], []  // 清空 residual
  ```

  **张量计算细节**：

  布局对齐数学表达：
  $$X = X_{\text{pack}} \cup X_{\text{res}}, \quad X_{\text{pack}} = X[:L-N_r], \quad X_{\text{res}} = X[L-N_r:]$$
  $$N_r = P_n \times W_n \times R, \quad R = \omega / \beta$$

  Bank conflict-free shared memory layout：
  $$\text{col}_{id} = \text{row}_{id} \oplus \text{col}_{id}$$

  Cooperative Softmax（Algorithm 1）：
  $$S_i = Q_i K_j^T, \quad S_i \in \mathbb{R}^{T_m \times T_n}$$
  $$m_i^{new} = \max(m_i, \text{rowmax}(S_i, sTMP))$$
  $$P_i = \exp(S_i - m_i^{new}), \quad P_i \in \mathbb{R}^{T_m \times T_n}$$
  $$sAcc = \text{tiled\_copy\_r2s}(P_i)$$
  $$P_i' = \text{tiled\_copy\_s2r}(sAcc)$$
  $$O_i^{new} = P_i' V_j + \text{diag}(e^{m_i - m_i^{new}}) O_i$$

  Hopper WGMMA 关键路径（PTX 级别）：
  ```
  ldmatrix → dequant（CUDA Cores）
  STSM → shared memory（存储 dequantized FP16）
  wgmma_SS → Tensor Cores（B matrix from shared memory）
  // STSM 和 wgmma 利用 Hopper 异步执行重叠
  ```

  **关键性能与精度 trade-off（Table I）**：
  | KV Cache | Throughput      | LongBench Acc |
  |----------|-----------------|---------------|
  | FP16     | 49.25 tok/s     | 48.25         |
  | INT4     | 147.21 (2.98×)  | 48.16 (-0.2%) |
  | INT2     | 209.48 (4.25×)  | 47.38 (-2.7%) |

  **量化+打包 Overhead（Table II）**：
  | Inference Phase | Marlin | Ladder | BitDecoding |
  |-----------------|--------|--------|-------------|
  | Prefill         | 58.02ms| 4.79ms | 0.0599ms    |
  | Decode          | 0.41ms | 0.65ms | 0.008ms     |

  **Multi-warp Ablation（Table III）**：
  W_n=1 → TC utilization 10.91%, latency 3.746ms（低效）
  W_n=4 + Coop Softmax → TC utilization 19.66%, latency 0.613ms（高效，correctness valid）

## AdaSplash: Adaptive Sparse Flash Attention

- 属于算法pipeline的实现是什么？实验比较什么？
  ADASPLASH 提出基于 α-entmax 的自适应稀疏注意力机制替代 softmax，结合 Hybrid Halley-Bisection 算法快速求解 α-entmax 的阈值 τ，以及基于 Triton 语言实现的 GPU kernel（tiling + recomputation + block masking），使稀疏注意力在训练时获得实际加速。α-entmax 的核心公式为：α-entmax(s) = [(α-1)s - τ1]_+^{1/(α-1)}，其中 τ 为归一化常数。通过 α 参数控制稀疏度（α>1 得稀疏分布，α=1 即 softmax，α=2 即 sparsemax）。实验比较：(1) Halley-bisection vs. Torch 标准 bisection 在计算 τ 时的迭代次数、加速比和 GPU 内存；(2) ADASPLASH vs. FlashAttention-2 (CUDA/Triton) 和 naive α-entmax (Torch bisection/sorting) 在 synthetic data 上不同序列长度 (1K-64K) 的 training step 时间和 GPU 内存；(3) 在 RoBERTa/ModernBERT + 下游任务（长文档分类 ECtHR、BEIR 检索、GLUE）和 GPT-2 + HellaSwag 上的训练速度、内存与任务精度。

- 硬件平台是什么，配置是什么。
  Efficiency benchmark (Figure 1, 3) 和 GPT-2 训练在单张 Nvidia H100 GPU (80GB) 上进行。Masked language modeling、text classification、GLUE tasks、BIER tasks 和 ModernBERT runtime 实验在 Nvidia RTX A6000 GPU (48GB) 上进行。

- 模型是什么。数据集和bench分别是什么。
  模型：
  - RoBERTa-base (125M params)，在 FineWeb-Edu 2B tokens 上 continuous pretraining
  - ModernBERT-base (149M params)，在 FineWeb-Edu 2B tokens 上 continuous pretraining（保留 window attention 层，仅替换 global attention 层）
  - GPT-2 (124M params)，在 FineWeb 10B tokens 上 from-scratch 训练，1024 context length
  数据集/Benchmark：
  - 长文档分类：ECtHR (European Court of Human Rights)，逐步扩展序列长度至 8192
  - 单向量检索：BEIR benchmark — SciFact, NFCorpus, FiQA-2018, TREC-COVID，评估 nDCG@10
  - 语言理解：GLUE benchmark — CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE
  - 语言建模：GPT-2 训练在 FineWeb (10B tokens)，评估 HellaSwag accuracy 和 validation loss
  - 连续预训练数据：FineWeb-Edu (2B tokens, English subset)

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/deep-spin/adasplash（ICML 2025）。基于 Triton 语言 + PyTorch + HuggingFace Transformers。

  **算法 Pipeline 核心**：

  **Step 1: α-entmax 变换替代 softmax**：
  softmax 总是对所有 token 分配非零概率，而 α-entmax 可产生稀疏概率分布：
  $$\alpha\text{-entmax}(\mathbf{s}) = [(\alpha - 1)\mathbf{s} - \tau \mathbf{1}]_{+}^{1/(\alpha - 1)}$$
  其中 τ 为归一化常数，通过求解 f(τ) = Σ_i[(α-1)s_i - τ]_+^{1/(α-1)} - 1 = 0 得到。α 越大稀疏度越高：α=1→softmax (dense)，α=1.5→moderate sparsity (~95% sparsity)，α=2→sparsemax (~99% sparsity)。

  **Step 2: Hybrid Halley-Bisection 快速求解 τ**：
  ```
  Input: logits s ∈ R^n, parameter α, iterations T
  1. s ← (α-1) * s
  2. τ_lo = max(s) - 1, τ_hi = max(s) - n^{1-α}, τ = (τ_lo + τ_hi)/2
  3. for t = 1..T:
       // Compute f, f', f'' using block-accumulated sums (no materialization of full S)
       f = Σ_j Σ_i [s_i^{(j)} - τ]_+^{1/(α-1)} - 1
       f' = -1/(α-1) * Σ_j Σ_i [s_i^{(j)} - τ]_+^{1/(α-1) - 1}
       f'' = (2-α)/(α-1)² * Σ_j Σ_i [s_i^{(j)} - τ]_+^{1/(α-1) - 2}
       
       // Bisection update for bounds
       if f(τ) < 0: τ_lo = τ else: τ_hi = τ
       
       // Halley's update (cubic convergence rate)
       τ_H = τ - 2*f*f' / (2*f'² - f*f'')
       
       // Fail-safe: use bisection if Halley goes out of bounds
       if τ_H ∈ [τ_lo, τ_hi]: τ = τ_H
       else: τ = (τ_lo + τ_hi)/2
  4. Output: [s - τ]_{+}^{1/(α-1)}
  ```
  Halley-bisection 仅需 3 次迭代达到 machine precision，vs. 标准 bisection 需 23 次，加速 ~15×（2.38ms vs 36.67ms at n=8192），内存节省 1.75×（512MB vs 896MB）。

  **Step 3: Triton kernel 实现 Block-wise Tiling（前向 pass）**：
  将 Q 分块为 T_r = ⌈n/B_r⌉ blocks，K,V 分块为 T_c = ⌈n/B_c⌉ blocks。对每个 Q_i block：
  ```
  for i = 1..T_r:                    // Load Q_i from HBM to SRAM
      // Compute τ_i using Halley-Bisection block version (over all K_j blocks)
      τ_i ← Halley-Bisection-Block(Q_i, K_1..K_Tc)  // no materialization of S
      for j = 1..T_c:                // Load K_j, V_j from HBM to SRAM
          S_i^{(j)} = Q_i K_j^T      // Compute on SRAM: B_r × B_c
          P_i^{(j)} = [(α-1)S_i^{(j)} - τ_i]_+^{1/(α-1)}  // sparse attention weights
          O_i += P_i^{(j)} V_j       // Accumulate output on SRAM
  ```

  **Step 4: Sparsity-aware Block Masking（关键加速技术）**：
  在 Halley-bisection 最后迭代中，动态构造 block mask M ∈ {0,1}^{T_r×T_c}：
  $$M_{ij} = \begin{cases} 1 & \text{if } \exists S_{i',j'} > \tau_{i'} \text{ in block (i,j)}, \\ 0 & \text{otherwise} \end{cases}$$
  M 仅需二进制值，可跨 attention 层共享。基于 M 创建 pointer-increment lookup tables：
  - K_j = {i | M_{ij}=1}：对 K_j block 贡献非零 P 的 Q_i 行索引
  - Q_i = {j | M_{ij}=1}：对 Q_i 有效的 K_j 列索引
  使用 torch.argwhere 提取 (i,j) 非零 entry 索引，后续前向/反向 pass 中跳过多余 block 的 HBM 读写和计算。

  **Step 5: 反向传播（利用 α-entmax 的稀疏 Jacobian）**：
  对于 p = α-entmax(s)，Jacobian 为：
  $$\frac{\partial α\text{-entmax}(s)}{\partial s} = \text{Diag}(u) - \frac{uu^T}{\|u\|_1}, \quad u_k = (p_k)^{2-α}$$
  反向 pass 分两 kernel：
  - **dK,dV kernel**：dS_i^{(j)} = U_i^{(j)} ⊙ (dP_i^{(j)} - δ_i)，用 K_j lookup table 仅迭代有效行
  - **dQ kernel**：dQ_i = Σ_{j∈Q_i} dS_i^{(j)} K_j，用 Q_i lookup table 仅迭代有效列
  前向需额外存储 O^{(2)} ∈ R^{n×d}（替代 softmax 的 O），以及 M（跨层共享）。

  **α 退火策略**：
  连续预训练时 α 从 1.0 线性增加至目标值（1.5 或 2.0）over 1B tokens（50,000 steps），实现 dense→sparse 平滑过渡。

  **关键性能数据**：
  - GPT-2 (1024 ctx, H100): ADASPLASH 1.03 s/step, FA2 0.98 s/step, Torch bisection 7.78 s/step, Torch sorting 3.61 s/step; peak memory 均 52.5 GB (vs 77.6 GB / 73.8 GB)
  - ModernBERT (8192 ctx, A6000): ADASPLASH 1.53s vs Halley-bisection (Triton only, w/o flash-block-masking) 1.61s vs Torch bisection 4.99s vs Torch sorting OOM
  - Synthetic (64K seq, H100): ADASPLASH 随稀疏度增长最终超越 FA2 (CUDA/Triton)
  - ModernBERT α=1.5 sparsity ~95%, α=2.0 sparsity ~99%
  - RoBERTa (8192 ctx): ADASPLASH 38h08m/epoch & 79.88GB memory vs Torch bisection 4h12m34s/epoch & 508.16GB memory

## Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs

- 属于算法pipeline的实现是什么？实验比较什么？
  提出两套KV cache压缩算法：(1) **Chunked Eviction**：将post-fill eviction方法（PyramidKV、SnapKV）适配到chunked pre-filling场景，使KV在pre-filling的每个chunk后可被提前evict，包括Naive Chunked Eviction（直接对每个chunk独立执行eviction heuristic）和Patched Chunked Eviction（在每个chunk末尾拼接最后k个prompt token作为query计算重要性分数）；(2) **PruLong**：一种end-to-end的attention head specialization方法，将attention heads二分类为retrieval heads（保留完整KV cache）和streaming heads（仅保留local window + attention sinks），通过hard concrete重参数化和next-token prediction loss端到端学习head类型，支持精确target sparsity正则化。实验比较：(1) PruLong vs DuoAttention vs PyramidKV（naive/patched）在HELMET和LongProc共8类task category上的critical KV footprint（保持≥90% full attention性能的最小KV footprint）；(2) 不同training data（Pre-training Mix vs BookSum Passkey vs Context Synthesis）和training stage（pre-SFT vs post-SFT）的消融；(3) chunk size sensitivity（8K vs 32K pre-filling chunk size）。

- 硬件平台是什么，配置是什么。
  PruLong训练：论文未明确说明具体GPU型号。评估：论文未明确说明具体GPU型号（使用PyTorch推理）。附录F提供了hardware metrics（throughput和peak memory），在装有特定GPU（未明确型号）的机器上测量，DuoAttention/PruLong peak memory约16-29 GiB，PyramidKV/SnapKV约17-47 GiB（因task而异）。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-3.1-8B-Instruct（主要）；ProLong-8B-Base（ablations for SFT stage analysis）。
  训练数据：
  - PruLong默认：Gao et al. (2025)的stage-II continued pre-training mix（长度512K，截断至128K），short:long data ratio 40%:60%
  - 对比数据：BookSum Passkey（DuoAttention原始数据）、Context Synthesis（Zhu et al., 2025）
  Benchmark：
  - HELMET（128K context setting，long input→short output）：Recall（JSON KV, RULER MK Needle/UUID, RULER MV）、RAG（NQ, TriviaQA, PopQA, HotpotQA）、Re-ranking（MS MARCO NDCG@10）、Many-shot ICL（TREC Coarse/Fine, NLU, BANKING77, CLINC150）、Long-document QA（NarrativeQA, ∞QA）、Summarization（∞Sum, Multi-LexSum）
  - LongProc（short/long input→long output）：HTML→TSV（structured prediction, 12K-38K input, 1K-10K output），Travel Planning（multi-city itinerary generation, 6K→3K）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/princeton-pli/PruLong

  **PruLong算法核心流程**：

  **Phase 1: Attention Head Type Classification**
  ```
  // 每层每头 i∈[1,L], j∈[1,H] 学习二值mask z_{i,j} ∈ {0,1}
  // z=1 → retrieval head (full attention), z=0 → streaming head (local window + sinks)

  // 混合attention机制
  Attn_{i,j}(Q,K,V) = z_{i,j} × Attn_full(Q,K,V) + (1-z_{i,j}) × Attn_streaming(Q,K,V)

  // Attn_full: 标准causal attention，key序列为所有历史token
  // Attn_streaming: 仅attend到local window (W=1024 token) + attention sinks (S=128 token)
  ```

  **Phase 2: Mask Learning via Hard Concrete Reparameterization**
  ```
  // 将z_{i,j}建模为Bernoulli随机变量，参数π_{i,j}
  // 使用hard concrete distribution [Louizos et al., 2018] 重参数化

  // 前向采样
  u ~ Uniform(0, 1)  // truncated at (1e-6, 1-1e-6)
  s = σ( (1/τ) × log(u/(1-u)) + log α )  // τ = 3/2, Gumbel reparameterization
  g̃ = l + s × (r - l)  // stretch to [-0.1, 1.1]
  z̃ = min(1, max(0, g̃))  // hard sigmoid → support {0,1}

  // 参数log α_{i,j}可训练，通过梯度下降优化
  ```

  **Phase 3: End-to-End Training Objective**
  ```
  // 公式(2): min-max optimization

  max_{λ1,λ2} min_{π} E_{z~Bern(π)} [
      1/N Σ_{n=0}^{N-1} log p_θ(x_{n+1} | x_{:n}; z)
  ] + λ1(s(π) - t) + λ2(s(π) - t)²

  // 第一项: next-token prediction loss（LM loss）
  // 第二项: Lagrangian penalty，约束sparsity s(π) → target t
  // λ1, λ2: 可训练Lagrange乘子（gradient ascent优化）
  // s(π) = 1 - 1/(LH) Σ σ(α_{i,j} - log(-l/r))  // 闭式期望L0 sparsity
  ```

  **Phase 4: Sparsity Warmup & Discretization**
  ```
  // 训练配置
  Target sparsity t: 从0 warmup到t_∞（如0.7），over 800/1000 steps
  LR (log α): 1.0
  LR (λ1, λ2): 1.0
  LR schedule: linear warmup first 10% → linear decay to 1% peak
  Batch size (tokens): 1,048,576
  Sequence length: 131,072
  Training steps: 1,000
  Model weights: frozen（ablation中unfrozen时LR=1e-5）
  Adam (β1, β2): (0.9, 0.95)
  Window size: 1024, Sink size: 128

  // 训练后离散化
  // 对任意target sparsity k%，将top k%的log α设为+∞ (z=1)，其余设为-∞ (z=0)
  ```

  **Phase 5: Inference**
  ```
  // 对每个attention head (i,j):
  if z_{i,j} == 1:  // retrieval head
      attn_i_j = FlashAttention(Q, K_full_history, V_full_history)
  else:  // streaming head
      K_local = K[-1024:, :]  // 最近1024个token
      K_sinks = K[:128, :]     // 前128个attention sink token
      attn_i_j = FlashAttention(Q, concat([K_sinks, K_local]), concat([V_sinks, V_local]))
      // evict非local非sink的KV → memory saving
  ```

  **Chunked Eviction (PyramidKV/SnapKV) 核心流程**：
  ```
  // Naive Chunked Eviction
  // 在chunked pre-filling的每个chunk后立即执行eviction
  for each chunk c of size C:
      X_c = tokens[c*C : (c+1)*C]
      K_c, V_c = forward(X_c)  // 计算当前chunk的KV
      scores = attention_score(K_c[-k:], V_c[-k:])  // 最后k=64个token的attention收分
      smoothed_scores = moving_average(scores)  // 平滑
      keep_indices = top-p%(smoothed_scores)  // 保留p%最重要KV
      evict(KV[keep_indices之外])  // 逐出其余KV

  // Patched Chunked Eviction
  // 关键区别：每个chunk末尾拼接prompt的最后k个token作为query
  for each chunk c of size C:
      X_patched = concat([X_c, prompt_tail_k])  // 拼接最后k个prompt token
      K_c, V_c = forward(X_patched)
      scores = attention_score(K_c[-k:], V_c[-k:])  // 用拼接的prompt tail计算重要性
      // 仅保留X_c对应的KV（丢弃补丁token的KV，除非是最后一个chunk）
      keep_indices = top-p%(smoothed_scores)
      evict(KV[keep_indices之外])

  // PyramidKV: 按pyramidal结构分配各层KV budget
  // 浅层budget多，深层budget少（后续层压缩率更高）
  budget[l] = base_budget × (1 - l/L)^γ  // γ控制pyramid陡峭程度

  // KV Group优化（GQA场景）
  // 对每个KV group内的多个query head取attention mean后再做eviction
  // 避免为每个query head独立选择KV → 减少8x内存（Llama-3.1-8B GQA g=4, 2 query groups/key）
  ```

  **全栈执行流程（PruLong inference on Llama-3.1-8B-Instruct）**：
  1. Pre-filling阶段（chunked, chunk_size=32K）：每个chunk中，各attention head根据z_{i,j}决定使用full attention还是streaming attention；streaming heads evict非local/sink的KV，retrieval heads保留全部KV
  2. Decoding阶段：对每个新生成的token，所有heads重新计算attention；streaming heads维持fixed-size KV cache（window + sinks），retrieval heads的KV cache线性增长
  3. KV Footprint计算：聚合所有timestep的un-evicted KV entries数量，归一化至full causal attention

  **关键性能数据（Table 2 - Critical KV Footprint %）**：
  | Task      | DuoAttention | PruLong | PyramidKV(Naive) | PyramidKV(Patched) |
  |-----------|-------------|---------|------------------|--------------------|
  | Recall    | 58.0        | 46.0    | >93.0            | 64.0               |
  | RAG       | 49.0        | 37.0    | 44.0             | <34.0              |
  | Re-Rank   | 69.0        | 61.0    | >94.0            | 94.0               |
  | ICL       | 49.0        | 38.0    | 42.0             | <36.0              |
  | LongQA    | 60.0        | 49.0    | 62.0             | <35.0              |
  | Summ      | 63.0        | 59.0    | 53.0             | 49.0               |
  | HTML      | 87.0        | 83.0    | 97.0             | 97.0               |
  | Travel    | 91.0        | 93.0    | >98.0            | >98.0              |

  **PruLong vs DuoAttention 消融（Table 3, 70% sparsity）**：
  - Pre-training Mix data: PruLong Recall 91.4 vs DuoAttention 38.6（+52.8, key differentiator）
  - PruLong在natural long-context data上表现优异，DuoAttention依赖synthetic passkey data
  - PruLong不更新model weights，保持instruction-following能力

  **Real Hardware Metrics（Appendix F, 70% sparsity）**：
  - DuoAttention: throughput 10.0×10⁻² req/s (Recall), peak memory 26.6 GiB
  - PruLong: throughput 10.8×10⁻² req/s (Recall), peak memory 26.3 GiB
  - PyramidKV+P+C: throughput 8.0×10⁻² req/s, peak memory 33.7 GiB
  - PruLong consistently achieves highest throughput and lowest peak memory across tasks

## Cost-Optimal Grouped-Query Attention for Long-Context LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  提出一种基于 scaling law 的 GQA 配置搜索方法，通过三步搜索过程找到给定目标 loss 和上下文长度下的 cost-optimal GQA 配置（nh, nkv, N）。核心设计：(1) **解耦 head 数量与 hidden size**：传统 GQA Transformer 强制 nh × dh = d（dh 固定为 64），本文解除此约束，使 nh 成为独立超参数自由控制 attention 计算 FLOPs（time-variant cost）；(2) **联合优化模型大小与 GQA 配置**：同时调整模型大小 N（time-invariant cost）、query head 数 nh 和 KV head 数 nkv（time-variant cost），实现推理资源在 attention 和非 attention 组件之间的最优分配；(3) **三步搜索过程（Step 1→3）**：Step 1 定义候选 GQA 配置集 H_cand = {nh=1,2,4,...,max(d)/dh} × {nkv=1,2,4,...,max(d)/dh}（满足 nkv ≤ nh），共 21 个候选；Step 2 对每个 H 训练系列不同大小模型（3M→1.2B），用 power-plus-constant 函数 L(N;H) = (a/N)^b + E 拟合 scaling curves（R² > 0.999）；Step 3 对目标 loss L* 和上下文长度 T，求解 N*(H) = a/(L* - E)^{1/b} 并计算硬件感知成本 Z = λM_infer^α + (1-λ)C_infer^β（λ=0.9, α=1/2, β=1/3），选择 Z 最小的 (N*, H*)。实验比较：(1) Loss vs. inference costs（M_infer, C_infer, Z）对不同 GQA 配置的 tradeoff 曲线（T=8K/16K/32K/64K/128K/512K）；(2) cost-optimal GQA vs Llama-3 GQA（nh=d/dh, nkv=8）在 T=128K 下的 training/inference throughput 和下游性能（common-sense reasoning + NIAH）；(3) 对齐 training FLOPs 的 comparison（用更少 head 的配置获得更多训练数据）；(4) nh 和 nkv 对 loss 的 power-plus-constant scaling law 验证（Section 5.4 + Appendix I）。

- 硬件平台是什么，配置是什么。
  NVIDIA A800 GPU，8 GPU 集群，BF16 训练，FP16 评估。单张 A800 测试 downstream throughput（batch_size=1, T=128K）。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-3 架构（RoPE θ=500,000 + SwiGLU FFN + RMSNorm + pre-norm，无 bias/dropout），使用 GPT-2 tokenizer（V=50,304），dh=64，dff≈8d/3。训练 3M→1.2B 参数模型，具体配置见 Table 9（L/d 从 4/256 到 36/1536）。数据集：SlimPajama（627B tokens，RedPajama 的去重版本），每 batch 512K tokens，训练数据与参数量之比为 20:1（Chinchilla law）。Benchmark：zero-shot common-sense reasoning（8 任务：ARC-Challenge, ARC-Easy, BoolQ, HellaSwag, Lambada, PIQA, SocialIQA, Winograd，使用 LM-Evaluation-Harness）、Needle-in-a-Haystack（RULER benchmark，1K-128K context）。Training 含两阶段——第一阶段 T=4K 20B tokens + 第二阶段 T=128K 1B tokens。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/THUNLP/cost-optimal-gqa

  **算法 Pipeline 详解（三步搜索 + 推理）**：

  **前提：解耦 nh 与 d**
  传统 GQA: nh × dh = d（nh = d/dh，不可调），本文解除此约束：
  ```
  # 传统 GQA（Llama-3）: nh 由 d 决定
  d = 1536, dh = 64 → nh = d/dh = 24（四舍五入到最近 power-of-2 → 32）

  # 本文: nh 独立于 d，可自由调整
  d = 1536, dh = 64, nh ∈ {1,2,4,8,16,32}  # 任意选择
  # 每个 head dimension 仍为 dh=64
  # 输出投影 W_o^h ∈ R^{d×dh} 仍映射回 d 维
  ```
  解耦后 QKV 投影维度变为 [d, nh×dh]（而非 [d, d]），当 nh < d/dh 时参数量减少。

  **Step 1: Candidate Selection**
  ```
  Input: max(d)/dh = 32  # 最大模型的 hidden size 对应
  H_cand = []
  for nh in {1, 2, 4, 8, 16, 32}:
      for nkv in {1, 2, 4, 8, 16, 32}:
          if nkv <= nh:
              H_cand.append((nh, nkv))
  # |H_cand| = 21 个候选配置
  ```

  **Step 2: Scaling Curves Fitting**
  ```
  # 对每个 H ∈ H_cand，训练系列模型（3M→1.2B）with T=8K
  for (nh, nkv) in H_cand:
      for N in [3M, 19M, 85M, 150M, 200M, 470M, 680M, 1.2B]:
          model = build_model(N, nh, nkv)
          loss = train(model, SlimPajama, ratio=20:1)
          record (N, loss)

      # 拟合 power-plus-constant scaling law
      L(N) = (a/N)^b + E
      # a, b: 配置相关参数
      # E: 语言的自然熵（跨配置共享）
      # 拟合 R² > 0.999
  ```

  **Step 3: Cost Minimization**
  ```
  Input: target loss L*, context length T
  # 推理成本公式
  C_infer(T) = 2N + 4TL dh nh          # 时间不变 FLOPs + 时间相关 FLOPs
  M_infer(T) = N + 2TL dh nkv          # 参数内存 + KV cache 内存

  for (nh, nkv) in H_cand:
      # 从 scaling law 求解满足 L* 的最小 N
      N*(nh,nkv) = a(nh,nkv) / (L* - E)^{1/b(nh,nkv)}

      # 计算硬件感知成本
      Z(nh,nkv) = 0.9 * M_infer(T)^(1/2) + 0.1 * C_infer(T)^(1/3)

  # 选择 Z 最小的配置
  H* = argmin Z
  return (N*, nh*, nkv*)
  ```

  线性插值计算 N* 对应的精确 (L,d) —— 使用 Table 7 的预定义配置插值。
  实际部署时选择最接近 (N*, nh*, nkv*) 的实际整数配置。

  **推理流程（以 H=8,1 at T=128K 为例）**：
  ```
  # 模型配置: N=1.8B (vs Llama-3 1.2B), nh=8 (vs 32), nkv=1 (vs 8)
  # Prefill 阶段:
  for each layer l in 1..L:
      Q = X @ W_Q  # [T, nh*dh] = [T, 8*64] = [T, 512]
      K = X @ W_K  # [T, nkv*dh] = [T, 64] — 仅 1 个 KV head!
      V = X @ W_V  # [T, nkv*dh] = [T, 64]
      # GQA: 8 个 query head 共享 1 组 KV
      for g in 1..8:  # nh/nkv = 8 groups
          Q_g = Q[:, (g-1)*64 : g*64]  # [T, 64]
          attn_g = softmax(Q_g @ K^T / sqrt(64))  # [T, T]
          out_g = attn_g @ V  # [T, 64]
      output = concat([out_1, ..., out_8]) @ W_O

  # Decode 阶段 (single token):
      # KV cache 大小: 2 * L * T * dh * nkv = 2 * 36 * 128K * 64 * 1
      # vs Llama-3 GQA: 2 * 36 * 128K * 64 * 8 = 8× 更大!
  ```

  **推理成本对比（T=128K, L*=2.615, BF16）**：
  | 配置 | N | nh | nkv | M_infer | C_infer | 节省 |
  |------|---|----|-----|---------|---------|------|
  | Llama-3 GQA | 1.2B | 32 | 8 | baseline | baseline | — |
  | Cost-Optimal | 1.8B | 8 | 1 | **-50.8%** | **-57.8%** | 同 loss |

  **Throughput 实测（Table 5, A800）**：
  - Training: Llama-3 GQA (32,8) = 18,655 tok/s → Ours (8,1) = 31,260 tok/s（+67.6%）
  - Inference: Llama-3 GQA (32,8) = 12,921 tok/s → Ours (8,1) = 20,643 tok/s（+59.8%）

  **Key hyperparameter constraints**：
  - dh = 64（固定），V = 50,304（固定），dff ≈ 8d/3
  - AdamW optimizer (β1=0.9, β2=0.95, weight_decay=0.1, gradient_clip=1.0)
  - WSD LR scheduler (10% warmup, 20% cosine decay to 0.1× max_lr)
  - Max LR: grid-searched per model size on MHA baseline (1e-3 for 3M/19M/85M, 5e-4 for 150M/200M/470M, 2e-4 for 680M/1.2B)
  - Precision: BF16 (training), FP16 (evaluation)

## DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 DuoAttention，将 attention head 分为 Retrieval Heads（关键长上下文处理，需 full attention across all tokens）和 Streaming Heads（主要关注 recent tokens 和 attention sinks，仅需 constant-length KV cache）。核心设计：(1) **基于优化的 Retrieval Head 识别**：为每个 KV head 分配可训练 gate value α_{i,j} ∈ [0,1]，前向 pass 中混合 full attention 和 streaming attention 输出：attn_{i,j} = α_{i,j}·full_attn + (1-α_{i,j})·streaming_attn。在合成 passkey retrieval 数据集上以 L2 distillation loss + L1 正则化项训练 gate values（仅数千参数，所有模型权重冻结），2,000 steps on 8×A100 完成。(2) **部署时二值化**：按阈值 τ（由 sparsity quantile 决定）将 gate value 二值化，高于 τ 为 retrieval head（使用 full KV cache），否则为 streaming head（仅保留 sink + recent tokens，constant memory）。(3) **Head 重排序**：预处理时按 head 类型重排 Q/K/V 投影的输出通道，将 retrieval/streaming heads 分为连续簇，以高效 slicing/concat 替代 scatter/gather。(4) **Chunked Pre-filling 兼容**：streaming heads 的 pre-filling 复杂度从 O(L²) 降至 O(LK)（K 为 chunk size），memory 从 O(L) 降至 O(K)。实验比较 Needle-in-a-Haystack（NIAH）、LongBench（14 任务）、MMLU/MBPP/MT-Bench（短上下文）上与 H2O、TOVA、StreamingLLM、FastGen 在相同 KV cache budget 下的准确率；以及单 A100 上不同 context length 下的 decoding/pre-filling latency 和 memory。

- 硬件平台是什么，配置是什么。
  Retrieval head identification training: 8× NVIDIA A100 GPU servers。Decoding/pre-filling efficiency 评测: 单张 NVIDIA A100 GPU（80GB）。默认数值格式：BFloat16 权重和激活。KV cache pre-allocation 避免动态内存分配开销。结合量化时使用 QServe（8-bit weight + 4-bit KV cache quantization）。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-2-7B-chat、Llama-2-7B-32K-Instruct（MHA，32 heads/layer）、Llama-3-8B-Instruct、Llama-3-8B-Instruct-Gradient-1048k、Llama-3-70B-Instruct（GQA，8 KV heads/layer）、Mistral-7B-Instruct-v0.2（GQA，8 KV heads/layer）。
  Retrieval head identification 训练数据：BookSum 数据集嵌入 10 个 32-word passkeys，50 个长度区间（1K tokens → 模型最大长度），passkeys 随机插入 1000 个位置。
  Benchmark：
  - Long-context: Needle-in-a-Haystack (NIAH, 至 1048K tokens)、LongBench（21 任务含 Single-Doc QA, Multi-Doc QA, Summarization, Few-shot Learning, Synthetic, Code）
  - Short-context: MMLU（1-shot）、MBPP（0-shot）、MT-Bench（0-shot）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/mit-han-lab/duo-attention。基于 PyTorch + FlashInfer（RoPE/RMSNorm kernels）+ FlashAttention-2。

  **Phase 1: Retrieval Head Identification（训练，仅优化 gate values）**
  ```
  # 初始化
  α_{i,j} = 1.0 for all heads  # 初始假设所有 head 都是 retrieval head
  optimizer = AdamW(lr=0.02, warmup 400 steps 0.002→0.02, decay 400 steps 0.02→0.002)

  # 合成数据集构造
  for each sample:
      context = BookSum excerpt (1K→model_max_len tokens)
      for i in 1..10:
          insert random 32-word passkey at random position in context
      target = recall all 10 passkeys
      # 仅计算最后 l 个 passkey token 的 loss

  # 前向 pass (per KV head j in layer i)
  full_attn = softmax(Q @ K^T ⊙ M_causal) @ V          # 标准 causal attention
  streaming_attn = softmax(Q @ K^T ⊙ M_streaming) @ V  # Λ-like mask: 仅 sink + recent tokens
  attn_{i,j} = α_{i,j} · full_attn + (1-α_{i,j}) · streaming_attn  # 混合输出

  # Loss
  L_distill = (1/N) Σ_i Σ_{j=T-l+1}^T (H_full[i][j] - H_mixed[i][j])²  # L2 on last hidden states
  L_reg = Σ_i Σ_j |α_{i,j}|                                                # L1 sparsity
  L = L_distill + 0.05 · L_reg

  # 可训练参数仅数千个浮点数（N_layers × N_heads），所有模型权重冻结
  # 使用 FSDP2 + DeepSpeed Ulysses sequence parallelism 支持长序列
  # 2,000 steps on 8×A100 完成
  ```

  **Phase 2: Deployment**
  ```
  # 二值化（按 sparsity quantile 阈值 τ）
  for each head (i,j):
      if α_{i,j} > τ:
          type_{i,j} = "retrieval"    # full attention, 全量 KV cache
      else:
          type_{i,j} = "streaming"    # streaming attention, constant KV cache

  # Head 重排序 — 预处理时重排 Q/K/V 投影权重
  # 将 retrieval heads 和 streaming heads 分为两个连续簇
  # 推理时使用 slicing/concat 而非 scatter/gather

  # Decoding（per layer）
  Q_ret, Q_str = split(Q, head_dim)  # 沿 head 维度切分
  K_ret = full_KV_cache_ret          # 全量历史 KV
  V_ret = full_KV_cache_ret
  K_str = const_KV_cache_str         # 仅 sink (64 tokens) + recent (256 tokens)
  V_str = const_KV_cache_str

  out_ret = FlashAttention(Q_ret, K_ret, V_ret)      # full attention for retrieval heads
  out_str = FlashAttention(Q_str, K_str, V_str)      # streaming attention for streaming heads
  output = concat([out_ret, out_str], head_dim) @ W_O

  # Chunked Pre-filling（streaming heads 优化）
  for each chunk of K tokens:
      K_chunk, V_chunk = compute_KV(chunk)
      # 仅保留 sink + recent tokens，其余立即 evict
      K_str = prune_to_sink_and_recent(K_str, K_chunk)
      V_str = prune_to_sink_and_recent(V_str, V_chunk)
      # 下一 chunk 仅需 attend 到 constant 数量的 contextual tokens
  # 复杂度: time O(LK) instead of O(L²), memory O(K) instead of O(L)
  ```

  **张量维度**（以 Llama-2-7B MHA 为例，25% retrieval ratio）：
  - Full KV cache: 所有 32 heads × 32K tokens × 128 dim × 2 (K+V) × 2 bytes (BF16) = ~512 MB
  - DuoAttention KV cache: 8 retrieval heads × 32K × 128 × 2 × 2 = ~128 MB + 24 streaming heads × (64+256) × 128 × 2 × 2 = ~39 MB → 总计 ~167 MB（节省 ~2.55×）

  **关键配置**：
  - Llama-2-7B (MHA): retrieval ratio 25%（可选更低至 ~10% for 2.55× memory/2.18× latency reduction）
  - Llama-3-8B (GQA): retrieval ratio 50%（可选更低至 ~50% for 1.67× memory/1.50× latency reduction）
  - Sink tokens: 64（deployment, from 128 in identification）
  - Recent tokens: 256（deployment, from 256 in identification）
  - Pre-filling chunk size: 32,000

  **结合量化**：DuoAttention + QServe (W8A8KV4) → Llama-3-8B 单 A100 容纳 3.3M tokens（6.4× capacity increase vs full attention BF16）。

## Cross-Self KV Cache Pruning for Efficient Vision-Language Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 Cross-Self Pruning (CSP)，一种 training-free 的 KV cache 剪枝方法，专为多模态视觉语言模型 (VLM) 设计。核心设计包含两部分：(1) **Cross-Self Attention Decomposition**：将原始注意力矩阵 A ∈ [0,1]^{L×L} 分解为 intra-modality attention（同一模态内的 self-attention: A^{st} ∈ [0,1]^{L_t×L_t}, A^{sv} ∈ [0,1]^{L_v×L_v}）和 inter-modality attention（跨模态的 cross-attention: A^{ct} ∈ [0,1]^{L_v×L_t}, A^{cv} ∈ [0,1]^{L_t×L_v}）。对两类注意力分别沿 query 轴求和得到重要性分数 A^s 和 A^c，然后独立进行 top-K 选择得到 M^s 和 M^c，最终 mask M = M^s ∧ M^c 取交集（即 token 必须在 intra- 和 inter- 两个维度都被判定为重要才保留）。同时使用 observation window O（最近 O 个 query token）和 recent window R 来剪裁注意力矩阵 A[-O:, :-R]，聚焦于最近上下文的实际需求；(2) **n-Softmax 平滑恢复**：剪枝后 attention 分布的 denominator 从 Σ_{j∈I^+ ∪ I^-} e^{O_j} 变为 Σ_{j∈I^+} e^{O_j}，导致注意力分数变得更加尖锐（sharpness-shift）。引入 n-softmax：A_i = e^{O_i} / (n + Σ_{j∈I^+} e^{O_j})，通过加上偏置 n（默认 n=1）恢复原始分布的平滑性。实验比较 MileBench 基准上与 H2O、SnapKV、ReCo、LOOK-M 系列方法的性能（29 个多模态子任务），以及不同 cache budget（10%/20%/30%/60%/100%）下的效率（解码延迟 + GPU 内存）。

- 硬件平台是什么，配置是什么。
  LLaVA-v1.5-7b 实验：NVIDIA RTX 4090 GPU，flash-attn-2.4.3post1。LLaVA-v1.5-13b 实验：NVIDIA A100 GPU，flash-attn-2.6.3。抽样温度 0（确定性生成），最大上下文长度 4096 tokens。MMCoQA/NeedleInAHaystack/GPR1200 数据集 batch_size=1，其余数据集 batch_size=24。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaVA-v1.5-7b（主要）、LLaVA-v1.5-13b、InternVL-v1.5-7B、MobileVLM-V2-3B。
  Benchmark：MileBench（29 个多模态数据集），分为：
  - Temporal Multi-Image Tasks (T1-T4)：Action Localization/Prediction/Sequence, Object Existence/Interaction/Moving Attribute/Shuffle, Egocentric Navigation/Moving Direction, Counterfactual Inference/State Change/Character Order/Scene Transition
  - Semantic Multi-Image Tasks (S1-S5)：Webpage QA/Textbook QA/Complex Multimodal QA, Slide QA/OCR QA/Document QA, Spot-the-Diff/CLEVR-Change, MMCoQA/ALFRED, nuScenes
  - Needle in a Haystack (NH)：Text & Image NeedleInAHaystack
  - Image Retrieval (IR)：GPR1200
  评估指标：各子任务内数据集平均准确率/ROUGE-L。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/TerryPei/CSP。算法 pipeline 如下：

  **Algorithm 1 核心伪代码**：
  ```
  Input: O ∈ R^{H×L_q×L_k} (attention logits), K, V caches, budget T, recent size R, observation window O
  for each decoding iteration:
      if L_k < T: return K, V  // cache 未满，不剪枝
      
      // Step 1: n-Softmax 计算注意力权重（平滑恢复）
      A = n-Softmax(O)  // A_i = e^{O_i} / (n + Σ_{j∈I^+} e^{O_j}), n=1
      
      // Step 2: Cross-Self 分解
      // 从 A 中分出 4 个子矩阵：
      A^{st} = A[:L_t, :L_t]           // text→text self-attention
      A^{sv} = A[L_t:, L_t:]           // visual→visual self-attention
      A^{ct} = A[L_t:, :L_t]           // visual→text cross-attention
      A^{cv} = A[:L_t, L_t:]           // text→visual cross-attention
      
      // Step 3: 分别计算 intra- 和 inter- 重要性分数
      A^s = Σ_{k=1}^{L_t} A^{st}_k ⊕ Σ_{k=1}^{L_v} A^{sv}_k  // 沿 query 轴求和
      A^c = Σ_{k=1}^{L_t} A^{ct}_k ⊕ Σ_{k=1}^{L_v} A^{cv}_k
      
      // Step 4: 独立 top-K 选择
      M^s = TopK(A^s, K^s)  // 从 intra-modality 角度选 top-K^s
      M^c = TopK(A^c, K^c)  // 从 inter-modality 角度选 top-K^c
      
      // Step 5: 取交集 + 拼接 recent tokens
      M = M^s ∧ M^c   // token 必须在两个维度都重要
      K = (K ⊙ M) ⊕ K[-R:]  // element-wise mask + 拼接 recent tokens
      V = (V ⊙ M) ⊕ V[-R:]
  ```

  **张量维度说明**：
  - A ∈ [0,1]^{L×L}, L = L_t + L_v（text + visual token 总数）
  - A^{st} ∈ [0,1]^{L_t×L_t}, A^{sv} ∈ [0,1]^{L_v×L_v}
  - A^{ct} ∈ [0,1]^{L_v×L_t}, A^{cv} ∈ [0,1]^{L_t×L_v}
  - M^s, M^c ∈ {0,1}^L（binary mask），M = M^s ∧ M^c
  - 实际使用 A[-O:, :-R] 剪裁版本（在 observation window + 最近的 token 范围内计算）

  **跨 self 比率选择**：K^s 和 K^c 的比例根据数据集特征调整。大多数数据集使用平衡比例（50% intra + 50% inter）。特殊数据集：EgocentricNavigation 使用 bias=0.5（多为 inter-attention），SlideVQA 使用 bias=1.5（偏 self-attention）。IR（Image Retrieval）任务 cross_ratio=0.9（90% inter-attention）。

  **n-Softmax 关键公式**：
  原始 softmax：A_i = e^{O_i} / Σ_{j∈I^+∪I^-} e^{O_j}
  剪枝后：A_i = e^{O_i} / Σ_{j∈I^+} e^{O_j}（分母变小 → 分数增大 → 分布变尖锐）
  n-Softmax：A_i = e^{O_i} / (n + Σ_{j∈I^+} e^{O_j})（加偏置 n 恢复平滑性），n=1

  **关键性能数据**（LLaVA-v1.5-7b, RTX 4090）：
  | Budget | Decoding Latency | GPU Mem   |
  |--------|-----------------|-----------|
  | 100%   | 26.023 ms/token | 1.571 GiB |
  | 60%    | 24.377 ms/token | 1.207 GiB |
  | 30%    | 21.027 ms/token | 0.523 GiB |
  | 10%    | 16.287 ms/token | 0.208 GiB |

## CSKV: Training-Efficient Channel Shrinking for KV Cache in Long-Context Scenarios

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：基于低秩分解的 KV Cache 通道收缩（Channel Shrinking）。核心为：(1) 对 Key/Value 权重矩阵 W^K, W^V 进行低秩分解 W^K ≈ A^K B^K，其中 A^K ∈ R^{hin×hcomp}，B^K ∈ R^{hcomp×hout}，hcomp < hout，存储中间低维特征 hcomp 作为压缩 KV Cache；(2) 双分支 KV Cache：近期 m 个 token 保留完整精度（SVD 不降维），历史 token 使用压缩表示；(3) ASVD (Activation-aware SVD) 初始化 + 逐层 MSE 重建损失微调，仅需 90 分钟/A100。
  - 实验比较：在 LongEval、LongBench、LVEval 三个长上下文 benchmark 上，对比 CSKV 与 StreamingLLM（token pruning）、H2O（token pruning）、ASVD（channel shrinking）在 50% 和 80% 压缩率下的性能。消融：初始化方法（Random vs SVD vs ASVD）、窗口大小（2-4096）、KV 压缩率分配、4-bit 量化兼容性。

- 硬件平台是什么，配置是什么。
  - 训练：单张 NVIDIA A100-80G GPU；微调一个 7B 模型耗时 90 分钟。
  - 推理评测硬件：论文未明确说明具体 GPU 型号。

- 模型是什么。数据集和bench分别是什么。
  - 模型：LongChat-7B-v1.5-32k（LLaMA 架构）、Mistral-7B-Instruct-v0.2。
  - 微调数据集：scaled-down version of the Pile（HuggingFace: ola13/small-the_pile），epoch=1，batch_size=1，AdamW optimizer，lr=5e-5。
  - Benchmark：LongEval（200/300/400/500 lines 子集，平均长度 4k-10k）、LongBench-E（qasper, hotpotqa, multifieldqa_en, gov_report, triviaqa）、LVEval（16K 子集）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/wln20/CSKV

**核心张量计算流程：**

```
# ========= 权重低秩分解 =========
# 对每层的 W_K, W_V 做 ASVD 低秩分解
W_K ∈ R^{hin×hout} → A_K ∈ R^{hin×hcomp}, B_K ∈ R^{hcomp×hout}
W_V ∈ R^{hin×hout} → A_V ∈ R^{hin×hcomp}, B_V ∈ R^{hcomp×hout}
# hcomp << hout（压缩率 = (hout - hcomp) / hout）

# ========= Prefilling 阶段 =========
# 输入 X ∈ R^{n×hin}（prompt 有 n 个 token）
K = X @ W_K                         # 完整 Key，用于 attention 计算
K_C = X @ A_K                       # 压缩 Key → 存入 Compressed Key Cache (n, hcomp)
K_local = K[-m:, :]                 # 保留最后 m 个 token 完整精度

# Value 同理

# ========= Decoding 阶段（以第 (n+1) 个 token 为例）=========
# 当前 token x ∈ R^{1×hin}
k = x @ W_K                         # 完整 Key
k_C = x @ A_K                       # 压缩 Key

# 更新缓存（Compressed Key Cache 有 n+1 token，Full Cache 有 m+1 token）
Compressed_Key_Cache.append(k_C)    # → (n+1, hcomp)
Full_Key_Cache.append(k)            # → (m+1, hout)

# 重建完整 Key 矩阵用于 Attention
K_hat_empty = Compressed_Key_Cache[:(n-m), :]  # 旧 token 的压缩特征
K_hat = K_hat_empty @ B_K                       # 低维 → 高维重建
K_final = concat([K_hat, Full_Key_Cache])       # 拼接得到完整 Key

# 保持窗口大小 m：移除 Full Cache 中最旧 token
```

**逐层重建训练流程：**

```
# SVD-based 初始化（ASVD, α=0.5, Absolute Mean Value scaling）
# 从标定数据采样 256 个样本计算缩放矩阵 S
for layer in range(n_layers):
    A_K[layer], B_K[layer] = ASVD_decompose(W_K[layer], calib_data)
    A_V[layer], B_V[layer] = ASVD_decompose(W_V[layer], calib_data)

# 逐层训练
for layer in range(n_layers):
    for X in train_loader:
        K = X @ W_K[layer].T          # 原始 Key 激活
        K_hat = X @ A_K[layer].T @ B_K[layer].T  # 低秩重建 Key
        loss_K = MSELoss(K, K_hat)
        
        V = X @ W_V[layer].T
        V_hat = X @ A_V[layer].T @ B_V[layer].T
        loss_V = MSELoss(V, V_hat)
        
        loss = loss_K + loss_V
        loss.backward()
        optimizer.step()

# 全局损失: L_all = Σ_{j=1}^{n_l} (L_{K,j} + L_{V,j})
```

**量化集成（KIVI 4-bit）：**
- 80% 通道压缩 + 4-bit 量化 = 95% 总压缩率
- PTQ 直接量化崩溃（Avg.Acc 0.00），必须使用 QAT
- QAT 模式：80% 通道压缩 + 4-bit → 95% 总压缩 → Avg.Acc 0.90（vs baseline 0.99）

## MoBA: Mixture of Block Attention for Long-Context LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  MoBA（Mixture of Block Attention）是一种将 Mixture of Experts（MoE）原理应用于注意力机制的长上下文稀疏注意力架构。核心设计：(1) **Block Partitioning and Routing**：将完整上下文划分为 n 个等大小的 KV block（block size B=N/n），每个 query token 通过 gating 网络（inner product with mean-pooled K）计算与每个 block 的 affinity score s_i = ⟨q, mean_pool(K[I_i])⟩，使用 top-k gating 选择最相关的 k 个 block；(2) **Causality Preservation**：禁止 query 关注 future blocks（s_i=−∞），强制每个 token 必须关注当前 block 并施加 causal mask，当前 block 类似 MoE 中的 shared expert；(3) **Hybrid of MoBA and Full Attention**：MoBA 与 full attention 参数等价（无参数增减），支持在训练阶段无缝切换——两阶段训练先 MoBA 后 full attention，或 layer-wise hybrid（最后几层保留 full attention）；(4) **Fine-Grained Block Segmentation**：类似 MoE 中 fine-grained expert segmentation 可提升性能。

  实验比较 MoBA vs Full Attention 在：(a) scaling law 实验（5 个模型规模 568M-2.1B，Chinchilla scaling，seqlen 8K/32K，block size=512, top-k=3，sparsity 81.25%-95.31%）；(b) hybrid training 策略（MoBA 90% tokens + Full Attn 10% tokens vs full-only vs MoBA-only）；(c) layer-wise hybrid SFT（最后 N 层为 full attention）；(d) 大规模下游评估（Llama-8B-1M-MoBA vs Llama-8B-1M-Full，从 Llama 3.1 8B 出发做 continual pre-training 至 1M context，block size=4096, top-k=12, sparsity 95.31%，最后 3 层保留 full attention）；(e) 效率基准（forward pass 时间 vs FlashAttention，1M-10M context，speedup 6.5× at 1M, 16× at 10M）。

- 硬件平台是什么，配置是什么。
  Scaling law 实验：论文未明确说明 GPU 型号。大规模评估（Llama-8B-1M）：多个 GPU，使用 tensor parallelism（将 K/V broadcast 到 distributed query heads 解决 10M context 显存限制）。效率测试（Section 3.4）：对比 FlashAttention baseline 在单 GPU 上的 forward pass 时间（图 2）。基于 FlashAttention 和 DeepSpeed-MoE 的实现。

- 模型是什么。数据集和bench分别是什么。
  模型：5 个 scaling law 模型（568M/822M/1.1B/1.5B/2.1B，配置见 Table 1），Llama 3.1 8B Base 作为继续预训练起点→Llama-8B-1M-MoBA（32 layers, 最后 3 层 full attention, 29 层 MoBA）。
  
  Benchmark：AGIEval, BBH, CEval, GSM8K, HellaSWAG, Loogle, Competition Math, MBPP, MBPP Sanitized, MMLU, MMLU Pro, OpenAI HumanEval, SimpleQA, TriviaQA, LongBench@32K, RULER@128K, Needle in the Haystack (up to 1M)。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/MoonshotAI/MoBA。基于 PyTorch + FlashAttention + DeepSpeed-MoE 实现。

  **Algorithm 1 张量级 pipeline（MoBA 前向）**：
  ```
  输入: Q, K, V ∈ R^{N×h×d}, block size B, top-k
  n = N/B  # number of blocks

  # Step 1: Split KV into blocks
  K̃_i, Ṽ_i = split_blocks(K, V, B)  # i ∈ [n], K̃_i ∈ R^{B×h×d}

  # Step 2: Compute gating scores (Eq. 6)
  K̄ = mean_pool(K, B)  # R^{n×h×d}, mean pooling along seq dim per block
  S = Q @ K̄^T          # R^{N×h×n}, affinity score per query per head per block

  # Step 3: Causal mask + top-k gating (Eq. 5)
  M = create_causal_mask(N, n)     # mask future blocks to -inf
  G = topk(S + M, k)               # G: binary gating matrix R^{N×h×n}

  # Step 4: Organize query-to-block assignments
  # Self-attention block: current block (always attended, causal=True)
  Q^s, K̃^s, Ṽ^s = get_self_attn_block(Q, K̃, Ṽ)
  # MoBA attention blocks: top-k selected historical blocks (causal=False)
  Q^m, K̃^m, Ṽ^m = index_select_moba_attn_block(Q, K̃, Ṽ, G)

  # Step 5: Compute attention via FlashAttention varlen
  O^s = flash_attn_varlen(Q^s, K̃^s, Ṽ^s, causal=True)
  O^m = flash_attn_varlen(Q^m, K̃^m, Ṽ^m, causal=False)

  # Step 6: Combine with online softmax (tiling)
  O = combine_with_online_softmax(O^s, O^m)
  ```

  **计算复杂度**：每个 query 仅关注 k 个 block（每个 block B tokens）+ 当前 block = (k+1)B tokens，复杂度从 O(N²) 降至 O((k+1)B·N) = O(kN²/n)。例如 N=1M, B=4096, k=12 时 sparsity = 1−(4096×13)/1M = 94.7%。

  **Hybrid Training Recipe（两阶段）**：
  - Stage 1: 90% tokens 使用 MoBA 训练
  - Stage 2: 10% tokens 切换到 full attention 训练
  - 切换时无显著 loss spike

  **Layer-wise Hybrid (推理/SFT)**：
  - 最后 3 层保留 full attention，其余层使用 MoBA
  - SFT 阶段 prompt tokens 被 mask 掉 loss，稀疏梯度从 unmasked tokens backprop 受限
  - 使用推理时 switching：prefill 用 MoBA，generation 用 full attention

  **具体配置 (Llama-8B-1M-MoBA)**：
  - Context: 128K→256K→512K→1M continual pre-training
  - MoBA: block size=4096, top-k=12, sparsity=95.31%
  - Layer-wise: 最后 3 层 full attention, 29 层 MoBA
  - Position interpolation (Chen et al. 2023) 用于 256K transition

## Elastic Attention: Test-time Adaptive Sparsity Ratios for Efficient Transformers

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 Elastic Attention，一种 test-time 自适应稀疏注意力机制，通过轻量级 Attention Router 根据输入动态调整模型的整体稀疏度。核心设计：(1) **Attention Router**：每层引入一个由 Task MLP + Router MLP 组成的小型路由模块（仅 0.27M 参数/层），以 Key hidden states 为输入，通过 pooling 沿序列维压缩得到 task representation，经两阶段 MLP 输出 head-wise 的二值路由决策 r_{hard}^{(ℓ,h)} ∈ {0,1}，决定每个 KV head 使用 FA（r=0）还是 SA（r=1）计算模式；(2) **Gumbel-Softmax + STE 优化**：训练时使用 Gumbel-Sigmoid 连续松弛 + Straight-Through Estimator 解决 argmax 不可微问题，配合温度退火（τ=τ_init·exp(-r·p) 衰减至 τ_min），实现训练-推理一致性；(3) **Lagrangian 约束训练目标**：min-max 优化 max_{λ1,λ2} min L_language + λ1·L_diff + λ2·L_diff²，其中 L_diff = Ω_MSR - t，t 为任务相关的 target sparsity（sparsity-sensitive tasks t=0.7，sparsity-robust tasks t=1.0），λ 为可训练 Lagrange 乘子；(4) **backbone 冻结**：仅训练 Attention Router 参数（~0.27M/层），所有预训练模型权重冻结。采用 decoupled LR 策略（router LR=5e-4，regularization LR=1e-3）。实验比较 LongBench-E（14 tasks, 6 categories）、RULER（8K-256K 长度外推）和 LongBench-V2（long-form reasoning）三个长上下文 benchmark 上与 DuoAttention、PruLong、InfLLM-V2、MoBA、NSA、XAttention 的性能和 Ω_MSR sparsity。

- 硬件平台是什么，配置是什么。
  训练：8× NVIDIA A800 GPU，每轮训练 12 小时完成，BF16 精度，FSDP + Hybrid Sharding。推理评测：论文未明确说明推理用 GPU 型号（基于 LOOM-Eval 框架，使用单 GPU）。

- 模型是什么。数据集和bench分别是什么。
  模型：Qwen3-4B、Qwen3-8B（Yang et al., 2025）和 Llama-3.1-8B-Instruct（Grattafiori et al., 2024）。训练数据集：ChatQA2-Long-SFT-data、MuSiQue、CoLT-132K、GovReport、XSum 五源混合，覆盖 sparsity-sensitive（Single-Doc QA, Multihop QA）和 sparsity-robust（code completion, summarization, ICL）任务，序列长度 8K-64K，总约 0.74B tokens。Benchmark：LongBench-E（14 sub-tasks across 6 categories）、RULER（8K/16K/32K/64K/128K/256K）、LongBench-V2（Easy + Hard settings）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/LCM-Lab/Elastic-Attention。模型：https://modelscope.cn/collections/LCM_group/Elastic-Attention

  **算法 Pipeline 详细流程**：

  **Phase 1: Attention Router 前向（per layer ℓ）**
  ```
  Input: Key hidden states x_K ∈ R^{s×H×d'}（s=seq_len, H=KV heads, d'=head_dim）

  # Step 1: Pooling along sequence dimension → task representation
  # 默认 boundary-pooling: 仅聚合前 100 + 后 100 tokens
  x_K' = Pooling(x_K)  # [H, d']

  # Step 2: Task MLP → 提取 task-specific 特征
  z_task = MLP_task(x_K')  # [H, d_task]

  # Step 3: Router MLP → 输出 head-wise routing logits
  z = MLP_router(z_task)  # [H, 2]
  # z[h, 0] = FA score, z[h, 1] = SA score

  # Step 4: Gumbel-Sigmoid 连续松弛（训练时）
  g = -log(-log(u + ε) + ε)  # u ~ Uniform(0,1), Gumbel noise
  r_soft = σ((z + g) / τ)   # [H, 2], soft routing probabilities
  # 温度 τ 按 τ(t) = max(τ_min, τ_init · exp(-r·p)) 退火

  # Step 5: Hard routing via argmax
  r_hard[h] = argmax_c(r_soft[h, c])  # c ∈ {0,1}, 0=FA, 1=SA

  # Step 6: STE gradient
  r_hard[h] = r_hard[h] + (r_soft[h] - gradient_detach(r_soft[h]))
  ```

  **Phase 2: Hybrid Attention 计算（per layer ℓ）**
  ```
  # 按 head 类型分组
  I_full = {h | r_hard[h] = 0}   # FA heads
  I_sparse = {h | r_hard[h] = 1}  # SA heads

  # FA heads: 标准 causal attention
  O_r[h] = softmax(Q[h] @ K_full^T / √d') @ V_full  # h ∈ I_full

  # SA heads: Streaming Sparse Attention (SSA) 或 XAttention (XA)
  # SSA: 仅保留 sink tokens (128) + local window (2048)
  K_tilde = K[sink ∪ recent], V_tilde = V[sink ∪ recent]
  O_s[h] = softmax(Q[h] @ K_tilde^T / √d') @ V_tilde  # h ∈ I_sparse

  # Concat all heads
  O = concat([O_r[h for h in I_full], O_s[h for h in I_sparse]], dim=head_dim)
  ```

  **Phase 3: 训练目标（min-max optimization）**
  ```
  # 计算 model sparsity ratio
  Ω_MSR = (1/(H·L)) · Σ_l Σ_h I[r_hard[l,h] = SA]

  # 损失函数
  L_language = CE_Loss(y | f_θ(x))
  L_diff = Ω_MSR - t  # t: target sparsity
  Total = L_language + λ1·L_diff + λ2·L_diff²

  # Lagrange multipliers 更新（gradient ascent）
  λ1 += lr_λ · ∂Total/∂λ1
  λ2 += lr_λ · ∂Total/∂λ2
  ```

  **关键超参数**：
  - Sequence length: 65536 (training)
  - Global batch size: 48
  - Training steps: 300
  - Router LR: 5e-4, Reg LR: 1e-3
  - AdamW (β1=0.9, β2=0.95), weight decay=0.1
  - Cosine LR schedule, 20% warmup
  - Gumbel temperature: τ_init→τ_min via exp decay (r=0.6)
  - Sparsity targets: t_robust=1.0, t_sensitive=0.7
  - SA config: SSA (sink=128, local=2048) or XA (τ=0.9, default params)
  - Block-Sparse-Attention: block_size=64, chunk_size=16384, sink=128

  **算法核心创新**：将下游任务分为 sparsity-robust（summarization 等粗粒度任务）和 sparsity-sensitive（QA 等细粒度检索任务）两类，通过 Attention Router 在 test-time 自动判断任务类型并分配相应 sparsity level，无需 per-task 手动调参。实验表明 Elastic Attention 在长上下文 benchmark 上实现了与 full attention 可比甚至超越的性能（如 Llama-3.1-8B 在 LongBench-E avg 53.35 vs backbone 53.28），同时 Ω_MSR 达 0.69（FA-SSA）或 0.77（FA-XA），实现 efficient inference。

## Discovering the Gems in Early Layers: Accelerating Long-Context LLMs with 1000x Input Token Reduction

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 GemFilter，一种 training-free 的 inference 策略，利用 LLM 早期层的 attention 矩阵作为 filter 来选择和压缩输入 token，将长上下文输入从 128K 压缩到 ~100-4096 个 token，然后仅将选中的 token 送入完整模型进行生成。核心机制：(1) 第一遍（Prompt Computation Phase）：仅运行 LLM 的前 r 层（filter layer），获取第 r 层的 query 和 key 矩阵 Q^{(r)}, K^{(r)}，对多 head attention 的最后一 query token 对所有 key token 的 attention scores 求和（J ← topk_index(Σ_{j=1}^h Q_n^{(r,j)} K^{(r,j)^T}, k)），选出 top-k 个最高 attention 的 token 索引；(2) 对索引排序回原始输入顺序；(3) 第二遍（Iterative Generation Phase）：将选中的 k 个 token T_J 送入完整 LLM 进行标准生成。实验比较 Needle in a Haystack 和 LongBench benchmark 上与 standard attention (full KV cache)、SnapKV 和 H2O 的准确率、运行时间、GPU 内存消耗。GemFilter 在 Needle in a Haystack 上显著优于 standard attention 和 SnapKV，LongBench 上与 SnapKV/H2O 可比；实现 2.4× 加速和 30% GPU 内存减少。

- 硬件平台是什么，配置是什么。
  Needle in a Haystack 和 LongBench 实验：NVIDIA A100-40GB GPU。运行时间和 GPU 内存实验：NVIDIA H100-80GB GPU。LLaMA 3.1 8B 在双 A100-40GB 上运行（需双卡支持 128K 上下文）。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA 3.1 8B Instruct (32 layers, 128K context)、Mistral Nemo 12B Instruct (40 layers, 128K context)、Phi 3.5 Mini 3.8B Instruct (32 layers, 128K context)。Benchmark：(1) Needle in a Haystack——压力测试检索能力，LLaMA 3.1 使用 120K 输入长度，Mistral Nemo 使用 60K 输入长度；(2) LongBench——多任务长上下文理解 benchmark，涵盖 14 个数据集：Single-Doc QA (NarrativeQA, Qasper, MultiFieldQA-en)、Multi-Doc QA (HotpotQA, 2WikiMultihopQA, Musique)、Summarization (GovReport, QMSum, MultiNews)、Few-shot Learning (TREC, TriviaQA, SAMSum)、Synthetic (PassageCount, PassageRetrieval-en)。评估使用 greedy decoding (num_beams=1, do_sample=False)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/SalesforceAIResearch/GemFilter。依赖：transformers==4.43.3, flash-attn==2.6.3, Python 3.12。算法 pipeline 如下：

  **核心算法（PyTorch 伪代码）**：
  ```python
  # Step 1: 第一遍——用前 r 层做 filter，选出 top-k token
  def find_context(self, query_states, key_states, k):
      key_states = repeat_kv(key_states, self.num_key_value_groups)
      # 仅用最后一个 query token 做 top-k 选择
      top_k_indices = top_index(key_states, query_states[:, :, -1:, :], k)
      return torch.sort(top_k_indices, dim=-1).indices

  def top_index(keys, queries, k, kernel=5):
      # 计算最后一 query token 与所有 key token 的内积
      in_pro = torch.matmul(queries, keys.transpose(-1, -2))  # [1, h, 1, n]
      # 跨所有 head 求和
      in_pro = torch.sum(in_pro, dim=1, keepdim=True)         # [1, 1, 1, n]
      # 1D average pooling 做聚类（kernel=5, stride=1）
      in_pro = F.avg_pool1d(in_pro, kernel=kernel, padding=kernel//2, stride=1)
      # 取 top-k
      return torch.topk(in_pro, k, dim=-1).indices
  ```

  **张量计算流程**：
  1. 输入 token 序列 T ∈ V^n（n=128K），选定 filter layer index r（如 LLaMA 3.1 的 r=13/32）和压缩目标 k（如 1024）
  2. 运行前 r 层 forward：F_{1:r}(T) → 获取第 r 层的 Q^{(r)} ∈ R^{n×d}, K^{(r)} ∈ R^{n×d}
  3. 对多 head attention：J ← topk_index(Σ_{j=1}^h Q_n^{(r,j)} K^{(r,j)^T}, k)，其中 Q_n^{(r,j)} 是第 j 个 head 最后一 query token 的 query 向量
  4. 对 J 排序回原始顺序，得 sorted_J
  5. 构造压缩输入 T_J ∈ V^k（仅保留 sorted_J 中索引对应的 token）
  6. 送入完整 LLM 生成：Gen(F_{1:m}, T_J)，使用标准 greedy decoding

  **时间复杂度对比**（n=128K, k=1024, m=32, r=13）：
  - Prompt Computation: Standard = Θ(mhn²d), SnapKV/H2O = Θ(mhn²d), GemFilter = Θ(rhn²d) → 约 r/m = 40% 的 prompt 计算量
  - Iterative Generation: Standard = Θ(mh(nt+t²)d), SnapKV/H2O = Θ(mh(kt+t²)d), GemFilter = Θ(mh(k²+t²)d)
  - Prompt 阶段 GPU 内存: Standard/SnapKV = mw + 2mhnd, GemFilter = rw + 2hnd（仅需要加载前 r 层权重）

  **Filter Layer 选择**：
  - LLaMA 3.1 8B (32 layers): r=13
  - Mistral Nemo 12B (40 layers): r=19
  - Phi 3.5 Mini 3.8B (32 layers): r=19
  - 消融实验（Table 2）：性能随 layer index 先升后降，layer 13-25 之间性能鲁棒

  **关键性能数据**（LLaMA 3.1 8B, H100）：
  - Speedup: 2.4× vs SnapKV/Standard attention
  - GPU Memory: 30% reduction vs SnapKV, 70% reduction vs Standard attention
  - Needle in a Haystack: GemFilter-1024 average score 0.887 (LLaMA), 0.838 (Mistral Nemo)，显著优于 Standard attention 和 SnapKV
  - LongBench (LLaMA 3.1, k=1024): GemFilter avg 34.50 vs SnapKV 35.25 vs Standard 36.72; (key=2048): GemFilter 35.87 vs SnapKV 35.80
  - LongBench (Mistral Nemo, k=4096): GemFilter avg 46.79 vs SnapKV 46.04 vs Standard 46.36
  - 与 SnapKV/H2O 的本质差异：GemFilter 使用单一 token 索引集 J（可打印供人工审查），SnapKV/H2O 使用 m·h 个独立索引集

  **使用示例**：
  ```
  python needle_eval.py \
    --model hf_model_id \
    --modified gemfilter \
    --topk 1024 \
    --ctx_len 32000
  ```

## Exploiting Sparsity for Long Context Inference: Million Token Contexts on Commodity GPUs

- 属于算法pipeline的实现是什么？实验比较什么？
  提出一种基于 attention 稀疏性的 top-k attention 机制：将完整 KV cache 存放在 CPU 内存中，对每个 query token 使用 Faiss 向量数据库做 approximate nearest neighbor search（ANN），仅检索 top-k 个最相关的 key-value pair 传输到 GPU 进行 attention 计算，将 GPU 内存占用从 O(N) 降至 O(k)。核心算法如 Algorithm 1：(1) 在 CPU 上对所有 layer、所有 head 的 key 张量分别构建 Faiss ANN index（支持 exact search 或 approximate HNSW graph）；(2) decoding 时每层 Query 在 GPU 上完成 QKV projection 后，query 向量传输到 CPU，通过 ANN search 获取 top-k 的 key indices 和 scores；(3) 仅传输选中的 k 个 value 向量和 scores 到 GPU，在 GPU 上执行 softmax+加权求和，与 GPU 本地的近期生成 token 的 KV cache（windowed attention）合并。Prefill 阶段使用 FlashAttention + H100 构建完整 KV cache（1M token 级采用 chunking strategy）。实验比较：(a) RULER benchmark（4K-128K context，13 tasks）上不同 k 值与 Full Attention 的性能对比；(b) Open LLM Leaderboard（MMLU, ARC, HellaSwag, Winogrande, OpenbookQA, BoolQ, PiQA）上多模型多 k 值的平均准确率；(c) AlpacaEval 2.0 上的 LC win rate；(d) 1M token Needle In A Haystack 对比 StreamingLLM (attention sink) cache eviction；(e) Uniform k vs 自适应 layer-wise k 分配的 RULER 性能对比。

- 硬件平台是什么，配置是什么。
  Prefill（构建 KV cache）：NVIDIA H100 GPU（用于 1M token 级 prefill，利用 FlashAttention + chunking）。
  Decoding（top-k attention）：单张 commodity GPU（约 16GB VRAM，论文以 ~16GB GPU RAM 运行 1M token 上下文解码），CPU 侧为通用 host memory 用于存放完整 KV cache 和执行 Faiss ANN search。
  CPU-GPU 数据传输：仅传输 k 个 value 向量（k 远小于 N），避免 full KV cache 往返传输的瓶颈。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-1 7B、Llama-2 7B、Llama-3 8B、Llama-3.1 8B、Vicuna-v1.3 7B、Llama-2 Chat 7B、Llama-3 Instruct 8B、Llama-3.1 Instruct 8B、Llama-3.2 1B Instruct、Llama-3.2 3B Instruct、GradientAI Llama-3-8B（训练至 262K/1M context length，用于 RULER 和 1M NIAH）。
  数据集/Benchmark：
  - RULER（Hsieh et al., 2024）：13 tasks in 4 categories（Needle In A Haystack + 7 NIAH variants, Summarization proxies, Multi-hop proxies, QA），context length: 4K, 8K, 16K, 32K, 64K, 128K
  - Open LLM Leaderboard v1：MMLU, ARC-Easy, ARC-Challenge, HellaSwag, Winogrande, OpenbookQA, BoolQ, PiQA（使用 lm-eval-harness, zero-shot）
  - AlpacaEval 2.0：805 queries，LLM-as-a-Judge (GPT-4 Turbo)，metric: LC win rate percentage
  - 1M token Needle In A Haystack（RULER subset，用于 extreme scaling 实验）
  注意力稀疏性分析数据集：50× 4000-token concatenated Wikipedia article snippets

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/ryansynk/topk-decoding。算法 pipeline 如下：

  **Stage 1 — KV Cache Prefill（一次性，CPU 侧存储）**：
  给定输入 tokens x ∈ R^{N×D}，使用标准 forward pass（FlashAttention + H100，或 Ring Attention 分布式）计算所有 L 层各 head 的 key/value activations：
  ```
  for ℓ in 1..L:
      K_ℓ, V_ℓ = model.forward_layer_ℓ(x)  // shape: (N, D) per head, stored on CPU
  ```
  长序列采用 chunking strategy：将 N 切分为 chunks，逐 chunk prefill 后 concatenate KV cache。

  **Stage 2 — Top-k 索引构建（CPU 侧）**：
  对每层、每个 head 的 key 张量 K_ℓ^h ∈ R^{N×d_k} 构建 Faiss ANN index：
  ```
  for ℓ in 1..L:
      for h in 1..H:
          K_index[ℓ][h] = faiss.IndexFlatIP(d_k)  // exact inner product search
          // 或 faiss.IndexHNSWFlat(d_k, M)  // approximate HNSW graph search
          K_index[ℓ][h].add(K_ℓ^h)  // 插入所有 N 个 key 向量
  ```
  dot product metric 直接对应 attention score 计算 q·K^T。

  **Stage 3 — Top-k Decoding Loop（CPU+GPU 协同）**：
  ```
  K_gen = [[] for ℓ in 1..L]  // GPU 侧近期生成 token 的 key cache
  V_gen = [[] for ℓ in 1..L]  // GPU 侧近期生成 token 的 value cache

  for each new token x ∈ R^{1×D}:
      for ℓ in 1..L:
          // Step 1: QKV projection on GPU
          q = x @ W_Q^ℓ     // (1, d_k)
          k = x @ W_K^ℓ     // (1, d_k)
          v = x @ W_V^ℓ     // (1, d_v)

          // Step 2: 将 q 传至 CPU，执行 ANN search
          q_cpu = q.to(cpu)
          vals, I = K_index[ℓ][h].search(q_cpu, k_per_head)
          // vals ∈ R^k: top-k inner product scores
          // I ∈ Z^k: top-k key indices

          // Step 3: 仅传输选中的 value 和 scores 到 GPU
          V_sel = V_ℓ[I].to(gpu)   // (k, d_v)
          vals_gpu = vals.to(gpu)  // (k,)

          // Step 4: GPU attention — context 部分（CPU 检索的 top-k）
          attn_ctx = softmax(vals_gpu / sqrt(d_k)) @ V_sel   // (1, d_v)

          // Step 5: GPU attention — 近期生成 token 部分（windowed，直接计算）
          attn_gen = softmax(q @ K_gen[ℓ][h]^T / sqrt(d_k)) @ V_gen[ℓ][h]  // (1, d_v)

          // Step 6: 合并两部分 attention 输出
          attn_out = attn_ctx + attn_gen

          // Step 7: 更新 GPU 侧 window cache
          K_gen[ℓ][h] = concat(K_gen[ℓ][h], k)
          V_gen[ℓ][h] = concat(V_gen[ℓ][h], v)

          // Step 8: 后续 FFN 等 transformer layer 计算...

      x_new = sample(attn_out @ W_out)
  ```

  **复杂度**：Full attention decoding 每 token O(N·D) 计算 + O(N·D) 显存；Top-k attention 每 token O(k·D) GPU 计算 + O(N·D) CPU 内存（完整 KV cache）+ O(k·D) CPU-GPU 数据传输 + O(log N) ANN search on CPU（HNSW）。

  **Per-layer adaptive k budget**：给定总 budget K_total = Σ_ℓ k_ℓ，按 linear increasing from first to last layer 分配 k_ℓ（vs uniform），在 fixed total budget 下获得 non-trivial performance boost。

  **使用示例**：
  ```
  # 构建 KV cache（prefill）
  python prefill.py --model meta-llama/Llama-3-8B --context documents.txt --max_length 1000000

  # Top-k decoding
  python generate.py --model meta-llama/Llama-3-8B --cache cache.pkl --k 128 --max_new_tokens 256
  ```

## FastKV: KV Cache Compression for Fast Long-Context Processing with Token-Selective Propagation

- 属于算法pipeline的实现是什么？实验比较什么？
  FastKV 提出一种解耦预填充计算与 KV Cache 压缩的两阶段推理框架，核心包含：(1) Token-Selective Propagation (TSP)：在模型中间层（如 LLaMA-3.1-8B 的 layer 15）根据基于注意力权重的 saliency score 选择关键 token 向后层传播，早期层保持完整上下文计算；(2) Layer-wise KV Retention：每个 decoder layer 独立根据注意力重要性分数压缩 KV cache，KV 保留率与预填充计算解耦（两个独立超参数：TSP rate 和 KV retention rate）。实验比较 LongBench（16 个子任务，含单文档 QA、多文档 QA、摘要、少样本学习、合成任务、代码补全）、RULER（检索/聚合/多跳追踪，最长 128K）和 Needle-in-a-Haystack（16K-128K）上的准确率，以及单张 A100 SXM GPU 上端到端时延（预填充 + 256 token 解码）的加速比。Baseline 包括 StreamingLLM、H2O、SnapKV（仅解码加速）和 PyramidInfer、GemFilter（预填充感知加速）。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 SXM GPU（单卡）；FlashAttention-2 kernel。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA-3.1-8B-Instruct（32 decoder layers，GQA，支持 128K context window）、Ministral-8B-Instruct（36 decoder layers，GQA，128K context window）、Mistral-Nemo-12B-Instruct（40 decoder layers，GQA，128K context window）。数据集/benchmark：LongBench（英文子集 + 代码，16 个子任务：NrtvQA、Qasper、MF-en、HotpotQA、2WikiMultihopQA、MuSiQue、GovReport、QMSum、MultiNews、TREC、TriviaQA、SAMSum、LCC、RepoBench-P、PassageCount、PassageRetrieval-en）、RULER（11 项子任务，context length: 8K/16K/32K/64K/128K）、Needle-in-a-Haystack（16K-128K，步长 16K）。标定数据集：论文未明确说明具体标定数据，仅说明使用 Equation 3 基于少量标定输入的 hidden state L2 距离自动选择 TSP 层。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/dongwonjo/FastKV。基于 HuggingFace Transformers 的 self-attention 实现 + FlashAttention-2 kernel 集成。

  **算法 pipeline 详解（两阶段预填充 + 独立 KV 压缩）：**

  **阶段一 —— 完整上下文预填充（Layer 0 到 TSP Layer L_TSP）：**
  ```
  for l = 0 to L_TSP:
      X, Att_l, K_X, V_X = layer_l(X)                        # 完整上下文注意力，构建 (N_I, d) 的 K_X, V_X
      K, V = KV_Compress(K_X, V_X, Att_l, R_KV)               # 每层基于 attention score 独立压缩 KV cache
      if l == L_TSP:
          # 计算 saliency score：使用 window tokens (N_obs=8) 作为 query
          S_i^{l,h} = Pooling(Σ_{n=0}^{N_obs} Att_l[h, N_I - n, i + m])  # Eq(1): MaxPooling kernel_size=7
          S_i^{TSP_layer} = (1/H) * Σ_h S_i^{TSP_layer,h}                 # Eq(2): 跨 head 均匀平均
          I_TSP = TopK(S^{TSP_layer}, N_I * R_TSP) ∪ window_indices      # 选取 top-R_TSP 的 token + window tokens
          x = X[I_TSP]                                                     # 仅传播选中的 hidden states (shape: N_I*R_TSP, d)
  ```
  其中 `R_TSP = 0.2`（默认），`N_obs = 8`（观察窗口大小），pooling kernel size = 7。

  **阶段二 —— 压缩上下文预填充（TSP Layer+1 到 Last Layer）：**
  ```
  for l = L_TSP+1 to L-1:
      x, Att_l, K_x, V_x = layer_l(x)                         # 仅在压缩后的 hidden states x 上计算注意力
      K, V = KV_Compress(K_x, V_x, Att_l, R_KV)               # 对压缩上下文同样执行 KV 保留
  ```

  **KV_Compress 核心逻辑：**
  ```
  KV_Compress(K_X, V_X, Att_l, R_KV):
      # 基于 group-wise saliency score（head-wise saliency 在 KV group 内平均）
      for each KV group g:
          S_g = (1/|heads_in_group|) * Σ_{h∈g} S_h           # head-wise 聚合为 group-wise
      I_KV = TopK(S, context_length * R_KV)                    # 按 R_KV 比率选择 top critical tokens
      return K[I_KV], V[I_KV]
  ```

  **TSP Layer 自动选择（Eq 3）：**
  ```
  L_TSP = argmin_{L ≤ L_max} (1/N) Σ_{i=1}^{N} ||H_i - H'_{L,i}||₂²
  ```
  其中 H_i 为完整上下文下最终层 hidden state，H'_{L,i} 为在 layer L 处应用 TSP 后的最终层 hidden state。L_max 约束防止 TSP 层过晚导致预填充节省有限。LLaMA-3.1-8B 选择 layer 15（共 32 层），Ministral-8B 选择 layer 17（共 36 层），Mistral-Nemo-12B 选择 layer 19（共 40 层）。

  **解耦设计核心：** TSP rate（控制预填充计算量）与 KV retention rate（控制解码时 KV cache 大小）完全独立。TSP rate=20% → 预填充计算率 60%，KV retention rate 可独立设为 10% 或 20%。

  **数值结果：**
  - LLaMA-3.1-8B，128K 上下文：预填充加速 1.82×，解码加速 2.87×
  - Ministral-8B，128K 上下文：端到端加速 >2×
  - LongBench 平均准确率（LLaMA，TSP=20%, KV=20%）：49.07 vs Full-context 50.19（下降 1.12 个百分点）
  - Token importance estimation 开销（128K 上下文）：0.15s，仅占预填充总时延 0.88%
  - Needle-in-a-Haystack（LLaMA，KV=10%）：FastKV 99.9 vs Full-context 99.0（TSP 帮助模型聚焦全局关键 token，甚至超越完整上下文）

## GTA__Grouped-head_latenT_Attention

- 属于算法pipeline的实现是什么？实验比较什么？
  论文提出 **GTA (Grouped-head latenT Attention)**，一种新的注意力机制，包含两个组件：(1) **Shared Attention Map**：将 query 和 key 按 heads 分组，同一 group 内的 heads 共享同一套 query-key 注意力计算，从而减少 MHA 中每个 head 独立计算的冗余——每个 head 映射到某个 Q group 和某个 K group，仅对 (Q_{q(i)}, K_{k(i)}) 计算注意力分数，而非每个 head 独立计算 Q_i K_i^T；(2) **Nonlinear Value Decoder**：引入一个压缩的 latent value representation C ∈ R^{N × n_c × d_l}（d_l ≥ d_h），对 latent value 先用 head-specific 投影矩阵 W_{P,i} ∈ R^{d_l × d_h} 生成 value，再通过 element-wise 乘以 sigmoid gate Sigmoid(x_t W_{G,i})（对当前 token 输入自适应）调制，实现非线性、上下文自适应的 head-specific value 生成。实验比较 MHA、GQA、MLA 三种 baseline，评估下游任务准确率、预填充/解码时延、KV cache 大小。

- 硬件平台是什么，配置是什么。
  训练：4 节点 × 8 NVIDIA A800 80GB GPU（共 32 GPU），分布式训练，支持 1~4 节点弹性扩展。
  推理（LLM-Viewer 模拟）：NVIDIA A100 40GB、A100 80GB、H100 80GB、H100 PCIe 80GB。
  推理（实际部署）：NVIDIA H100 80GB、NVIDIA A800 80GB、NVIDIA RTX 3060 12GB、Apple M2、BCM2712（移动处理器）。

- 模型是什么。数据集和bench分别是什么。
  模型：160M 参数（24 层，hidden=768，n_h=12），500M 参数（24 层，hidden=1280，n_h=20），1B 参数（54 层，hidden=1280，n_h=20）。GTA 变体 GTA1~GTA4 采用不同 n_q/n_k/n_c 分组数（如 GTA1: n_q=3, n_k=1, n_c=1, d_l=128；GTA4: n_q=10, n_k=1, n_c=2, d_l=256）。
  预训练数据集：C4（160M/500M 验证实验）、smollm-corpus 220B tokens（1B 扩展实验）。
  微调数据集：tulu3-sft-mixture。
  Benchmark：PIQA、HellaSwag、ARC-e、ARC-c、Winogrande、BoolQ、MathQA、TruthfulQA、SIQA、LogiQA、BBH、MBPP、IFEval、Wikitext PPL。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/plm-team/GTA。训练框架基于自定义 PyTorch 实现，微调使用 LlamaFactory [39]，评估使用 lm-evaluation-harness [25]。

  **GTA 算法详细张量计算流程（以 1B 模型为例，n_h=20, n_q=5, n_k=1, n_c=1, d_h=64, d_l=128）：**

  **Step 1: 输入投影（Eq 5）**
  ```
  X ∈ R^{N × 1280}   # N tokens, hidden=1280

  Q = X @ W_Q        # W_Q ∈ R^{1280 × 320},  Q ∈ R^{N × 320}  (5 Q groups × 64 head_dim)
  K = X @ W_K        # W_K ∈ R^{1280 × 64},   K ∈ R^{N × 64}   (1 K group × 64 head_dim)
  C = X @ W_C        # W_C ∈ R^{1280 × 128},  C ∈ R^{N × 128}  (1 C group × 128 latent_dim)
  ```

  **Step 2: 按分组映射 head**
  ```
  q(i): {0..19} → {0..4}   # 20 heads 映射到 5 Q groups
  k(i): {0..19} → {0}      # 20 heads 共享 1 套 key
  c(i): {0..19} → {0}      # 20 heads 共享 1 套 latent value
  ```

  **Step 3: 非线性 Value 解码（Eq 6）**
  ```
  for each head i in 0..19:
      # 从共享 latent value C 投影生成 head-specific value
      V_i = (C @ W_{P,i}) ⊙ sigmoid(x_t @ W_{G,i})
      # W_{P,i} ∈ R^{128 × 64}: d_l → d_h
      # W_{G,i} ∈ R^{1280 × 64}: H → d_h, x_t 是当前 token
      # ⊙ 为 element-wise 乘法
  ```

  **Step 4: 注意力计算（Eq 7, 等价公式 Eq 8）**
  ```
  for each head i in 0..19:
      # 共享 attention map：使用 Q group 的 query 对 K group 的 key 计算 score
      Q_group = Q[q(i) * 64 : (q(i)+1) * 64]    # (N, 64)
      K_group = K                                 # (N, 64), 所有 heads 共享
      C_group = C                                 # (N, 128)

      # Eq 8: 先对 latent value 做 attention，再 gate
      attn_scores = Q_group @ K_group^T / sqrt(64)     # (N, N)
      attn_weights = softmax(attn_scores)               # (N, N)
      # 在 latent 空间计算 attention
      O_i_raw = attn_weights @ C_group                   # (N, 128)
      # 投影到 head_dim 并 gate
      O_i = (O_i_raw @ W_{P,i}) ⊙ sigmoid(x_t @ W_{G,i}) # (N, 64)
      # 输出投影
      O_i = O_i @ W_{O,i}                                # (N, 1280)
  ```

  **Step 5: 合并输出**
  ```
  O = sum(O_i for i in 0..19)    # (N, 1280)
  ```

  **KV Cache 写入**：仅 cache K (64 dims/token) 和 C (128 dims/token)，共 192 dims/token/layer vs MHA 的 2560 dims/token/layer (=7.5%)。decode 时仅需追加 1 个 token 的 K 和 C 并重新计算 gate。

  **关键优化 tricks：**
  - Equation 8 的 reformulation 使得 decode 时无需重新计算所有历史 V_i（仅需存 C 并执行 latent-space attention），大幅减少 decode 的 FLOPs。
  - Shared attention matrix 减少了 QK^T 计算次数：从 MHA 的每个 head 独立计算（n_h 次）降至每个 Q-K group 组合的计算（n_q 次），当 n_q << n_h 时显著节省。
  - 数值结果（1B 模型）：预填充 FLOPs 降至 GQA 的 37.5%，KV cache 降至 GQA 的 30%，预填充时延 2× 加速，解码时延也显著改进。

## HATA: Trainable and Hardware-Efficient Hash-Aware Top-k Attention for Scalable Large Model Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  HATA提出Hash-Aware Top-k Attention，将learning-to-hash系统性地集成到top-k attention中。与现有方法追求qk scores精确数值估计不同，HATA将query和key映射为二进制hash codes（rbit=128），通过Hamming距离获取相对qk score排序，以极低成本实现top-k key选择。核心实现：(1) Hash Modeling：定义query-key hashing优化问题 min Σ_i s_i||h(q)-h(k_i)||² + η||Σh(k_i)||² + λ||W_H^T W_H-I||，使用sigmoid松弛sign函数支持梯度训练，每head独立训练hash权重W_H∈R^{d×128}；(2) Training Data Construction：prefill阶段从Q/K pairs采样，top 10%标记为正样本(线性衰减标签[1,20])，其余为负样本(标签-1)；(3) HATA Prefill：额外计算K_H=HashEncode(K)缓存hash codes；(4) HATA Decode：HashEncode新Q和K→bitwise_xor+bitcount计算Hamming距离→TopK选N个最近keys→sparse attention。实验比较LongBench-e/RULER accuracy vs Dense/Loki/Quest/MagicPIG/StreamingLLM/H2O/SnapKV，end-to-end/prefill/decode效率，HATA-off vs MagicPIG KVCache offloading，scalability to 14B/32B models和256K context，hash bits/token budget/optimizations ablation。

- 硬件平台是什么，配置是什么。
  48GB HBM GPU (最高149.7 TFLOPS FP16)，96 cores。Ubuntu 24.04，CUDA 12.1，PyTorch 2.4，FlashInfer。效率评估使用batch_size=1~8，sequence length=8K~256K。Offloading实验使用PCIe 4.0 + 48 CPU threads。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-2-7B-32K-Instruct (MHA, 32 layers×32 heads, d=4096, ctx=32768)，Llama-3.1-8B-Instruct (GQA, 32 layers×32 heads/8 KV heads, d=4096, ctx=131072)，Qwen2.5-14B-Instruct-1M (GQA, 48 layers×40 heads/8 KV heads, d=5120, ctx=1M)，Qwen2.5-32B-Instruct (GQA, 64 layers×40 heads/8 KV heads, d=5120, ctx=131072)。
  Benchmark：LongBench-e (12 tasks)，RULER (11 tasks, 32K-256K)，InfiniteBench，LongBench-v2，Needle-in-a-Haystack。
  Hash训练数据：Qasper(5短序列)、LSHT和RepoBench-P(各2中序列)、LongBench-v2(2超长序列)，覆盖中英文QA和code understanding，最终150K-300K qk pairs。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/gpzlx1/HATA。代码量：1470行C++/CUDA + 940行Python。算法pipeline：

  **Phase 1: Hash Training**
  ```
  # 对每个attention head独立训练W_H ∈ R^{d×128}
  for each head:
      W_H = init(d, 128)
      for epoch in 1..15:
          for iter in 1..20:                         # 20 iterations/epoch
              # h(x) = 2 * Sigmoid(σ * x @ W_H) - 1, σ=0.1
              loss = ε*Σ_i s_i*||h(q)-h(k_i)||²     # similarity preservation
                   + η*||Σ_i h(k_i)||²               # bits balance
                   + λ*||W_H^T @ W_H - I||           # bits uncorrelation
              W_H = SGD(lr=0.1, momentum=0.9, wd=1e-6)(loss)
  ```
  超参数：σ=0.1, ε=0.01, λ=1.0, η=2.0, chunk_size=32K, 2-3 chunks/epoch

  **Phase 2: HATA Prefill**
  ```
  Q, K, V = Proj(X)                      # standard QKV projection
  K_H = HashEncode(K)                    # [s, 128/32] = [s, 4] INT32
  K_H_cache = K_H                        # cache hash codes
  K_cache, V_cache = K, V                # standard KVCache
  O = DenseAttention(Q, K, V)            # dense output for prefill
  
  # HashEncode(K):
  #   K_H = Sign(K @ W_H)                # [s, d] × [d, 128] → [s, 128] binary
  #   K_H = BitPack(K_H)                 # pack 128 bits → 4 INT32
  ```
  Prefill额外开销：O(s×d×rbit)，rbit=128 ≪ s，实际<1%

  **Phase 3: HATA Decode (核心加速)**
  ```
  Q, K, V = Proj(x)                      # projection for single new token
  K_cache = [K_cache; K]                 # update KVCache
  V_cache = [V_cache; V]
  Q_H = HashEncode(Q)                    # [1, 4] INT32
  K_H = HashEncode(K)                    # [1, 4] INT32
  K_H_cache = [K_H_cache; K_H]           # update hash cache
  
  # Hamming distance computation
  S = bitcount(bitwise_xor(Q_H, K_H_cache))  # [1, s], s=seq_len
  # GQA: aggregate S across shared KV head queries
  
  Idx = TopK(S, N)                       # N = top-k token budget
  K_sparse = Gather(K_cache, Idx)        # [N, d]
  V_sparse = Gather(V_cache, Idx)        # [N, d]
  O = FlashAttention(Q, K_sparse, V_sparse)  # [1, d]
  ```
  Decode复杂度：O(s×rbit/32 + s log N + N×d) vs Dense O(s×d)，N≪s

  **关键性能数据**：
  - Llama2 batch=8 seq=32K: 7.20× speedup over Dense, 1.99× over Loki
  - Llama2 batch=1 seq=256K: 6.51× over Dense, 2.21× over Loki, 1.19× over Quest
  - HATA-off vs MagicPIG on Llama2: 6.04× prefill + 2.54× decode speedup
  - Accuracy: LongBench-e avg 34.60 (Llama2) / 53.94 (Llama3.1) vs Dense 34.47/54.10

## HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 HISA（Hierarchical Indexed Sparse Attention），一种免训练的即插即用式层级索引器，替代 DeepSeek Sparse Attention (DSA) 中 O(L²) 复杂度的 flat token scan indexer。核心设计为两阶段层级搜索：(1) **Block-level 粗过滤**：将前缀划分为大小为 B 的连续 block，对每个 block 内 indexing keys 做 mean pooling 得到代表向量，query 对所有 ⌈L/B⌉ 个 block 代表向量打分，保留 top-m 个 block；(2) **Token-level 精筛**：在保留的候选 block（最多 mB 个 token）内，使用与原始 DSA 相同的 token-level indexer 打分，选出最终 top-k token。HISA 输出与 DSA indexer 完全相同的数据结构（每个 query 的 k 个 token 索引集），下游 Sparse MLA 算子完全不变。每 query 索引复杂度从 O(L) 降至 O(L/B + mB)，每层从 O(L²) 降至 O(L²/B + LmB)。block size B 控制粗过滤粒度，top-m 控制候选池大小，满足 mB ≥ k 的可行性约束。首尾 block 强制保留以处理 attention sink 和局部上下文。实验比较三种索引策略：(a) DSA (原始 full-prefix token-level indexer)、(b) Block-Sparse (仅 Stage 1，无 token 精筛)、(c) HISA (完整两阶段)。在 kernel-level latency（8K-64K context）、Needle-in-a-Haystack (NIAH, 8K-648K context)、LongBench（6 类任务）上比较，以及 attention score 可视化和超参数敏感性分析。

- 硬件平台是什么，配置是什么。
  Kernel-level latency 测试：单张 NVIDIA A100 GPU，使用 TileLang kernel 实现。End-to-end 评测：vLLM online serving framework，FP8 精度部署 DeepSeek-V3.2 和 GLM-5。NIAH 评测使用基于 RULER (https://github.com/NVIDIA/RULER) 修改的评估代码库。LongBench 评测使用 lm-eval-harness framework (https://github.com/EleutherAI/lm-evaluation-harness)。对于 GLM-5，因 OOM 问题调整了部分任务的 concurrency（longbench_single concurrency=1，longbench_summary concurrency=2），默认 num_concurrent=20。

- 模型是什么。数据集和bench分别是什么。
  模型：DeepSeek-V3.2（采用 DSA + Sparse MLA，MQA mode）、GLM-5（同样采用 DSA 范式）。Benchmark：(1) Needle-in-a-Haystack (NIAH)——8K 至 648K tokens context，needle 深度 0%-100%，评估 retrieval accuracy，基于 RULER 修改的评估代码，不使用 chat template；(2) LongBench——bilingual multi-task long-context understanding benchmark，覆盖 6 类任务：Single-Doc QA (SQA)、Multi-Doc QA (MQA)、Summarization (Sum)、Few-shot Learning (FS)、Synthetic Retrieval (Syn)、Code Completion (Code)。所有评估均为 zero-shot 设置。DeepSeek-V3.2 使用标准 chat template，GLM-5 不使用 chat template。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/MuLabPKU/TransArch（论文声称的仓库地址，但截至当前仓库内 HISA 代码标记为"Release HISA code ☐"尚未发布）。DSA 的参考 TileLang kernel 实现在 https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32。算法 pipeline 如下：

  **HISA 两阶段层级索引（Algorithm 1 核心流程）**：
  ```
  输入: query indexing representations {q_{t,j}^I}, gating weights {w_{t,j}^I},
        token indexing keys {k_s^I}_{s=1}^L, block size B, block budget m, token budget k
  输出: 每 query 的 selected token set T_t (size k)

  // Stage 0: Block 划分与 Pooling
  M = ceil(L / B)
  for b = 1 to M:
      k̃_b^I = MeanPool({k_s^I | s ∈ B_b})

  // 对每个 query position t
  for each query position t:
      // Stage 1: Block-level 粗过滤（公式 5-7）
      for b = 1 to M (B_b causally precedes t):
          J_{t,b} = Σ_j w_{t,j}^I · ReLU(q_{t,j}^I · k̃_b^I)
      C_t = TopK(J_{t,:}, m) ∪ {first block, last block}
      Ω_t = ⋃_{b ∈ C_t} B_b            // 候选 token 集，|Ω_t| ≤ mB

      // Stage 2: Token-level 精筛（公式 8-9，与 DSA 相同机制）
      for s ∈ Ω_t:
          I_{t,s} = Σ_j w_{t,j}^I · ReLU(q_{t,j}^I · k_s^I)
      T_t = TopK({I_{t,s} | s ∈ Ω_t}, k)

  // T_t 送入 Sparse MLA（公式 3，与 DSA 完全相同）
  u_t = Attn(h_t, {c_s | s ∈ T_t})
  ```

  复杂度：per-query O(L/B + mB)，per-layer O(L²/B + LmB)。vs DSA indexer O(L²)。

  三 regime 边界行为：
  - t ≤ k: 等价 dense attention
  - k < t ≤ mB: 等价 DSA（粗过滤器全选）
  - t > mB: HISA 层级优势激活

  默认超参数：B=128, m=64 (candidate=8192), k=2048。也测试 (B=64,m=128) 和 (B=256,m=32)，均保持 mB=8192。Block-Sparse baseline: B=128, m=16 (candidate=2048, 无 token 精筛)。

  **关键性能数据**：
  - 64K context kernel speedup: 2.16× (4:1 ratio) ~ 3.75× (fixed 8K budget) vs DSA indexer
  - NIAH: HISA 接近 DSA 性能，远超 Block-Sparse（Block-Sparse 在 needle 位于中间位置时显著退化）
  - LongBench: DeepSeek-V3.2 Avg: DSA 51.05, HISA 50.78, Block 49.54; GLM-5 Avg: DSA 46.01, HISA 46.32, Block 42.67
  - 超参数敏感性：B=64/128 优于 B=256（更细粒度 block 精筛更准），所有 HISA 配置均远优于 Block-Sparse

## Hardware-Efficient Attention for Fast Decoding

- 属于算法pipeline的实现是什么？实验比较什么？
  提出两种新的硬件高效注意力变体：(1) **GTA (Grouped-Tied Attention)**：将 key 和 value 的投影参数绑定为单一的 *tied KV* 状态，对每组 query head 共享一个 distinct tied KV head。Key 路径仅使用 tied KV 的前半维度（不加 RoPE），另一半 key 维度来自独立的单头 RoPE 投影并广播到组内所有 head；Value 路径使用 tied KV 的完整维度。KV cache 大小相对于同 group 数的 GQA 减半、算术强度翻倍。(2) **GLA (Grouped Latent Attention)**：将 MLA 的单头 latent 压缩扩展为多 latent head，每个 latent head 维度 d_c = 2d_h（MLA 为 4d_h），latent head 可在 TP rank 间分片，避免 MLA 的 latent 在全设备复制的问题。解码时吸收低秩 up-projection 矩阵，每个 latent head 仅服务于其 group 内的 query head。实验比较：在 Small (183M)、Medium (433M)、Large (876M)、XL (1.471B) 四个 GPT-3 规模上与 MHA、MQA、GQA-4、MLA 对比 perplexity（FineWeb-Edu + 5 数据集平均）和 downstream accuracy（SciQ, OpenBookQA, ARC-Easy, HellaSwag, PIQA, WinoGrande, MMLU 共 7 benchmark）。

- 硬件平台是什么，配置是什么。
  训练：论文未明确说明具体 GPU 型号和数量。推理 kernel benchmark：NVIDIA H100 80GB SXM5 GPU（BF16 峰值 989 TFLOPS/s，HBM 带宽 3350 GB/s）。多 GPU serving benchmark：8× H100 80GB GPU，NVLink 互联，使用 DeepSeek-Coder-V2 Base（236B 参数，21B active）FP8 量化模型。

- 模型是什么。数据集和bench分别是什么。
  模型：GPT-3 配置 + Llama 3 架构（SwiGLU FFN, RMSNorm, RoPE），四个规模——Small (183M, 12 layers, d=768, hq=12, dh=64)、Medium (433M, 24 layers, d=1024, hq=16, dh=64)、Large (876M, 24 layers, d=1536, hq=16, dh=96)、XL (1.471B, 24 layers, d=2048, hq=16, dh=128)。各 variant 通过加宽 FFN 匹配 MHA 的参数总量。训练数据：FineWeb-Edu-100B（small 25B tokens，其余 50B tokens），Llama 3 tokenizer（vocab 128K），AdamW (β1=0.9,β2=0.95, weight decay=0.1, grad clip=1.0)，cosine LR decay to 1%。Perplexity 评估：FineWeb-Edu validation、Cosmopedia、RedPajama v1 C4、RedPajama v1 Wikipedia、Pile（各 100M tokens）。Downstream benchmark：SciQ、OpenBookQA、ARC-Easy、HellaSwag、PIQA、WinoGrande、MMLU（zero-shot）。Serving benchmark 模型：DeepSeek-Coder-V2 Base (236B, 21B active, FP8)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/Dao-AILab/grouped-latent-attention（permissive license）。

  **GTA 算法 pipeline**：
  ```
  # 输入: hidden states X ∈ R^{B×L×d}
  # hq query heads, hkv KV heads (group size gq = hq/hkv), dh per-head dim

  # 1. Q 投影
  Q = X @ W^Q          # [B, L, hq, dh]

  # 2. Tied KV 投影（单一投影矩阵替代 W^K 和 W^V）
  KV = X @ W^{KV}      # [B, L, hkv, dh] — 单一 tied state

  # 3. 构造 K 和 V
  V = KV               # [B, L, hkv, dh] — 完整维度用于 value
  K_NoPE = KV[:,:,:,:dh/2]      # 前半维度，不加 RoPE
  K_RoPE = X @ W^{K_RoPE}       # [B, L, 1, dh/2] — 单头 RoPE key
  K_RoPE = apply_rope(K_RoPE)
  K_RoPE_bcast = broadcast(K_RoPE, hkv)  # 广播到所有 KV head
  K = concat([K_NoPE, K_RoPE_bcast], dim=-1)  # [B, L, hkv, dh]

  # 4. Attention (GQA-style grouping)
  for each group g of gq query heads:
      attn[g] = softmax(Q_g @ K_g^T / sqrt(dh)) @ V_g

  # 5. Output projection
  O = concat(all groups) @ W^O
  ```
  KV cache per token: hkv × dh × sizeof(dtype)（vs GQA: hkv × 2 × dh），算术强度 ≈ 2gq（vs GQA: ≈ gq）。结合了低秩 key 的洞察（仅部分维度需要 RoPE）、KV 状态共享（key 和 value 源自同一 state），以及 GQA 的分组并行设计。

  **GLA 算法 pipeline (GLA-2, hc=2 latent heads)**：
  ```
  # hc=2 latent heads, dc=2dh, gq = hq/hc

  # --- 训练时: down+up projection ---
  c_0^{KV} = X @ W^{DKV}_0   # [B, L, 2dh]
  c_1^{KV} = X @ W^{DKV}_1   # [B, L, 2dh]
  K_0 = c_0^{KV} @ W^{UK}_0   # [B, L, gq*dh]
  V_0 = c_0^{KV} @ W^{UV}_0
  # 类似地 K_1, V_1

  # --- 解码时: weight absorption ---
  # W^{UK} 被吸收进 W^Q, W^{UV} 被吸收进 W^O
  # 直接对 latent c^{KV} 计算 attention:
  Q_0 ∈ R^{B×1×gq×(2dh)}, Q_1 ∈ R^{B×1×gq×(2dh)}
  O_0 = softmax(Q_0 @ c_0^{KV}^T / sqrt(2dh)) @ c_0^{KV}
  O_1 = softmax(Q_1 @ c_1^{KV}^T / sqrt(2dh)) @ c_1^{KV}

  # --- 分布式执行 (TP=2) ---
  # rank 0: c_0^{KV}, Q_0, W^{VO}_0 → partial O_0
  # rank 1: c_1^{KV}, Q_1, W^{VO}_1 → partial O_1
  O = AllReduce(O_0 @ W^{VO}_0 + O_1 @ W^{VO}_1)
  ```
  KV cache: unsharded = hc × d_c = 2 × 2dh = 4dh（与 MLA d_c=4dh 相同）。但 TP≥2 时每 device 仅 d_c = 2dh（MLA 因单头 latent 全复制仍为 4dh）。算术强度 ≈ 2gq（双倍于 GQA）。RoPE 维度 d_R = 32（默认），通过 decoupled RoPE 机制保留位置信息。

  **关键结果**：XL (1.471B) 上 GTA-4 达到 PPL 10.129（vs GQA-4 10.202）、downstream avg 60.2%（与 GQA-4 持平）；GLA-2 达到 PPL 10.218（vs MLA 10.256）、downstream avg 60.0%（vs MLA 59.1%）。GLA kernel 在 speculative decoding (L_q=2) 下比 FlashMLA 快 2×，标准 decoding (L_q=1) 快约 20%。端到端 serving：GLA-8 (TP=8) 在 64 并发下 throughput 1461 tok/s（vs MLA TP=8 的 859 tok/s，提升 70%）；在 131K 长 prefill 不平衡负载下 GLA-8 吞吐 100 tok/s（vs MLA hybrid TP+DP 的 37 tok/s，提升 2.7×）。

## KV-Distill: Nearly Lossless Learnable Context Compression for LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  KV-Distill 提出一种可训练的 KV cache 压缩框架，通过 token 重要性打分（FFN scorer）、LoRA 条件计算适配、以及前向+反向 KL 散度蒸馏目标来将长上下文的 KV cache 压缩为更短的表示。具体流程：(1) 通过一个 FFN 对第 η=6 层的 hidden states 打分得到每个 token 的重要性分数 s ∈ R^N；(2) 取 top-k 重要 token 索引，通过 hard selection matrix S ∈ {0,1}^{k×N} 从 KV cache 中提取 ˜X = SX；(3) 将上下文通过带有 LoRA adapters 的 LM_θ 编码，其中被选中的 token 路由到可训练的 W^Q/W^O 矩阵，未选中的 token 通过冻结的原始矩阵；(4) 使用加权 KL 散度 L(θ) = λ·D_KL(p||q_θ) + (1-λ)·D_KL(q_θ||p) 匹配压缩前后的 next-token 分布。实验比较 KV-Distill 与 H2O (H2A 问题感知/H2I 问题无关)、DODO、ICAE 在提取式 QA、长文本 QA、抽象式摘要、Needle-in-a-Haystack 上的性能。

- 硬件平台是什么，配置是什么。
  训练：8 × NVIDIA A100 80GB GPU 集群，使用 DeepSpeed Stage 2 分布式训练，bf16 精度。推理评估：论文未明确说明推理硬件，但从模型规模（7B-27B）推断在单/多 GPU 上进行。

- 模型是什么。数据集和bench分别是什么。
  模型：LLAMA-2 7B、LLAMA-3 8B、MISTRAL 7B、GEMMA-2 9B、GEMMA-2 27B（均使用 instruction-tuned 版本）。训练数据：从 Self-Instruct、P3、LongAlpaca、Super-Natural Instructions 中 curated 的大规模指令数据集，拆分为 (Context, Instruction, Answer) 三元组。基准测试：SQuAD（提取式 QA，平均长度 225 tokens）、QuALITY（长文本多选题 QA，平均 6K tokens）、SQuALITY（长文本抽象式摘要，平均 7K tokens）、GovReport（长文档摘要，平均 10K tokens）、Needle-in-a-Haystack（长文本检索）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  官方仓库 https://github.com/vnchari/kv-distill，论文声明代码和 checkpoint 即将发布，但截至本文时仅有 README，完整代码尚未公开。算法 pipeline 如下：
  ```
  # === KV-Distill 压缩流程 ===
  # 输入: context c ∈ V^N, 预训练 LM, LoRA adapter LM_θ

  # Step 1: Token Importance Scoring
  # η=6 层的 hidden states X'_η ∈ R^{N×d}
  s = FFN_θ(X'_η)              # s ∈ R^N, 每个 token 的重要性分数
  indices_topk = topk(s, k)     # 取 top-k 索引, k = N × retention_ratio

  # Step 2: 编码上下文通过 LoRA-adapted LM_θ
  for each transformer layer l:
      # 正常 forward pass
      X_l^K, X_l^V = encode(context)  # [N, d]

      # 条件计算路由:
      for selected tokens i ∈ indices_topk:
          Q_i = (z_i @ W^Q_lora)       # 使用可训练的 LoRA W^Q
          O_i = attention(Q_i, K, V) @ W^O_lora  # 使用可训练的 LoRA W^O
      for unselected tokens j:
          Q_j = (z_j @ W^Q_frozen)     # 使用冻结的原始 W^Q
          O_j = attention(Q_j, K, V) @ W^O_frozen  # 使用冻结的原始 W^O

  # Step 3: 从 LM_θ 的输出中提取压缩 KV cache
  ˜X = S @ X    # S ∈ {0,1}^{k×N}, hard selection
  # 即只保留 indices_topk 对应 token 在所有层的 KV

  # Step 4: 梯度传播（non-differentiable topk 的替代方案）
  # 在注意力计算中，对 attention weights 按重要性衰减:
  α' = σ(s) ⊙ α    # σ=sigmoid, ⊙=Hadamard product
  # 被选中 token 的 attention weight 不变（sigmoid(高分)≈1）
  # 未选中 token 的 attention weight 被衰减

  # Step 5: KL 散度蒸馏损失
  p = softmax(LM(y | X_full) / T)    # teacher: 完整 cache
  q_θ = softmax(LM(y | ˜X) / T)    # student: 压缩 cache
  L(θ) = λ·Σp·log(p/q_θ) + (1-λ)·Σq_θ·log(q_θ/p)
  # λ=0.6 (偏向 forward KL 以稳定训练)

  # Step 6: 训练细节
  # LoRA: rank=128, 应用于 Q,K,V,O 矩阵 (rsLoRA)
  # 优化器: AdamW, lr=5e-5, batch_size=32
  # 训练时随机采样 retention ratio ∈ [0.1%, 80%]
  # 长上下文 (>1536 tokens) 折叠为 batch of N×1536
  # 前几个 tokens (<10) 始终保留（sink tokens）
  ```
  关键设计：KV retention ratio 在训练时随机采样（0.1%-80%），因此单个 KV-Distill 模型支持任意压缩率。训练参数仅 150M（LoRA adapter），压缩后的 KV cache 在自回归解码时零额外开销。前 k% 的 token 选择可跨层共享索引。forward + reverse KL 混合损失（λ=0.6）优于纯 forward KL（λ=1, SQuAD 83.4%）、纯 reverse KL（λ=0, 82.7%）和 auto-encoding + CE loss（79.1%）。No routing（用可学习 embedding 替代条件计算 routing）= 67.4%。

  关键结果：LLAMA-3 8B SQuAD: KVD 25% retention 86.6%（vs uncompressed 87.6%, H2A 25% 84.0%, H2I 25% 56.6%）。Needle-in-a-Haystack: 90% compression 下近乎完美准确率。QuALITY: 10x compression 下与 uncompressed 性能接近。SQuALITY: >20% retention 时 ROUGE-L 等于或超过 uncompressed。GovReport fine-tuning: 1% retention (100x compression) ROUGE-L 22.8（vs uncompressed 23.7）。各模型蒸馏训练 3-4 天（GEMMA 27B 需 4 天）。

## LOOK-M: Look-Once Optimization in KV Cache for Efficient Multimodal Long-Context Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  LOOK-M 是一种免微调（fine-tuning-free）的多模态 KV cache 压缩方法，核心实现分为两部分：(1) **Text-Prior KV Pair Eviction**：在 prompt prefilling 阶段，利用累积注意力分数（cumulative attention scores）动态更新 KV cache，但区别于传统的无差别积累方法（如 H2O），LOOK-M 对文本 token 赋予 text-prior 值 T_p = Max(A_s) 加到其累积注意力分数上，确保文本 token 在 eviction 阶段优先保留，图像 token 中仅保留 attention score 最高的 top-N 个；(2) **KV Pairs Merging**：对被 evicted 的 KV pair 使用 many-to-one nearest-neighbor matching 找到其最相似的 conserved token，然后通过三种合并策略（averaged merging、pivotal merging、weighted merging）将 evicted token 的信息融入 conserved token 中。实验比较 LOOK-M（含 A-Merge/W-Merge/P-Merge 三种合并策略，以及叠加 text-prior TP 的组合共 6 种变体）与 text-only KV cache eviction baselines（H2O、SnapKV、RoCo）和 Full Cache 在 MileBench 基准上的准确率/ROUGE-L，以及不同 KV cache budget（5%-100%）、不同压缩率比例（α¹:α²）、不同模型架构下的性能，和 decoding latency/GPU memory 效率。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 (80GB) 和 RTX 3090 (24GB) GPU。延迟和 GPU 内存测试在 RTX 3090 单卡上进行。FlashAttention-2 加速注意力计算。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaVA-v1.5-7B、LLaVA-v1.5-13B（LLM backbone: Vicuna-7B/13B）、InternVL-v1.5-7B、MobileVLM-V2-3B。
  数据集/Benchmark：MileBench，含 4 类子任务——T: Temporal Multi-image Tasks（T1-T4，含 Action Localization/Prediction/Sequence, Object Existence/Interaction, Egocentric Navigation, Counterfactual Inference/State Change 等）、S: Semantic Multi-image Tasks（S1-S5，含 Webpage QA/Textbook QA, Slide QA/OCR/Document QA, Visual Change Captioning, Multimodal Dialogue, Space Understanding）、N: Needle in a Haystack Tasks（N1 Text Needle, N2 Image Needle）、I: Image Retrieval。评估指标为 Accuracy 和 ROUGE-L。默认 recent ratio α¹=0.1，important ratio α²=0.1。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/SUSTechBruce/LOOK-M

  **算法 Pipeline（基于 Section 3 Methodology）**：

  **输入**：多模态 prompt X = {X₁^T, X₁^I, ..., X_N^T, X_M^I}，L_prompt 为 prompt 长度，D 为 hidden dimension。最近窗口大小 M = α¹ × L_prompt，重要 token 数 N = α² × L_prompt。

  **Phase 1: Multimodal Prompt Encoding（Prefilling）**：
  ```
  for each transformer layer:
      K = X @ W_K   # shape: [L_prompt, D]
      V = X @ W_V   # shape: [L_prompt, D]
      Q = X @ W_Q   # shape: [L_prompt, D]
      A_p = softmax(Q @ K^T / sqrt(D))   # [L_prompt, L_prompt], causal
  ```

  **Phase 2: Text-Prior KV Pair Eviction（3.2 节，公式 4-8）**：
  ```
  # Step 1: 计算累积注意力分数
  A_s = sum(A_p[i,:] for i in 0..L_prompt)   # [L_prompt], 沿 query 维求和

  # Step 2: Text-Prior 增强——文本 token 获得优先级
  T_p = max(A_s)                              # 取最大 attention score 作为 text-prior
  for each textual token index t in T:
      A_s[t] = A_s[t] + T_p                   # 文本 token 分数 += text-prior

  # Step 3: 选择保留 token
  recent_kv = KV[-M:]                         # 最近 M 个 KV pair 始终保留
  I = Top_N(A_s[:-M], N)                      # 从前 L_prompt-M 个 token 中选 top-N
  K_c = concat(K[I], recent_K)                # conserved K: [N+M, D]
  V_c = concat(V[I], recent_V)                # conserved V: [N+M, D]
  K_e = K \ K_c                               # evicted K
  ```

  **Phase 3: KV Pairs Merging（3.3 节，公式 9-12）**：
  ```
  # Step 1: 计算 similarity matrix between K_e and K_c
  for i in evicted_indices I_e:
      for j in conserved_indices I_c:
          s_ij = cosine_sim(K[i], K[j])       # k_i^T k_j / ||k_i|| ||k_j||

  # Step 2: 对每个 conserved token j，找其 maximum similarity set
  for j in I_c:
      k_sim[j] = {K_e[i] | argmax_i matches j}

  # Step 3: 合并策略（三选一）
  # (a) Averaged Merging: 直接平均
  k_c[j] = 1/(L_sim + 1) * (k_c[j] + sum(k_sim[j]))

  # (b) Pivotal Merging: 先融合 evicted↔closest，再平均
  pivotal = avg(k_e[i], k_closest)            # 对每个 evicted token
  k_c[j] = 1/(L_sim + 1) * (k_c[j] + sum(pivotal))  # 再加权 conserved

  # (c) Weighted Merging: 基于 similarity 矩阵动态加权
  k_c[j] = 1/(L_sim + 1) * (k_c[j] + sum(k_sim[i] * S[x][y]))

  # Value 合并使用与 Key 相同的 similarity matrix 和权重（alignment property）
  ```

  **Phase 4: Token Generation（Decoding）**：
  ```
  for each new token x_t:
      q_t = x_t @ W_Q
      k_t, v_t = x_t @ W_K, x_t @ W_V
      K = concat(K_c, k_t)   # compressed KV + new token
      V = concat(V_c, v_t)
      x_t_out = softmax(q_t @ K^T / sqrt(D)) @ V
  ```

  **默认配置**：recent ratio α¹=0.1，important ratio α²=0.1（总 cache budget = 20%），最佳合并策略为 TP + P-Merge（text-prior + pivotal merging）。在 extreme compression ratio 99%（仅保留 1%）下 LOOK-M 仍维持接近 Full Cache 的性能。

  **关键效率数据（Table 4, RTX 3090）**：
  - Full Cache (100%): Decoding Latency 28.16 ms/token, GPU Memory 1.52 GiB
  - LOOK-M (20% budget): Decoding Latency 20.98 ms/token, GPU Memory 0.32 GiB (≈80% memory reduction)
  - LOOK-M (5% budget): Decoding Latency 18.22 ms/token, GPU Memory 0.13 GiB (≈92% memory reduction)

## LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  LaCache 是一种 training-free 的 KV cache 压缩算法，核心实现包含两个关键创新：(1) **Ladder-shaped KV cache pattern**：不同于 StreamingLLM 在所有层缓存相同 tokens 的滑动窗口策略，LaCache 将 KV 状态跨层分布——浅层保留早期 token 的 KV cache，深层逐步将焦点转移到更近期的 token，形成阶梯状（ladder-shaped）存储模式。具体通过两个超参数控制：Span S（同一 token 的 KV 状态被保留的连续层数）和 Overlap O（每层保留的 token 数量）。在相同 KV cache budget 下，这种跨层异质存储可覆盖更长的上下文；(2) **Iterative compaction mechanism**：当 KV cache 达到预设容量时，对已压缩的 KV cache 再次应用 ladder-shaped pattern，老 token 被更激进地压缩，新 token 保留更多，实现 O(1) 内存复杂度的无限连续生成。两个超参数 S、O 分别校准以在存储效率和生成精度间取得最优 trade-off。实验比较与 Full KV cache、StreamingLLM、H2O、TOVA、PyramidInfer、SnapKV 在 language modeling (PPL) 和 long-context understanding (LongBench, Needle-In-A-Haystack, RULER) 各 benchmark 上的 accuracy-efficiency trade-off，以及 score-throughput Pareto 曲线。

- 硬件平台是什么，配置是什么。
  单卡 NVIDIA A100 80GB GPU（PG19 长上下文评估，PG19 600K tokens 时使用 FlashAttention-2 加速）。单卡 NVIDIA H200 GPU（LongBench score-throughput trade-off 评估，batch size=1）。Bfloat16 精度。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama2-7B/13B、Llama2-7B/13B-Chat、Llama3-8B、Llama3.2-3B-Instruct（128K）、SmolLM2-1.7B-Instruct、LongChat-7b-v1.5（32K）。
  数据集/Benchmark：(1) Language modeling：Wikitext-2（token-by-token generation, decoding length 1K-16K）、PG19（100 books, 10M tokens, sliding window 256）；(2) LongBench（21 个数据集，bilingual long-context understanding，上下文 5K-15K）；(3) Needle-In-A-Haystack（up to 128K context, 50 repetitions）；(4) RULER（13 tasks, 16K context, 100 repetitions）。
  评估指标：Perplexity (PPL)、LongBench 各子任务 score、NIAH accuracy、RULER accuracy、throughput (tokens/s)。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/GATECH-EIC/LaCache（BSD-3-Clause license，PyTorch 实现）

  **算法 Pipeline（Ladder-Shaped KV Cache Pattern + Iterative Compaction）**：

  给定：LLM 共 L 层，每层 H 个 KV head，输入序列长度 T，KV cache budget C（以 token 数量计）。

  **Phase 1: Ladder-Shaped Pattern Eviction（Prefill 后）**：
  ```
  超参数：Span S（同一token跨层保留层数）, Overlap O（每层保留token数）
  
  for layer l in 1..L:
      # 每层保留的 token 范围（形成 ladder 形状）
      # 浅层保留更早的 token，深层逐步右移
      start_token = (l - 1) * (S - O)   # 每层右移 S-O 个 token
      end_token = start_token + O        # 保留 O 个 token
      
      # 每层仅保留 [start_token, end_token) 范围内的 KV cache
      K_cache[l] = K_full[l, start_token:end_token]
      V_cache[l] = V_full[l, start_token:end_token]
      
      # 为覆盖边界，在 ladder 起始和末尾位置额外保留更多 token（避免 gaps）
  ```
  张量维度：K_full ∈ R^{L×H×T×d}, 压缩后 K_cache ∈ R^{L×H×O×d}
  总 cache size = L × O × H × d（vs Full: L × T × H × d）

  **Span S 和 Overlap O 的校准**：
  - LongBench 理解任务：S ≈ L × compression_ratio（均匀压缩分布，50% budget → S = L/2）
  - Language modeling 任务：S = L/4（消融实验最优，Fig. 10）
  - O = S/2（language modeling，保证语义连续性）；O = 0~S/2（long-context understanding，取决于是否需要 global vs local 信息）

  **Phase 2: Iterative Compaction（持续解码时）**：
  ```
  # 当 KV cache 达到 budget C 时触发
  compacted_KV = apply_ladder_pattern(current_KV, S, O)
  # 新一轮 token 的 KV 填充 freed space
  # 随迭代次数增加，老 token 被越来越激进地压缩（经历更多次 ladder eviction）
  # 语义：距离当前 token 越远的 token，被压缩比越大
  ```
  内存复杂度：O(1)（constant KV cache size），支持理论上无限长序列的连续生成。

  **FlashAttention 兼容性（关键设计选择）**：
  LaCache 故意不依赖 attention maps 来识别重要 token（与 H2O/TOVA/SnapKV 不同），而是使用基于位置的静态 ladder pattern。这意味着：
  - 无需 materialize attention scores → 与 FlashAttention 完全兼容
  - 实际设备上可实现更高 throughput（Fig. 7 实验验证：LaCache Pareto-optimal in score-throughput trade-off on H200）

  **核心 Insight 的形式化（信息保留下界分析）**：
  - Ladder pattern 确保每个 token 至少被 S 个不同层覆盖 → 每个 token 的信息保留下界被提升
  - 所有 layer 的 token 覆盖分布尽可能均匀 → 最坏情况（重要 token 出现在覆盖最少的层）的精度损失被最小化
  - 相邻 token 在自然语言中语义关联性高 → ladder 的平滑过渡（partial overlap 在相邻层间）实现 old token 的 smooth fade-out

## LagKV: Lag-Relative Information of the KV Cache Tells Which Tokens Are Important

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：LagKV，一种无需注意力权重的 KV Cache 压缩/驱逐方法。核心机制为递归分区压缩——将 KV cache 按 lag size L 分区，使用下一个相邻 chunk 的统计量（max/min）对当前 chunk 归一化，计算 channel-wise 标准差后 softmax 得到 token 重要性分数，再对 K 和 V 的分数求和，使用 top-K 策略选择保留 Token。同时保留 attention sink（前 S 个 token）和滑动窗口（最后一个分区）。
  - 实验比较：(1) RULER benchmark（16K context）对比 SnapKV、StreamingLLM 在 0.25/0.5/0.75/0.875 压缩比下表现；(2) LongBench 和 64-digit Passkey Retrieval 消融实验，测试不同 L（128/512/1024）和 r（2×/4×/6×/8×）组合；(3) chunk-by-chunk prefill 模式下 FGT 准确率和 needle score；(4) 不同 scoring 变体对比（LocalKV、L2 norm vs LagKV）。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明硬件平台和具体 GPU 型号/数量。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Llama-3.1-8B-Instruct、Qwen2.5-7B-Instruct（均使用 GQA 以减少 KV cache 大小）。
  - Benchmark：RULER（16K，含 13 个子任务：Single-Key/Single-Value/Multi-Key/Multi-Query/Variable Tracking/Common Word Extr/Freq. Word Extr/QA1/QA2）、LongBench（含 Single-doc QA、Multi-doc QA、Summarization、Few-shot、Synthetic、Code 子任务）、Needle-in-a-Haystack（64-digit Passkey Retrieval，背景为 Paul Graham Essays）。

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：代码仓库 https://github.com/AI-Lab-China-Merchants-Bank/LagKV，集成于 NVIDIA KVPress 框架 (https://github.com/NVIDIA/kvpress)。
  - 算法 pipeline（伪代码）：
    ```
    输入: K_i, V_i ∈ R^{n×d_h} for each head i
    超参: S=16 (sink size), L (lag size), r (retention ratio)

    def LagKV_compress(K_i, V_i, S, L, r):
        # 1. 保留 attention sink（前 S 个 token）
        compressed = K_i[:, :S], V_i[:, :S]

        # 2. 统计剩余长度，若 < 2L 则不压缩
        remaining_len = n - S
        if remaining_len < 2*L:
            return concat(compressed, K_i[:, S:], V_i[:, S:])

        # 3. 按 L 分区（最后一个分区作为滑动窗口）
        n_partitions = floor(remaining_len / L)
        partitions = split(K_i[:, S:], n_partitions, dim=seq)
        V_partitions = split(V_i[:, S:], n_partitions, dim=seq)

        # 4. 递归压缩每个分区（最后一个保留不压缩）
        for p in range(n_partitions - 1):
            K_cur = partitions[p]      # 当前 chunk: (h, L, d_h)
            K_ref = partitions[p+1]    # 参考 chunk（下一分区）

            # 4a. 使用参考 chunk 计算 token-wise min/max
            min_k = min(K_ref, dim=seq)     # (h, d_h) 每个 channel 在参考分区中的最小值
            max_k = max(K_ref, dim=seq)     # (h, d_h) 每个 channel 在参考分区中的最大值

            # 4b. 归一化当前 chunk
            K_norm = (K_cur - min_k) / (max_k - min_k + eps)  # (h, L, d_h)

            # 4c. 计算 channel-wise std + softmax
            K_std = std(K_norm, dim=channel)  # (h, L)
            score_K = softmax(K_std, dim=seq) # (h, L)

            # 对 V 做同样操作
            min_v = min(V_ref, dim=seq)
            max_v = max(V_ref, dim=seq)
            V_norm = (V_cur - min_v) / (max_v - min_v + eps)
            V_std = std(V_norm, dim=channel)
            score_V = softmax(V_std, dim=seq)

            # 4d. 求和得到最终 token score
            score = score_K + score_V  # (h, L)

            # 4e. Top-K 选择（每个 head 独立）
            topk_indices = topk(score, k=r*L, dim=seq)
            kept_K = gather(K_cur, topk_indices, dim=seq)
            kept_V = gather(V_cur, topk_indices, dim=seq)
            compressed_K.append(kept_K)
            compressed_V.append(kept_V)

        # 5. 加上滑动窗口（最后一个分区）和 Mod 余数
        compressed_K.append_all(partitions[-1])
        compressed_V.append_all(V_partitions[-1])

        return concat(compressed_K), concat(compressed_V)
    ```
  - 压缩比计算公式：L_R = S + rL*(⌊(L_s - S)/L⌋ - 1) + L + Mod(L_s - S, L)；C = 1 - L_R/L_s
  - 数学直觉：token-wise locality 使得相邻 token 的 K/V 值相似，用下一 chunk 归一化可消除 channel 偏移，保留 channel-wise variance 作为重要性度量。与 KIVI 量化思路类似但用于驱逐而非量化。完全不依赖 query 态或 attention weight → 与 FlashAttention 兼容且无指令依赖偏差。

## LOCRET: Enhancing Eviction in Long-Context LLM Inference with Trained Retaining Heads on Consumer-Grade Devices

- 属于算法pipeline的实现是什么？实验比较什么？
  LOCRET 是一种轻量级训练式 KV cache 淘汰框架。在每层 transformer 注入一个小型 retaining head（两层 MLP，中间维度 d_R=1024），训练 retaining head 预测每个 KV cache unit 的 Causal Importance Score (CIS)。CIS 定义为 answer token 对该 prefix token 的最大 attention score (softmax 前)。推理时在 chunked prefill 过程中每处理一个 chunk，使用 retaining head 对 KV cache 打分，evict 低 CIS 的 cache unit 以维持固定 budget b。同时保留最后 n_s 个 token 的 KV cache 作为 stabilizers 以缓解上下文不连续性。还提出 LOCRET-Q 变体：训练时将 query token 前置以感知 query，推理时 query 插入序列首部实现 query-aware eviction。实验比较 LOCRET vs FULLATTN、InfLLM（offloading）、HF-2BITS（KV cache 量化）、SIRLLM（eviction）、MINFERENCE（sparse attention）在 ∞Bench 和 L-Eval 上的 task accuracy + peak memory，以及 NVIDIA 4090 上的推理速度（tok/s）。LOCRET-Q 与 SNAPKV、H2O、SIRLLM 在 RULER benchmark 上比较 query-driven task 性能。

- 硬件平台是什么，配置是什么。
  训练：单张 NVIDIA A800 GPU（<1 GPU hour）。推理评估（非 4090 实验）：工作站 8×NVIDIA A800/H800 GPU + 104 Intel Xeon Platinum 8470 CPU + 1.0 TB CPU 内存，Red Hat 4.8.5；单 GPU 运行除 FULLATTN 外所有实验，FULLATTN 用 2 GPU（vLLM tensor parallelism）。消费级设备速度实验：单张 NVIDIA 4090 24GB + 512 AMD EPYC 9754 CPU + 1.0 TB CPU 内存，PCIe Gen 4 (16GT/s)，Ubuntu 9.4.0。

- 模型是什么。数据集和bench分别是什么。
  模型：Phi-3-mini-128K（MHA，3.8B 参数）和 Llama-3.1-8B-instruct（GQA，8B 参数）。训练数据：LongAlpaca（QA SFT 数据集），3000 steps，seq_len=10240，lr=5e-4，AdamW，warmup=2000 steps，α=0.0025。Benchmark：(1) ∞Bench——R.PassKey、R.Number、E.Sum、E.QA、E.MC、Z.QA、E.Dia、C.Debug、M.Find（平均 ~100K tokens，Z.QA ~2000K）；(2) L-Eval——CodeU、NQ、CUAD、NarrativeQA、QMSum、SPACE（>16K tokens）；(3) RULER——13 子任务 128K context（LOCRET-Q 评估）；(4) LongBench（附录）；(5) 自定义 10M-token R.PassKey（附录 J）；(6) Rock-Paper-Scissors 多轮对话 benchmark（附录 K）。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/huangyuxiang03/Locret

  **训练阶段**：
  1. 在每层 transformer 注入 retaining head R：一个小型 FFN，包含两个线性变换 W1 ∈ R^{(d_m + 2d_kv) × d_R} 和 W2 ∈ R^{d_R × h/g}，激活函数 σ 对齐原模型的非线性函数。d_R=1024。
  2. CIS 预测：Ŝ = R([Q, K, V]) = σ([Q, K, V]·W1)·W2。Ŝ[k] ∈ R^{h/g}，第 j 个分量为 token k 在 head j 的预测 CIS。
  3. Ground truth CIS：对于训练实例 d，S[k]_j := max_p (Q_j K_j^T)_{p,k}，其中 p 遍历所有 answer token，k 遍历所有 prefix token。对 GQA 模型，取同一 group 内不同 query head 的最大 attention score。
  4. 训练 Loss：Smooth-L1(Ŝ, S) + α·L2(Ŝ[k], Ŝ[k+1])，后者为相邻 token 平滑项。
  5. 保留 LLM backbone 冻结，仅训练 retaining head 参数。训练开销 < 1 GPU 小时。

  **推理阶段（Algorithm 1）**：
  ```
  Input: Model M, Prompt tokens x, Local length n_loc, Stabilizer length n_s, Budget b, Chunk size B
  // 保留最后 n_loc 个 token 不被 evict
  chunk_positions = split_chunk(0, x.length - n_loc, B)
  K_cache, V_cache, score_cache = [], [], []
  for chunk ∈ chunk_positions:
      begin, end = chunk.begin_pos, chunk.end_pos
      K_chunk, V_chunk, score_chunk =
          M(x[begin:end], K_cache, V_cache)  // forward pass with retaining heads
      K_cache = Concat(K_cache, K_chunk)
      V_cache = Concat(V_cache, V_chunk)
      score_cache = Concat(score_cache, score_chunk)
      if chunk is not the last:
          score_cache[-n_s:] = +∞  // stabilizers: never evict last n_s tokens
      indices = top-b(score_cache).indices  // keep top-b highest CIS tokens
      K_cache, V_cache, score_cache = K_cache[indices], V_cache[indices], score_cache[indices]
  // 处理最后 n_loc 个 token
  K_cache, V_cache, score_cache = M(x[-n_loc:], K_cache, V_cache)
  x_gen = M.generate(K_cache, V_cache)  // decoding with compressed KV cache
  ```

  **LOCRET-Q 变体**：训练时将 query 最后 l_a 个 token 前置到序列首部，收集 CIS labels。推理时 query 插入序列首部确保所有 eviction 操作感知 query。

  **关键超参数**：
  - Phi-3-mini-128K: b=6000, B=3072, n_s=2500, n_loc=100
  - Llama-3.1-8B-instruct: b=16384, B=1024, n_s=2500, n_loc=100
  - Training: lr=5e-4, batch_size=1, max_seq_len=10240, 3000 steps, AdamW, linear scheduler with warmup=2000, α=0.0025

  **关键结果**：KV cache 压缩比 up to 20× (<10% perf loss)；128K+ 长上下文推理在单张 NVIDIA 4090 上可行；10M token 上下文评估（1747.6× 压缩比）100% 准确率；LOCRET-Q >2× prefill speedup on RULER。

## LogQuant: Log-Distributed 2-Bit Quantization of KV Cache with Superior Accuracy Preservation

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是一种 training-free 的 2-bit KV Cache 量化算法，利用对数分布的 token 选择策略保留重要 token 为全精度（BF16）而将其余 token 量化为 INT2，同时基于 position-agnostic 特性重组缓存以提升内存局部性。实验比较：LogQuant vs KiVi（主要 baseline）在 2-bit/4-bit 精度下跨多模型、多任务、多压缩比的准确性（GSM8K Exact Match + LongBench 6 类任务）和吞吐量/内存效率（H100 HuggingFace pipeline）。

- 硬件平台是什么，配置是什么。
  NVIDIA H100 48G MIG（单卡，用于效率 benchmark：HuggingFace pipeline，平均 prompt 长度 512，最大输出长度 2000，递增 batch size 记录峰值内存和吞吐量直到 48GB 上限）。准确性评估的 GPU 论文未明确说明具体型号，使用 HuggingFace transformers 推理 pipeline。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama3.1-8B-Instruct、Llama3-8B-Instruct (GQA)、Qwen1.5-7B-Chat (MHA)、Qwen1.5-7B-Chat-AWQ、Qwen1.5-14B-Chat-AWQ (MHA)、Qwen2-7B-Instruct (GQA, 仅保留 1/8 KV heads)、Phi-3-mini-128k-instruct (MHA, 3.8B)。
  数据集/Benchmark：
  - GSM8K（5-shot，输入 token 600-1700，Exact Match 评估）
  - LongBench（全部 21 个数据集，覆盖 6 类任务：Math、Code Completion、Few-shot Learning、Multi-Document QA、Single-Document QA、Summarization、Synthetic Tasks）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/Concyclics/LogQuantKV
  
  **算法 Pipeline**：
  
  1. **Log-distributed token 选择（Algorithm 1）**：维持 2W~3W 个全精度 token。当长度超过 3W 时，对前 2W 个 token 做步长=2 的子采样（A[0:2W:2]），密度减半至 W 个，再追加 W 个新 token，天然形成 log₂ 稀疏性——最新窗口密度 p，次新窗口密度 p/2，再往前 p/4……
  
  ```
  Input: A (list of original precision tokens), a* (new token), W (window length)
  Output: A (updated list of tokens)
  procedure APPENDTOKEN(A, a*, W):
    if length(A) < 3W:
      A ← concat(A, a*)
    else:
      A ← concat(A[0:2W:2], A[2W:3W])  // 将前 2W 个 token 密度减半
      A ← concat(A, a*)                 // 追加新 token
    end if
    return A
  end procedure
  ```
  
  2. **量化策略**：非保留 token 量化为 INT2。每 channel 独立量化（Key-per-channel），group size=64（HuggingFace 默认值）。量化后端使用 Quanto（也可换 HQQ）。
  
  3. **Position-Agnostic 重组**：由于 Attention 输出 O = A·V = softmax(QK^T)·V 对 K、V 中 token 顺序具有置换不变性（即 A_P·V_P = A·V，P 为任意置换），可将全精度 token 与量化 token 连续拼接存储，无需保留原始位置顺序，提升内存局部性和处理效率。
  
  4. **压缩率计算**：对于序列长度 L、保留 2W 个全精度 token 的 BF16 模型 + 2-bit 量化：compression ratio = 16L / (2(L-2W) + 16×2W)。
  
  5. **与 KiVi 的关系**：LogQuant 的 W 受限于 ⌊R/3⌋（KiVi 保留 R 个全精度 token），确保不超过 KiVi 的全精度 token 数量。对于 R=128，LogQuant 使用 3⌊128/3⌋=126 个全精度 token，略少于 KiVi 的 128 个。
  
  6. **集成方式**：继承 HuggingFace transformers 的 Cache 类，通过 derived class 实现。与 HuggingFace 推理 pipeline 无缝兼容。
  
  7. **张量计算流程**（单次解码步骤）：
     - Q ∈ R^(1×d)，K_cache ∈ R^(N×d)，V_cache ∈ R^(N×d)
     - 对 quantized token 的 K/V 做 dequantize：K_deq = dequant(K_quantized, scale, zero_point)
     - K_full = concat([K_deq, K_full_precision])  // position-agnostic 重排后存储
     - A = softmax(Q × K_full^T / √d)  // 标准 scaled dot-product attention
     - O = A × V_full  // 加权求和，token 顺序不影响结果
  
  关键结果：LogQuant 在相同压缩比下，Math/Code 任务准确率比 KiVi 高 40%-200%；吞吐量提升 25%，batch size 增加 60%（H100 48G）。

## MEDA: Dynamic KV Cache Allocation for Efficient Multimodal Long-Context Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  MEDA 是一种免训练（training-free）的动态层间 KV cache 分配方法，专为多模态长上下文推理设计。核心实现包含三部分：(1) **跨模态注意力熵（Cross-Modal Attention Entropy）**：每层计算文本→视觉（A_TV）和视觉→文本（A_VT）的跨模态注意力矩阵，计算其注意力熵 E_CM^l = -(E_TV^l + E_VT^l)，以此量化该层注意力分布的不确定性和分散程度。熵越低表示注意力越集中于关键跨模态 token 对，该层对 KV cache 需求较小；熵越高表示注意力越分散，需要更多 KV cache。基于此通过 inverse entropy softmax allocation 公式 S_l = α_l · S, α_l = exp(E_CM^l) / Σ_k exp(E_CM^k) · L · ρ 动态分配各层的 KV cache 大小。(2) **多模态 KV Pair 选择（Multimodal KV Pair Selection）**：在 prefill 阶段计算累积注意力分数 A_s，对文本 token 的注意力分数加 max(A_s) 偏置优先保留文本 KV pairs，保留最近 M 个 token 的上下文窗口，从剩余 token 中选取 top-N 个最高注意力分数的 token 组成保守 cache (K_c, V_c)。(3) **多模态 KV Pair 合并（Multimodal KV Pair Merging）**：对未选中的 less important tokens，通过 many-to-one nearest-neighbor matching 基于 cosine similarity 匹配到最近的保守 token，使用平均合并策略 k_j ← (k_j + Σ_{i∈N_j} k_i) / (|N_j| + 1) 合并 KV pair，保留全局上下文信息而非简单丢弃。实验比较在 MileBench（多图像多文本 benchmark）、Video-ChatGPT、DREAM-1K、WorldQA 上与 H2O、SnapKV、PyramidKV（text-centric baselines）和 LOOK-M（multimodal KV cache baseline）在不同压缩比 ρ 下的 accuracy/ROUGE-L/F1 等性能指标以及 decoding latency 和 GPU memory usage。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 GPU（单卡）。实验环境详见于附录 A.2：AMD EPYC 7643 48-Core Processor + NVIDIA A100 GPU。精度测试使用 HuggingFace Transformers。速度测试使用 DREAM-1K 前 20 个 YouTube 视频样本。解码速度测量为 decoding 阶段时间除以总生成 token 数。KV cache 内存使用计算：Memory = (input_len + decoding_len) × 2 × 32 × 32 × 128 × 2 / (1024³) GiB，其中 2 表示 FP16 精度 2 bytes，32 为 attention head 数和 layer 数，128 为每 head 维度，第二个 2 表示 K 和 V 各一份。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaVA-v1.5-13B（32 layers）、LLaVA-NeXT-7B（32 layers）、InternVL-v1.5-7B（用于 multi-images 任务）；LLaVA-Video-7B/32B、LongVA-7B、LongVILA-8B（用于 long-video 任务）。数据集：(1) MileBench——6440 个多模态长文本样本，平均 15.2 张图片和 422.3 words/sample，含四类子任务：Temporal Multi-image Tasks (T)、Semantic Multi-image Tasks (S)、Needle in a Haystack Tasks (NH)、Image Retrieval Tasks (IR)，评估指标为 accuracy 和 ROUGE-L；(2) Video-ChatGPT——基于 ActivityNet-200 的视频描述 benchmark，GPT-3.5 评估正确性、细节、上下文理解、时序理解四个维度；(3) DREAM-1K——1000 个视频片段（真人电影、动画、stock footage、YouTube、TikTok），AutoDQ 指标评估 F1/Precision/Recall；(4) WorldQA——开放式 QA 数据集，GPT-4 评估生成质量。压缩比 ρ 在 multi-images 任务默认为 0.1，long-video 任务默认为 0.2。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/AIoT-MLSys-Lab/MEDA（论文中声明）

  **算法 Pipeline 核心流程**：

  **Stage 1 - Prefill + 跨模态注意力熵计算（Prompt Encoding）**：
  ```
  输入：多模态 prompt X ∈ R^{L_prompt × D}（含 text tokens X_n^T, image tokens X_m^I, video tokens X_q^V）
  
  for each layer l = 1..L:
      # 标准 QKV 投影
      Q^l = X W_Q^l, K^l = X W_K^l, V^l = X W_V^l    # [L_prompt, D]
      
      # 获取 text / visual token 子集索引
      Q_T^l = Q^l[text_indices]    # [n_T, D]
      K_T^l = K^l[text_indices]    # [n_T, D]
      Q_V^l = Q^l[visual_indices]  # [n_V, D]
      K_V^l = K^l[visual_indices]  # [n_V, D]
      
      # 跨模态注意力计算
      A_TV^l = Softmax(Q_T^l (K_V^l)^T / √D)    # [n_T, n_V]
      A_VT^l = Softmax(Q_V^l (K_T^l)^T / √D)    # [n_V, n_T]
      
      # 跨模态注意力熵（公式 5-6）
      E_TV^l = (1/|T|) Σ_i Σ_j A_TV^l[i,j] · log(A_TV^l[i,j])
      E_VT^l = (1/|V|) Σ_i Σ_j A_VT^l[i,j] · log(A_VT^l[i,j])
      E_CM^l = -(E_TV^l + E_VT^l)
  
  # 动态层间 KV cache 分配（公式 7）
  for each layer l:
      α_l = exp(E_CM^l) / Σ_k exp(E_CM^k) · L · ρ   # ρ: 压缩比
      S_l = α_l · S                                   # S: 总 KV cache budget
  ```

  **Stage 2 - KV Pair 选择与合并**：
  ```
  对每层 l，分配 budget S_l：
      # 1. 累积注意力分数（公式 8）
      A_p = Attn(Q_p K_p^T)                    # prefill 阶段的 attention
      A_s = Σ_i A_p[i, :]                      # 沿 query 维度求和 [L_prompt]
      
      # 2. Text-prior 偏置（公式 9）
      A_s[T] = A_s[T] + max(A_s)               # T = text token 索引
      
      # 3. 保留最近 + 选 top-N 重要 token（公式 10）
      I = Top_N(A_s[:-M])                       # 排除最近 M 个 token
      K_c = [K[I, :]; K[-M:, :]]               # 保守 cache
      V_c = [V[I, :]; V[-M:, :]]
      
      # 4. Many-to-one 最近邻匹配（公式 11）
      for each less important token i in I_less:
          for each conserved token j in I_c:
              u_{i,j} = cos_sim(k_i, k_j)      # key token 间的余弦相似度
  
      # 5. 平均合并（公式 12）
      for each j in I_c:
          N_j = {i | j = argmax cos_sim(k_i, k_j)}
          k_j ← (k_j + Σ_{i in N_j} k_i) / (|N_j| + 1)
          v_j ← (v_j + Σ_{i in N_j} v_i) / (|N_j| + 1)
  ```

  **Stage 3 - Decoding with Compressed KV Cache**：
  ```
  for each new token x_t:
      q_t = x_t W_Q
      # 使用压缩后的 K_c, V_c 计算 attention
      x_{t,out} = Softmax(q_t K_c^T / √D) V_c
      # 新 token 的 KV 追加到 cache
      K_c ← [K_c, x_t W_K], V_c ← [V_c, x_t W_V]
  ```

  **关键参数**：β₁ : β₂ = 3 : 1（recent context tokens M 和 important tokens N 的比例），memory overhead per layer 与 β₁ + β₂ 成正比。ρ 为总压缩比（如 0.1 即保留 10% KV cache）。所有实验在单张 A100 上完成，精度为 FP16。

  **复杂度**：跨模态熵计算仅在 prefill 阶段执行一次（O(n_T · n_V) per layer），KV pair 选择为 O(L_prompt) TopK 操作，合并为 O(L_less · L_c) 最近邻搜索。与 prefill 的 O(L_prompt²) 相比额外开销极小。Decoding 阶段直接使用压缩后的 KV cache，内存和延迟均降低。

  **关键效果**：
  - MileBench, LLaVA-NeXT-7B, ρ=0.1：MEDA 在所有 11 个 sub-task 上均优于或接近 Full Cache，整体显著优于 H2O/SnapKV/PyramidKV/LOOK-M
  - LLaVA-Video-7B, ρ=0.2：F1 31.3 vs Full Cache 32.5（H2O: 27.7, SnapKV: 28.8）
  - 20% budget 下 GPU memory 从 2.42 GiB 降至 0.67 GiB（72% 减少），decoding latency 从 14.61 ms/token 降至 8.23 ms/token
  - 5% budget 下 decoding latency 降至 5.18 ms/token（2.82× speedup）
  - 无需任何 fine-tuning，即插即用兼容所有 MLLM

## MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 Multi-Head Linear Attention (MHLA)，将 token 序列沿 token 维度划分为 M 个 non-overlapping blocks（"heads"），每个 block 计算局部 KV summary，再通过可学习的系数矩阵 Mc（Multi-Head Mixing）让每个 query block 对各 block 的 summary 进行加权混合，恢复 query-conditioned 的 token 级别选择性，同时保持 O(Nd²) 线性复杂度。
  - 实验比较：在 DeiT/VLT（图像分类）、DiT/DiG（类别到图像生成）、SANA（文本到图像生成）、Wan2.1（视频生成）、Transformer++（NLP）五个领域，将 MHLA 替代原始 attention 模块（self-attention 或 linear attention），对比准确率/FID/生成质量/perplexity 等指标。

- 硬件平台是什么，配置是什么。
  - NVIDIA H100 Tensor Core GPU（吞吐量测试使用）
  - 不同设备上的吞吐量对比（DiT-S/2 at 4096 resolution across different devices）
  - 训练硬件：论文未明确说明具体 GPU 型号和数量

- 模型是什么。数据集和bench分别是什么。
  - 图像分类模型：DeiT-T/S、VLT-T/S；数据集：ImageNet-1K
  - 图像生成模型：DiT-S/B/L/XL、DiG-S；数据集：ImageNet-1K (C2I)
  - 文本到图像生成模型：SANA-0.6B（从官方 checkpoint fine-tune）；数据集：31,292 张互联网图片
  - 视频生成模型：Wan2.1-1.3B（替换 FlashAttention 为 MHLA）；评测：VBench；序列长度 31,500 tokens（81 frames at 480×800）
  - NLP 模型：340M 参数语言模型；数据集：FineWeb-Edu 10B tokens 训练，SlimPajama 5B tokens；评测：MMLU, Commonsense Reasoning (WinoGrande, PIQA, ARC-c, ARC-e, OBQA, BoolQ), Wiki ppl, LMB ppl, LongBench

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源链接：https://github.com/DAGroup-PKU/MHLA（MIT license）
  - 代码结构：五个子项目 mhla_image_classification、mhla_dit、mhla_nlp、mhla_videogen、mhla_sana
  - 基于 PyTorch 实现，依赖 flash-linear-attention、timm 等库
  - 算法 pipeline 张量计算流程：
    1. 输入 X ∈ R^(N×d)，线性投影得到 Q, K, V = XW_Q, XW_K, XW_V
    2. Kernelized 特征映射：Q̃ = φ(Q), K̃ = φ(K)（如 ReLU 或 elu+1）
    3. 将序列分为 M 个 blocks，每 block b 含 N_b 个 tokens
    4. 每个 block 计算局部 KV summary：S_b = Σ_{j∈b} K̃_j^T V_j ∈ R^(d×d)，z_b = Σ_{j∈b} K̃_j ∈ R^d
    5. Multi-Head Mixing：通过可学习系数矩阵 Mc ∈ R^(M×M)，query block i 的混合 summary：S̃_i = Σ_{b=1}^M m_{i,b} S_b，z̃_i = Σ_{b=1}^M m_{i,b} z_b
    6. 输出计算：o = (q̃^T S̃_i) / (q̃^T z̃_i)
    7. 初始化策略：m_{i,j}^(0) ∝ 1 - dist(i,j)/max_k(dist(i,k))（locality-biased），训练中 clip 到 (0,1) 确保非负
    8. 复杂度 O(Nd² + M²d²)，当 M² ≤ N 时主导项为 O(Nd²)

## MOM: Memory-Efficient Offloaded Mini-Sequence Inference for Long Context Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  MOM（Memory-efficient Offloaded Mini-sequence Inference）是一种针对长上下文 LLM 推理的内存高效算法，包含两项核心实现：(1) Mini-Sequence Processing——将 MLP 层的输入沿序列维度划分为 M 个 mini-sequences（每个大小 N≈S/M），逐个处理以降低中间激活内存，且仅对最后一个 MLP 层和 LM Head 处理最后一个 token 的表示；(2) KV Cache Offloading——在 prefill 阶段将 KV cache 从 GPU offload 到 CPU，decode 阶段重新加载回 GPU。Mini-sequence 仅作用于 MLP 和 LM Head，attention 层保持不变，因此可与 FlashAttention 和 GQA 等现有优化无缝集成。

  实验比较四种配置：(a) Standard（无优化）、(b) Offload Only（仅 KV cache offloading）、(c) Mini-sequence Only（仅 MLP mini-sequence 分区）、(d) MOM（Mini-sequence + Offloading 组合），以及 Chunked Prefill（chunk size=8192 和 512）。评估维度：峰值 VRAM 使用量、最大可扩展上下文长度、总推理延迟（prefill+decode）、TTFT（Time to First Token）、decode 速度（tokens/s）、准确率（logit equivalence + Needle-in-a-Haystack test）。

- 硬件平台是什么，配置是什么。
  主实验：单张 NVIDIA A100 80GB GPU，bfloat16 精度。额外实验：单张 RTX 4080 mobile 12GB GPU，bitsandbytes 4-bit 量化（context lengths [16000, 20000, 24000]）。

- 模型是什么。数据集和bench分别是什么。
  主要模型：Meta-Llama-3.2-8B（bfloat16）。额外模型：Qwen2.5-7B、Mistral NeMo (12B)、Llama3.2-3B（4-bit 量化）、Qwen2.5-3B（4-bit 量化）。Benchmark：Needle-in-a-Haystack（评估长上下文检索准确率，needle depth × context length 矩阵）、Logit Equivalence Test（随机输入验证输出 logits 完全一致）。Context lengths：A100 上 [48000, 80000, 112000, 144000]，最大可达 455000 tokens。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/TianyiZhu877/MOM（基于 HuggingFace Transformers，使用 transformers.cache_utils.OffloadedCache）

  **算法 Pipeline（Algorithm 1）**：
  
  输入：X ∈ R^{B×S×d}，Mini-sequence size C，offloaded KV cache K

  ```
  # 对每个 Transformer Block:
  A = Attention(X)                    # attention 层保持完整，使用 FlashAttention/GQA
  Update and offload KV cache: K ← offload(K, A)
  
  if last MLP layer:
      A_last = A[:, -1, :]            # 仅取最后一个 token 的表示 [B, d]
      O_last = MLP(A_last)            # 仅处理最后一个 token 的 MLP
      L = LM_Head(O_last)             # 仅对最后 token 计算 logits [B, vocab_size]
      Reload KV cache from CPU to GPU for decode stage
      return L
  else:
      M = ceil(S / C)                 # 划分 mini-sequences
      Partition A into {A_i}_{i=1}^M, each A_i ∈ R^{B×N×d}, N ≈ C
      for i = 1 to M:
          O_i = MLP(A_i)              # 逐个处理 mini-sequence
      O = concat([O_1, ..., O_M])     # 拼接输出
      return O
  ```

  **张量计算示例（Llama-3-8B, S=128K, C=8192, d=4096, I=4d=16384）**：
  ```
  # Standard: 中间激活 = S × I = 128K × 16384 ≈ 2.1B floats ≈ 4.2GB (bfloat16)
  # MOM: M = ceil(128K/8192) = 16, 中间激活 ≈ N × I = 8K × 16384 ≈ 131M floats ≈ 262MB
  # 内存节省：4.2GB → 262MB（约 16× reduction per MLP layer）
  ```

  **与 Chunked Prefill 的关键差异**：
  - Chunked Prefill 将整个 prefill（attention + MLP + LM Head）按 chunk 分多次前向，导致 forward-pass 重复开销
  - MOM 仅拆分 MLP 层，所有 mini-sequences 在单次前向 pass 中处理，attention 层保持完整序列计算，仅 MLP 逐 mini-sequence 执行

  **KV Cache Offloading 集成**：
  ```
  # Prefill: 每层 attention 后 offload KV cache to CPU
  K[layer], V[layer] → CPU (via OffloadedCache)
  # Decode: 所有层 KV cache 重新加载到 GPU
  K[all], V[all] → GPU
  # Decode 阶段使用 GPU 上完整 KV cache 进行 autoregressive generation
  ```
  Decode 阶段 KV cache 已全部在 GPU 上，因此 decode 速度几乎无退化（Table 4: MOM decode 25.712 tok/s vs Standard 25.804 tok/s @ 48K context）。

## MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding

- 属于算法pipeline的实现是什么？实验比较什么？
  MagicDec 提出使用基于压缩 KV cache 的推测解码（speculative decoding）在长上下文、大批量场景下同时提升吞吐和延迟。核心算法 pipeline 分为三层：

  **Layer 1 — Bottleneck 分析与临界序列长度判定**：通过 roofline 模型分析 LLM 推理瓶颈随 batch size 和 sequence length 的转移。在长短序列下 batch 增大时推理变为 compute-bound（线性层饱和），验证成本高导致 SD 失效；但在 $S \ge S_{\text{inflection}}$ 时，KV cache 加载成为主导瓶颈（memory-bound），验证成本 $T_V/T_T$ 接近 1。此时若 draft 的 KV cache 增长速度慢于 target，$T_D/T_T$ 随 batch 增大反而下降，因此 SD 在大 batch 下也能加速。$S_{\text{inflection}}$ 取决于模型 FLOPS-to-memory 比和 GPU FLOPS-to-bandwidth 比（GQA 模型更高，H100 比 A100 更低）。

  **Layer 2 — 压缩 KV 自推测（Self-Speculation）**：使用 target 模型自身加上稀疏 KV cache 作为 draft model（self-speculation），替代传统的小型独立 draft model。关键洞察：(a) 长上下文下 KV cache 超过参数内存占用，小 draft model 的 KV 可能占 target 的 38%~140%（如 LLaMA-3.1-8B draft for LLaMA-3.1-70B）；(b) 压缩 KV 比压缩 model weights 能获得更高的 token acceptance rate——在相同 memory budget 下，Top-K KV sparsification 的接受率远超 model compression（如 LLaMA-3.1-70B 上 90%+ vs 80%+）。

  **Layer 3 — 最优 Drafting 策略选择**：在给定模型/硬件/任务下，根据公式 $\min_{T_{select}, K, \gamma, \alpha} [\frac{1}{\Omega(\gamma,\alpha)}(\frac{\gamma \cdot (T_D(B,K) + T_{select}(B,S,K))}{T_T(B,S)} + \frac{T_V(B,S,\gamma)}{T_T(B,S)})]$ 选择最优的 KV 压缩算法（static vs dynamic）、KV budget $K$、推测长度 $\gamma$。比较了 static 方法（StreamingLLM、SnapKV）和 dynamic 方法（PQCache、TopK），dynamic 接受率更高但 search cost $T_{select}$ 随 batch 增大而增长。

  实验比较：SnapKV self-speculation vs StreamingLLM self-speculation vs 小 draft model（Llama-3.2-1B + StreamingLLM KV）vs autoregressive decoding baseline，在 PG-19、RULER（niah-multikeys-3、cwe、qa-1）任务上评估 speedup。

- 硬件平台是什么，配置是什么。
  NVIDIA 8×A100 80GB（8-way tensor parallelism，主实验平台）；NVIDIA 8×H100 80GB + 4×H100（高 FLOPS/bandwidth 平台）；NVIDIA 8×L40（低成本 GPU 平台）。bfloat16 精度。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA-3.1-8B（主 target model，GQA），LLaMA-2-7B-32K（非 GQA，对比 FLOPS-to-memory ratio 的影响），LLaMA-3.1-70B（KV compression vs model compression 实验），Qwen-2.5-7B、Qwen-2.5-32B、Mistral-7B-v0.3（泛化验证）。Draft model：LLaMA-3.2-1B、TinyLlama-1.1B（小 draft model 对比）。
  
  数据集：PG-19（语言建模 perplexity，主评估数据集），RULER benchmark（niah-multikeys-3/needle in a haystack with passkeys 3, cwe/common word extraction, qa-1/question answering 1，context length 32K），各任务 context length 从 1K 到 100K tokens。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/Infini-AI-Lab/MagicDec（ICLR 2025）。基于 PyTorch + GPT-Fast + FlashInfer + torch.compile + CUDA graphs + Triton matmul。

  **算法 pipeline（以 LLaMA-3.1-8B self-speculation + SnapKV, batch=128, S=32000, 8×H100 为例）**：

  ```
  # ===== Phase 1: Prefill（仅一次）=====
  输入: prompt_tokens ∈ [B, S]  # B=128, S=32000
  
  # 完整 dense attention + 生成完整 KV cache
  K_full, V_full = DenseAttention(Q, K, V)  # [B, S, n_heads, d_head]
  # KV cache 大小: B × S × n_layers × 2 × n_heads × d_head
  # = 128 × 32000 × 32 × 2 × 8 × 128 = ~25.2 GB (bf16)
  
  # ===== Phase 2: KV 压缩（SnapKV static algorithm）=====
  # SnapKV: 基于最后一层 attention score 选择重要 KV
  attn_weights_last = Q_last @ K_last^T  # [B, n_heads, 1, S]
  # 对每个 head，沿 S 维度 pooling (kernel_size=5)
  pooled_attn = AvgPool1d(attn_weights_last, kernel_size=5)
  # 保留 observation window (size=32) + top-(K-32) 最高分位置
  obs_indices = [-32:]  # 最近的 32 位置
  sparse_indices = TopK(pooled_attn[:, :, :-32], K-32)  # 剩下的 K-32 个
  draft_indices = obs_indices ∪ sparse_indices  # |draft_indices| = K

  # 构建压缩 KV cache
  K_draft = gather(K_full, draft_indices)  # [B, K, n_heads, d_head]
  V_draft = gather(V_full, draft_indices)
  # 压缩 KV 大小: B × K × n_layers × 2 × n_heads × d_head
  # K=2049 时: 128 × 2049 × 32 × 2 × 8 × 128 = ~1.6 GB

  # ===== Phase 3: Decoding Loop（逐 step）=====
  gamma_optimal = 6  # 由 MagicDec 框架根据公式 (4) 选择
  
  while not all_done:
      # ---- Draft Phase: 用压缩 KV 推测 γ 个 token ----
      draft_tokens = []
      for step in range(gamma):
          # Self-speculation: target model 使用压缩 KV cache 生成
          q_new = W_q @ embed(token)
          # 仅对压缩 KV attend
          s = q_new @ K_draft^T / sqrt(d_head)  # [B, n_heads, 1, K]
          a = Softmax(s)
          o = a @ V_draft
          # FFN + LM head
          next_token = LMHead(FFN(o))
          draft_tokens.append(next_token)
          # 更新 draft KV cache（追加新 token 的 KV）
          k_new, v_new = compute_kv(next_token)
          K_draft = concat(K_draft, k_new)
          V_draft = concat(V_draft, v_new)
      
      # ---- Verify Phase: target 并行验证 γ 个 token ----
      # 使用完整 KV cache，一次 forward pass 验证所有 draft tokens
      K_full, V_full = concat(K_full, new_k_all), concat(V_full, new_v_all)
      # 对连续的 γ+1 个位置（原 last + γ 个 draft）做 attention
      q_all = W_q @ embed([current_token] + draft_tokens)  # [B, γ+1, d]
      s_full = q_all @ K_full^T / sqrt(d_head)  # [B, n_heads, γ+1, S_full]
      logits_all = LMHead(FFN(Softmax(s_full) @ V_full))
      
      # 逐个比对 draft token 与 verified token
      verified_tokens = []
      for i in range(gamma):
          if draft_tokens[i] == argmax(logits_all[i]):
              verified_tokens.append(draft_tokens[i])
          else:
              # 第一个不匹配 token 仍是正确的（从 target 来）
              verified_tokens.append(argmax(logits_all[i]))
              break
      # Ω(γ,α) = (1 - α^{γ+1})/(1 - α) ≈ 5.07 (α=0.85, γ=6)
      # 即平均每步验证生成 5.07 个 token
      
      output_tokens.extend(verified_tokens)
  ```

  **Speedup 计算（公式 2）**：
  $$\frac{T_{Avg}^{SD}}{T_T} = \frac{1}{\Omega(\gamma, \alpha)} \left( \frac{\gamma \cdot T_D}{T_T} + \frac{T_V(\gamma)}{T_T} \right)$$
  - 当 $T_V/T_T \approx 1$（memory-bound, KV dominant）且 $T_D/T_T \to 0$（压缩 KV 远小于完整 KV），$\frac{T_{Avg}^{SD}}{T_T} \approx \frac{1}{\Omega(\gamma,\alpha)}$ → speedup $= \Omega(\gamma,\alpha) > 1$
  - SnapKV self-speculation @ batch=128, S=32000, 8×H100: $T_T$=26.07ms, $T_{SD}$=12.96ms, speedup=2.01x
  - 最高 speedup: SnapKV self-speculation @ batch=41, S=100000, 8×H100, cwe task: 2.51x

## MagicPIG: LSH Sampling for Efficient LLM Generation

- 属于算法pipeline的实现是什么？实验比较什么？
  提出基于Locality Sensitive Hashing (LSH) 的采样方法来近似self-attention计算，替代传统的TopK稀疏注意力。将attention output视为从attention score分布w中采样value的期望值（o = E_{i~w}[v_i]），利用Self-normalized importance sampling + LSH SimHash从key分布中高效采样，实现对attention输出的无偏估计。实验比较了MagicPIG vs Full Attention、Quest（dynamic sparse attention）在lm-eval-harness (GSM8K-CoT, MMLU, COQA)、LongBench (QASPER, LCC, Repobench-P, TriviaQA等)和RULER (13个合成任务，16K-256K context)上的准确率，以及不同硬件下(A100, L20, RTX 4090)的解码吞吐量和延迟。

- 硬件平台是什么，配置是什么。
  GPU: NVIDIA A100-80GB (搭配CodeLlama-34B, 16K context)、NVIDIA L20-48GB (搭配CodeLlama-13B, 16K context)、模拟RTX 4090-24GB (L20带宽限制, 搭配Llama-3.1-8B-Instruct, 96K context)；CPU: Intel Platinum 8480+ (A100场景)、Intel 8563C (L20场景)。

- 模型是什么。数据集和bench分别是什么。
  模型: Llama-2-7b-chat、Llama-3.1-8B-Instruct、Code-Llama-13b-16K、Code-Llama-34b-16K、MegaBeam-Mistral-7B-512K、Llama3-8B-Prolong-512K、Llama-3.1-70B-Instruct。
  数据集/benchmark: lm-eval-harness (GSM8K-CoT, MMLU-Flan-Cot-Fewshot, COQA)、LongBench (QASPER, LCC, Repobench-P, TriviaQA, PRE, TREC)、RULER (13个合成任务含NIAH single/multi-key, CWE, FWE等)、infini_igsm (4K/8K close reasoning tasks)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源链接: https://github.com/Infini-AI-Lab/MagicPIG。算法pipeline核心流程：

  1. **预处理(centering)**：对K cache做中心化，K = K - mean(K)，解决q和k方向几乎相反导致LSH采样失效的问题。
  2. **LSH哈希表构建**：用K×L个随机投影向量W (SimHash)，将每个k_i投影到K-bit哈希码，构建L张哈希表。
  3. **解码时每步**：
     a. GPU上计算q的哈希码：q_code = Sign(q @ W)，得到K×L bit。
     b. CPU上查询L张哈希表，收集至少在2张表中与q碰撞的key索引集合S。
     c. CPU上计算采样概率u_i = 1 - (1-p_i^K)^L - L·p_i^K·(1-p_i^K)^{L-1}，其中p_i = 1 - arccos(q·k_i/(|q|·|k_i|))/π。
     d. 计算注意力输出估计：ō = Softmax(w_S/√d - log(u))·V_S，其中w_S = q·K_S^T。
  4. On-device cache: sink tokens和local tokens保留在GPU，不经过LSH采样，通过recursive attention合并GPU/CPU结果。
  张量计算维度：q∈R^{1×d}, K,V∈R^{n×d}, W∈R^{d×(K×L)}, 典型参数K=8~10, L=75~300, 计算量为全注意力的2%~5%。

## Mustafar: Promoting Unstructured Sparsity for KV Cache Pruning in LLM Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  核心实现：(1) per-token magnitude-based unstructured pruning——对每个 token 的 KV cache vector 按元素绝对值排序，移除最低 magnitude 的元素达到目标稀疏度 s；(2) 对 Key cache：使用 per-token magnitude-based 或 output-aware 非结构化剪枝，探索 pruning direction（per-token vs per-channel）和 output-awareness 的影响；(3) 对 Value cache：per-token magnitude-based pruning 即等价于 output-aware per-token pruning（因 V 的每个元素乘以同一个 attention score），per-channel 则需额外计算 output-aware score；(4) 保留 local dense window（最近 32 token 不剪枝）；(5) 利用 bitmap-based 稀疏格式（扩展自 Coruscant）对剪枝后的稀疏 KV cache 进行最大程度压缩，每 tile 为 1×64 列，64-bit bitmap 表示非零位置，tile offset 寻址。

  实验比较：(a) Mustafar per-token magnitude-based unstructured pruning vs ThinK structured pruning，在 Key cache、Value cache、以及 Joint KV cache 上的 LongBench 精度；(b) 不同剪枝度 K_s/V_s=0.5, 0.7 下的精度退化；(c) unstructured vs 2:4 semi-structured sparsity 对比；(d) Mustafar 与正交压缩方法联合：+H2O token eviction, +KIVI (2-bit/4-bit KV cache quantization)；(e) 扩展至 Llama-2-13B-chat 大模型；(f) RULER benchmark 65K context 下 vs ThinK；(g) 80%/90% 极高稀疏度下的精度评估。

- 硬件平台是什么，配置是什么。
  效率评估：NVIDIA RTX 6000 Ada GPU（48GB VRAM）。精度评估：GPU 论文未明确说明型号（使用 HuggingFace Transformers 推理）。性能测量使用 NVIDIA Nsight Profiling Tool。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-2-7B（MHA）、Llama-3-8B-Instruct（GQA）、Mistral-7B-Instruct-v0.2（GQA）、Llama-2-13B-chat（MHA）、Llama-3.1-8B-Instruct（RULER 评测）。数据集/benchmark：LongBench（6 类任务：Single-Doc QA, Multi-Doc QA, Summarization, Few-shot Learning, Synthetic, Code）和 RULER（13 个任务含 Needle-in-a-Haystack, context 65,536 tokens）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源链接：https://github.com/dhjoo98/mustafar

  **Mustafar 剪枝算法 pipeline（per-token magnitude-based，以 LLaMA-3-8B 单层 decode step 为例）**：

  ```
  Step 1 - 输入: Q_t ∈ R^{1×d}，当前 token query
         K_cache ∈ R^{T×d}，V_cache ∈ R^{T×d}（T 个已缓存 token 的全精度 KV）

  Step 2 - 剪枝 Key cache（per-token magnitude-based）:
    for each token i = 1..T-W（W=local window size=32）:
        abs_K_i = |K_cache[i, :]|          # element-wise absolute, shape [d]
        threshold = top_k_threshold(abs_K_i, sparsity=K_s)
        mask_K[i, j] = 1 if abs_K_i[j] >= threshold else 0
    # 最近的 W=32 token 的 mask 全部保留（mask[i, :]=1 for i > T-W）

  Step 3 - 剪枝 Value cache（per-token magnitude-based）:
    for each token i = 1..T-W:
        abs_V_i = |V_cache[i, :]|
        threshold = top_k_threshold(abs_V_i, sparsity=V_s)
        mask_V[i, j] = 1 if abs_V_i[j] >= threshold else 0
    # per-token magnitude 等价 output-aware（见 Figure 4 分析）

  Step 4 - 稀疏化 KV Cache:
    K_sparse[i] = K_cache[i] ⊙ mask_K[i]  # element-wise masked
    V_sparse[i] = V_cache[i] ⊙ mask_V[i]

  Step 5 - Bitmap 压缩（per tile = 1×64）:
    for each token i:
        for each tile t（每 64 个连续元素为一组）:
            bitmap = pack_bits(mask[i, t*64 : (t+1)*64])
            nonzeros = gather(K_sparse[i, t*64:(t+1)*64], mask)
            compressed[i].append((tile_offset, bitmap, nonzeros))
  ```

  剪枝公式：
  - Key cache per-token magnitude：S = |K_i|，按 |K_i| 排序保留 top-(1-s) 元素
  - Key cache output-aware：S = |K_i| ⊙ broadcast(Σ_{t} |Q_t|)，Q_t 累加当前和下31个 query
  - Value cache per-token magnitude（等于 output-aware）：S = |V_i|
  - Value cache per-channel output-aware：S = |V| ⊙ broadcast(Σ_{t} |α_t|)，α_t 为 attention score

  MHA 下 GQA 映射：多个 Q head 对应同一 KV pair 时，对每个 KV 的 Q 组求和所有剪枝分数。

## PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  PM-KVQ 提出三个关键技术实现 KV Cache 后训练量化以降低长 CoT 推理的内存开销：(1) **Progressive Quantization**：逐步降低 KV Cache 位宽（16→8→4→2 bit），充分利用目标硬件内存预算，而非每步直接量化到目标位宽；(2) **Block-wise Memory Allocation**：对不同敏感度的 transformer block 分配不同位宽，通过一阶泰勒近似估计敏感度，建模为整数规划问题并用 CVXPY 求解；(3) **Calibration with Positional Interpolation**：对短标定数据施加位置插值（RoPE 中位置索引乘以缩放因子 s），在不增加标定开销的情况下近似长上下文数据分布。
  实验比较对象为 RotateKV、MiKV、KIVI，在 AIME-2024/2025、CMIMC-2024、LiveCodeBench 上评测数学推理和代码生成能力。

- 硬件平台是什么，配置是什么。
  性能评测使用 8×A100-80G GPU 服务器进行 fake quantization 实验。目标 GPU 配置取决于模型规模：DeepSeek-Qwen-7B 使用 1×4090-24G；DeepSeek-LLaMA-8B 使用 1×4090-24G；DeepSeek-Qwen-14B 使用 1×A100-40G；DeepSeek-Qwen-32B、QwQ-32B 使用 1×A100-80G；DeepSeek-LLaMA-70B 使用 1×A100-80G（论文未明确说明 70B 的具体 GPU 配置，但从上下文推断为 A100-80G）。

- 模型是什么。数据集和bench分别是什么。
  模型：DeepSeek-R1-Distill-Qwen-7B/14B/32B、DeepSeek-R1-Distill-LLaMA-8B/70B、QwQ-32B。
  标定数据集：RedPajama arXiv 子集，随机选取 512 个样本，每个长度 2048 tokens。位置插值 s=4（嵌入 8192 上下文到 2048 tokens），α 通过网格搜索在 [0,1] 区间寻优（grid size=20），最小化自注意力算子的重建损失。
  评测 Benchmark：AIME-2024、AIME-2025（各30道数学竞赛题）、CMIMC-2025（代数/组合/几何，各10道标准题）、LiveCodeBench（2025年1月1日至4月6日的代码生成题）。数学题每道采样 16 个回答，代码题每道采样 4 个回答，temperature=0.6，top-p=0.95，最大输出长度 32768 tokens。指标为 pass@1 和 Voting accuracy。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/thu-nics/PM-KVQ

  算法 Pipeline（三步，推理前预处理 + 推理时执行）：

  **Step 1（预处理）—— Block-wise Memory Allocation：**
  对每个 transformer block i 和每个候选位宽 b，用校准数据前向传播记录 KV Cache 梯度 G_Ki, G_Vi。
  计算敏感度 s_{i,b} = ||G_Ki ⊙ (Ki - Q_b(Ki))||_1 + ||G_Vi ⊙ (Vi - Q_b(Vi))||_1
  求解整数规划：min Σ_i Σ_b x_{i,b} · s_{i,b}, s.t. Σ_b x_{i,b}=1, Σ_i Σ_b x_{i,b} · Mem(Q_b(Ki)+Q_b(Vi)) ≤ M。用 CVXPY 求解器（几秒内完成）。
  可选位宽集合 B：DeepSeek-LLaMA-8B 使用 {4,8}，其他 LLM 使用 {2,4}。

  **Step 2（预处理）—— Calibration with Positional Interpolation：**
  在 RoPE 中对位置索引 m 乘以缩放因子 s：θ' = s · m · θ^{-2i/d}
  然后用修正后的 RoPE 进行通道级重参数化标定：
  Λ = diag(λ_i)，λ_i = (max_m K_{m,i})^α
  P = (QΛ) · Q((KΛ^{-1})^T)，将 Key 中的 outlier 迁移到 Query 中。

  **Step 3（推理时）—— Progressive Quantization + Equivalent Right Shift：**
  初始阶段：以 FP16/INT16 存储 KV Cache。
  当内存预算耗尽时，执行位宽收缩：
  从 16bit → 8bit → 4bit → 2bit（Fbit），每次用 Equivalent Right Shift：
  X_b = ((2^{2b} - 2^b + 1)(X_{2b} + 2^{b-1})) >> 3b
  保持零点不变 (Z_b = Z_{2b})，缩放因子放大为 S_b = (2^b + 1)S_{2b}。
  同时保留首 token 为 INT16，最近 128 tokens 用滑动窗口保留 INT16（继承 KIVI/SKVQ 策略）。
  量化方式：非对称分组量化（group size=128），如公式 X_asym = ⌊(X_FP16 - Z) / S_asym⌋, S_asym = (max(X_FP16) - Z) / (2^N - 1)。

  **性能结果：**
  DeepSeek-Qwen-7B (2-bit): PM-KVQ pass@1 40.00% vs KIVI 32.08% on AIME-2024，提升 ~8%。
  DeepSeek-LLaMA-8B (4-bit): PM-KVQ pass@1 47.71% (BS=6, block-wise) vs KIVI 41.25%，甚至超过 16-bit 的 44.17%。
  DeepSeek-LLaMA-70B (2-bit): PM-KVQ pass@1 64.79% vs KIVI 51.88%，提升 12.91%。

## PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling

- 属于算法pipeline的实现是什么？实验比较什么？
  PyramidKV 是一种动态 KV cache 压缩算法，基于"Pyramidal Information Funneling"现象（底层注意力分散在全局上下文、中层注意力逐步收窄到局部区域、顶层注意力集中在少量关键 token 上），提出：(1) 跨层不均匀 KV cache budget 分配：底层分配更多 cache、顶层分配更少 cache，按算术序列递减；(2) 基于 attention score 的 token 选择：保留最后 α 个 token（instruction tokens）的 KV cache，然后根据这些 token 对其他 token 的 attention score（sum over instruction tokens）选择 top-k^l tokens 保留。公式：k^l = k^0 - (k^0 - k^{m-1})/(m-1) × l，其中 k^{m-1} = k^{total}/(β·m)，k^0 = 2·k^{total}/m - k^{m-1}。超参 β=20（控制金字塔形状）、α=8（instruction tokens 数）。

  实验比较：(a) PyramidKV vs H2O / SnapKV / StreamingLLM / FullKV 在 LongBench 17 个数据集上的性能（KV cache size 64/96/128/256/2048）；(b) Needle-in-a-Haystack 长上下文检索实验（Mistral-7B @32k, LLaMa-3-8B @8k, LLaMa-3-70B @8k, cache size 64/96/128）；(c) KV cache 内存节省实验（LLaMa-3-8B, seq_len=8192, batch=1, fp16）；(d) 推理速度对比（H2O/SnapKV/StreamingLLM/PyramidKV latency）；(e) 额外开销分析（allocation time vs selection time vs total inference time）；(f) PyramidKV+MInference 混合方法；(g) 与 PyramidInfer 对比（arithmetic vs geometric decay, token re-evaluation vs discard）；(h) Ablation: 算术/几何/指数/熵/Gini 分配策略对比，α 和 β 超参数敏感性；(i) 128K context 扩展实验（Llama-3-8B-Instruct-Gradient-1048k）；(j) vLLM 集成 throughput 实验。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 GPU。评估模型：LLaMa-3-8B-Instruct、Mistral-7B-Instruct、LLaMa-3-70B-Instruct。推理精度：fp16。batch size=1（内存消耗实验），greedy decoding（性能评估）。延迟测量实验 prompt length 512/1024/2048/4096，generation length 512/1024/2048/4096。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMa-3-8B-Instruct、LLaMa-3-70B-Instruct、Mistral-7B-Instruct（Jiang et al., 2023）。数据集和 Bench：(a) LongBench（Bai et al., 2023），17 个数据集覆盖 6 类任务——Single-Document QA (NarrativeQA, Qasper, MultiFieldQA-en)、Multi-Document QA (HotpotQA, 2WikiMultihopQA, MuSiQue)、Summarization (GovReport, QMSum, MultiNews)、Few-shot Learning (TREC, TriviaQA, SAMSum)、Synthetic (PassageCount, PassageRetrieval-en)、Code Completion (LCC, RepoBench-P)。平均输入长度 1,235-18,409 tokens。Metrics: F1 (QA), Rouge-L (Summarization), Accuracy (Synthetic), Edit Sim (Code)；(b) Needle-in-a-Haystack（Fact Retrieval Across Context Lengths），最多 32K context (Mistral-7B)、8K (LLaMa-3)；(c) 128K context 实验使用 Llama-3-8B-Instruct-Gradient-1048k。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/Zefan-Cai/PyramidKV。算法 pipeline 如下：

  **Step 1 - Budget Allocation（Pre-computed，在推理前一次性计算）**：
  ```python
  # m = 总层数, k_total = 总 KV cache budget, beta = 20 (超参)
  k_top = k_total / (beta * m)                    # 顶层 budget
  k_bottom = 2 * k_total / m - k_top              # 底层 budget
  # 按算术序列分配各层 budget
  for l in range(m):
      k_l = k_bottom - (k_bottom - k_top) / (m - 1) * l  # k^l = k^0 - Δ · l
      budgets[l] = int(k_l)
  # 最后 α 个 token 的 KV 在所有层均保留（instruction tokens）
  ```

  **Step 2 - Attention Score Calculation（Prefill 阶段）**：
  ```python
  # Q, K: [batch, heads, seq_len, d_k]
  # α = 8 (instruction tokens)
  A = softmax(Q @ K.T / sqrt(d_k))                # [heads, seq_len, seq_len]
  # 对每层每个 head 计算 token 重要性分数
  for h in range(num_heads):
      s_h = A_h[-α:, :].sum(dim=0)                 # sum attention from instruction tokens
      # s_h[i] = Σ_{j ∈ [n-α, n]} A_{ij}^h
  ```

  **Step 3 - KV Cache Selection**：
  ```python
  # 对每层 l，每个 head h，选 top-k^l tokens
  for l in range(m):
      k_l = budgets[l]
      for h in range(num_heads):
          _, top_indices = torch.topk(s_h, k_l)    # 选最高分的 k^l 个 token
          K_selected[l, h] = K[l, h, top_indices]
          V_selected[l, h] = V[l, h, top_indices]
      # 使用 torch.gather 执行 eviction（非 in-place 操作）
  ```

  **Step 4 - 推理时使用压缩后的 KV cache**：
  ```python
  # Decoding 阶段仅使用 K_selected, V_selected
  # 位置编码保持原始位置不变（不滚动 position）
  # 各层独立维护其 compressed KV cache
  output = attention(Q_new, K_selected[l], V_selected[l])
  ```

  **Step 5 - vLLM 集成（附录 R）**：
  每个 sequence 的 block table 扩展为 per-layer block table，使得每层可以独立检索其 KV cache，而非使用固定内存偏移。解决 naive 实现中不同层不同 budget 导致的 cache fragmentation 问题。Throughput 结果：PyramidKV 在 compression 下 throughput 随 input context length 增长而降低（因小 chunk 的内存分配/释放/移动/访问导致碎片化），需 per-layer page-out 解决。

## Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：Q-Filters 是一种训练无关（training-free）的 KV Cache 压缩方法。核心算法分为两步：(1) **离线校准阶段**：从校准数据集（Pile 子集，约 3000 样本）中收集各层各头的 Query 表示 $Q^h$，对每个头的 $Q^h$ 矩阵进行 SVD 分解 $Q^h = U \Sigma V^\top$，取第一右奇异向量 $v_1$ 作为该头的 Q-Filter（并对符号做规范化 $v_1^+ = \operatorname{sgn}(\mathbf{1}u_1^\top)v_1$）；(2) **推理阶段**：对每个头计算所有 Key 向量在 Q-Filter 上的投影 $\langle K_t^h, v_1^+ \rangle$，保留投影值最大的 KV pairs，丢弃投影值最小的。该方法基于定理 3.3：$\mathbb{E}_{Q_i^h}(\langle Q_i^h, K_i^h \rangle) \approx \kappa^h \langle K_i^h, u^h \rangle$，即 Key 在 Query 主方向上的投影可近似其期望注意力分数。对于 GQA，对每组 Query 的 Q-Filters 取平均。
  - 实验比较：在 Language Modelling（Pile 数据集上 perplexity）、Needle-in-a-Haystack（检索准确率）、Ruler 数据集（多子任务得分）上与 StreamingLLM、SnapKV、K-Norm（L2 范数方法）、Expected Attention 对比，压缩比从 2× 到 64×。同时测量 Time to First Token (TTFT) 以对比延迟。

- 硬件平台是什么，配置是什么。
  - 2 块 NVIDIA A100-80GB GPU（用于校准和推理实验）。Q-Filters 校准在 Llama-3.2-70B 上耗时不到 3 分钟（2×A100-80GB）。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Llama-3.1-8B、Llama-3.1-70B（Dubey et al., 2024）、Qwen-2.5-7B-Instruct（Qwen et al., 2025）、Llama-3.2-1B。
  - 数据集/benchmark：Pile（Gao et al., 2020）用于语言建模 perplexity 评估和 Q-Filters 校准；Needle-in-a-Haystack（合成检索任务，needle 深度 1k-64k tokens）；Ruler（Hsieh et al., 2024，含 CWE、FWE、Multi-Key、Multi-Query、Multi-Value、Single、QA、VT 子任务，序列长度 4096/8192/16384）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：代码在 https://github.com/NathanGodey/qfilters ；基于 KVPress 库（https://github.com/kvpress）和 HuggingFace Transformers。
  - 算法 pipeline 伪代码：
    ```
    # 离线校准阶段 (只执行一次)
    Q_bank = []  # 存储各层各头的 Query 激活
    for sample in calibration_dataset[:3000]:
        for layer in model.layers:
            for head in layer.heads:
                Q_bank[layer][head].append(Q_activations)
    
    q_filters = {}
    for layer in model.layers:
        for head in layer.heads:
            Q_matrix = stack(Q_bank[layer][head])  # [N*d_k]
            U, S, Vt = SVD(Q_matrix)
            v1 = Vt[0, :]  # 第一右奇异向量，d_k 维
            sign = sign(mean(ones @ U[:,0]))  # 保证正期望投影
            q_filters[layer][head] = sign * v1  # Q-Filter
    
    # GQA 处理：对每组共享 Query 的 head 取平均
    if model uses GQA:
        for group in kv_groups:
            q_filters[group] = mean(q_filters[heads_in_group])
    
    # 推理阶段
    def q_filters_compress(kv_cache, max_size):
        for layer, head in layers_and_heads:
            K = kv_cache[layer][head].keys  # [seq_len, d_k]
            scores = K @ q_filters[layer][head]  # [seq_len]
            keep_indices = topk(scores, max_size)
            kv_cache[layer][head] = kv_cache[layer][head][keep_indices]
    ```
    张量计算：给定 Key 矩阵 $K^h \in \mathbb{R}^{L \times d_H}$ 和 Q-Filter $v_1^+ \in \mathbb{R}^{d_H}$，重要性得分为 $s = K^h \cdot v_1^+ \in \mathbb{R}^L$。保留 $s$ 最大的 $k$ 个 KV pairs。该操作仅涉及一次矩阵-向量乘法和一次 top-k 选择，与 FlashAttention 兼容（无需物化注意力矩阵）。

## Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  Quest 提出 query-aware KV cache 稀疏注意力算法。核心思想：KV cache 中 token 的关键性高度依赖于当前 query token（如 Fig. 2 所示 "A is B. C is D. A is" 中，"B" 仅在 query="is" 时关键），因此不能预先静态裁剪 KV cache。Quest 采用 PageAttention（vLLM, Kwon et al. 2023）的 page 粒度管理 KV cache，为每 page 维护 per-channel Key 向量的最小值 $m_i$ 和最大值 $M_i$ 作为元数据。推理时，对给定 Query 向量 $Q$，Quest 计算每 page 的 attention score 上界：$U_i = \max(Q_i \cdot m_i, Q_i \cdot M_i)$，求和 $\sum_i U_i$ 作为该 page 的 criticality 估计。然后选择 Top-K pages 执行近似 self-attention（仅加载选中 page 的 K、V），其余 page 不加载。前两层保持 full attention（因稀疏度低 <10%），其余层使用 Quest。

  实验比较：(a) PG19 语言建模困惑度（32K tokens, token budget=4096）：Quest vs H2O/TOVA/Full Cache；(b) Passkey Retrieval（10K 和 100K 长度）：Quest vs H2O/TOVA/StreamingLLM，不同 token budget (32/64/128/256/512 和 256/512/1024/2048/4096)；(c) LongBench 六数据集（NarrativeQA, HotpotQA, Qasper, TriviaQA, GovReport, MultifieldQA）：Quest vs H2O/TOVA/StreamingLLM，不同 KV cache budget；(d) Kernel 级效率：Quest vs FlashInfer，NVBench 测量 criticality estimation / Top-K filtering / approximate attention 延迟；(e) 端到端推理延迟：Quest vs FlashInfer，单 batch decode 阶段平均每 token 延迟（16K-32K）。

- 硬件平台是什么，配置是什么。
  Kernel 评估：NVIDIA RTX 4090，CUDA 12.2。端到端评估：NVIDIA Ada 6000 GPU（48GB）。模型：LongChat-7b-v1.5-32k（基于 Llama-2-7B，32 layers），Yarn-Llama-2-7b-128k（基于 Llama-2-7B，128K context），Llama2-7B。FP16 权重，同时测试 4-bit 权重量化。

- 模型是什么。数据集和bench分别是什么。
  模型：LongChat-7b-v1.5-32k（Li et al., 2023），Yarn-Llama-2-7b-128k（Peng et al., 2023），Llama2-7B（Touvron et al., 2023）。数据集：PG19 测试集（100 本书，平均 70K tokens，perplexity 评估），Passkey Retrieval（Yarn, Peng et al. 2023，10K 和 100K 长度），LongBench（Bai et al., 2023，含 NarrativeQA/HotpotQA/Qasper/TriviaQA/GovReport/MultifieldQA 共六个子任务）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/mit-han-lab/Quest。基于 FlashInfer（https://flashinfer.ai）kernel 库实现，使用 RAFT（https://github.com/rapidsai/raft）的 batched Top-K CUDA operator。

  算法 pipeline（两阶段）：

  **阶段 0：KV Cache 插入时维护元数据**
  ```
  对于每个 page p（含 S 个 token）每个 channel i:
    M_i^p = max(M_i^p, k_i)   # page p 中所有 token Key 的 channel-wise 最大值
    m_i^p = min(m_i^p, k_i)   # page p 中所有 token Key 的 channel-wise 最小值
  ```
  元数据大小：2 × d_head × num_pages（相比完整 KV cache 的 2 × seq_len × d_head，压缩比为 1/PageSize，如 page_size=16 时仅 1/16）。

  **阶段 1：推理时 Criticality Estimation**
  ```
  输入：Query 向量 Q ∈ R^{d_head}, 所有 page 的 M^p, m^p
  for each page p:
    score_p = 0
    for each channel i:
      U_i = max(Q_i * m_i^p, Q_i * M_i^p)  # 保证 U_i ≥ 任意 K_i * Q_i
      score_p += U_i
  top_k_pages = TopK({score_p}, k=K)
  ```
  张量计算：给定 $Q \in \mathbb{R}^{d}$，$M^p \in \mathbb{R}^{d}$，$m^p \in \mathbb{R}^{d}$，每 page 上界 $s_p = \sum_{i=1}^{d} \max(Q_i \cdot m_i^p, Q_i \cdot M_i^p)$。由于 $\max(Q_i m_i^p, Q_i M_i^p) \geq Q_i K_i^{(t)}$ 对 page 内所有 token $t$ 成立，$s_p$ 是 page 内任意 token attention score 的上界。选 score 最高的 K 个 page。

  **阶段 2：Approximate Attention on Top-K Pages**
  ```
  加载 Top-K pages 的完整 K, V 向量
  S = Q @ K_selected^T / sqrt(d_head)  # 仅计算选中 tokens 的 attention
  A = softmax(S)
  O = A @ V_selected
  ```

  内存加载量：完整 KV cache = 2M × L bytes；Quest = 2M × L/S（元数据）+ 2M × K × S（Top-K 页）= 1/PageSize + K/PageNum of total KV cache。例如 page_size=16, 64K context, K=4K pages → 约 8× 内存加载减少。

## R-KV: Redundancy-aware KV Cache Compression for Training-Free Reasoning Models Acceleration

- 属于算法pipeline的实现是什么？实验比较什么？
  R-KV 提出一种面向推理模型（如 DeepSeek-R1）的训练无关、冗余感知的 KV Cache 压缩算法。核心思路：现有 attention-based KV cache 压缩方法（如 SnapKV）仅依赖 attention score 判断 token 重要性，但推理模型的长 CoT（Chain-of-Thought）输出中存在大量自反射和自我重复内容，这些冗余 token 同样获得高 attention score，导致缓存被冗余内容填满而关键推理 token 被错误淘汰。R-KV 通过三个核心组件联合解决：(1) Importance Scoring（重要性评分，§3.2）：基于最后 α 个 observation tokens 的 attention weight，对 MHA 直接计算 softmax 后的注意力矩阵，对 GQA 则额外对同一 KV head 组内各 query head 的 attention score 做 max-pooling 聚合，再对每个 token 的 per-query importance 做滑动窗口 max-pooling（窗口大小 2W）稳定化后取均值得到 per-head importance score I_i^h；(2) Redundancy Estimation（冗余估计，§3.3）：对各 head 内的 key vectors 做 L2 归一化后计算余弦相似度矩阵 S^h = K̄^h (K̄^h)^T ∈ R^{n×n}，抑制自相似（对角线置零）和最近 β 个高相似 token 的链接（保留最新出现的高相似 token 以避免丢失近期信息），再通过 softmax 归一化得到 per-head redundancy score R_i^h；(3) Joint Selection Strategy（联合选择，§3.4）：最终 selection score Z_i^h = λ·I_i^h − (1−λ)·R_i^h，取 top-B_budget tokens 保留。解码时每 B_buffer 步压缩一次，始终保留最后 α 个 observation tokens。

  实验比较：(a) R-KV vs SnapKV vs FullKV on MATH-500 和 AIME 2024，不同 KV cache budget（128/256/512/768/1024/1536/2048/2560/3072/4096），模型为 DeepSeek-R1-Distill-Llama-8B 和 DeepSeek-R1-Distill-Qwen-14B，pass@1 指标（每问题采样 64 responses）；(b) Throughput 和 Memory Saving 对比：R-KV vs FullKV vs SnapKV，8K 和 16K 生成长度，fixed budget (1024/1536/3072) 和 ratio budget (10%/34%/54%)；(c) λ 消融实验：λ ∈ {0, 0.01, 0.05, 0.1, 0.5, 1} on MATH-500，证明 λ=0.1 最优；(d) Token selection 可视化（Fig. 7）：R-KV vs SnapKV 在相同输入下的 selected KV token 分布对比。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 80GB GPU。模型：DeepSeek-R1-Distill-Llama-8B（32 layers, 32 heads, GQA, hidden_size=4096），DeepSeek-R1-Distill-Qwen-14B（40 layers, 40 heads, GQA, hidden_size=5120）。超参：B_buffer=128, α=8, λ=0.1, similarity threshold T（论文未明确给出具体值），β（论文未明确给出具体值），max-pooling 滑动窗口 W（论文未明确给出具体值）。

- 模型是什么。数据集和bench分别是什么。
  模型：DeepSeek-R1-Distill-Llama-8B（简称 R1-Llama-8B），DeepSeek-R1-Distill-Qwen-14B（简称 R1-Qwen-14B）。数据集：(a) MATH-500（Hendrycks et al., 2021，最大生成长度 16384 tokens），(b) AIME 2024（MAA, 2024，最大生成长度 32768 tokens）。评估方式：pass@1 使用非确定性采样（temperature=0.6, top-p=0.95），每问题采样 64 responses。Baselines: SnapKV（原为 prefilling 设计，适配到 decoding 使用相同压缩间隔），FullKV（无压缩，作为 golden standard）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/Zefan-Cai/R-KV。论文页面：https://zefan-cai.github.io/R-KV.page/。项目为 PyTorch 实现。

  算法 pipeline（decoding-time KV cache compression）：

  **Step 1 - Decoding-Time Compression 触发判断**：
  设 KV cache 总长度为 L_full，budget 为 B_budget，buffer 为 B_buffer。每生成 B_buffer 个 token 后触发压缩。压缩时保留最后 α 个 tokens 作为 observation tokens，将现有 B_budget 个 cache tokens 与前 (B_buffer − α) 个 buffer tokens 合并为 N_c = B_budget + B_buffer − α 个候选 KV tokens。

  **Step 2 - Importance Scoring via Attention**（per head）：
  ```python
  # Q_obs ∈ R^{α × d}, K_cand ∈ R^{N_c × d} per head h
  A = softmax(Q_obs @ K_cand.T / sqrt(d))  # [α, N_c], Eq. (1)
  # 若为 GQA 则先对各 query head 独立计算再 max-pooling 聚合: Eqs. (2)-(3)
  # 稳定化：滑动窗口 max-pooling + 均值
  for j in range(α):
      A_tilde[j, i] = max(A[j, i-W:i+W])  # 窗口大小 2W, Eq. (4)
  I_i = mean(A_tilde[:, i])  # Eq. (4), per-token importance
  ```

  **Step 3 - Redundancy Estimation via Cosine Similarity**（per head）：
  ```python
  K_norm = K_cand / (norm(K_cand, dim=-1, keepdim=True) + 1e-8)  # Eq. (5)
  S = K_norm @ K_norm.T  # [N_c, N_c], 余弦相似度矩阵
  S.diagonal().fill_(0)  # 抑制自相似
  # 保留最近 β 个高相似 token 不被标记为冗余
  for i in range(N_c):
      similar = where(S[:, i] > T)  # similarity threshold T
      recent_beta = topk_indices(similar, k=β, largest=True)  # 最近 β 个
      S[recent_beta, i] = 0
  S_bar_i = mean(S[:, i])  # 平均相似度
  R = softmax(S_bar)  # [N_c], 归一化冗余分数, Eq. (6)
  ```

  **Step 4 - Joint Selection**（per head）：
  ```python
  Z_i = λ * I_i - (1-λ) * R_i  # Eq. (7), λ=0.1
  # 跨 head 聚合：AggScore_k = mean_h(Z_{k,h})
  # 选择 AggScore 最高的 B_budget tokens
  topk_indices = argmax(AggScore, k=B_budget)
  # 拼接 selected tokens + α observation tokens
  K_comp = cat([K_cand[topk_indices], K_obs])
  V_comp = cat([V_cand[topk_indices], V_obs])
  ```

  **关键数值**：以 R1-Llama-8B on AIME24 为例，平均生成长度 ~15,536 tokens。10% ratio budget → B_budget=1,536, B_buffer=128。压缩后仅保留约 1554 KV tokens vs 15536 FullKV，节省 ~90% KV cache 内存，batch=1 时 throughput 从 75.44 tok/s (FullKV) 提升至 80.46 tok/s（略提升），最大 batch 从 62 提升至 402（6.5×），端到端 throughput 从 849 tok/s 提升至 3252 tok/s（3.8×）。16K 生成长度下优势更明显：最大 batch 从 30 提升至 402（13.4×），throughput 从 347 tok/s 提升至 3189 tok/s（9.2×）。

## SeerAttention-R: Sparse Attention Adaptation for Long Reasoning

- 属于算法pipeline的实现是什么？实验比较什么？
  SeerAttention-R 提出一种自蒸馏 Attention Gate (AttnGate) 稀疏注意力框架，在 post-training 阶段只训练轻量级 gating module（冻结原始模型权重），实现解码阶段 block-level 稀疏注意力。实验比较了 SeerAttention-R 与 Full Attention baseline 和 Quest（training-free 稀疏注意力）在长序列推理 benchmark 上的准确率，同时通过 oracle sparsity 实验验证 attention 本身存在稀疏性。消融实验研究了 block size、hybrid dense layers、threshold vs token budget 等设计选择。

- 硬件平台是什么，配置是什么。
  训练：AMD MI300x GPU，DeepSpeed ZeRO-2 优化。
  推理精度评估：论文未明确说明 GPU 型号（推理精度实验）。
  Kernel 性能 benchmark：NVIDIA H100 GPU（Section 4.4）。

- 模型是什么。数据集和bench分别是什么。
  模型：Qwen3-4B, Qwen3-8B, Qwen3-14B（GQA, g=4）, DeepSeek-R1-Distill-Qwen-14B。
  训练数据集：OpenR1-MATH-220K（HuggingFace），packed to 32k token sequences。
  Benchmark：AIME24, AIME25（各64样本pass@1），MATH-500（8样本pass@1），GPQA-Diamond（16样本pass@1）。
  Baseline对比：Full Attention（dense），Quest（training-free sparse，block size调整为64与SeerAttention-R对齐）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/microsoft/SeerAttention

  算法 Pipeline（伪代码级描述）：

  **Stage 1 - AttnGate 前向推理（稀疏块选择）：**
  ```
  输入: Q ∈ R^{1, num_kv_heads, d_head}   (decode阶段单token)
        K ∈ R^{seq_len, num_kv_heads, d_head}
        K_compression_cache ∈ R^{num_blocks, num_kv_heads, d_gate}

  // 1. Q分支：GQA query head 聚合
  Q_nope, Q_pe = split_rope(Q)
  Q_reshaped = reshape(Q_nope, [num_kv_heads, g * d_head])
  Q_gate = RoPE(W_q_gate @ Q_reshaped)  // [1, num_kv_heads, d_gate]

  // 2. K分支：pooling + 压缩（使用 K Compression Cache 加速）
  //    新token累积到block_size倍数时才更新cache
  K_new_block = concat[MaxPool(K[-block_size:]), MinPool(K[-block_size:]), AvgPool(K[-block_size:])]
  update K_compression_cache[-1] = W_k_gate @ K_new_block  // [1, num_kv_heads, d_gate]

  // 3. 块级注意力分数预测
  S = softmax(Q_gate @ K_compression_cache^T / sqrt(d_gate))  // [1, num_kv_heads, num_blocks]

  // 4. Top-K 选择 (token budget 方法)
  block_budget = token_budget / block_size
  selected_block_indices = topk(S, k=block_budget, dim=-1)  // [1, num_kv_heads, block_budget]
  // 始终选中最后一个未完成的block（K Compression Cache 未更新时）
  selected_block_indices = selected_block_indices ∪ {last_block_index}
  ```

  **Stage 2 - 块稀疏 Flash Decoding（使用selected blocks计算attention）：**
  ```
  // 对每个 KV head group，只对 selected_block_indices 中的块计算 attention
  对于每个选中的 block i:
    K_block = K_cache[i * b : (i+1) * b]
    V_block = V_cache[i * b : (i+1) * b]
    S_i = Q @ K_block^T / sqrt(d_head)     // [1, b]
    // FlashAttention tiled softmax & rescaling
    O = online_softmax_rescale(O, S_i, V_i)
  输出 O ∈ R^{1, d_head}
  ```

  **Stage 3 - 训练（自蒸馏）：**
  ```
  // 只训练 AttnGate 参数（W_q_gate, W_k_gate），冻结原始模型权重
  对于每个 training step:
    1. 修改版 FlashAttention-2 kernel 同时计算 full attention output 和 block-level ground truth
    2. Ground truth = 1D column-wise maxpool(full_attention_scores)
    3. 对 GQA 组内 query heads 的 ground truth 再做 maxpool → KV-head 级别
    4. Normalize ground truth to sum 1
    5. L_KL = KL(AttnGate(S) || ground_truth)
    6. 反向传播仅更新 AttnGate 参数
  ```

  训练配置：global batch size=16, 800 steps (0.4B tokens), AdamW lr=1e-3, cosine decay, MI300x GPUs, DeepSpeed ZeRO-2。

  K Compression Cache 内存开销：block_size=64 时仅需原始 KV cache 的 1/128 (<1%)。

## SentenceKV: Efficient LLM Inference via Sentence-Level Semantic KV Caching

- 属于算法pipeline的实现是什么？实验比较什么？
  SentenceKV 是一种句子级语义 KV 缓存管理方法。在 prefilling 阶段，将输入按标点符号分割成句子桶，用最后 N 个 token（observation window，默认 N=32）对前面所有 token 计算注意力分数并求和得到重要性 α_i，保留 top ⌊r·τ⌋ 个 token（r 为 semantic keeping factor，默认 2-3；τ 为 token budget，默认 1024），对每个句子桶计算平均 key 向量 $\bar{k}_{s,h}$ 存于 GPU，完整 KV 对 offload 到 CPU。在 decoding 阶段，维护 sentence cache $Q_s$ 累积当前生成句子的 query 向量，一旦遇到句子边界就计算平均 query $\bar{q}$，与所有句子桶的 $\bar{k}_{s,h}$ 做内积相似度排序，按相似度从高到低选取句子桶直到 token 数达到 τ，将对应 KV 对从 CPU 加载回 GPU 计算 attention。实验比较 Full KV、H2O、SnapKV、Quest、InfLLM、ShadowKV 等方法在 LongBench（单文档/多文档 QA、few-shot、合成、代码任务）、PG-19（PPL）、NIAH（检索准确率）、RULER（8 个子任务检索准确率）上的表现，以及 GPU 内存和推理延迟的消融。

- 硬件平台：2 块 NVIDIA H100 80GB GPU + 2 块 NVIDIA H100 NVL 96GB GPU。

- 模型：Llama-3-8B、Llama-3.1-8B-Instruct、Llama-3.2-3B-Instruct、Longchat-v1.5-7B。

- 数据集和 benchmark：PG-19（语言建模 PPL，最长 32k tokens）、LongBench（单文档 QA/Multi-doc QA/few-shot/合成/代码）、NIAH（Needle-In-A-Haystack，最长 8000 tokens）、RULER（8 个子任务，最长 64k tokens）。

- 开源情况：代码开源 https://github.com/zzbright1998/SentenceKV。

  算法 pipeline 使用例子（伪代码）：
  ```
  # === Prefilling Phase ===
  sentences = split_by_punctuation(prompt)  # 按标点分句
  obs_window = prompt[-N:]                  # 最后 N=32 个 token
  
  for token_i in prompt[:-N]:
      alpha[token_i] = sum over heads, obs_tokens of attn(obs_t, token_i)
  
  selected_tokens = top_k(alpha, k=int(r * tau))  # r=2-3, tau=1024
  discard rest
  
  for sentence_s in sentences:
      retained_in_s = selected_tokens ∩ sentence_s
      for head_h in range(H):
          k_bar[s][h] = mean(k[x][h] for x in retained_in_s)  # Eq.1
          # k_bar 存 GPU 作为语义向量
  
  offload_to_CPU(K_selected, V_selected)  # KV 对存 CPU
  
  # === Decoding Phase ===
  Q_s = []  # sentence cache
  for t in range(max_new_tokens):
      x_t, q_t = generate_next_token()
      Q_s.append(q_t)
      q_bar = mean(Q_s)  # Eq.2
      
      scores = []
      for sentence_s in sentences:
          for head_h in range(H):
              S = dot(q_bar, k_bar[s][h])  # 内积相似度
          scores.append((s, aggregate_similarity))
      
      sorted_buckets = sort_by_similarity(scores, descending=True)
      retrieved = []
      for bucket in sorted_buckets:
          if len(retrieved) + len(bucket.retained_tokens) <= tau:
              retrieved.extend(bucket.retained_tokens)
          else:
              break
      
      load_from_CPU_to_GPU(K[retrieved], V[retrieved])  # shape: (tau, H, d)
      O_t = softmax(q_t @ K_retrieved.T / sqrt(d)) @ V_retrieved  # Eq.3
      
      if is_sentence_boundary(x_t):  # 遇到句号/问号等
          Q_s = []  # 重置 sentence cache
  ```

  关键参数：τ=1024（约为 32k 上下文全量 KV 的 3%），r∈{2,3}，observation window N=32。在 256k tokens 时 GPU 内存从 Full KV 的 89.71GB 降至 52.71GB，延迟从 84.9ms 降至 17.8ms。NIAH 上检索准确率 97.5%（τ=128），远超 SnapKV 的 78.2%。

## ShadowKV__KV_Cache_in_Shadows_for_High-Throughput_Long-Context_LLM_Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  ShadowKV 提出两阶段算法：(1) **Prefilling**：对 pre-RoPE key cache 做 SVD 低秩分解（rank r=160），仅保留低秩投影矩阵 A∈ℝ^{b×s×r} 和 B∈ℝ^{b×h_kv×r×d} 在 GPU 上，将 value cache 全部 offload 到 CPU；同时对 post-RoPE key cache 按 chunk size=8 切分并计算 chunk 均值作为 landmarks L∈ℝ^{b×h_kv×s/c×d}；通过 chunk 内 cosine similarity 检测 outlier chunks（约占 0.3%，即 48/128K），outlier 的 KV 对保留在 GPU 作为 static cache。(2) **Decoding**：对每个 query，先用 landmarks 计算 chunk attention score P = MatMul(Q, L^T)，Softmax 后按 KV head 聚合选出 top-k chunks（k=256，1.56% sparse budget=256×8/128K），从 CPU fetch 对应 value cache，同时从低秩投影 MatMul(Gather(A, I), B) 重建 key cache，利用 CUDA multi-stream 重叠二者；额外维护 cache mechanism 利用 KV cache temporal locality（相邻 decoding step 命中率约 60%），通过 index scan 检测 miss chunks 减少重复计算和数据搬运。实验比较 Full Attention、Quest、Loki、InfiniGen 等 baseline，在 RULER（128K-1M）、LongBench（>4K）、Needle In A Haystack（16K-1M）、InfiniteBench 上评估准确率，在 A100 上评估吞吐（tokens/s）和 batch size。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 80GB PCIe GPU（GPU 显存带宽 2 TB/s，PCIe 带宽 31.5 GB/s）。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-3.1-8B（8 KV heads）、Llama-3-8B-1M（8 KV heads）、GLM-4-9B-1M（4 KV heads）、Yi-9B-200K（4 KV heads）、Phi-3-Mini-128K、Qwen2-7B-128K、Llama-3-70B-1M。Benchmarks：RULER（13 tasks，包括 N-S1、N-S2、N-MK1、N-MK2、N-MQ、N-MV、QA-1、QA-2、VT、FWE）、LongBench（NarratQA、MultiFQA、HotpotQA、MuSiQue、DuRead、GovRep、SAMSum、PassRetr、LCC）、Needle In A Haystack（16K-1M）、InfiniteBench。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/ByteDance-Seed/ShadowKV（Apache 2.0，301 stars，Python 35.1% + C++ 25.7% + CUDA 17.4%）。
  
  **Prefilling 算法（Algorithm 1）**：
  ```
  Input: K, K_RoPE, V ∈ R^{b × h_kv × s × d}, rank r, chunk size c, outlier num o
  // Step 1: SVD on pre-RoPE key cache
  A ∈ R^{b × s × r}, B ∈ R^{b × h_kv × r × d} ← SVD(K, rank=r)
  // Step 2: Segment post-RoPE keys into chunks, compute mean as landmarks
  C ∈ R^{b × h_kv × s/c × d} ← Reduce(K_RoPE, chunk_size=c)
  // Step 3: Compute cosine similarity within each chunk to find outliers
  S ← CosineSimilarity(C, K_RoPE)  // each chunk's tokens vs chunk mean
  I ← ArgTopK(-Min(S, dim=-1), o)  // lowest cosine similarity = outliers
  K_outlier, V_outlier ← Gather(K_RoPE, V, I)
  // Step 4: Offload values to CPU, keep non-outlier landmarks on GPU
  V_CPU ← V \ V_outlier
  L ← C \ Gather(C, I)
  // GPU retains: A, B, L, K_outlier, V_outlier
  ```
  
  **Decoding 算法（Algorithm 2）**：
  ```
  Input: A, B, L, V_CPU, Q ∈ R^{b × h_q × s_q × d}, K_outlier, V_outlier
  // Step 1: Compute approximate attention scores via landmarks
  P ∈ R^{b × h_q × s_q × n_c} ← MatMul(Q, L^T)
  S ∈ R^{b × h_q × s_q × n_c} ← Softmax(P / sqrt(d))
  S1 ∈ R^{b × h_q × n_c} ← sum(S, dim=-2)        // aggregate over query tokens
  S2 ∈ R^{b × h_kv × n_c} ← max_kv_group(S1)      // max over GQA group
  I ∈ R^{b × h_kv × k} ← ArgTopK(S2, k)           // top-k chunk indices
  // Step 2: Overlapped ops (multi-stream)
  V_sparse ← Gather(V_CPU, I)                       // PCIe fetch from CPU
  K_sparse ← MatMul(Gather(A, I), B)                // reconstruct from low-rank
  K ← [K_outlier; RoPE(K_sparse); K]                // concat outlier + sparse + new token
  V ← [V_outlier; V_sparse; V]
  // Step 3: Standard FlashAttention on sparse KV pairs
  Output ← FlashAttention(Q, K, V)
  ```
  
  核心设计：pre-RoPE keys 极低秩（rank 160 即可 6× 压缩无精度损失）→ 存低秩投影；post-RoPE keys 空间局部性 + outlier 稀少（0.3%）→ chunk 均值做 landmarks + outlier 静态缓存；temporal locality（60% hit rate）→ cache mechanism 减少重复操作；CUDA multi-stream → overlap key 重建与 value 抓取。理论等效带宽 7.2 TB/s（A100 原生带宽 2 TB/s 的 3.6×）。

## SnapKV: LLM Knows What You are Looking for Before Generation

- 属于算法pipeline的实现是什么？实验比较什么？
  SnapKV 是一种无需微调的 KV cache 压缩算法，通过 prompt 末尾的 "observation window" 计算注意力权重，对 prefix 中的关键 KV 位置进行投票和聚类选择，从而在生成阶段使用恒定大小的压缩 KV cache。实验比较：(a) SnapKV 压缩 vs 全量 KV cache（Full KV）在 LongBench 16 个数据集上的准确率；(b) SnapKV vs H2O（Heavy-Hitter Oracle）在 LongBench 上的准确率对比；(c) Needle-in-a-Haystack 压力测试（最长 380K tokens）；(d) 解码延迟与 batch size/序列长度 scaling；(e) 消融实验——pooling 对 LongEval-Lines 检索准确率的影响；(f) Command-R 上的 RAG 任务（citation、generation、end-to-end）；(g) SnapKV + Medusa 并行解码的兼容性实验。

- 硬件平台是什么，配置是什么。
  单卡 NVIDIA A100-80GB GPU。解码速度和内存实验均在此平台上完成，HuggingFace 原生实现配合少量代码修改即可运行。

- 模型是什么。数据集和bench分别是什么。
  模型：LWM-Text-Chat-1M（1M 上下文）、LongChat-7b-v1.5-32k、Mistral-7B-Instruct-v0.2、Mixtral-8x7B-Instruct-v0.1（32K 上下文）、Command-R（35B，128K 上下文）。
  数据集/Benchmark：LongBench（16 个子任务：MultiFieldQA-en、Qasper、HotpotQA、2WikiMQA、Musique、GovReport、QMSum、MultiNews、TREC、TriviaQA、SAMSum、PassageCount、PassageRetrieval-en、RepoBench-P、LCC、PREC）、Needle-in-a-Haystack（扩展至 380K tokens）、LongEval-Lines、NarrativeQA、bioasq、HotpotQA（RAG）、QASPER。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源：https://github.com/FasterDecoding/SnapKV。安装后 monkey-patch 模型即可使用：`from snapkv.monkeypatch.monkeypatch import replace_mistral; replace_mistral()`。已集成到 KVCache-Factory（https://github.com/Zefan-Cai/KVCache-Factory），vLLM 有相关 PR。

  **算法 pipeline 伪代码（基于论文 Listing 1）**：

  ```
  输入: query_states (B, H, L_prompt, D), key_states, value_states,
        window_size (L_obs), max_capacity_prompt, kernel_size
  输出: 压缩后的 key_states, value_states

  def snap_kv(query_states, key_states, value_states, window_size,
              max_capacity_prompt, kernel_size):
      # 仅 prompt 阶段执行压缩
      if q_len < max_capacity_prompt:
          return key_states, value_states

      # === Phase 1: Vote for important prefix positions ===
      # 计算 observation window 内 queries 对 prefix keys 的注意力权重
      # attn_weights: (B, H, L_obs, L_prefix) 其中 L_prefix = L_prompt - L_obs
      attn_weights = compute_attn(
          query_states[..., -window_size:, :],  # Q_obs: (B, H, L_obs, D)
          key_states,                            # K_full: (B, H, L_prompt, D)
          attention_mask
      )

      # 沿 query 维度求和得到每个 prefix 位置的累积注意力分数
      # vote: (B, H, L_prefix) — 每个 head 对每个 prefix token 的重要性投票
      vote = attn_weights[..., -window_size:, :-window_size].sum(dim=-2)

      # 1D 池化实现聚类——保留高注意力 token 周围上下文
      pool_vote = pool1d(vote, kernel_size=kernel_size,
                         padding=kernel_size//2, stride=1)

      # TopK 选择：选出 max_capacity_prompt - window_size 个最重要的 prefix 位置
      k = max_capacity_prompt - window_size
      indices = pool_vote.topk(k, dim=-1).indices  # (B, H, k)

      # === Phase 2: Compress and store ===
      # 扩展 indices 匹配 head_dim
      indices = indices.unsqueeze(-1).expand(-1, -1, -1, D)

      # 按 indices 收集压缩后的 prefix KV
      k_past_compress = key_states[..., :-window_size, :].gather(dim=2, index=indices)
      v_past_compress = value_states[..., :-window_size, :].gather(dim=2, index=indices)

      # 保留完整 observation window 的 KV（不做压缩）
      k_obs = key_states[..., -window_size:, :]
      v_obs = value_states[..., -window_size:, :]

      # 拼接: 压缩的 prefix KV + 完整的 observation window KV
      key_states = torch.cat([k_past_compress, k_obs], dim=2)
      value_states = torch.cat([v_past_compress, v_obs], dim=2)

      return key_states, value_states
  ```

  **核心张量计算流程**：
  1. Prefill 阶段：将完整 prompt 输入模型，在每层计算 QKV 投影
  2. 对于每层 attention，取 Q 的最后 `window_size` 个 token（observation window），计算它们对所有 K 的注意力权重 `W_obs ∈ R^{H × L_obs × L_prefix}`
  3. 沿 query 维度求和对每个 prefix 位置投票：`C_h = Σ_i W_obs[h, i, :]`，得到每个 head 的重要性向量 `C ∈ R^{H × L_prefix}`
  4. 1D max/avg pooling 平滑邻域（kernel_size=5~13，依模型调整）→ `pool_vote`
  5. TopK 选择保留 `max_capacity_prompt - window_size` 个位置 → 成压缩后的 prefix KV
  6. 拼接 observation window 的完整 KV → 形成最终压缩 KV cache
  7. 生成阶段：仅使用压缩后的 KV cache 进行 attention 计算，KV cache 大小恒定不变

  **关键超参数**：
  - `window_size`（observation window 大小）：Mistral 用 32，Command-R 用 64，LWM 用 16
  - `kernel_size`（pooling kernel）：Mistral 用 7，Command-R 用 13，LWM 用 5
  - `max_capacity_prompt`（压缩后 KV 大小）：1024/2048/4096
  - 压缩比：平均 input 13K tokens，1024 prompt KV → 92% 压缩率；4096 → 68% 压缩率
  - 极端压力测试：380K tokens → 1024 prompt KV → 380× 压缩比

## ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 ShadowKV，一种面向长上下文 LLM 推理的高吞吐系统，核心算法包括三部分：(1) **低秩 Key Cache 压缩**：对 pre-RoPE key cache 执行在线 SVD 分解（rank r=160），仅存储低秩投影矩阵 A∈R^{b×s×r} 和 B∈R^{b×h_kv×r×d}，压缩比约 6× 且无精度损失；(2) **Landmark 近似稀疏 Attention**：将 post-RoPE key cache 按 chunk_size=8 分块，每块计算均值作为 compressed landmark L∈R^{b×h_kv×s/c×d}，解码时用 Q 与 L 计算近似注意力分数选择 top-k chunk；(3) **Outlier 静态缓存**：通过 chunk 内 cosine similarity 检测近似最差的 outlier chunk（仅 0.2-0.3%），将其完整 KV 对作为 static cache 保留在 GPU 上以保证精度。总体 sparse budget 仅为 1.56%（k=256, o=48, S=128K 下）。

  实验比较：(1) Accuracy 实验：在 RULER (128K)、LongBench (>4K tokens)、Needle In A Haystack (16K-1M) 上对比 Full Attention、Quest、Loki、InfiniGen，ShadowKV 以 1.56% sparse budget 保持与 Full Attention 一致的精度；(2) 消融实验：sparse budget 变化、chunk size 选择 (4/8/16/32)、pre-RoPE key rank 选择 (32/64/96/128/160/192/256)、outlier 数量影响；(3) 兼容性实验：与 MInference 结合加速 prefill 阶段；(4) 多轮对话实验：Multi-turn NIAH 对比 SnapKV、StreamingLLM；(5) 精度敏感性实验：FP8 精度下与 baseline 对比。

  伪代码核心流程：

  ```
  # Pre-filling 阶段 (Algorithm 1):
  K = X @ W_k^T                    # pre-RoPE key, shape: [b, h_kv, s, d]
  A, B = SVD(K, rank=r)            # A: [b, s, r], B: [b, h_kv, r, d]
  K_RoPE = RoPE(K)                 # post-RoPE key
  C = Reduce(K_RoPE, chunk_size=c) # chunk mean landmarks [b, h_kv, s/c, d]
  S = CosineSimilarity(C, K_RoPE)  # 每 chunk 内 cosine similarity
  I = ArgTopK(-Min(S, dim=-1), o)  # 选择 o 个最差近似的 chunk 作为 outlier
  K_outlier, V_outlier = Gather(K_RoPE, V, I)
  V_CPU = V \ V_outlier            # 其余 value 下放 CPU
  L = C \ Gather(C, I)             # 非 outlier chunk 的 landmarks 保留 GPU

  # Decoding 阶段 (Algorithm 2):
  P = MatMul(Q, L^T)               # 用 landmarks 近似注意力分数 [b, h_q, s_q, n_c]
  S = Softmax(P / sqrt(d))
  S1 = sum(S, dim=-2)              # 沿 query 维度求和
  S2 = max_kv_group(S1)            # GQA 情况下聚合到 KV heads
  I = ArgTopK(S2, k)               # 选择 top-k chunk
  V_sparse = Gather(V_CPU, I)      # 从 CPU 取回 value（PCIe 传输）
  K_sparse = MatMul(Gather(A, I), B) # 从低秩投影重建 key（与上面重叠执行）
  K = [K_outlier; RoPE(K_sparse); K_new]  # 拼接 outlier + 重建 + 新 token
  V = [V_outlier; V_sparse; V_new]
  O = FlashAttention(Q, K, V)      # 标准 attention 计算
  ```

- 硬件平台是什么，配置是什么。
  NVIDIA A100 GPU (80GB)。GPU 内存带宽 2 TB/s，PCIe 带宽 31.5 GB/s (PCIe 4.0 x16)。部分实验在 8×A100 上进行。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-3.1-8B、Llama-3-8B-1M (Gradient extended)、GLM-4-9B-1M、Yi-9B-200K、Phi-3-Mini-128K、Qwen2-7B-128K、Llama-3-70B-1M。所有模型均使用 GQA (Grouped Query Attention)。
  数据集/Benchmark：RULER（13 个子任务，含 NIAH 变体、multi-key、multi-query、multi-value、variable tracking 等）、LongBench（6 大类 21 任务：单/多文档 QA、摘要、代码补全、信息检索等）、Needle In A Haystack（上下文窗口 16K-1M tokens）、InfiniteBench（10 任务，平均长度 214K）。RULER 128K 上下文为主，LongBench 仅使用超过 4K tokens 的样本。部分测试使用 PG-19 数据集分析低秩特性。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/ByteDance-Seed/ShadowKV 。

## Star_Attention__Efficient_LLM_Inference_over_Long_Sequences

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：Star Attention 是一种两阶段 block-sparse attention 近似算法，将注意力计算分为 (1) 阶段一 Context Encoding（blockwise-local attention + anchor block）和 (2) 阶段二 Query Encoding & Token Generation（分布式 global attention via distributed softmax）。anchor block 机制将每段 context block 前缀拼接第一个 block 作为 anchor，使 block-local attention 的 attention sink 集中在 anchor token 上，从而逼近 global attention 分布。实验比较 Star Attention vs Ring Attention（分布式 global attention）、StreamingLLM（sink tokens + sliding window）、MInference（动态稀疏 attention pattern）。评估指标：准确率（RULER/BABILong/InfiniteBench）和推理加速比。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 GPU，bfloat16 精度。8B 模型：16K-128K 用 8 GPU + 4 workers，256K-512K 用 16 GPU + 8 workers，1M 用 32 GPU + 16 workers。70B 模型：16K-32K 用 8 GPU + 4 workers，64K 用 16 GPU + 4 workers，128K 用 32 GPU + 8 workers。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-3.1-8B-Instruct、Llama-3.1-8B-Base、Llama-3.1-70B-Instruct（Meta-AI）、gradientai-Llama-3-8B-Instruct-262K、gradientai-Llama-3-8B-Instruct-1048K（Gradient.ai 扩展上下文版本）。Benchmark：(1) RULER — 13 任务，含 NIAH 检索、Multi-Hop Tracing、Aggregation、QA；(2) BABILong — 5 任务，多事实推理；(3) InfiniteBench — 10 任务，含摘要、多语言 QA、代码调试、检索。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/NVIDIA/Star-Attention 。基于 HuggingFace Transformers 和 NVIDIA TRT-LLM 实现，集成 Flash Attention（Dao, 2024）。算法 pipeline 如下：

  **阶段一：Context Encoding（blockwise-local attention with anchor block）**
  输入 context c，block size b。Split c into n = ceil(L/b) blocks: c = [c1, c2, ..., cn]。对 i = 2..n，构造 augmented block c'_i = (c1, ci)（prefix anchor block c1 到每个 context block 前）。n 个 augmented block 分发到 H 个 hosts 并行处理：
  ```
  for each host h concurrently:
    for each assigned block c'_i:
      compute self-attention over 2b tokens of c'_i
      generate KV cache for c'_i
      discard KV cache of anchor block c1 (保留 ci 的 KV)
      append remaining KV cache to kv_h
  ```
  每个 host 仅对分配到的 block(s) 计算 local blockwise attention（O(n * b^2) vs O((n*b)^2) 的 global attention），无 host 间通信。阶段一的 attention 复杂度为 O(Lb)，相对 full attention 的 O(L^2) 线性化。

  **阶段二：Query Encoding & Token Generation（distributed global attention via online softmax）**
  输入 query tokens q。广播 q 到所有 hosts。指定一个 query-host h_q。对每个 decoder layer 和每个 output token：
  ```
  for each host h concurrently:
    compute Q, K, V from input tokens
    compute local attention A_h = softmax(QK_h^T/sqrt(d)) V_h (使用 Flash Attention)
    compute s_h = sum(exp(QK_{h,k}^T / sqrt(d))) (local softmax denominator)

  gather all A_h and s_h at query-host h_q
  compute s_global = Σ_{h=1..H} s_h
  compute A_global = Σ_{h=1..H} (s_h / s_global) * A_h  (weighted aggregation)
  generate next token from A_global
  仅在 h_q 更新 KV cache
  ```
  实际实现使用 log-sum-exp trick（online softmax, Milakov & Gimelshein 2018）保证数值稳定性：
  ```
  s_global ← s_1, A_global ← A_1
  for h = 2..H:
    s_global ← s_global + log(1 + exp(s_h - s_global))
    A_global ← exp(s_h - s_global) * A_global + exp(A_h - s_global) * A_h
  ```
  通信开销：每个 token 仅需传递一个 scalar s_h 和一个 vector A_h ∈ R^d 从各 context host 到 query host。基于 PyTorch 实现，集成 FlashAttention (v2) 用于注意力计算，使用 CUTLASS 编写自定义 CUDA kernel。算法 pipeline 核心：pre-RoPE key cache 的 SVD 分解是 online 的（prompt-dependent），与 data-independent 的离线 weight 分解（如 Palu）不同。低秩 key 存储占用 S×r 而非 S×d（r=160, d=128 for Llama），压缩比约 6×。Landmark 允许 O(S/c × d) 的近似注意力计算替代 O(S × d) 的全量计算。K cache 重建 MatMul(Gather(A, I), B) 与 V_CPU 的 PCIe 取回通过 CUDA multi-stream 重叠执行。Temporal locality cache 机制利用相邻解码步 KV 选择的高重复率（>60%），跳过重复 chunk 的取回和重建。

## TokenButler: Token Importance is Predictable

- 属于算法pipeline的实现是什么？实验比较什么？
  TokenButler 提出一个轻量级预测器（<1% 参数开销），通过 attention distillation 学习预测 token 重要性，实现 query-aware 的细粒度 KV-cache 稀疏访问。核心实现：在固定的深度间隔 G（producer_frequency=4）处放置 producer layer，用二层 MLP（hidden=512）从 hidden states 预测低维 importance queries（d'=16），与经过学习投影矩阵 W_K 降维后的真实 KV-cache keys 做点积得到 token 重要性分数，在固定 budget 下筛选 top-k token 送入 attention kernel。训练时冻结 LLM，仅训练预测器，用 cross-entropy loss 蒸馏 masked causal attention distribution。推理时将 KV-cache 划分为 Sink Buffer（前 S 个 token）、Local Window Buffer（最近 N 个 token 循环缓冲区）和 Important Buffer（TokenButler 动态填充），保证 attention kernel 访存连续。引入 prediction interval（每 N 步运行一次预测器）和 neighbor fetching（对选中 token 扩展空间邻居）以摊销预测开销。
  实验比较：vs StreamingLLM（静态 recency）、H2O（注意力分数驱逐）、SnapKV（滑动窗口池化注意力驱逐）、Quest（逐页 min-max token 幅度）、PyramidKV、KIVI（KV 量化）、SingleSVD、xKV、MiniCache、KVzip、TokenSelect。评估指标：synthetic co-reference accuracy/coverage、RULER 各项子任务得分、LongBench 各项子任务得分、AIME24 accuracy、per-token decode latency、throughput。

- 硬件平台是什么，配置是什么。
  Nvidia A6000（主要评测平台，用于 latency 测量和训练），Nvidia H100（throughput 评测，Figure 8）。训练在单张 A6000 上完成。CPU offloading 场景的 latency 在 >=256K context 下评测。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-3.1-8B-Instruct、Qwen2.5-7B-Instruct-1M、Llama-3.2-1B/3B、DeepSeek-R1-Distill-Llama-8B。
  训练数据：C4 (RealNewsLike, 90k)、FineWeb-Edu (sample-10BT, 90k)、CodeParrot-Clean (90k)、BABILong (context {2k,4k,8k,16k})，训练序列长度 1024（预测器通过 key-cache 投影直接泛化到 64K context）。
  Benchmark：synthetic co-reference resolution（100 个虚构地点名，100^4 组合空间）、RULER（64K context）、LongBench（64K context）、AIME24（reasoning）、WikiText2（perplexity/recall 评测）。

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/abdelfattah-lab/TokenButler
  算法 pipeline 伪代码：
  ```
  # 训练阶段（冻结 LLM）
  for each producer layer p in {0, G, 2G, ...}:
      H = hidden_states_at_layer_p          # [B, L, E]
      Q_imp = MLP(LayerNorm(H))             # [B*H, G, L, d']
      for consumer layer l in [p+1, p+G]:
          slot = (l-1) % G
          K_proj = K_cache[l] @ W_K[l]      # [B, H_kv, L, d']
          A_pred = Q_imp[:,slot] @ K_proj.T # [B*H, L, L]
          A_true = teacher_attention[l]      # 冻结模型产出
          loss += CrossEntropy(softmax(A_true), softmax(A_pred))
  
  # 推理阶段（decode step t）
  # 每 i 步运行一次预测器（prediction interval）
  if t % i == 0:
      for each producer layer p:
          H = hidden_states_at_layer_p
          Q_imp = MLP(LayerNorm(H))
          for consumer layer l:
              K_proj = K_cache[l] @ W_K[l]
              scores = Q_imp[slot] @ K_proj.T  # [H, L]
              # 排除 sink + local_window 中的 token
              scores[candidates] = top_k(scores[candidates], B)
              # 将选中的 KV pairs 迁移到 Important Buffer
              # neighbor fetching: 为每个选中 token 扩展空间邻居
              selected_tokens = expand_with_neighbors(selected_tokens)
  # Attention Kernel: 拼接 [Sink | Important | Local_Window]
  attn_output = FlashAttention(Q, K[selected], V[selected])
  
  # 延迟投影优化：新 token 在 local window 中停留 N 步后，
  # 才批量投影其 key 到 d' 维空间加入预测器搜索空间
  if token_evicted_from_window:
      K_proj_batch = K[recent_N] @ W_K  # 批量投影，利用 HBM 带宽
  ```
  关键张量维度：Q_imp ∈ R^{(B*H) x G x L x d'}，K_proj ∈ R^{B x H_kv x L x d'}，d'=16 << D=128/head_dim。预测器参数占比：Llama-3.1-8B 为 29.4M (0.368%)，Qwen2.5-7B 为 20.9M (0.299%)。

## Trainable_Dynamic_Mask_Sparse_Attention

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 Dynamic Mask Attention (DMA)，一种可训练的 content-position 双感知稀疏注意力机制。核心由两部分组成：(1) **Content-Aware Dynamic Mask**：从 value 向量表示采样生成动态 mask。使用采样权重矩阵 Δ∈R^{n_h×d_h×n_h} 和 per-head 门控系数 A∈R^{n_h}，通过 δ=exp(τ(v·Δ)×A) 计算重要性分数，其中 τ 为非负激活函数（softplus），sparsification 函数 f(·) 保留 per-head Top-w 位置的原始分数、其余置 −∞。门控系数 A 可做成 query-dependent 以自适应输入。(2) **Position-Aware Sparse Weights**：用动态 mask 对 scaled dot-product attention 做稀疏化，mask 值为 −∞ 的位置 attention weight≈0，kernel 实现对这些位置直接填零跳过计算。整个过程完全可微，支持端到端训练。复杂度从 O(n²d_h) 降为 O(nwd_h)，内存从 O(n²) 降为 O(nw)。实验比较所有 attention 变体（MHA、SWA、MLA、NSA）在 Scaling Laws（80M→1.7B 参数下的 SmolLMCorpus Perplexity）、Multi-Query Associative Recall（512 KV pairs，更长序列+更小模型维度）、下游 Benchmark（LAMBADA、MMLU、TriviaQA、ARC、PIQA、HellaSwag、OBQA、Winogrande、LongBench、RULER、BBH、GSM8K、MATH、MBPP）以及 Needle-in-a-Haystack 长文本信息检索上的表现。

- 硬件平台是什么，配置是什么。
  NVIDIA A100-SXM4-80GB GPU。Kernel 性能 benchmark 使用 32 heads、8 KV heads、d_h=128、bf16 精度，3 次 warmup + 1000 次 run 取平均。

- 模型是什么。数据集和bench分别是什么。
  模型：Qwen3 1.7B 结构（仅修改 self-attention 部分为 DMA）。Scaling Laws 实验从 80M/200M/680M/1.7B 参数规模，12/16/24/28 层，d_model=768/1024/1536/2048，heads=6/8/12/16。训练数据集：SmolLMCorpus。下游 benchmark 实验：先用 32B tokens（Web/TextBook/Code/Math）做基础预训练（seq_len=2048），再用 8B tokens（seq_len=8K）做长上下文续训（RoPE base freq 10K→100K）；微调阶段 seq_len=16K（RoPE base freq→400K）。Benchmark：LAMBADA、MMLU、TriviaQA、ARC、PIQA、HellaSwag、OBQA、Winogrande、LongBench（英文任务）、RULER、BBH、GSM8K、MATH、MBPP。优化器：AdamW + WSD LR scheduler。Tokenizer：NeoX。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/flash-algo/flash-sparse-attention（CUDA kernel 实现）。论文提供了完整 PyTorch 参考实现（Listing 1）。算法 pipeline 分两步：
  **Step 1 — 动态 Mask 生成（Content-Aware）**：
  ```
  # V ∈ R^{b, n_h, n, d_h}，Δ ∈ R^{n_h, d_h, n_h}，A ∈ R^{n_h}
  dt = W_dt( V.transpose(1,2).reshape(b, n, n_h*d_h) )  # → [b, n, n_h]
  dt = exp( A * softplus(dt) ).transpose(-1, -2)          # → [b, n_h, n]
  m_t = dt.expand(-1, -1, q_len, -1)                      # → [b, n_h, q_len, n]
  m_t = m_t.masked_fill(causal_mask, -inf)                 # apply causal mask
  topk_indices = topk(m_t, w, dim=-1).indices              # per-head top-w selection
  m_t = m_t.masked_fill(not_in_topk, -inf)                 # mask out non-top-w positions
  ```
  **Step 2 — 稀疏 Attention（Position-Aware）**：对每个 (b, h, q_idx)，取 topk_indices 位置的 K/V 向量 [w, d_h]，计算 q_elem·K_selected^T / sqrt(d_h) + mask_selected，softmax 后加权求和 V_selected，输出 o_t∈R^{n_h×d_h}。复杂度从 O(n²d_h) 降为 O(nwd_h)。kernel 在 block 级别判断，若整个 K block 对应的 mask 全为 −∞，则直接跳过该 block 的加载和计算。

## WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 WindowKV —— 一种任务自适应的 KV cache 窗口选择方法。核心实现包含三个组件：(1) **Task-Adaptive Window Selection**：将输入 context 分为 observation window（最后 α 个 token）和 review windows（其余 context 按固定窗口大小 ω 切分）。使用 observation window 对 review context 中各 token 计算累积注意力分数 t_j = Σ A_ij（i ∈ observation window），再按窗口聚合为 window-level score s_k = (1/min(p,ω)) · sum(Top-p(W_k))。训练一个 bert-base-cased 任务自适应分类器（9551 样本，8:1:1 划分）将输入 context 分为 Information Localization 任务（QA 类，p=ω，保留整个窗口以理解完整语义）和 Information Aggregation 任务（摘要/代码类，p<ω，仅保留窗口内 top-p 高注意力 token）；(2) **Intra-Group Layer KV Cache Indices Sharing**：将 m 层 Transformer 分为 H=m/γ 组，每组 γ 层。仅在每组第一层 l_g 执行完整的 task-adaptive window selection，组内其余层共享同一套 KV cache indices I_lg，大幅降低计算开销；(3) **Dynamic Budget Allocation**：受 PyramidKV 启发，按等差数列跨组分配 budget——底层组分配更多 budget，顶层组分配更少，形成金字塔结构。总 budget b^total 分布在 H 个组上，通过超参数 λ（实验中 λ=14）控制金字塔形状。

  实验比较：(1) LongBench benchmark 上对比 StreamingLLM (SLM)、H2O、PyramidKV (PKV)、Full KV (FKV)，在 Qwen2.5-1.5B-Instruct 和 LLaMA3-8B-Instruct 上分别测试 KV cache size=512/1024/2048 下 6 类任务（Single-Doc QA、Multi-Doc QA、Summarization、Few-shot Learning、Synthetic、Code）的 16 个子数据集，WindowKV 在 LongBench 上以 12% 原始 KV cache 取得最多 SOTA 结果；(2) Needle-in-a-Haystack 测试 LLaMA3-8B-Instruct 在 8K context length、512 KV cache size 下的长上下文检索能力（Rouge-1 F1）；(3) Throughput test：在单张 A100 40G 上对比 Vanilla、Vanilla+WindowKV、Vanilla+WindowKV+Classifier 的吞吐量和延迟；(4) 消融实验：不同 γ（共享层数 = 1/4/7/8/14/16）对 LongBench 性能的影响；不同 review window size（8/16/32/64/128）的影响；任务类型（localization vs aggregation）匹配窗口选择策略的必要性。

- 硬件平台是什么，配置是什么。
  训练：8× NVIDIA A100 40GB GPU（训练 task-adaptive classifier）。
  推理/评估：单张 NVIDIA A100 40GB GPU（throughput test 和 benchmark 评估）。

- 模型是什么。数据集和bench分别是什么。
  模型：Qwen2.5-1.5B-Instruct（28 层，γ=7）、LLaMA3-8B-Instruct（32 层，γ=8）。
  Benchmark：LongBench（包含 6 类 16 子任务——Single-Doc QA: NarrativeQA/Qasper/MultiFieldQA-en; Multi-Doc QA: HotpotQA/2WikiMQA/Musique; Summarization: GovReport/QMSum/MultiNews; Few-shot Learning: TREC/TriviaQA/SAMSum; Synthetic: PCount/PassageCount; Code: RepoBench-P/RepoBench-L）；Needle-in-a-Haystack（Rouge-1 F1 评估检索准确率）。
  分类器训练数据集：自建数据集，9551 样本，train/val/test = 8:1:1。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源：代码公开在 https://github.com/optim996/WindowKV

  算法 Pipeline 伪代码级解释：
  ```
  # 超参数
  n = input_context_length
  α = observation_window_size       # Qwen2.5: α=4(loc) or 16(agg); LLaMA3: α=16 or 32
  ω = review_window_size            # Qwen2.5: ω=32; LLaMA3: ω=8(loc) or 16(agg)
  p = top_p_for_aggregation         # p=ω for localization; p<ω for aggregation
  γ = shared_layers_per_group       # Qwen2.5: γ=7; LLaMA3: γ=8
  H = num_layers / γ                # number of groups
  λ = 14.0                          # pyramid shape control
  b_total = target_kv_cache_size

  # Step 1: 任务分类
  task_type = classifier(input_context)  # "localization" or "aggregation"

  # Step 2: 前向传播 - 第一层计算 attention scores
  # 对于每组的第一层 l_g:
  Q, K = W_q @ h, W_k @ h                    # [n, d_head]
  A = softmax(Q @ K^T / sqrt(d_k))            # [n, n], causal masked

  # Step 3: Observation window 评估 review context token 重要性
  # observation window tokens: [n-α, n]; review context: [0, n-α]
  for j in [0, n-α]:
      t_j = sum_{i in [n-α, n]} A[i, j]      # token-level attention score

  # Step 4: Window-level scoring
  num_windows = ceil((n - α) / ω)
  for k in [1, num_windows]:
      W_k = {t_j, ..., t_{j+ω-1}}             # review window k
      top_p = select_top_p(W_k, p)             # Top-p tokens within window
      s_k = (1 / min(p, ω)) * sum(top_p)       # window score

  # Step 5: 根据 group budget 选择 top windows
  b_0 = (2 * b_total) / H - b_{H-1}           # bottom group budget (largest)
  b_{H-1} = b_total / (λ * H)                  # top group budget (smallest)
  for h in [0, H-1]:
      b_h = b_0 - (b_0 - b_{H-1}) / (H-1) * h  # arithmetic sequence
      per_layer_budget = b_h / γ
      n_windows_h = per_layer_budget / ω
      I_h = indices_of_top_n_windows(scores, n_windows_h)

  # Step 6: 组内共享 indices
  for h in [0, H-1]:
      group_layers = [h*γ, h*γ+1, ..., h*γ+γ-1]
      for layer in group_layers:
          if layer == group_layers[0]:
              selected_KV = gather(KV_cache[layer], I_h)    # 仅首层计算
          else:
              selected_KV = gather(KV_cache[layer], I_h)    # 复用首层 indices
  ```

  关键张量计算路径（以 LLaMA3-8B-Instruct、n=7950、KV size=2048 为例）：
  1. 输入 token embedding [7950, 4096] 经 32 层 Transformer
  2. 每层 attention: Q,K,V = Linear(h), dim=[7950, 128] per head (32 heads in 8 KV groups via GQA)
  3. 第一层计算 full attention A = softmax(Q@K^T/√128), [7950, 7950]
  4. Observation window (最后 α=16 tokens) → 累积 attention → t_j per review token [7950-16]
  5. 切分 review windows (ω=8 for localization) → ~992 windows → s_k per window
  6. 4 groups (γ=8), b_total=2048, λ=14 → b^0=704, b^3=320 → 每层 budget: group0=704, group1=576, group2=448, group3=320
  7. Window selection + indices sharing → 仅 4 次 selection（每组首层），其余 28 层复用 → 以 12% 原始 KV cache 保持 LongBench 性能 41.35 vs FKV 41.51

## X-EcoMLA: Upcycling Pre-Trained Attention into MLA for Efficient and Extreme KV Compression

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 X-EcoMLA——一种轻量级后训练方法，将预训练 Transformer 的 MHA/GQA 注意力模块 upcycle 为 MLA (Multi-head Latent Attention)，实现 KV cache 压缩。核心实现：(1) **SVD-based 权重初始化**：对预训练 Q、K、V 权重矩阵执行 SVD 分解，用分解后的低秩矩阵初始化 MLA 的 down/up-projection 矩阵。具体为：对 W^Q 做 SVD → U_q 作为 W^{DQ}，Σ_q V_q^T reshape 后分割为 W^{UQ} 和 W^{QR}；对 [W^K, W^V] 做 joint SVD → U_{kv} 作为 W^{DKV}，Σ_{kv} V_{kv}^T 分割为 W^{UK} 和 W^{UV}；W^{KR} 由所有 head 的 W^K 平均后取最后 d_r 列初始化；(2) **两种 rank 选择策略**：Fixed Rank Selection（所有层统一 r_q 和 r_{kv}）和 Dynamic Rank Selection（通过能量阈值 δ_q, δ_{kv} 基于累积奇异值平方能量自动确定每层 rank）；(3) **两阶段训练**：Stage 1 — 端到端知识蒸馏，使用 teacher-student KL 散度损失在 SFT 数据上训练；Stage 2 — DPO (Direct Preference Optimization)，以蒸馏后模型自身为 reference model 进一步偏好对齐；(4) **统一 RoPE 设计**：所有 head 共享单一 Key-RoPE 向量（与 DeepSeek MLA 一致），使每个 head 能完全利用全部 d_r 维位置编码，相比 MHA2MLA 的 per-head RoPE 提供 n_h× 的位置编码容量。

  实验比较：(1) **初始化方法对比**（Table 1）：Random vs SVD Fixed vs SVD Dynamic，在 SmolLM 135M/360M 和 Llama 3.2 1B/3B 上，SVD 初始化在 SmolLM 上提升 22.8%-30.91%，在 Llama 3.2 上提升 6.5%-8.1%；(2) **极端 KV 压缩 + 不同 Teacher 大小**（Table 2）：Llama3.2-1B 为 base，teacher 分别为 1B/3B/8B，KV cache 从 53.1% 压缩到 7.81%，8B teacher 在 15.6% KV size 和 3.6B token 下恢复 1.56 平均分，在 9.4% KV size 和 7B token 下仅 0.38 分下降；(3) **8B 模型扩展**（Table 3）：Llama3-8B 压缩到 15.63% KV（65.16 vs 65.78），Llama3.1-8B 压缩到 10.94% KV（65.85 vs 66.63）；(4) **与 MHA2MLA 对比**（Table 4）：SmolLM 1.7B 上 continual pretraining（52.87 vs 51.69，+1.18）和 SFT（49.34 vs 48.19 at 12.5% KV）；(5) **与 PALU 对比**（Table 5）：Llama3-8B 上 X-EcoMLA 15.63% KV (67.34) vs PALU-J-LRD 50% KV (66.19)，KV 用量约 3× 更少仍 +1.15 分；(6) **与 H2O 对比**（Table 9）：同 base 同 KV size 下 X-EcoMLA 持续大幅优于 H2O（9.4% KV: 50.49 vs 45.05）；(7) **消融实验**（Appendix A.5）：蒸馏损失 vs 交叉熵（Table 12，纯 CE 退化至 48.54 vs 52.77）、LayerNorm 影响（Table 13，去掉 LN 一致更好）、更大 teacher vs 更多数据（Table 14，8B teacher + 3.6B token 优于 1B teacher + 7B token）；(8) **长上下文评估**（Table 8）：LongBench 上压缩模型匹配或超越 full-cache baseline；(9) **Hybrid MLA**（Table 10）：50% 层 upcycle 为 MLA，KV size 78.1% 下性能超越 baseline（53.67 vs 52.77）。

- 硬件平台是什么，配置是什么。
  AMD MI300 GPU（8× MI300 用于训练和推理 evaluation）。训练耗时：3.6B tokens SFT + DPO 约 70 GPU hours（约 8.96 hours on 8× MI300 with 8B teacher）；7B tokens 约 140 GPU hours。推理吞吐/内存测试在 single AMD MI300 和 8× AMD MI300 上。

- 模型是什么。数据集和bench分别是什么。
  模型：SmolLM-135M-Instruct, SmolLM-360M-Instruct, SmolLM-1.7B-Instruct（MHA-based）; Llama 3.2-1B-Instruct, Llama 3.2-3B-Instruct, Llama 3.1-8B-Instruct（GQA-based）; Llama3-8B, Llama3.1-8B。Teacher 模型：Llama3.2-1B-Instruct, Llama3.2-3B-Instruct, Llama3.1-8B-Instruct。
  数据集：SFT 阶段使用 OpenHermes-2.5 + GenQA + Infinity-Instruct（约 6.8B tokens）；DPO 阶段使用 Llama3-ultrafeedback + orca_dpo_pairs + ultrafeedback_binarized（约 0.2B tokens）。
  Benchmark：LM Harness Eval benchmark (big-refactor branch)，9 个 zero-shot 任务：ARC-Challenge, ARC-Easy, HellaSwag, MMLU, OpenBookQA, PIQA, PubMedQA, RACE, WinoGrande；LongBench（长上下文评估：LCC, Qasper, QMSum, Multi-News, SamSum, RepoBench-P）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/AMD-AGI/AMD-Hybrid-Models

  算法 Pipeline（SVD 初始化核心伪代码）：
  ```
  # 输入: 预训练 MHA/GQA 权重 W_Q, W_K, W_V ∈ R^{d × n_h·d_h}
  # 输出: MLA 权重 W_DQ, W_UQ, W_QR, W_DKV, W_UK, W_KR, W_UV

  # === Query 侧初始化 ===
  U_q, Σ_q, V_q^T = SVD(W_Q)                         # W_Q = U_q Σ_q V_q^T
  W_DQ = U_q                                          # [d, r_q] down-projection
  W_UQR_bar = (Σ_q @ V_q^T).view(r_q, n_h, d_h)       # reshape
  W_UQ = W_UQR_bar[:, :, :d_qk].view(r_q, n_h*d_qk)  # NoPE query up-proj
  W_QR = W_UQR_bar[:, :, -d_r:].view(r_q, n_h*d_r)   # RoPE query up-proj

  # === KV 侧初始化（Joint SVD）===
  W_KV = concat(W_K, W_V, dim=-1)                     # [d, 2·n_h·d_h]
  U_kv, Σ_kv, V_kv^T = SVD(W_KV)
  W_DKV = U_kv                                        # [d, r_kv] down-projection
  W_UKV = Σ_kv @ V_kv^T                               # [r_kv, 2·n_h·d_h]
  W_UK_bar = W_UKV[:, :n_h*d_h].view(r_kv, n_h, d_h) # key part
  W_UK = W_UK_bar[:, :, :d_qk].view(r_kv, n_h*d_qk)  # key up-proj (NoPE)
  W_UV = W_UKV[:, n_h*d_h:]                           # value up-proj

  # === RoPE Key（所有 head 共享）===
  W_K_avg = W_K.view(d, n_h, d_h).mean(dim=1)         # [d, d_h]
  W_KR = W_K_avg[:, -d_r:]                             # [d, d_r]

  # === Dynamic Rank Selection ===
  # 对 W_Q: Σ_q 平方累积能量 ≥ δ_q · total_energy 确定 r_q
  # 对 [W_K,W_V]: Σ_kv 平方累积能量 ≥ δ_kv · total_energy 确定 r_kv
  ```

  前向传播（MLA 推理时 KV cache 仅需 (r_kv + d_r)·l）：
  ```
  # 输入: hidden state H ∈ R^{l×d}
  C_KV = H @ W_DKV                                   # [l, r_kv] 压缩 latent（缓存）
  K_C = C_KV @ W_UK                                  # [l, n_h*d_qk] NoPE key
  V_C = C_KV @ W_UV                                  # [l, n_h*d_h] value

  C_Q = H @ W_DQ                                     # [l, r_q] query 压缩
  Q_C = C_Q @ W_UQ                                   # [l, n_h*d_qk] NoPE query
  Q_R = RoPE(C_Q @ W_QR)                             # [l, n_h*d_r] RoPE query
  K_R = RoPE(H @ W_KR)                               # [l, d_r] 共享 RoPE key

  Q = concat(Q_C, Q_R, dim=-1)                        # [l, n_h*(d_qk+d_r)]
  K = concat(K_C, repeat(K_R, n_h), dim=-1)           # RoPE key 每 head 复制
  O = softmax(QK^T/√(d_qk+d_r)) @ V_C
  # 上行 W_UK 可吸收进 W_Q, W_UV 可吸收进 W_O，消除推理时 up-projection 开销
  ```

  训练流程：
  ```
  # Stage 1: Knowledge Distillation (SFT)
  for batch in sft_dataloader:
      student_logits = X_EcoMLA(batch.input_ids)
      teacher_logits = frozen_teacher(batch.input_ids)
      loss = KL(teacher_logits || student_logits)
      loss.backward(); optimizer.step()
  # Stage 2: DPO
  for batch in dpo_dataloader:
      # reference_model = 蒸馏后的 student（冻结）
      loss = -log σ(β·(log π_student(y_w|x) - log π_ref(y_w|x)
                      - log π_student(y_l|x) + log π_ref(y_l|x)))
      loss.backward(); optimizer.step()
  ```

## XAttention: Block Sparse Attention with Antidiagonal Scoring

- 属于算法pipeline的实现是什么？实验比较什么？
  实现 XAttention，一种 training-free 的 block-sparse attention 框架。核心创新是用注意力矩阵的反对角线和（antidiagonal sum）作为 block 重要性的轻量级代理指标，通过三步流程加速长上下文推理：(1) Strided Antidiagonal Scoring：按步长 S 沿反对角线采样元素求和作为 block 得分；(2) Threshold Block Selection：选择累计反对角线 softmax 概率之和超过阈值 τ 的最小 block 集合；(3) Minimum Threshold Prediction：通过动态规划为每个注意力头预测最优阈值，进一步优化稀疏度。

  实验比较：(1) RULER benchmark 上对比 Full Attention (FlashInfer/FlashAttention)、FlexPrefill、MInference、SeerAttn，在 4k-128k 序列长度下，XAttention (S=8) 平均分 88.47 vs Full 87.52，且优于 FlexPrefill (87.72)；(2) LongBench 真实长文本任务对比 MInference 和 FlexPrefill，XAttention 取得最高平均分；(3) Video-MME 视频理解任务上对比 Full Attention、MInference、FlexPrefill，XAttention 在长视频上优于 Full Attention；(4) VBench 视频生成任务（HunyuanVideo），XAttention + 5-step warmup 达到 PSNR 23.5 / SSIM 0.822 / LPIPS 0.155，密度仅 45.5%；(5) 效率对比：256k 上下文下 prefill 注意力加速最高 13.5×（S=16，密度 7.32%），pattern selection 比 MInference 快 24.9×、比 FlexPrefill 快 5.9×。

- 硬件平台是什么，配置是什么。
  NVIDIA GPU（论文致谢 NVIDIA DGX 服务器捐赠）。使用 FlashInfer 框架进行注意力计算。具体 GPU 型号论文未明确说明。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-3.1-8B-Instruct（文本任务）、Qwen2-VL-7B-Instruct（视频理解）、HunyuanVideo（视频生成，DiT 架构，non-causal attention）。
  数据集/Benchmark：RULER（合成长上下文 benchmark，4k-128k）、LongBench（真实长文本任务含 Single-Doc QA、Multi-Doc QA、Summarization、Few-shot Learning、Code）、Video-MME（900 视频、254 小时）、VBench（946 GPT-augmented 文本提示词）。

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/mit-han-lab/x-attention

  算法 Pipeline（XAttention 三步流程）：

  ```
  # 输入: Q, K ∈ R^{L×d}，block size B，stride S，threshold τ
  # 输出: Sparse mask M, Sparse attention output O

  # === Step 1: Strided Antidiagonal Scoring ===
  N_B = L // B  # number of blocks
  for b = 0 to N_B - 1:
      # 沿反对角线 reshape Q 和 K
      Q_reshaped = []  # shape: [S, B//S, d] per block
      for i = S-1 down to 0:
          Q_reshaped.append(Q[b*B:(b+1)*B, :][i::S, :])
      K_reshaped = []
      for i = 0 to S-1:
          K_reshaped.append(K[i::S, :])

      # 计算近似注意力分数
      A_approx = Softmax((Q_reshaped @ K_reshaped^T) / sqrt(d_h) / S)
      # A_approx 的反对角线和作为 block 重要性代理
      M_b = find_blocks(A_approx, τ)

  # find_blocks: 选择累计概率超过 τ 的最小 block 集合
  # find_blocks(A, τ) = argmin_{B} |B|  s.t. Σ_{b∈B} Σ_{(i,j)∈b} A_{i,j} ≥ τ

  M = concat(M_0, ..., M_{N_B-1})  # 稀疏 mask

  # === Step 2: Threshold Block Selection ===
  # 使用累积反对角线 softmax 概率超过 τ 的 block 作为选中 block
  # 不同注意力头可设置不同 τ，实现动态稀疏度

  # === Step 3: Minimum Threshold Prediction (Optional) ===
  # 动态规划为每个头寻找最优阈值
  # D[h][m]: h 个头、m 次调整的最佳性能
  # D[h][m] = max(D[h-1][m], P(h, m))
  # t_h(m) = t_h(m-1) * 0.9  # 每次调整降低 10%

  # === Step 4: Sparse Attention Computation ===
  # 仅对 M 中标记的 block 执行精确注意力计算
  O = BlockSparseAttention(Q, K, V, mask=M)
  ```

  关键洞察：反对角线交叉每个 block 内所有可能的垂直和斜线注意力模式（vertical & slash patterns），确保不遗漏任何关键模式。每个 token 至少参与一条反对角线，保证信息完整性。

  加速来源：Block 级计算从 O(L²) 降至 O(L² × density)，如 128k 时密度仅 6.89%（S=8），理论加速 ~14.5×。

## ZSMerge: Zero-Shot KV Cache Compression for Memory-Efficient Long-Context LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 ZSMerge —— 一种零样本、无参数的动态 KV Cache 压缩框架，包含四个核心组件：(1) **三分区 budget 分配**：将总缓存预算 B 划分为 proximity component (Bp，保留最近 token)、context component (Bc，保留高分 token)、residual component (Br，动态合并历史被驱逐 token)，B = Bp + Bc + Br；(2) **贡献评估**：每个 token t 在解码步 T 的贡献分数 s_t^(T) = λ·s_t^(T-1) + a_t^(T)，其中 a_t^(T) 为 attention 分数，λ=0.98 为指数衰减因子，类似 RL 信用分配；(3) **残差 token 合并**：将被驱逐的候选 token (k_t, v_t) 通过最大内积选择最兼容的 residual slot r̂ = argmax(k_r^T · k_t)，然后用增量均值聚合更新：k_r̂ ← (w_r̂·k_r̂ + k_t)/(w_r̂+1)，v_r̂ ← (w_r̂·v_r̂ + v_t)/(w_r̂+1)，w_r̂ ← w_r̂ + 1；(4) **补偿注意力评分**：修订注意力计算为 â_t^(T) = exp(q_T^T k_t/√d + α·log w_t) / Σ_i exp(...)，其中 w_i 为 token i 的融合计数（未压缩 w_i=1），α∈[0,1] 为 scale factor（论文固定 α=1），log w_t 补偿合并 token 的表示偏差，并保证 Theorem 1：未压缩 token i 的 â_i^(T) ≥ a_i^(T)（原注意力分数），即未压缩 token 保留相对注意力优势。

  实验比较：
  (1) Memory Reduction（§4.2.1）：LLaMA2-7B on A800-80GB，FullKV 在 54K tokens 时 OOM，ZSMerge 用 18K cache budget 保持 VRAM 恒定 43GB，消除 OOM；
  (2) Throughput（§4.2.2）：ZSMerge 维持 9 tokens/sec 解码速率（54K tokens），FullKV 从 9 降至 4 tokens/sec；在 13B model/4096+4096/batch 16 上 ZSMerge 是唯一可运行方法（178.2 tokens/s）；
  (3) Workload-Scalable 吞吐/延迟（Table 1/4）：7B 和 13B 模型，多种 seq length × batch size 组合，ZSMerge 在压缩场景下全面超越 H2O(5%) 和 LESS(5%)，经常超越 FullKV；
  (4) Numerical Error Analysis（§4.3）：残差合并 vs 纯驱逐的注意力输出相对误差，在 ≤20% cache size 下 ZSMerge 减少误差 60.5% (20%), 43.8% (10%), 37.4% (5%)，在 50% 压缩下减少 89.1%；
  (5) XSum ROUGE 质量（Table 2）：LLaMA2-7B 和 Falcon-7B，20%/10%/5% 三种 cache budget，ZSMerge 全面超越 H2O 和 LESS。5% 压缩下 LLaMA2-7B ROUGE-1 30.60 vs FullKV 30.59（基本无损），Falcon-7B 上 ZSMerge 保留超 50% baseline 性能而 LESS 几乎崩溃（ROUGE-1 7.75）；
  (6) LongBench 六类任务（Table 3/5）：LLaMA2-7B 和 Mistral-7B，cache size 512/1024，覆盖 CODE/FSHOT/MDQA/SDQA/SUMM/SYNC，ZSMerge 与 SnapKV 持平，远超 H2O 和 StreamingLLM；
  (7) InfiniteBench 100K+ tokens（Table 6）：LLaMA3-8B，8 个任务，ZSMerge 平均 52.95 vs FullKV 54.69（96.8%），超过 H2O (46.63)、OmniKV (50.33)、InfLLM (42.64)；
  (8) GSM-Infinite-8k（Table 7）：Qwen2.5-7B/Yi-1.5-6B/LLaMA-3.1-8B 三种架构，ZSMerge 整体匹配或超越 SnapKV；
  (9) Hyperparameter Sensitivity（§C.2）：LLaMA2-7B + XSum，sweep Bp/B (0.1-0.9), Br/(B-Bp) (0-0.08), α (0.0-1.0)，推荐 Bp/B=0.5, Br/(B-Bp)=0.02, α=1.0。

- 硬件平台是什么，配置是什么。
  NVIDIA A800-80GB GPU（核心实验平台）。使用 NVIDIA A800-80GB 进行所有吞吐、延迟和 OOM 测试。13B 模型实验也在同一平台完成。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA2-7B (MHA)、Falcon-7B (MQA)、Mistral-7B-Instruct (GQA)、LLaMA-3.1-8B-Instruct (GQA)、Qwen2.5-7B-Instruct (GQA)、Yi-6B/1.5-6B，共六种模型系列，覆盖 MHA/MQA/GQA 三种注意力机制。
  数据集：XSum（16k 新闻文章摘要）、LongBench（21 任务 6 类别，中英文，5K-15K tokens）、InfiniteBench（100K+ tokens，8 任务）、GSM-Infinite-8k（数学推理，3 难度级别）。
  Benchmark/指标：ROUGE-1/2/L（XSum 摘要质量）、LongBench 各任务准确率/得分、InfiniteBench 8 任务准确率、GSM-Infinite-8k symbolic/medium/hard 三级准确率。
  合成数据集：效率测试使用合成数据。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/SusCom-Lab/ZSMerge。基于 Transformers 库实现，仅替换 `scaled_dot_product_attention` 函数。支持 LLaMA、Falcon、Mistral 模型系列。评估使用 KVCache-Factory 框架 (https://github.com/Zefan-Cai/KVCache-Factory)。

  算法 Pipeline 伪代码：

  ```
  # 初始化（每个 attention head）
  B_p = proximity_budget    # 最近 token 保留数
  B_c = context_budget      # 高分 token 保留数
  B_r = residual_budget     # 残差合并 slot 数
  λ = 0.98                  # 衰减因子
  α = 1.0                   # 补偿 scale factor
  s = zeros(T)              # 贡献分数 [T]
  w = zeros(B_r)            # 融合计数 [B_r]
  K_r = zeros(B_r, d)      # 残差 key cache [B_r, d]
  V_r = zeros(B_r, d)      # 残差 value cache [B_r, d]

  # 解码步 T 的前向传播
  def zsmerge_forward(Q, K, V, T):
      # K ∈ R^{T×d}, V ∈ R^{T×d}, Q = q_T ∈ R^d

      # Step 1: 贡献评估（Eq. 5）
      attn_scores = softmax(Q @ K.T / sqrt(d))  # [1, T]
      s = λ * s + attn_scores[0]                 # 指数衰减累积

      # Step 2: Budget 分配与 cache 构建
      # Proximity: 最近 B_p 个 token
      idx_p = [T-B_p+1, ..., T]
      K_p, V_p = K[idx_p], V[idx_p]

      # Context: top-B_c 按 s 排序
      idx_c = topk(s[:T-B_p], B_c)
      K_c, V_c = K[idx_c], V[idx_c]

      # Residual: 将剩余 token 合并到 B_r 个 slot
      idx_evicted = remaining tokens not in idx_p ∪ idx_c
      for (k_t, v_t) in zip(K[idx_evicted], V[idx_evicted]):
          # 选择最兼容的 residual slot（Eq. 6）
          r_hat = argmax(K_r @ k_t)  # maximum dot product
          # 增量均值聚合（Eq. 7）
          K_r[r_hat] = (w[r_hat]*K_r[r_hat] + k_t) / (w[r_hat] + 1)
          V_r[r_hat] = (w[r_hat]*V_r[r_hat] + v_t) / (w[r_hat] + 1)
          w[r_hat] += 1

      # Step 3: 拼接压缩 cache（Eq. 4）
      K_B = concat([K_p, K_c, K_r], dim=0)  # [B, d]
      V_B = concat([V_p, V_c, V_r], dim=0)

      # Step 4: 补偿注意力计算（Eq. 8）
      # 构建权重向量 w_all: uncompressed=1, compressed=w[r_hat]
      scores = Q @ K_B.T / sqrt(d) + α * log(w_all)
      attn = softmax(scores)
      output = attn @ V_B
      return output
  ```

  关键设计要点：
  - 时间复杂度 O(T + B·d)，与纯驱逐方法（O(T)）相近，远优于 full attention 的 O(T²)
  - Br=0 时退化为纯驱逐策略（H2O-like）
  - 残差合并通过 Jensen 不等式保证 attention mass 守恒（Theorem 1）
  - Implementation 细节：prefilling 阶段可用 `window_size` 限制 s 初始化范围（同 SnapKV），大幅加速 prefill


## dKV-Cache: The Cache for Diffusion Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 dKV-Cache —— 首个为 Diffusion Language Models (DLMs) 设计的 KV-Cache 机制。核心实现基于对扩散去噪过程中 token 表征动态的观察：(1) 一旦 token 被解码（从 [MASK] 变为具体 token），其 K/V 表征趋于稳定；(2) token 表征最大的变化发生在解码步和去噪早期。基于此提出 **延迟缓存策略**（delayed caching）：仅缓存已解码 token 的 K/V，掩码 token 每步重新计算；并引入 **一步延迟缓存**（one-step delayed caching）：使用上一步的掩码集合 M_{t-1} 而非当前步 M_t 来决定缓存哪些 token，避免缓存刚解码 token 的不稳定 K/V。设计两种变体：**(1) dKV-Cache-Decode**：近乎无损加速，每 N 步刷新缓存，保持 O(L³) 复杂度；**(2) dKV-Cache-Greedy**：激进缓存，仅对当前 token D_t、上一步 token D_{t-1} 和局部窗口 W(t) 内的 token 计算 QKV，将复杂度降至 O(L²)，代价是轻微性能下降。还提出 dKV-Cache-Prefill 和 dKV-Cache-PD 变体处理长 prefill 场景。底层实现引入 concat_reorder 算子：将缓存 token 重排到序列一端（左侧），避免非连续位置的 gather/scatter 操作，将索引开销从 [B,L,D] 矩阵层级转移到 [B,L] token 层级。

  实验比较：(1) 与 Few-Steps/Half-Steps baseline（减少去噪步数加速）对比（Table 1）：在 LLaDA-8B-Instruct 上，dKV-Cache-Greedy 在 random remasking 下超越 Few-Steps baseline（MMLU 45.77 vs 43.19，GSM8K 67.93 vs 65.58），dKV-Cache-Decode 在 confidence remasking 下接近无损（MMLU 51.00 vs Base 51.11，GSM8K 78.85 vs Base 77.56，HumanEval 46.34 vs Base 39.63 甚至超越）；(2) Dream-7B 实验（Table 2-3）：dKV-Cache-Decode 在 GSM8K 上加速 2.09-4.13×，dKV-Cache-Prefill 加速高达 10.19×（GPQA, L=128），dKV-Cache-PD 在 MBPP 上加速 5.35×；(3) 长 prefill 设置（Table 3）：MLMU T=4 时 dKV-Cache-Prefill 加速 7.40×，GPQA 上 18.91×；(4) 与 cache ratio 和 One-step delay 消融（Figure 3-4）；(5) Memory/Speed tradeoff 分析（Figure 5）：加速 1.75-3.3×，内存开销适中；(6) Batch size 对加速的影响（Figure 7）：batch size=1 时加速有限甚至退步，batch size 增大后加速比显著提升。

- 硬件平台是什么，配置是什么。
  NVIDIA A6000（LLaDA 测速）；NVIDIA H20（Dream 测速）。训练硬件论文未明确说明，方法为 training-free。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaDA-8B-Instruct（masked diffusion LLM，8B 参数）、Dream-Base-7B（从 AR 模型适配的 diffusion LLM，7B 参数）。
  数据集/Benchmark：MMLU（通用语言理解）、GSM8K（数学推理）、Math500（数学推理）、GPQA（研究生级 QA）、HumanEval（代码生成）、MBPP（代码生成）。LLaDA 使用 zero-shot evaluation，Dream 使用 few-shot in-context learning（3-shot/5-shot/8-shot）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/horseee/dKV-Cache（NeurIPS'25 accepted），包含 Dream 和 LLaDA 的修改版模型和生成脚本。

  使用方法：
  ```python
  # Dream 模型加载
  from models.dream import DreamModel
  model = DreamModel.from_pretrained("Dream-7B", use_cache=True, cache_type="decode", cache_steps=4)
  # LLaDA 模型加载
  from models.llada import LLaDAModelLM
  model = LLaDAModelLM.from_pretrained("LLaDA-8B-Instruct", use_cache=True, cache_type="decode", cache_steps=8)
  ```

  算法 Pipeline 伪代码（dKV-Cache-Decode，步 t）：
  ```
  Require: x^{1:L}_{c(t)}, M_t (masked token set), K^{I\M_{t-1}}_{t-1}, V^{I\M_{t-1}}_{t-1}
  1: x' ← x[M_{t-1}]             # 只取未缓存 token（上一步的掩码集），一步延迟
  2: PE' ← [PE[I \ M_{t-1}]; PE[M_{t-1}]]  # 重排位置编码：缓存 token 在左，未缓存 token 在右
  3: Q_t^{M_t}, K_t^{M_t}, V_t^{M_t} ← Transformer(x')  # 仅计算未缓存 token 的 Q/K/V
  4: K_t^I ← Concat(K_{t-1}^{I\M_{t-1}}, K_t^{M_{t-1}})  # 拼接缓存 K 与新计算 K
  5: V_t^I ← Concat(V_{t-1}^{I\M_{t-1}}, V_t^{M_{t-1}})  # 拼接缓存 V 与新计算 V
  6: K_t^{I\M_t} ← Reorder(K_t^I, I'), V_t^{I\M_t} ← Reorder(V_t^I, I')  # 重排序提取下一步缓存
  7: p' ← Attention(Q_t^{M_t}, K_t^I, V_t^I)  # 完整注意力计算
  8: p ← Scatter(p', M_{t-1})  # 将 logits 散播回原始位置
  9: Return p, K_t^{I\M_t}, V_t^{I\M_t}
  ```

  核心张量计算（Eq. 4，dKV-Cache-Decode 的数学形式）：
  ```
  z_t = softmax(Q_t^{M_{t-1}} (K_t^I)^T / sqrt(d_k)) V_t^I
  where:
    K_t^I = concat_reorder(K_{t-1}^{I \ M_{t-1}}, K_t^{M_{t-1}})
    V_t^I = concat_reorder(V_{t-1}^{I \ M_{t-1}}, V_t^{M_{t-1}})
  ```

  dKV-Cache-Greedy 将 M_{t-1} 替换为三个组件的并集：
  ```
  M_t = {D_t} ∪ {D_{t-1}} ∪ W(t)
  where W(t) = {x_i : i ∈ [D_t - ceil(w/2), D_t + floor(w/2)]}  # 以 D_{t-1} 为中心的局部窗口
  ```

  cache ratio 度量：1/T Σ_{i=1}^T |T_i^{cache}| / |T_i|，其中 T_i^{cache} 为步 i 从缓存复用的 token 子集。
  concat_reorder 的关键优化：将 token 序列重排使得缓存 token 在左侧连续排列，从而用 concat（而非 gather/scatter）操作实现 KV 合并，索引开销从 O(BLD) 降至 O(BL)。

## xKV: Cross-Layer SVD for KV-Cache Compression

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 xKV，一种无需训练的 plug-and-play KV-Cache 压缩方法：将多个相邻层的 KV-Cache 水平拼接后执行跨层 SVD，提取共享的 left singular vectors（共享基 A）和各层独立的 reconstruction matrices（B_ℓ_i），从而将多层 KV-Cache 压缩到同一个低秩子空间中。实验比较：vs MiniCache（inter-layer SLERP merging）、vs Single SVD（per-layer SVD）。在 RULER（64K context）上评估 Llama-3.1-8B-Instruct、Qwen2.5-7B/14B-Instruct-1M；在 RepoBench-P 和 LCC 上评估 DeepSeek-Coder-V2-Lite-Instruct（MLA 架构）。指标：各子任务 accuracy 及 compression rate。xKV 在 8× 压缩比下比 MiniCache 达 6.8× 更高压缩率且 accuracy 提升 2.7%；在 MLA 架构上进一步实现 3× 压缩。

- 硬件平台是什么，配置是什么。
  论文仅提及使用 HuggingFace 实现，未明确说明 GPU 型号或硬件配置。prefill 阶段在线 SVD 开销在 128K context 下 <10% of prefill time。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-3.1-8B-Instruct (8 KV heads, GQA)、Qwen2.5-7B-Instruct-1M (4 KV heads)、Qwen2.5-14B-Instruct-1M (8 KV heads)、DeepSeek-Coder-V2-Lite-Instruct (16B MoE, 2.4B activated, MLA)。
  数据集/Benchmark：RULER（subtasks: NIAH-S1/S2/MK1/MK2/MQ/MV, QA-1/2, VT, FWE）、RepoBench-P、LCC（from LongBench）。

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  已开源：https://github.com/abdelfattah-lab/xKV

  **算法 Pipeline (伪代码)**：
  ```python
  # 跨层 SVD 压缩
  # 输入: 一组层的 KV-Cache [X_l1, X_l2, ..., X_l|G|] 各 ∈ R^{L×d}
  # G: 组大小(如2或4), L: 序列长度, d: 隐藏维度

  # 1. 水平拼接
  X_cat = concat_horizontal([X_l for l in group])  # shape: [L, |G| * d]

  # 2. SVD 分解, 保留前 r 个秩
  U, S, Vt = SVD(X_cat)
  U_r = U[:, :r]        # [L, r]
  S_r = S[:r, :r]       # [r, r]
  Vt_r = Vt[:r, :]      # [r, |G| * d]

  # 3. 矩阵融合: A = U_r @ S_r, B_li = Vt_r 的对应列块
  A = U_r @ S_r                      # 共享基, [L, r]
  B = [Vt_r[:, i*d:(i+1)*d] for i in range(|G|)]  # 各层重构矩阵, [r, d]

  # 4. 存储: A + {B_li} for each group
  # 原始存储: |G| * L * d
  # 压缩后: L * r + |G| * r * d
  # 压缩率 ≈ (|G| * L * d) / (L * r + |G| * r * d)
  # 当 L >> r*d 时近似于 L/r

  # 5. Decode 重构
  # X_l_i ≈ A @ B_li = [U_r @ S_r] @ B_li
  ```

  **分组策略**：Stride-based contiguous grouping（N 层分 N/G 组，每组 G 层相邻）。
  **Key/Value 差异处理**：keys 和 values 压缩敏感度不同，固定 rank ratio key:value = 1:1.5（如 key rank=96, value rank=144）。
  **RoPE 处理**：对 pre-RoPE key states 执行 SVD，decode 时重新应用 RoPE。
  **MLA 处理**：对 non-RoPE latent representations 应用 xKV，解耦的 RoPE keys 不压缩。
  **在线分解**：prefill 阶段按请求在线执行 SVD（非离线统计），更好捕捉上下文动态。

## TreeKV: Smooth Key-Value Cache Compression with Tree Structures

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 TreeKV —— 一种 training-free 的 KV cache 压缩方法，使用树形结构（tree structure）实现平滑的 cache 压缩。核心实现：(1) 通过 wavelet 分析发现 token 贡献从远到近呈现平滑递增，越靠近序列末尾信息越丰富且与相邻 token 差异越大，因此设计"左侧稀疏、右侧密集"的树形结构；(2) decoding 阶段：当 KV cache 达到容量上限 c 时，使用循环 eviction scope {idx, idx+1} 决定待淘汰的相邻 token 对，比较两者的平均 attention 权重（importance score），淘汰较低分者，然后 idx 循环递增 (idx mod c + 1)，使得淘汰范围从远到近循环移动，形成从 coarse-grain 到 fine-grain 的平滑过渡树结构；(3) prefilling 阶段：将 prompt 切分为 blocks（block size = b），以最后一个 block 作为 observation window 查询整个序列得到各 block 的 attention-based importance score，然后在 block 级别执行与 decoding 相同的树形淘汰策略，同时计算（而非逐 token 顺序处理）以提升效率；(4) position encoding re-assignment：淘汰后按缓存中剩余 token 的相对顺序重新分配位置编码。

  实验比较：
  (1) Language modeling on PG19 & OpenWebText2（Table 1-2）：与 StreamingLLM、H2O、TOVA、Full Attention 对比，在 4k/8k/16k 上下文下使用 cache size=1024（含 4 sink + 508 recent + 512 selected），TreeKV 在 16k PG19 上 PPL 6.91（vs TOVA 7.15，提升 3.6%），在 16k OpenWebText2 上 PPL 5.18（vs TOVA 5.24，提升 1.1%）；
  (2) 超长序列（10M tokens，Figure 4）：在 PG19 拼接的 10M token 序列上，TreeKV 的 NLL 最低且随长度增长稳定，而 TOVA 和 H2O 性能退化；
  (3) Longbench 长上下文理解（Table 3）：使用 Llama-3.2-1B-Instruct，与 H2O、SnapKV、FullKV 对比，cache size 2k/8k 下 TreeKV 平均得分 31.70/32.80，优于 H2O (30.23/32.79) 和 SnapKV (31.50/32.04)；
  (4) Ablation study（Figure 5）：对比 H2O、TreeKV 和 TreeKV_Select_Left_Token（仅用树结构不用 attention 权重），验证树结构本身对性能的贡献远超 attention-weight-based selection。

- 硬件平台是什么，配置是什么。
  NVIDIA RTX 4090 GPU（24GB），使用 bf16 精度。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-2-7B（pretrained with 4K context length，用于 language modeling），Llama-3.2-1B-Instruct（用于 Longbench 长上下文理解）。
  数据集：PG19 test set（100 本完整书籍，平均 113k tokens/本）、OpenWebText2（从 Pile 随机抽取 100 个 test set 样本，平均 18k tokens/样本）。
  Benchmark：Longbench（16 任务 6 类别：Single-Document QA、Multi-Document QA、Summarization、Few-shot Learning、Synthetic Tasks、Code Completion，平均长度 ~11k tokens）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文声明开源：https://github.com/ZiweiHe/TreeKV（截至检索时仓库为空，代码尚未发布）。

  算法 Pipeline 伪代码（对应论文 Algorithm 1）：

  ```
  # TreeKV Decoding Stage
  # 输入: x^{(1..T)} ∈ R^{1×d}, cache size c
  # 初始化
  S = zeros(c)          # 累积 attention scores
  C = zeros(c)          # 计数
  K_cache = []           # KV cache 存储
  V_cache = []
  idx = 1                # eviction scope 起始索引

  for t in 1..T:
      # 1. 标准 QKV projection（不变）
      q = x[t] @ W_Q    # [1, d]
      k = x[t] @ W_K    # [1, d]
      v = x[t] @ W_V    # [1, d]

      # 2. 追加新 KV 到 cache
      K_cache.append(k)
      V_cache.append(v)

      # 3. 标准 attention 计算
      a = softmax(q @ K_cache^T / sqrt(d))  # [1, len(cache)]

      # 4. 更新 importance scores
      C = (C ∪ {0}) + 1           # 每个 token 的参与计数 +1
      S = (S ∪ {0}) + a           # 累积 attention weights

      # 5. 若超出容量，执行树形淘汰
      if len(K_cache) > c:
          S_avg = S / C            # 平均 attention weight

          # 在 eviction scope {idx, idx+1} 中淘汰较低分者
          if S_avg[idx] > S_avg[idx+1]:
              淘汰 K_cache[idx+1], V_cache[idx+1], C[idx+1], S[idx+1]
          else:
              淘汰 K_cache[idx], V_cache[idx], C[idx], S[idx]

          # idx 循环右移，形成树结构
          idx = (idx + 1) mod c + 1

      # 6. Position encoding re-assignment
      # 按 cache 中剩余 token 的相对顺序重新分配位置编码
  ```

  **Wavelet 分析驱动设计**：
  - 将 attention 输出 s = a^T ∘ V 沿序列长度维度做 multi-level Haar wavelet 分解
  - 观察：越靠近序列末尾，所有频率分量的幅度逐渐增大，尤其高频分量增长显著
  - 含义：token 贡献递增，且与邻居 token 的差异性递增 → 从远到近呈现平滑过渡
  - 因此设计"左侧稀疏(远)、右侧密集(近)"的树形淘汰结构

  **Prefilling Stage 差异**：
  - Prompt 切分为 block_size = b 的多个 block
  - 以最后一个 block 作为 observation window，query 整个序列得到 attention weights
  - 对每个 block 内所有 token 的 attention weights 取均值作为该 block 的 importance score
  - 在 block 级别执行与 decoding 相同的树形淘汰（所有 block 并行处理）

  **Cache 组成**：1k cache = 4 sink tokens + 508 recent tokens + 512 TreeKV-selected tokens

## Tree Attention: Topology-aware Decoding for Long-Context Attention on GPU clusters

- 属于算法pipeline的实现是什么？实验比较什么？
  实现 Tree Attention，将自注意力计算重新表述为能量函数（logsumexp）的梯度，利用 logsumexp 和 max 的结合律（associative property）将序列轴上的归约操作通过树形归约（tree reduction）并行化。核心算法创新：
  (1) **能量函数表述**：Observation 1 证明 self-attention = ∂F/∂ζ|_{ζ=0}，其中 F(ζ) = log Σ_a exp(q·k_a^T + ζ·v_a^T) 为 moment generating function。
  (2) **Tree Decoding 算法（Algorithm 3）**：每 GPU 使用 Flash Attention 2 计算局部分子和分母（lse），再通过 3 次 AllReduce（max reduction + sum reduction ×2）合并全局结果，复杂度 O(3(N/p + log p))，通信步数随设备数对数增长（vs Ring Attention 的线性增长）。
  (3) **拓扑感知通信**：利用现代 GPU 集群的两层网络拓扑（intra-node NVLink 高带宽 + inter-node InfiniBand 低带宽），NCCL 自动选择 intra-node ring reduce + inter-node tree reduce，Tree Attention 的 AllReduce 天然适配此拓扑。

  实验比较：(1) Latency：16-head attention block 下，Tree Attention vs Ring Attention 在不同序列长度（80K-5.12M）和 GPU 数量（8-128 H100）下的执行时间，128 GPU + 5.12M 序列时达到 ~8× speedup；(2) Peak Memory：两个 RTX 4090 上 Tree vs Ring Attention 的 peak memory，Tree Attention 峰值内存约 2× 更低（doubling hidden size from 2048 to 4096, gap doubles from 524MB to 1040MB）；(3) Communication Volume：理论分析 V_tree = 2(p-1)/p × (bd + 2bn_h)，低于 Ring Attention 的 V_ring = 2btd × p；(4) Llama 3.1-8B end-to-end：8×H100 上 Tree Attention 解码比 Ring Attention 快 2-4×（32K-256K），4×MI300X 上快 2-3×，2×RTX 4090 PCIe 上 Llama 3.2-1B 快 4-5×。

- 硬件平台是什么，配置是什么。
  DGX H100 集群：16 节点 × 8 H100 GPU，intra-node NVLink 4.0 (900 GBps) all-to-all，inter-node 8× InfiniBand NDR per node (每 GPU 1 条，400 Gbps，aggregate 3.2 Tbps node injection bandwidth)。AMD MI300X 集群：4 GPU，AMD Infinity Fabric intra-node + RoCE inter-node。NVIDIA RTX 4090：2 GPU，PCIe interconnect。所有计算 BF16 精度。

- 模型是什么。数据集和bench分别是什么。
  模型：标准 16-head attention block（head dim=128）用于 micro-benchmark；Llama 3.1-8B（8×H100, 4×MI300X）、Llama 3.2-1B（2×RTX 4090）用于端到端测试。数据集：合成序列（micro-benchmark）和真实 prompt（Llama 端到端，32K-256K）。Benchmark 非标准 NLP benchmark——主要度量 latency、peak memory、communication volume 和 throughput。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/Zyphra/tree_attention（JAX 实现）。同时提供匿名开源版本：https://anonymous.4open.science/r/tree_attention-7C32。基于 JAX + Flash Attention 2 (JAX binding: https://github.com/nshepperd/flash_attn_jax)。

  算法 pipeline 伪代码（Tree Decoding, Algorithm 3）：
  ```
  # 输入: q ∈ R^{1×d_h}, K,V ∈ R^{N×d_h} 分片在 p 个 GPU 上
  # 每 GPU 持有 chunk: k_hat, v_hat ∈ R^{t×d_h}, t = N/p

  # Step 1: 每 GPU 本地 Flash Attention 2 计算
  o_local, lse_local = flash_attn2(q, k_hat, v_hat)  # o_local ∈ R^{1×d_h}, lse_local ∈ R

  # Step 2: AllReduce(max) 获取全局 max
  m_global = AllReduce(max, lse_local)  # tree reduction, O(log p) 通信步

  # Step 3: 计算本地修正的分子和分母
  n_local = o_local * exp(lse_local - m_global)  # [1, d_h]
  d_local = exp(lse_local - m_global)            # scalar

  # Step 4: AllReduce(sum) 获取全局分子和分母
  n_global = AllReduce(sum, n_local)  # tree reduction, O(log p)
  d_global = AllReduce(sum, d_local)  # tree reduction, O(log p)

  # Step 5: 归一化输出
  z = n_global / d_global  # [1, d_h]
  ```

  张量计算详解：
  - 与 Ring Attention 的关键差异：Ring Attention 将 K,V chunks 在 GPU 间环形传递（P2P），每个 GPU 依次处理所有 chunks。Tree Attention 不移动 K,V chunks，而是每 GPU 本地计算 partial result 后通过 AllReduce 合并——通信量从 O(2btd×p) 降至 O(2(p-1)/p × (bd + 2bn_h))。
  - 理论加速来源：Theorem 1 证明 associative reduction 的时间复杂度为 O(N/p + log p)。当 p 趋近 N，复杂度为 O(log N)。Ring Attention 为 O(N)（需依次传递所有 chunks）。
  - 内存节省来源：Ring Attention 需存储相邻 GPU 传来的 K_chunk, V_chunk + 输出 chunk (4btd + 2bd)；Tree Attention 仅需存储本地 n, d + 结果 (2btd + 2bd + 2bn_h)。由于 2bn_h << 2btd，Tree Attention 峰值内存约减半。
