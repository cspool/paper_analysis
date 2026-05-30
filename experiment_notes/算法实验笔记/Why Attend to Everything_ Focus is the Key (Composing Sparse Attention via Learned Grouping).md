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
