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
