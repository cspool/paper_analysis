## SqueezeLLM Dense-and-Sparse Quantization

- 属于算法pipeline的实现是什么？实验比较什么？
  Post-training weight-only quantization framework with two novel techniques: (i) **sensitivity-based non-uniform quantization** using Fisher information (diagonal Hessian approximation) to weight k-means clustering — centroids are pulled closer to weights with higher second-order sensitivity to the final loss, achieving far better perplexity than uniform quantization at equal bitwidth; (ii) **Dense-and-Sparse decomposition** that extracts 0.45% weight values (0.05% most sensitive + 0.4% outliers) into a sparse FP16 matrix stored in CSR format, with the remaining 99.55% dense matrix quantized at 3-4 bits with significantly contracted value range. Both dense and sparse components participate in inference via separate but concurrently-launched custom CUDA kernels.

  实验比较：Perplexity on C4 and WikiText2 (LLaMA 7B/13B/30B/65B, LLaMA2 7B/13B/70B, OPT 1.3B/2.7B/6.7B/13B/30B) against RTN, GPTQ (with/without activation ordering, with/without grouping g128), AWQ (g128), SpQR, QuIP, and OmniQuant; MMLU zero-shot and 5-shot accuracy on Vicuna v1.1/v1.3 (7B/13B/33B) vs AWQ; instruction-following ability via GPT-4 pairwise scoring vs GPTQ/AWQ.

- 硬件平台是什么，配置是什么。
  量化：NVIDIA A100-80G (Fisher information computation via gradient backpropagation) + Intel Xeon Gold 6126 48-core (sensitivity-weighted k-means clustering)。推理延迟评测：NVIDIA A6000 GPU (primary, using Torch CUDA profiler for 128/1024 token generation), also A100 GPU (kernel-only matrix-vector runtime benchmark)。

- 模型是什么。数据集和bench分别是什么。
  Models: LLaMA (7B/13B/30B/65B), LLaMA2 (7B/13B/70B), OPT (1.3B/2.7B/6.7B/13B/30B), Vicuna v1.1 (7B/13B), Vicuna v1.3 (7B/13B/33B)。Datasets: C4 and WikiText2 (perplexity, chunk size 2048), MMLU benchmark (zero-shot and 5-shot), Vicuna evaluation (80 sample questions, GPT-4 pairwise scoring with order randomization, 160 total queries)。

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/SqueezeAILab/SqueezeLLM (MIT license)。

  算法Pipeline全流程（以LLaMA-7B 3-bit量化为例）：

  **Step 1: Fisher信息矩阵计算（Sensitivity Estimation）**
  ```
  # 对calibration dataset D (100 samples from C4或Vicuna training set)
  for each sample d in D:
      loss = cross_entropy(model(d), labels)
      compute gradients g_d = ∂loss/∂W for all weight matrices
  F = diagonal( (1/|D|) * Σ_d (g_d ⊙ g_d) )  # per-weight Fisher diagonal
  ```
  Fisher计算资源：7B模型~0.3分钟(A100)，65B模型~2.5分钟(A100)。峰值内存需求：7B=33GB, 13B=61GB, 30B=149GB, 65B=292GB。

  **Step 2: Sensitivity-weighted K-means Clustering（Per-channel, per weight matrix）**
  ```
  量化目标函数: argmin_Q Σ_{i=1}^{N} F_ii * (w_i - Q(w_i))^2
  # Fisher对角线值F_ii充当每个权重w_i的importance weight
  # 非均匀量化: 每个输出channel有独立LUT (2^b个FP16 centroid)
  
  for each output channel c in weight matrix W (shape [out_c, in_c]):
      k = 2^b  # e.g., k=8 for 3-bit
      centroids = k-means++初始化(W[c, :], k)
      用F[c, :]作为权重执行weighted k-means:
          repeat until convergence:
              assignment[i] = argmin_j ||w_i - centroid_j||^2
              centroid_j = (Σ_{i ∈ cluster_j} F_ii * w_i) / (Σ_{i ∈ cluster_j} F_ii)
      输出: LUT[c] = {c_0, ..., c_{k-1}} (FP16 values), indices[c] ∈ [0, k-1]^in_c (b-bit每个)
  ```
  K-means耗时：7B=11min, 13B=17min, 30B=45min, 65B=80min (Xeon Gold 6126 48核)。

  **Step 3: Dense-and-Sparse Decomposition**
  ```
  for each weight matrix W:
      # 识别outliers (基于百分位阈值)
      T_min = percentile(W, τ_low)    # τ_low ≈ 0.2%
      T_max = percentile(W, 1 - τ_high) # τ_high ≈ 0.2%
  
      # 识别sensitive values (基于Fisher信息排名)
      top_k_sensitive = topk_indices(F, k=0.05% of total elements)
  
      # 稀疏矩阵S = outliers ∪ sensitive values (去重)
      S_indices = {i | W_i < T_min or W_i > T_max or i ∈ top_k_sensitive}

      # 存储S为CSR格式 (FP16)
      S_csr.values = W[S_indices]             # FP16
      S_csr.col_indices = column indices       # int16
      S_csr.row_ptrs = row boundary offsets    # int32

      # 密集矩阵D = W中非S元素 (99.55%的参数)
      D = W.copy(); D[S_indices] = 0
  
      # 对D执行Step 2的sensitivity-weighted k-means
      D_indices, LUTs = weighted_kmeans_quantize(D, F, b=3)
  
  存储格式:
  - Dense分量: 3-bit indices + per-channel 8-entry LUT (8×FP16 per channel)
  - Sparse分量: CSR格式带FP16 values
  - 总avg bits ≈ 3.24 bit (3-bit dense + ~0.24 bit sparse overhead for 0.45% sparsity)
  ```

  **Step 4: 推理Forward Pass (GPU kernel)**
  ```
  def forward_layer(W_indices_3bit, LUTs, S_csr, activation_X):
      # X: activation vector (FP16), shape [in_features]
      # 两个kernel fused在单次launch中:
      
      # Dense部分: LUT-based dequant + matvec
      # 每个thread block加载index → LUT lookup → FP16 multiply-accumulate
      Y_dense = lut_dequant_matvec(W_indices_3bit, LUTs, X)
      
      # Sparse部分: Balanced CSR SpMV
      # 每线程10个non-zero元素 (balanced kernel避免row-skew问题)
      Y_sparse = balanced_csr_matvec(S_csr, X)
      
      return Y_dense + Y_sparse
  ```
  全部计算保持FP16精度，activations不量化。非均匀量化的LUT dequantization开销很小（相比uniform quantization增加仅~10% latency）。

  **关键结果（LLaMA-7B, C4 perplexity）**:
  | 方法 | Avg Bits | PPL (C4) | Speedup vs FP16 | Mem (GB) |
  |------|----------|----------|-----------------|----------|
  | FP16 Baseline | 16 | 7.08 | 1.0x | 12.7 |
  | RTN 3-bit | 3 | 28.26 | 2.3x | 2.9 |
  | GPTQ 3-bit (no group) | 3 | 9.55 | 2.3x | 2.9 |
  | SqueezeLLM dense-only | 3.02 | 7.75 | 2.1x | 2.9 |
  | GPTQ 3-bit (g128) | 3.24 | 7.89 | 0.2x (permutation overhead) | 3.0 |
  | AWQ 3-bit (g128) | 3.24 | 7.90 | 2.0x | 3.0 |
  | **SqueezeLLM 0.45%** | **3.24** | **7.56** | **1.9x** | **3.1** |

  Dense-only SqueezeLLM 3-bit already outperforms GPTQ g128 (7.75 vs 7.89), demonstrating that sensitivity-based non-uniform quantization alone is more effective than group-wise uniform quantization. Adding 0.45% sparsity further reduces the gap from FP16 to only 0.48 PPL points.
