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
