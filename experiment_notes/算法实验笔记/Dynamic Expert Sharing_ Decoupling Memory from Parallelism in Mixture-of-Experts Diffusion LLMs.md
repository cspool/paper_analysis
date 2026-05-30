## Dynamic Expert Sharing: Decoupling Memory from Parallelism in Mixture-of-Experts Diffusion LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  DES (Dynamic Expert Sharing) 将 MoE 优化从 token-centric pruning 转变为 sequence-level coreset selection，最大化并行解码块内的 expert 复用。两种策略：
  1. **DES-Seq**（Intra-Sequence Sharing）：对每个 token 取 Top-k experts，取所有 token 的并集作为共享 coreset —— C_DES-Seq = ∪_{n=1}^{N} TopK(I_n, k)。
  2. **DES-Vote**（Saliency-Aware Voting）：所有 token 按加权 router saliency 投票选举 coreset —— 先 mask 每 token 的 local Top-K 之外权重，跨序列聚合加权投票 V_i = Σ_{n=1}^{N} Masked(I_{n,i})，再取 Top-M_core experts。
  实验比较 DES-Seq (k=2, k=3) 和 DES-Vote (β=0.10, β=0.15 for LLaDA2.0; β=0.4, β=0.6 for LLaDA-MoE-7B) vs Vanilla、Top-K、NAEE、MC-MoE 在生成 benchmark 上的 accuracy 和 expert load、latency 表现。

- 硬件平台是什么，配置是什么。
  NVIDIA B200 GPU，CUDA 13.1，Intel Xeon 6960P CPU。使用 NVIDIA Nsight Systems 进行 kernel profiling。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaDA2.0-Mini (16B) 和 LLaDA-MoE-7B-A1B-Instruct (7B)，均为 MoE dLLM 架构。推理框架：dInfer + Fast-dLLM (KV cache 方法，0.9 confidence-based sampling)。
  数据集/benchmark：HumanEval、MBPP、GSM8K、MATH500，评估 long-form generative decoding 和多样化推理能力。Block length 32 (16 prefix + 16 suffix cache tokens)。

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  论文未提供开源代码（arXiv 2602.00879，2026年1月）。Catalyzex 标记为 "Paper and Code" 但无公开 GitHub 链接。

  **DES 算法伪代码（Algorithm 1）**：
  ```
  输入: 序列信息 I, Coreset 选择函数 Φ, 激活函数 σ, 目标 K
  输出: 层输出 Y
  
  Stage 1: Sequence-level Consensus
    C ← Φ(I)                        // 识别高效用 expert coreset
  
  Stage 2: Constrained Local Routing
    for each token n ∈ {1, ..., N}:
      S_n ← TopK(I_n|_{i∈C}, K)     // 在 coreset 内路由
      g_n ← σ(I_n|_{i∈S_n})          // 重新归一化 gate weights
      y_n ← Σ_{i∈S_n} g_{n,i} · E_i(x_n)
    return Y = {y_1, ..., y_N}
  ```

  **DES-Vote 具体过程（Algorithm 3）**：
  ```
  输入: Router logits I (shape: N×M), Coreset size M_core, Top-K
  1: I_m ← Mask(I, K)               // 保留 local Top-K，其余置零
  2: V ← Σ_{n=1}^{N} I_{m,n}        // 跨 token 聚合加权投票 (shape: M)
  3: C ← TopK(V, M_core)            // 排序选 top-M_core experts
  ```

  **延迟模型**：L_MoE(Φ) ≤ b·|Φ(I)| + a·(N·K)，其中 b 为 HBM→SRAM weight fetching cost，a 为 marginal compute cost。优化目标：min |Φ(I)| s.t. A(Φ(I)) ≥ A_base - ε。

  **关键结果**：DES-Vote (β=0.15) 在 LLaDA2.0-Mini 上减少 unique expert activations 55%（T=84→38），保留 99.5% relative accuracy，MoE 层延迟降低 38.0%。DES-Vote 在相同 coreset size 下始终优于 DES-Seq（Top-K recall 更高，reconstruction loss 更低）。
