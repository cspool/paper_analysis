## Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是两个 MoE 模型的完整预训练与后训练流水线：**Ling-Lite**（总参数 16.8B，激活 2.75B）和 **Ling-Plus**（总参数 290B，激活 28.8B）。算法层面的核心创新包括：(1) **Fine-Grained Experts + Shared Expert**：扩展专家数量并等比缩小每个专家的中间维度，同时引入一个无需路由的 Shared Expert 提供通用能力。(2) **NormHead**：对 LM-Head 权重进行 L2 归一化，缓解训练中输出 norm 不稳定导致的 loss spike。(3) **Stochastic Routing Warmup**：在训练早期以线性衰减权重混合随机路由 logits 和学习到的路由 logits，防止训练初期路由崩溃和不均衡。(4) **Scaling Laws for MoE**：系统分析 MoE 架构的 batch size 和 learning rate 随 compute budget 的幂律关系，以及 MoE vs Dense 的 efficiency lever（~3x）。(5) **Skip Loss Spikes & Sample Retry**：检测到 loss spike 时跳过当前更新并将数据随机重注入后续 batch，持续 spike 则自动降低学习率。(6) **SFT→DPO 后训练流程**：含 quality assurance（rule-based filtering + LLM judge）、semantic deduplication、Vanilla DPO + Robustness Optimization + format-focused DPO。

  实验比较：(a) 与 Dense 架构对比 scaling law（FLOPs-to-Loss 曲线）；(b) Ling-Lite-Base vs Qwen2.5-7B、LLaMA-3.1-8B、Mistral-7B-v0.3；(c) Ling-Plus-Base vs DeepSeek-V2-Base、Qwen2.5-72B-Base、LLaMA-3.1-70B-Base；(d) Ling-Lite instruct vs Qwen2.5-7B-Instruct、LLaMA-3.1-8B-Instruct、Mistral-7B-v0.3-Instruct；(e) Ling-Plus instruct vs DeepSeek-V2.5-Chat、Qwen2.5-72B-Instruct、LLaMA-3.1-70B-Instruct、GPT4o-0806；(f) 不同加速器 (Device A vs Device D) 上的训练一致性；(g) Safety vs False Refusal trade-off；(h) Needle-in-A-Haystack 测试（最长 64K）。

- 硬件平台是什么，配置是什么。
  五种异构 AI 加速器（按可用性降序）：Device A (370 TFLOPS, 64GB, 无 FP8)、Device B (120 TFLOPS, 96GB, 无 FP8)、Device C (312 TFLOPS, 80GB, 无 FP8)、Device D (989 TFLOPS, 80GB, 支持 FP8)、Device E (147 TFLOPS, 96GB, 支持 FP8)。训练共使用 9T tokens 跨五种硬件配置混合训练。Scaling law 实验 compute budget 从 1e18 到 6e20 FLOPs。Ling-Plus 在高性能硬件 (Device D) 上训练 1T tokens 成本约 635 万 RMB，低规格硬件降至约 508 万 RMB（节省 ~20%）。

- 模型是什么。数据集和bench分别是什么。
  模型：Ling-Lite (16.8B/2.75B active) 和 Ling-Plus (290B/28.8B active)，均为 MoE 架构，使用 fine-grained experts + shared expert，dropless 路由，支持 4K→16K context（RoPE θ 从 10K→600K）。预训练数据：9T tokens（1T 中文 + 5.5T 英文 + 2.5T 代码），来源包括 Common Crawl、书籍、学术论文、社交媒体、百科、数学、编程代码。Benchmarks：英文（MMLU、MMLU-Pro、MMLU-Redux、BBH、HellaSwag、PIQA、ARC-Challenge、WinoGrande、RACE-Middle/High）、中文（C-Eval、CMMLU）、数学（GSM8K、MATH）、代码（HumanEval、MBPP、CRUXEval-I/O）。Instruct 额外评估：IFEval、GPQA-Diamond、SimpleQA、C-SimpleQA、MultiPL-E、LiveCodeBench、AIME-2024、BFCL-v2、Nexus、T-eval、Arena-Hard、Arena Safety、Cvalues、Xstest、Orbench-Hard-1k。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  模型权重：[https://huggingface.co/inclusionAI](https://huggingface.co/inclusionAI)。核心算法流程：

  **MoE Forward with Fine-Grained Experts + Shared Expert**:
  ```
  # 输入: h_t ∈ R^d, N experts, top-k routing
  p_t = Softmax(R(h_t))              # 路由概率
  o_t = Σ_i p_{t,i} * E_i(h_t)      # top-k 专家输出加权和
  o_t' = o_t + E_share(h_t)         # 加上 Shared Expert
  ```

  **Stochastic Routing Warmup** (step i ≤ W):
  ```
  s_t = Linear(h_t)                  # 原始路由 logits
  μ_s, σ_s = running_stats(s_t)      # 运行时均值/标准差
  ϵ ~ N(0, I)                        # 标准正态噪声
  ŝ_t = α · s_t + (1-α) · (μ_s + σ_s · ϵ)
  α = min(i/W, 1.0)                  # 从 0 线性增长到 1
  ```

  **NormHead**:
  ```
  h_o = (W_lm_head / ||W_lm_head||_2) · h
  ```

  **Skip Loss Spikes & Sample Retry**:
  ```
  if detect_loss_spike(current_loss):
      skip current_update()
      save affected_data()
      randomly_reinject_data_to_future_batches()
      if spike_persists:
          lr *= decay_factor  # 自动降学习率
  ```

  **Efficiency Lever (Scaling Law)**：给定相同 training loss，MoE 所需 compute budget 约为 Dense 的 1/3，且随 compute budget 增大 efficiency lever 从 ~3× (@ 1e21 FLOPs) 增长到 >3.5× (@ 1e24 FLOPs)。
