## LYNX: Enabling Efficient MoE Inference Through Dynamic Batch-Aware Expert Selection

- 属于算法pipeline的实现是什么？实验比较什么？
  - LYNX 提出了一种新的算法 pipeline —— **AffinityBinning（亲和力分箱）** 技术，用于在 batch 级别动态减少 MoE 推理中激活的专家数量。核心算法包括四个步骤：
    1. **Confidence Analysis（置信度分析）**：对每层的每个 token，计算其相对于 top-1 expert 的 log-ratio（即 router logits 之差，等价于 softmax 概率比的对数）。对于 sigmoid-based router（如 DeepSeek），使用 pre-sigmoid scores 的差值。将这些值按模型 sparsity ratio（k/N）决定的 bin width（由参数 α 控制）和 bin count（由参数 β 控制）进行离散化。bin=0 表示最高亲和力（与 top-1 expert 的 log-ratio 为 0），越负的 bin 表示亲和力差距越大。α 和 β 仅由模型架构的 sparsity ratio 决定，无需 task-specific tuning。
    2. **Adaptive Expert Scoring（自适应专家评分）**：对 batch 中所有 token 的 binned 偏好进行加权汇总。使用 batch_size 为底数的指数加权方案：score(e) = Σ_t (batch_size)^{bin(t,e)}，其中 bin(t,e) 为 token t 对 expert e 的 AffinityBinning 结果。高置信度 token 的偏好专家获得指数级更高权重，低置信度 token 的偏好被大幅降权。动态确定最终 active expert set 的大小。
    3. **Expert Remapping（专家重映射）**：将低置信度 token 的 expert assignment 重映射到 minimal critical expert set 内。高置信度 token 始终保留其 top-ranked expert。Preserve top-k 激活语义：每个 token 仍然激活 k 个 expert。
    4. **Phase-Aware Gating（相位感知门控）**：仅在 memory-bound decode iterations 中启用上述 pipeline，prefill 等 compute-bound 阶段直接绕过。
  - 实验比较：
    - Baseline：vLLM 默认推理（标准 top-k routing，无 expert reduction）
    - LYNX vs Baseline：TPOT、准确率、系统吞吐量、SLO-aware throughput
    - 对比范围：4 个模型家族（Qwen, Mixtral, DeepSeek, Llama），8 个 benchmark（GSM8K, HumanEval, MBPP, MATH, ChartQA, MMMU, AIME, GPQA）
    - 关键结果：median TPOT 降低 1.09-1.30x，准确率偏差 <1%，平均情况甚至提升准确率

- 硬件平台是什么，配置是什么。
  - NVIDIA H200 GPU (141 GB HBM)，SXM NVLink 互联
  - 2x AMD EPYC 9554 64-Core CPU，1.5 TB DRAM
  - Ubuntu 22.04.4 LTS，NVIDIA driver 560.35.05，CUDA 12.6
  - TP=2/4 配置，EP=2/4 实验使用 A100

- 模型是什么。数据集和bench分别是什么。
  - **模型**：
    - Qwen2-57B-A14B-Instruct (k=8, N=64, sparsity ratio 0.125)
    - Qwen3-30B-A3B-Instruct
    - Qwen3-235B-A22B-Thinking-2507
    - Mixtral-8x7B-Instruct-v0.1 (k=2, N=8, sparsity ratio 0.25)
    - DeepSeek-V2-Coder (k=8, N=256, sparsity ratio 0.03)
    - Llama-4-Maverick-17B-128E-Instruct
    - Llama-4-Scout-17B-16E-Instruct
  - **数据集/Benchmark**：
    - 代码：HumanEval, MBPP
    - 数学：GSM8K, Minerva Math (Algebra)
    - 视觉推理：ChartQA, MMMU
    - 推理：AIME, GPQA
    - 真实 trace：ShareGPT, Mooncake
    - 准确率指标：Pass@1 (HE/MBPP), Exact Match (MATH/GSM8K)，遵循 EleutherAI LM evaluation harness 规范

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源情况**：论文未提供开源代码链接。arxiv ID 2411.08982。
  - **算法 Pipeline 伪代码**：
    ```
    输入: batch tokens T = {t1, t2, ..., tB}, router logits L (B x N), sparsity ratio k/N
    参数: α (bin width factor), β (max bin count) - 由 k/N 决定

    # Step 1: Confidence Analysis (per-token, per-layer)
    for each token t in batch:
        top1_logit = max(L[t])
        for each expert e in top-k(t):
            log_ratio = L[t][e] - top1_logit  # difference of logits
            bin[t][e] = clamp(floor(log_ratio * α), -β, 0)  # discretize to [negative, 0]

    # Step 2: Adaptive Expert Scoring (batch-level)
    for each expert e in union of all top-k selections:
        score[e] = 0
        for each token t in batch:
            if e in top-k(t):
                score[e] += B^{bin[t][e]}  # exponential weighting by batch_size

    # Step 3: Determine Active Expert Set
    # Keep high-confidence tokens' top-1 experts unconditionally
    # Select additional experts based on score threshold (dynamic)
    active_experts = select_top_by_score(scores, threshold=determine_by_distribution(scores))

    # Step 4: Expert Remapping
    for each token t in batch:
        if confidence(t) < threshold:  # low-confidence token
            remap low-ranked experts to alternatives in active_experts
    ```
  - **关键张量计算流**（单 MoE layer, Qwen2-57B, batch B=16, N=64, k=8）：
    1. Router logits (B x 64) → softmax → top-8 per token
    2. Confidence Analyzer：对每个 (token, expert in top-8) 计算 log_ratio = logit[e] - logit[top1] → 离散化为 bin ∈ [-β, 0]（例如 β=5，则 6 个 bin）
    3. Adaptive Scorer：score[64] = Σ_t B^{bin[t][e]}，B=16 → 高置信度 token (bin=0) 贡献 16^0=1，低置信度 (bin=-5) 贡献 16^{-5}≈0.0001
    4. 动态阈值筛选 → active set 大小（例如从 25 个降至 15 个）
    5. Remapper：low-confidence tokens 的 lower-ranked experts 重映射到 active set 内 → 仍保持每个 token 8 个 expert
    6. 最终 dispatch：(B x 8) → (active_count x ...) 的 GEMM
