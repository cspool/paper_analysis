## LExI: Layer-Adaptive Active Experts for Efficient MoE Model Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  - LExI 提出一种 data-free 的 post-training 优化技术，通过两阶段 pipeline 为预训练 MoE 模型的每一层静态分配最优的 active expert 数量（top-k），无需任何 calibration 数据集或微调：
    1. **Stage 1 — 逐层 Top-K 扰动敏感性分析（Monte Carlo Profiling）**：对每个 MoE layer，从标准正态分布 N(0,1) 采样随机输入张量 X ∈ R^{B×L×H}。先用 baseline top-k 计算输出 Y_base，再用候选 top-k ∈ {1, 2, ..., top-k_baseline} 计算扰动输出 Y_perturbed。用 Frobenius 范数 ||Y_perturbed - Y_base||_F 度量输出偏差，重复 N_iter 次取平均得到该层的 top-k 敏感性 profile D_j(k)。整个过程仅使用模型权重，无需真实数据。
    2. **Stage 2 — 进化搜索（Evolutionary Search with Proxy）**：将 Stage 1 的敏感性值作为 proxy，用进化算法搜索全局最优的逐层 top-k 分配 k* = (k_1, ..., k_L)。目标：最小化总敏感损失 Σ_j D_j(k_j)，约束：Σ_j k_j = B（总 active expert budget），k_min ≤ k_j ≤ k_max。使用 tournament selection + uniform crossover + mutation（每层 ±1 同时保证总和不变），迭代 G_max 代后返回最优分配。
  - 实验比较：
    - LExI vs Baseline（pretrained fixed top-k）vs Inter-Expert Pruning (NAEE, 12.5%/25%/50%) vs Intra-Expert Pruning (MoE-I², 12.5%/25%/50%)
    - 指标：Throughput (tokens/s) vs Accuracy/F1/Perplexity 的 Pareto trade-off
    - LM-Eval：9 个语言理解任务 (ARC-c, ARC-e, BoolQ, HellaSwag, MMLU, OBQA, RTE, WinoGrande) 平均准确率
    - LongBench (Qasper)：F1 score vs throughput
    - Passkey Retrieval：准确率 vs throughput（100 iterations，varying depths）
    - Perplexity：C4, PTB, WikiText-103 上的 PPL vs throughput
    - VLMEvalKit：MME, MMMU, ScienceQA（仅 DeepSeekVL2-Tiny）
  - 结果：OLMoE-1B-7B 上 LExI (B=100) 可达与 50% intra-pruning 相同的 throughput 同时准确率高 10%；Qwen1.5-MoE 上 LExI 获得比 inter/intra pruning 高 5.1% 的吞吐量同时准确率高 0.5%；Mixtral-8x7B 上 LExI 在相同吞吐量下比 inter-pruning 准确率高 10%。

- 硬件平台是什么，配置是什么。
  - **NVIDIA H100 80GB GPUs**（支持 Tensor Cores）
  - 大多数 LLM 模型使用 4 GPUs；DeepSeek-V2-Lite-Chat 和 DeepSeekVL2-Tiny 使用 2 GPUs
  - 多 GPU 间使用 Tensor Parallelism
  - 批量推理 batch size = 16，input/output 序列长度因模型而异（遵循各模型最大 context length 约束）
  - 论文未明确说明 CPU、内存、互联类型（NVLink/NVSwitch）等详细配置

- 模型是什么。数据集和bench分别是什么。
  - **LLM 模型**：
    - Mixtral-8x7B-Instruct-v0.1 (46.7B, 32 layers, 8 experts, top-k=2)
    - Qwen1.5-MoE-A2.7B-Chat (14.3B, 24 layers, 60 experts, top-k=4)
    - OLMoE-1B-7B-0924-Instruct (6.92B, 16 layers, 64 experts, top-k=8)
    - MiniCPM-MoE-8x2B (17B, 40 layers, 8 experts, top-k=2)
    - DeepSeek-V2-Lite-Chat (15.7B, 27 layers, 64 experts, top-k=6)
  - **VLM 模型**：DeepSeekVL2-Tiny (3B, 12 layers, 64 experts, top-k=6)
  - **Benchmarks**：
    - lm-eval-harness：ARC-c, ARC-e, BoolQ, HellaSwag, MMLU, OpenBookQA, RTE, WinoGrande（报告平均准确率）
    - LongBench：Qasper（F1 score）
    - Passkey Retrieval（准确率，100 iterations with varying depths）
    - Perplexity：C4, Penn Treebank (PTB), WikiText-103
    - VLMEvalKit：MME, MMMU, ScienceQA（VLM 评估）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - **未开源**。论文未提供代码仓库链接或代码可用性声明。
  - LExI 算法 pipeline 伪代码（基于论文 Algorithm 1 + Algorithm 2）：
    ```
    # Stage 1: Per-Layer Sensitivity Profiling
    D = {}  # key: top-k value, value: list of Frobenius norms
    for i in range(N_iter):
        X = randn(B, L, H)  # random Gaussian input
        set_topk(model, k_base)
        Y_base = moe_forward(X)
        for k in T:  # T = [1, 2, ..., k_base]
            set_topk(model, k)
            Y_k = moe_forward(X)
            D[k].append(||Y_k - Y_base||_F)
    for k in T:
        D[k] = mean(D[k])  # average perturbation loss

    # Stage 2: Evolutionary Search
    population = random_feasible_allocations(N_pop, L, B, k_min, k_max)
    for g in range(G_max):
        p1, p2 = tournament_select(population)  # min φ(k) = Σ D_j(k_j)
        offspring = uniform_crossover(p1, p2)    # each layer randomly from parent
        offspring = mutate(offspring, η_mut)      # ±1 per layer, ΣΔ = 0
        offspring = project_to_feasible(offspring, B, k_min, k_max)
        population.append(offspring)
    k_star = argmin_{k in population} Σ D_j(k_j)
    return k_star  # (k_1, ..., k_L) per-layer top-k
    ```
  - 核心张量计算：MoE 层输出 y = Σ_{i=1}^{top-k} G(x)_i · E_i(x)，其中 G(x) = Softmax(TopK[x·W_g])。LExI 通过改变每层的 top-k 参数（set_topk 操作）控制激活的 expert 数量。Frobenius 范数 ||Y_k - Y_base||_F = sqrt(Σ_{i=1}^{B×L×H} (Y_k[i] - Y_base[i])²) 衡量改变 top-k 对输出的扰动程度。
