## MoE-GPS: Guidelines for Prediction Strategy for Dynamic Expert Duplication in MoE Load Balancing

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MoE-GPS 提出两种 expert 预测策略用于 MoE 推理时的动态 expert duplication 实现负载均衡：
    1. **Distribution-Only Prediction**：仅预测 coarse-grained token 分布（各 expert 被激活的比例），不预测具体 token-to-expert 映射。使用 Multinomial Distribution + MLE (Maximum Likelihood Estimation) 对每层 MoE 的 expert 激活概率建模：$\hat{p}_i^l = n_i^l / N$，其中 $n_i^l$ 为训练集第 l 层 expert i 被激活的次数。预测无运行时 overhead（offline 估计），可平衡 FFN compute 但不能减少 All-to-All 通信。
    2. **Token-to-Expert Prediction**：将 expert selection 建模为多分类问题，预测每个 token 的激活 expert。探索三类模型：(a) Probability Model——始终选训练集中频率最高的 expert；(b) Conditional Probability Model——按 token index 或 position index 条件化选择最频繁 expert；(c) Neural Networks——FFN（2 层 MLP, 4096→128→64→8 logits）和 LSTM with Sparse Attention（2-layer LSTM, hidden 64, sparse attention + residual connection）。Token-to-Expert Prediction 可同时平衡 compute 和通信，但有 predictor inference overhead。
    3. 两种策略均配合 Algorithm 1（Expert Duplication 贪心算法）：通过迭代将 overloaded GPU 上的热门 expert 复制到 underloaded GPU，直至所有 GPU token 数差 ≤ 1。
  - 实验比较：(1) Baseline（无 prediction）vs Distribution-Only Prediction vs Token-to-Expert Prediction（多精度点）；(2) 不同 skewness 下的预测准确率（error rate）和系统性能（normalized performance）；(3) Token-to-Expert Prediction 不同 predictor 类型的 accuracy-overhead trade-off（probability model / conditional probability / FFN / LSTM）；(4) 不同 interconnect（NVLink 2TB/s vs PCIe 32GB/s）下的端到端 latency 对比。

- 硬件平台是什么，配置是什么。
  - GPU：4× NVIDIA A100，fully connected via NVLink 3.0（2 TB/s bandwidth）
  - 低带宽配置：PCIe 4.0（32 GB/s bandwidth）
  - 模拟器：LLMCompass [36]（block-level LLM inference simulator, ISCA 2024, validated with silicon measurements）
  - 增强：添加 MoE + Expert Parallelism 支持、Mixtral 架构支持（GQA, SwiGLU, Sliding Window）、Prediction Strategy modeling

- 模型是什么。数据集和bench分别是什么。
  - 模型：
    - Mixtral 8×7B（主要实验，32 layers, 8 experts/layer, Top-K=2）
    - LLaMA-MoE [37]（Appendix C，cross-validation）
    - Switch Transformer [7]（Appendix C，cross-validation）
  - 数据集：
    - MMLU（skewness=1.39, error rate=1.80%）
    - Alpaca Eval（skewness=1.40, error rate=0.98%）
    - SST2（skewness=1.99, error rate=16.00%）
  - 配置：batch size=1, sequence length=512（prefill stage）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：论文代码未开源。使用的 LLMCompass 模拟器（https://github.com/PrincetonUniversity/LLMCompass）开源（ISCA 2024）。
  - 算法 pipeline 伪代码：
    ```
    # === Distribution-Only Prediction (Offline) ===
    # 训练阶段：统计各层 expert 激活频率
    for layer l in 1..L:
        for batch in training_data:
            tokens = batch  # seq_len × N batches
            expert_assignments = MoE_Router(tokens, layer=l)  # Top-K routing
            for expert e in 1..E:
                n_e[l] += count(expert_assignments == e)
        # MLE estimation
        p_hat[e][l] = n_e[l] / total_tokens[l]

    # 推理阶段：使用 p_hat 指导 expert duplication
    for layer l in 1..L:
        # 预测各 GPU 应处理的 token 比例
        target_tokens_per_gpu = total_tokens / G  # G = num GPUs
        # 使用 Algorithm 1: Expert Duplication
        P, d = ExpertDuplication(f=token_expert_map, p_hat[:,l],
                                  M=GPU_memory, C_max=max_copies)
        # Scatter tokens（通信：随机分发，未针对 expert 位置优化）
        tokens = AllToAllScatter(tokens, d)
        # FFN compute（计算已均衡）
        for gpu in 1..G:
            output[gpu] = FFN_Experts(tokens[gpu], P[gpu])

    # === Token-to-Expert Prediction ===
    # 训练 Predictor（以 FFN 为例）
    # input: token embeddings ∈ R^{seq_len × 4096}
    # output: expert logits ∈ R^{seq_len × 8}
    class FFNPredictor:
        def forward(x):  # x: (batch, seq_len, 4096)
            h = ReLU(Linear(x, 4096→128))     # (batch, seq_len, 128)
            h = ReLU(Linear(h, 128→64))        # (batch, seq_len, 64)
            logits = Linear(h, 64→8)            # (batch, seq_len, 8)
            return logits  # 每层独立 classifier head

    # 推理阶段：predictor 插入 Attention 之前
    for layer l in 1..L:
        predicted_experts = Predictor[l](hidden_states)  # overhead
        # 直接路由 token 到对应 GPU（跳过 Scatter 通信）
        tokens = DirectRoute(tokens, predicted_experts)
        output = FFN_Experts(tokens)

    # === Algorithm 1: Expert Duplication ===
    # P: expert→GPU placement, d: token→GPU dispatch
    def ExpertDuplication(f, M, P_init, C_max):
        d[t] = min{g | (f(t), g) in P}  # assign token to any GPU with its expert
        L[g] = |{t | d(t)=g}|           # load per GPU
        while max(L) - min(L) > 1:
            g_h = argmax(L); g_c = argmin(L)
            Δ = ceil((L[g_h] - L[g_c]) / 2)
            e* = most_popular_expert_on(g_h)
            if (e*, g_c) not in P and copies(e*) < C_max:
                copy_weights(e* → g_c)   # 复制 expert 权重
                P = P ∪ {(e*, g_c)}
                reassign first Δ tokens of e* from g_h to g_c
            update L[g_h], L[g_c]
        return P, d
    ```
  - 关键设计：
    - Distribution-Only Prediction 的 overhead 为零（offline 估计，MLE 公式 $\hat{p}_i = n_i/N$ 极简单）
    - Token-to-Expert Prediction 的 overhead 来自 predictor 前向推理，accuracy 越高通常 overhead 越大（更复杂模型）
    - Error rate 建模：Distribution-Only 用 $|\hat{p}-p|/(1/E)$；Token-to-Expert 用 1−accuracy。性能影响模型分 Optimistic/Typical/Pessimistic 三档，默认使用 Typical（errors uniformly distributed）
