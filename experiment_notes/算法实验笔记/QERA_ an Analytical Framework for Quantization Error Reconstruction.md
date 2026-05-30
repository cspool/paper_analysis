## QERA: an Analytical Framework for Quantization Error Reconstruction

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：QERA 提出两个分析解（analytical solution）来求解量化误差重建（Quantization Error Reconstruction, QER）问题——给定量化权重 W̃，寻找最优低秩项 C_k = A_k B_k 使得层输出误差最小化：(1) QERA-exact (Theorem 1)：最小化 E[||y - ỹ||²] 的精确闭式解 C_k = (R_{XX}^{1/2})⁻¹ · SVD_k(R_{XX}^{1/2} (W - W̃))，其中 R_{XX} = E[x^T x] 为输入自相关矩阵；(2) QERA-approx (Theorem 2)：在"不同嵌入维度不相关"假设下的高效近似解 C_k = S⁻¹ · SVD_k(S (W - W̃))，其中 S = diag(√E[x₁²], ..., √E[x_m²]) 为对角标度矩阵。QERA 对量化函数 q(·) 无约束，可结合任意量化方法使用。QERA-approx 解释了 LQER 启发式方法的成功，并解决了 LQER 中校准样本数与恢复性能不一致的问题。
  - 实验比较：(1) QPEFT 实验：QERA-approx vs Full Fine-tuning, LoRA, QLoRA, LoftQ (5-iter) — 在 RoBERTa-base @ GLUE (4/3/2-bit, rank 8/64) 和 LLaMA-2-7B/LLaMA-3.1-8B @ SlimPajama & GSM8K (4/2-bit, rank 8/64) 上比较微调准确率/困惑度和收敛速度；(2) PTQ 实验：QERA-exact & QERA-approx vs BF16, w-only, ZeroQuant-V2, LQER, HQQ — 在 TinyLlama-1.1B, Gemma-2-2B, Phi-3.5-mini, LLaMA-2-7B/13B, LLaMA-3.1-8B/70B 上比较 WikiText2 PPL 和 6 个下游任务 (ARC, BoolQ, CommonSenseQA, Winogrande, MMLU, BBH) 准确率，以及 Vicuna-7b-v1.5 @ AlpacaEval 2.0 的 Win Rate；(3) 消融：模型输出误差 vs 权重逼近误差、LoftQ 迭代数 vs 模型输出误差、校准集大小对 LQER vs QERA 性能的影响、Assumption 1 的 R_{XX} 非对角元验证。

- 硬件平台是什么，配置是什么。
  - QPEFT 实验：4× NVIDIA A100 80GB GPU，AMD EPYC 64-Core Processor，1024GB RAM，总计约 2100 GPU hours。
  - PTQ 实验：8× NVIDIA A6000 48GB GPU，AMD EPYC 256-Core Processor，1024GB RAM，总计约 4500 GPU hours。
  - 矩阵平方根计算使用 SciPy 的 blocked Schur algorithm，在 CPU 上执行（FP64）。自相关矩阵外积在 FP32 累积，FP64 计算平方根。

- 模型是什么。数据集和bench分别是什么。
  - 模型：RoBERTa-base (QPEFT)；LLaMA-2-7B/13B, LLaMA-3.1-8B/70B, TinyLlama-1.1B, Gemma-2-2B, Phi-3.5-mini (PTQ)；Vicuna-v1.5-7B (指令跟随评估)。
  - 数据集（微调）：GLUE benchmark (MNLI, QNLI, RTE, SST-2, MRPC, CoLA, QQP, STS-B)；SlimPajama（连续预训练）；GSM8K（监督微调）。
  - 数据集（校准）：WikiText2（用于 RoBERTa-base QPEFT 校准）；论文中 QPEFT 校准集来自预训练数据集；PTQ 校准集论文未明确指定具体数据集名。
  - Benchmark（PTQ 评估）：WikiText2 (perplexity), ARC (challenge), BoolQ, CommonSenseQA, Winogrande, MMLU, BigBench-Hard (BBH)；使用 lm-evaluation-harness 评测。AlpacaEval 2.0 (GPT4-Turbo 作为 evaluator, length-controlled win rate)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源链接：https://github.com/ChengZhang-98/QERA。
  - 软件依赖：PyTorch, Transformers, PEFT, Accelerate, SciPy (blocked Schur algorithm), lm-evaluation-harness, Evaluate, AlpacaEval 2.0。
  - 量化格式：4-bit 使用 QLoRA 的 4-bit floating point (PEFT 实现)；3/2-bit 使用 emulated MXINT (block size=32/16)。
  - 核心伪代码（QERA-approx 初始化，PyTorch-like）：
    ```
    # W: pretrained weight [m, n]
    # q(), dq(): quantize, dequantize functions
    # X_calib: calibration dataset of input vectors
    # k: target rank

    def qera_approx_init(W, q, dq, X_calib, k):
        # Step 1: Compute activation statistics
        s_sq = torch.zeros(m)
        for x in X_calib:        # x shape: [1, m]
            s_sq += x.square().squeeze(0)
        s_sq /= len(X_calib)
        S = torch.diag(torch.sqrt(s_sq))  # shape: [m, m]

        # Step 2: Quantize and compute scaled error
        W_q = q(W)
        W_tilde = dq(W_q)
        E = W - W_tilde                # weight quantization error
        Q = S @ E                       # scaled error

        # Step 3: Truncated SVD on scaled error
        U, Sigma, Vt = torch.svd(Q)
        U_k = U[:, :k]
        Sigma_k = Sigma[:k]
        Vt_k = Vt[:k, :]

        # Step 4: Unscale to get low-rank terms
        A_k = torch.inverse(S) @ U_k   # shape: [m, k]
        B_k = torch.diag(Sigma_k) @ Vt_k  # shape: [k, n]

        return A_k, B_k

    # Forward pass (same for QERA-exact and QERA-approx):
    # y = x @ (W_tilde + A_k @ B_k)
    # At inference: pre-merge C_k = A_k @ B_k into W_tilde
    ```
  - QERA-exact 伪代码（区别于 approx 在于使用 R_{XX} 代替 S）：
    ```
    def qera_exact_init(W, q, dq, X_calib, k):
        # Step 1: Compute autocorrelation matrix
        R = torch.zeros(m, m, dtype=torch.float64)
        for x in X_calib:
            R += (x.T @ x).to(torch.float64)   # outer product
        R /= len(X_calib)
        R_sqrt = matrix_sqrt(R)                 # blocked Schur, CPU

        # Step 2-4: Same as approx but with R_sqrt instead of S
        W_q = q(W); W_tilde = dq(W_q)
        Q = R_sqrt @ (W - W_tilde)
        U, Sigma, Vt = torch.svd(Q)
        A_k = torch.inverse(R_sqrt) @ U[:, :k]
        B_k = torch.diag(Sigma[:k]) @ Vt[:k, :]
        return A_k, B_k
    ```
  - 张量计算流程：给定线性层 y = xW（x ∈ R^m, W ∈ R^{m×n}），量化 W → W̃ = dq(q(W))，QERA 寻找 C_k = A_k B_k（rank k << min(m,n)）使得 ||ỹ - y||₂ 最小化。QERA-exact 通过 R_{XX}^{1/2} 将最小化层输出误差转化为标准 SVD 低秩逼近问题：min ||R_{XX}^{1/2}(W̃ + C_k - W)||_F²。QERA-approx 在 Assumption 1（E[x_i x_j]=0, i≠j）下将 R_{XX} 简化为对角矩阵 S²，大幅降低计算开销。推理时 y = x(W̃ + A_k B_k)，低秩项可预合并进 W̃ 不引入额外推理开销。
