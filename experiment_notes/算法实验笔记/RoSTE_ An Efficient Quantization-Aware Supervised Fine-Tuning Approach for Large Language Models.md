## RoSTE: An Efficient Quantization-Aware Supervised Fine-Tuning Approach for Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：RoSTE 是一种量化感知监督微调（QA-SFT）算法，将 4-bit 权重量化、激活量化和 KV cache 量化与监督微调结合在单一训练阶段。核心算法 pipeline：(1) **Bilevel Optimization Formulation**：上层子问题通过 STE (Straight-Through Estimator) 优化量化后的权重矩阵以最小化 SFT loss；下层子问题通过最小化 weight-activation quantization error surrogate loss 来选择最优旋转矩阵。(2) **Adaptive Rotation Strategy**：对每一层，在 identity 矩阵 I（无旋转）和 random Walsh-Hadamard 矩阵 H 之间做离散搜索，选择使量化误差更低的选项。旋转矩阵 R 作用于线性层：`LIN_i(X; W_i, R_i) = σ(Q_x(X R_i) Q_w(R_i^T W_i))`，利用正交变换 R_i R_i^T = I 保持计算不变性。旋转分为可离线吸收的 between-block rotation R1/R2 和在线旋转 R3/R4。(3) **RoSTE 训练循环（Algorithm 1）**：外层 K 次迭代（论文设 K=1）交替执行 rotation configuration search（逐层比较 I vs H 的量化误差 E(12)）和内层 T 步 QAT via STE（`w^{t+1} = w^t - η g_ste^t`，其中 `g_ste^t = (⟨Q_x(Rx_t) | Q_w(Rw^t)⟩ - y_t) R^T Q_x(Rx_t)`）。量化方案：非对称均匀量化（asymmetric uniform quantizer），per-token activation quantization + per-channel weight quantization。旋转矩阵使用 fast Hadamard CUDA kernel 实现高效在线旋转。
  - 实验比较：(a) **Exp.1**：Pythia 1B/6.9B 和 Qwen2.5 0.5B/7B 在 Reddit TL;DR Summarization 任务上对比 RoSTE vs PTQ baselines（RTN, GPTQ, QuaRot, SpinQuant on fine-tuned models）和 QAT baseline（STE without rotation），W4A4KV4 及 W4A8KV4 配置，评价指标 ROUGE-1/2/L/LSum；(b) **Exp.2**：Llama 3.1 8B 在 Tulu 3 SFT mixture 上训练，6 个下游任务评估（TruthfulQA, MMLU-Pro, BigBenchHard, AGIEval, GSM8K, MATH），W4A4KV4 及 W4A8KV4 配置；(c) 消融实验：旋转策略对比（No Rotation / Complete Rotation / Adaptive Rotation (RoSTE)）；(d) 理论验证：量化误差随训练步数的变化轨迹（Fig. 4）、激活 outlier 分布可视化（Fig. 3, 6, 7）；(e) 与 QLoRA、LLM-QAT、DuQuant 的额外比较（Table 7-9）；(f) 训练开销统计（Table 10：training time + peak GPU memory）。

- 硬件平台是什么，配置是什么。
  - 8× NVIDIA A100 GPUs 集群。CUDA 环境论文未详细说明版本号。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Pythia 1B/6.9B (Biderman et al., 2023), Qwen2.5 0.5B/7B (Yang et al., 2024), Llama 3.1 8B (Dubey et al., 2024)。
  - Exp.1 数据集：Reddit TL;DR Summarization dataset (Huang et al., 2024)，训练集 117k 样本，评估用 TL;DR test dataset，指标 ROUGE-1/2/L/LSum (Lin, 2004)。
  - Exp.2 数据集：Tulu 3 SFT mixture dataset (Lambert et al., 2024)，训练集 100k 样本。评估使用 EleutherAI LM Evaluation Harness (Gao et al., 2021)，benchmarks：TruthfulQA (6-shot, Acc mc1), MMLU-Pro (0-shot, EM), BigBenchHard (3-shot, EM), AGIEval (0-shot, Acc), GSM8K (8-shot, EM), MATH (4-shot, EM)。
  - 量化配置：权重 W4/W4（4-bit），激活 A4/A8（4/8-bit），KV cache KV4/KV8（4/8-bit），asymmetric uniform quantizer，per-token activation + per-channel weight 量化组，clipping factor ∈ {1, 0.95, 0.9}。
  - 校准：从 fine-tuning dataset 抽取 n=128 样本计算量化误差 E(12)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 代码开源：https://github.com/OptimAI-Lab/RoSTE
  - **算法伪代码（对应 Algorithm 1 + 全文实现细节）**：
    ```
    输入: 预训练权重 {W_i^pt}_{i=0}^{ℓ-1}, 学习率 η, 校准样本 D_cal (n=128), SFT 数据集 D_sft
    输出: 量化微调模型 m_Q(·; W^{KT}, R^{K-1})

    // Step 1: 修改 normalization layer（吸收 mean subtraction + scale/shift 到相邻权重矩阵）
    for each normalization layer:
      if LayerNorm: absorb mean subtraction into prev weight, absorb scale/bias into next weight
      if RMSNorm: absorb scale into next weight

    // Step 2: 初始化
    W^0 = {W_i^pt}_{i=0}^{ℓ-1}

    // Step 3: RoSTE 外层循环 (论文设 K=1)
    for k = 0, ..., K-1:
      // -- Lower level: Rotation Configuration --
      计算 E_all_I = E(W^{kT}, {I}_{i=0}^{ℓ-1})   // 全部无旋转的量化误差
      计算 E_all_H = E(W^{kT}, {H}_{i=0}^{ℓ-1})   // 全部旋转的量化误差
      for each layer/module i = 0, ..., ℓ-1:
        比较 layer-wise quantization error:
          若 E_i(I) < E_i(H): R_i^k = I  (no rotation)
          否则: R_i^k = H  (random Walsh-Hadamard rotation)
          其中 H = H_diag · Diag(s), H_diag ∈ R^{d×d} 为 Walsh-Hadamard 矩阵, s ∈ {-1,1}^d 随机 sign vector

      // -- Upper level: QAT Stage via STE --
      for t = 0, ..., T-1:
        采样 mini-batch ξ ⊆ D_sft
        对每个 linear layer i (forward pass):
          X_in = input activation
          // 在线旋转（若 R_i 未被 merge 进权重）
          X_rot = X_in · R_i                  // 对非 mergeable rotation (R_3, R_4)
          X_q = Q_x(X_rot)                    // per-token asymmetric 量化
          W_rot = R_i^T · W_i                 // 若 R_i mergeable，在训练前已完成
          W_q = Q_w(W_rot)                    // per-channel asymmetric 量化
          output = σ(X_q · W_q)               // INT4 matmul via fast Hadamard kernel

        计算 SFT loss L = -E_i[Σ_t log P(y_{i,t} | x_i, y_{i,<t}; m_Q)]

        // Backward via STE
        梯度近似：∂Q_w(R_i^T W_i) / ∂W_i ≈ R_i  (STE: 量化器当作恒等)
        更新：W^{kT+t+1} = W^{kT+t} - η ∇_W L_SFT(m_Q(·; W, R^k); ξ)

      // 旋转矩阵吸收（merge offline rotations into weights for inference）
      merge R_1, R_1^T, R_2, R_2^T, R_4^T into corresponding weight matrices
      keep R_3, R_3^T, R_4 as online fast Hadamard kernel rotations
    ```
  - **量化误差计算 E(12)**（用于 rotation selection）：
    ```
    E({W_i}, {R_i}) = Σ_{i=0}^{ℓ-1} ||Q_w(R_i^T W_i) - R_i^T W_i||^2
                    + (1/n) Σ_{i=0}^{ℓ-1} Σ_{j=0}^{n-1} ||Q_x(X_{i,j} R_i) - X_{i,j} R_i||^2
    ```
    对 n=128 个校准样本计算 weight quantization error + activation quantization error，逐层逐 sample 求和。
  - **关键旋转位置**（Fig. 5）：R1（between-block, offline mergeable）作用于 Q/K/V projection、Up/Gate projection、O projection、Down projection、embedding、lm_head；R2（in-block, offline mergeable）作用于 Value projection 和 O projection（MHSA 内）；R3（in-block, online）作用于 Query 和 Key（消除 KV cache outliers）；R4（in-block, online）作用于 Down projection（MLP 内）。
