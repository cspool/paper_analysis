## Toward Inference-optimal Mixture-of-Expert Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：本论文提出三部分算法贡献：
    1. **MoE Scaling Law（公式4）**：将 dense Transformer 的 scaling law 扩展至 MoE 架构，引入 expert 数量 E 作为第三个 scaling 维度。核心公式为 $\log L(N, D, E) \triangleq \log(A/N^{\alpha} + B/\hat{E}^{\beta} + C/D^{\gamma} + F) + d \log N \log \hat{E}$，其中 $\hat{E}$ 通过 $E_{start}$ 和 $E_{max}$ 两个参数建模 expert 增长的饱和效应。
    2. **Inference Cost Estimation（Section 4）**：提出 cost per token 指标 $C_{Model,G} = GC_0 / T_{Model}(G)$，在 vLLM 上 profiling 8×40GB A100 GPU（NVLink），建立模型大小与推理成本的线性关系，并推导 MoE 模型总参数量为 $N_{MoE} = (1 + (E-1) \cdot 1/3)N$（因为每两层 Transformer 中仅一层为 MoE，MLP 占总参数 2/3）。
    3. **Over-training 策略（Section 5.2）**：在固定训练预算 B 下，不使用 loss-optimal 配置，而是训练一个更小的模型（70-85% reduction），利用节省的预算训练更多 token，达到接近 loss-optimal 的质量但显著降低推理成本。
  - 实验比较：
    - (a) Scaling law 拟合：100M-730M 参数的 dense 模型，每个配置 4/8/16/32 experts，2.5B-20B tokens 训练，验证 loss predicted vs actual（Figure 1）。
    - (b) Inference cost profiling：不同模型大小和 expert 数量的推理成本曲线（Figure 2）。
    - (c) 三难权衡（Training-Inferece-Quality Trilemma）：训练预算、推理成本、模型质量三者 trade-off（Figure 3），loss-optimal 下 4/8 expert MoE 在相同推理成本下质量最优，但训练成本是 16/32 expert 的 2.5-4.3 倍。
    - (d) Over-training vs loss-optimal：固定训练预算下，over-trained 8/16-expert MoE 在相同质量下推理成本仅为 loss-optimal 4-expert 的 47%-52%；相同推理成本下可节省 68.4% 训练预算（Figure 4, 5, 6）。
    - (e) 对比 base model（dense/4-expert）：8-expert over-trained MoE 推理成本为 dense 的 31.6%-38.1%，仅需 dense 23.3%-28.2% 的 activated parameters（Figure 5, 6）。

- 硬件平台是什么，配置是什么。
  - 训练：A100 GPU（最多 32 卡），使用 data parallelism + tensor parallelism + model parallelism（Megatron-DeepSpeed）。
  - 推理 profiling：8×40GB A100 GPU，NVLink 互联，使用 vLLM serving 系统。
  - 推理 cost 计算：以单 GPU 每秒运行成本 $C_0$ 为常数单位。

- 模型是什么。数据集和bench分别是什么。
  - **模型架构**：Llama-style（gated-MLP，MLP intermediate hidden 为 2.67× hidden dimension），每两层 Transformer 中替换一层为 MoE（Top-2 gating）。具体配置（Table 1）：
    | Name | d_model | n_layers | n_heads | Actual Params (w/o embedding) |
    |------|---------|----------|---------|-------------------------------|
    | 100M | 768 | 12 | 8 | 81,395,712 |
    | 200M | 896 | 14 | 8 | 184,064,768 |
    | 320M | 1024 | 16 | 12 | 289,406,976 |
    | 730M | 1536 | 16 | 16 | 679,477,248 |
    每个 dense 模型分别训练 4/8/16/32 experts 版本，以及一个 dense baseline。
  - **数据集**：SlimPajama（open-source LLaMA pretraining data blend：82% 互联网内容、4.5% code、4.5% Wikipedia、4.5% books、2.5% Arxiv、2% StackExchange）。训练使用最多 20B tokens，validation 使用 0.58B tokens。
  - **Benchmark 指标**：Validation loss（perplexity）、inference cost per token（dollars/token）、training FLOPs。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  **论文未明确说明代码/权重开源链接**。训练框架基于 Megatron-DeepSpeed（fork），推理 profiling 使用 vLLM（Kwon et al., 2023）。

  **Scaling Law 算法 pipeline（公式4 拆解）**：

  ```
  # === Step 1: 定义 effective expert count E_hat ===
  # 建模 expert 增长的饱和效应（E_start=专家线性增长起点, E_max=饱和上限）
  inv_E_start_max = 1/E_start - 1/E_max
  inv_E_hat = 1/(E - 1 + (inv_E_start_max)^(-1)) + 1/E_max
  # 当 E << E_start: E_hat ≈ E (linear growth)
  # 当 E >> E_max: E_hat ≈ E_max (saturated)

  # === Step 2: Scaling Law 预测 loss ===
  # log L = log(A/N^α + B/E_hat^β + C/D^γ + F) + d * log(N) * log(E_hat)
  N = model_dense_params    # e.g., 100M → 730M
  D = training_tokens       # e.g., 2.5B → 20B
  E = num_experts           # 4, 8, 16, 32

  term_1 = A / (N^alpha)    # 模型容量项
  term_2 = B / (E_hat^beta) # Expert 数量项 (饱和)
  term_3 = C / (D^gamma)    # 数据量项
  base_loss = term_1 + term_2 + term_3 + F
  interaction = d * log(N) * log(E_hat)  # N-E 交互项

  log_L = log(base_loss) + interaction
  # 最终 loss = exp(log_L)

  # === Step 3: 拟合参数 ===
  # 优化目标：min Σ Huberδ(log L_pred - log L_actual)
  # 使用 L-BFGS, δ=1e-3
  # 参数 α,β,γ ∈ [0,2]; A,B,C,D,F from grid search
  # 评估: RMSLE=3.908e-3, Huber=1.033e-3
  ```

  **Over-training 算法（Algorithm 1 & 2 张量计算路径）**：

  ```
  # === Algorithm 1: Optimal Inference Cost for Bounded Loss ===
  # 给定训练预算 B，base model (E experts)，target model (E' > E experts)
  # 求: 在保证 L_E' ≤ L_E_opt 的前提下，最小推理成本 I_E'_min

  (N_E, D_E) = loss_optimal_config(B, E)    # 求解损失最优配置
  L_E_opt = scaling_law(N_E, D_E, E)        # 基准 loss
  I_E = min_g Get_cost(N_E, E, g)          # 基准推理成本

  # Dichotomy search: 找满足 L_E'(N, B) = L_E_opt 的最小 N
  N_E' = dichotomy_search(E', L_E_opt, B)
  # 因为 L 在 loss-optimal 前单调递减，取刚好等于 L_E_opt 的 N 即可得到最小推理成本
  I_E'_min = min_g Get_cost(N_E', E', g)
  # N_E' 通常仅为 N_E 的 15%-30%

  # === Algorithm 2: Optimal Loss for Bounded Inference Cost ===
  # 给定训练预算 B，base model (E experts)，target model (E' > E experts)
  # 求: 在保证 I_E' ≤ I_E 的前提下，最低 validation loss L_E'_min

  (N_E, D_E) = loss_optimal_config(B, E)
  I_E = min_g Get_cost(N_E, E, g)          # 推理成本上限

  # Dichotomy search: 找满足 I_E'(N) = I_E 的 N
  N_E' = dichotomy_search_by_cost(E', I_E)
  D_E' = B / (6 * N_E')                    # 剩余预算全给数据
  L_E'_min = scaling_law(N_E', D_E', E')
  # N_E' 通常为 N_E 的 30%-85%
  ```

  **推理成本建模关键张量关系**：
  ```
  # 单 token KV-cache = 2hl (h=hidden_dim, l=n_layers)
  # 总 KV-cache 内存 = G*M_0 - N_m  (G GPU, M_0 per-GPU memory)
  # 最大并发请求数: b = (G*M_0 - N_m) / ((2p+n)hl)
  #   p = 平均 prompt length, n = 平均 output length
  # 
  # throughput: T_m = (G*M_0 - N_m) / (khl(L^P + L^D))
  #   k = 2p + n (常数, 由 traffic pattern 决定)
  #   L^P = prompt latency, L^D = decode latency
  #
  # cost per token: C = G*C_0 / T_m(G)
  # 最优 GPU 数: argmin_G C = G*C_0 / T_m(G)
  ```

  **关键发现汇总（定量）**：
  | 对比场景 | Base Model | Over-trained | 结果 |
  |---------|-----------|-------------|------|
  | Loss-optimal dense baseline | Dense Transformer | Over-trained 8-expert MoE | 推理成本 31.6%-38.1% of dense |
  | Loss-optimal 4-expert baseline | 4-expert MoE | Over-trained 8-expert MoE | 推理成本 47%-52% of 4-expert |
  | Loss-optimal 4-expert baseline | 4-expert MoE | Over-trained 16-expert MoE | 推理成本 48%-53% of 4-expert |
  | 相同推理成本 | Loss-optimal 4-expert | Over-trained 16-expert MoE | 训练 FLOPs 节省 68.4% |
  | Loss-optimal 4-expert baseline | 4-expert MoE | Loss-optimal 16-expert MoE | 训练 FLOPs 仅需 23.7%-42.8% |
