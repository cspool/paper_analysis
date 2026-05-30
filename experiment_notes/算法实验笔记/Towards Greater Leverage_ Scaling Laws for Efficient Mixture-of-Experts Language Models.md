## Towards Greater Leverage: Scaling Laws for Efficient Mixture-of-Experts Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：提出 **Efficiency Leverage (EL)** 指标，定义 MoE 架构相对于等性能 dense 模型的计算效率比（EL = C_dense / C_moe）；通过分阶段实证研究（300+ 模型，最大 28B 参数）建立 MoE 架构的统一 scaling law。核心算法贡献：(1) 推导 MoE 最优超参 scaling law（η^opt = 1.1576·C^{-0.1529}, B^opt = 0.0694·C^{0.3644}）；(2) 推导 MoE vs dense 的最优模型-数据分配策略（MoE 偏向更小 M、更多 D）；(3) 对 activation ratio (A)、expert granularity (G)、shared expert ratio (S) 进行系统性消融，拟合联合 scaling law：EL(A,G,C) = Â^{α + γ(log G)² + β log G}，其中 α = a + d·log C。系数拟合值：a=1.23, d=-7.61e-2, γ=1.67e-2, β=-1.17e-1, A_start=1.63e-2, A_max=5.28e+16。
  - 实验比较：(a) **Activation ratio 消融**：固定 E^a=2, E^s=1，E 从 2 到 256 变化（A=0.8%~100%），在 C=1e18~6e20 FLOPs 范围观测 IsoFLOPs 曲线和 EL scaling；(b) **Expert granularity 消融**：G=2~16（E 从 64 到 512，proportionally 减小 d_expert），观测 U 形 loss-G 关系，最优 G≈12；(c) **Shared expert ratio 消融**：S=0%~83.3%，固定 E=256, E^a+E^s=12，观测最优 S 随 C 从 16.7% 降至 8.3%；(d) **Dense layer proportion 消融**：60层模型中前 0~3 层用 dense FFN 替代 MoE；(e) **Attention-FFN compute allocation 消融**：attn FLOPs ratio 20%~50%；(f) **Ling-mini-beta 验证**：17.5B 总参/0.85B 激活 vs Dense-6.1B，1T tokens 训练，验证 7× EL 预测。

- 硬件平台是什么，配置是什么。
  - GPU 训练。论文明确说明使用 "Ling series models" 训练框架（基于 Ling-Team et al. 2025），参考其 300B MoE 模型使用非高端 GPU 训练。具体 GPU 型号论文未明确说明，但基于 Ant Group Ling Team 的公开技术报告（Every FLOP Counts, 2025），可能使用 NVIDIA A100/H800 等 GPU。训练精度论文未明确说明。

- 模型是什么。数据集和bench分别是什么。
  - **模型架构**：decoder-only Transformer with GQA (Grouped-Query Attention) + RoPE + BPE tokenizer (vocab=126,464)。MoE layer 使用 top-k routing（softmax gate + load balancing loss coeff=0.01 + router z-loss coeff=0.001）。消融实验模型规模从 8 layers/d_model=384 到 22 layers/d_model=1280，最大训练 FLOPs 6e20。Ling-mini-beta 验证模型：20 layers, d_model=2048, d_ffn=5120, d_expert=384, 16 heads/4 kv_heads, E=384, E^a=12, E^s=1, N=17.5B, N^a=0.85B。对比 dense 模型 Dense-6.1B：28 layers, d_model=4096, d_ffn=14336, 32 heads/8 kv_heads, N=6.11B。
  - **训练数据**：Ling Team 大规模多语言语料库（中英文为主），组成：Web 46%、Books 5%、Wiki 4%、Academic 6%、Code 25%、News 0.1%、Social 1.9%、Domain 1%、SFT 4%、Math 6%、Exam 1%。消融实验使用 2T token 子集，Ling-mini-beta 验证使用 1T token 子集。
  - **Benchmark**：(a) General Knowledge/Reasoning: ARC-challenge/easy, AGIEval, OpenBookQA, BBH, ProntoQA, Multi-LogiEval, HellaSwag, PIQA; (b) Language Understanding: RACE-middle/high; (c) Professional Knowledge: MMLU, MMLU-Pro, CMMLU, C-Eval, CommonsenseQA, GPQA; (d) Code: HumanEval, HumanEval-cn/Plus/FIM, MBPP, MBPP-Plus, LiveCodeBench, CruxEval; (e) Math: GSM8K, MATH, CMATH, MGSM-zh, CN-Middle School 24, Minerva-Math, MathBench, Gaokao2023-Math-En, GAOKAO-Math24。
  - **关键指标**：Training loss（cross-entropy），Efficiency Leverage (EL)，benchmark accuracy/F1。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  **代码未开源**（Ant Group Ling Team 内部代码）。论文基于 Ling 系列 LLM 内部训练框架（Ling-Team et al., 2025 "Every FLOP Counts"）。

  **Efficiency Leverage (EL) 计算伪代码**：
  ```
  # === EL 定义 ===
  # 给定 MoE 架构 X_MoE 和 dense 架构 X_Dense
  # EL(X_MoE | X_Dense; C_target) = C_dense / C_moe
  # s.t. |L(C_moe; X_MoE) - L(C_dense; X_Dense)| ≤ ε

  # === Step 1: 拟合各架构的 loss scaling 曲线 ===
  def fit_loss_scaling(configs, flops_budgets):
      """
      对每个架构配置，在多个 FLOPs budget 下训练模型，拟合 L(C) = α · C^{-β}
      使用 compute-optimal allocation: M_opt, D_opt determined by scaling law
      """
      loss_curves = {}
      for arch in configs:  # 不同 A, G, S 的 MoE 配置 + dense baseline
          losses = []
          for C in flops_budgets:  # e.g., 3e18, 6e18, 2e19, 6e19, 2e20, 6e20
              M_opt = α_M · C^{β_M}   # optimal model scale (FLOPs/token)
              D_opt = α_D · C^{β_D}   # optimal data size (tokens)
              η = 1.1576 · C^{-0.1529}  # optimal learning rate
              B = 0.0694 · C^{0.3644}   # optimal batch size
              model = build_model(arch, M_opt)
              loss = train(model, D_opt, η, B)
              losses.append((C, loss))
          loss_curves[arch] = fit_power_law(losses)  # α · C^{-β}
      return loss_curves

  # === Step 2: 计算 EL ===
  def compute_el(dense_curve, moe_curve, C_moe):
      """
      EL = C_dense / C_moe
      其中 C_dense 满足 L_dense(C_dense) = L_moe(C_moe)
      """
      L_moe = moe_curve(C_moe)  # α_moe · C_moe^{-β_moe}
      # 解 L_dense(C_dense) = L_moe → α_dense · C_dense^{-β_dense} = L_moe
      C_dense = (L_moe / α_dense)^{-1/β_dense}
      return C_dense / C_moe

  # === Step 3: 拟合联合 EL scaling law ===
  # EL(A, G, C) = Â^{α + γ(log G)² + β log G}
  # 其中 Â 是 A 的饱和变换:
  #   1/Â = 1/(A + (1/A_start - 1/A_max)^{-1}) + 1/A_max
  # α = a + d · log C
  # 使用 Huber loss + BFGS 优化拟合参数 (a, d, γ, β, A_start, A_max)
  ```

  **MoE 前向传播张量计算（per token）**：
  ```
  # Input: h^t ∈ R^{d_model}, 第 t 个 token 的 hidden state
  # MoE layer with E experts (index i=1..E), E^a activated, E^s shared

  # Step 1: Router gating
  g^t = Softmax(W^g @ h^t)  # W^g ∈ R^{E × d_model}, g^t ∈ R^E
  selected = TopK(g^t, E^a)  # 选择 top-E^a 个 expert indices

  # Step 2: Expert computation (每个激活的 expert)
  # expert_i FFN: W_up_i ∈ R^{d_expert × d_model}, W_gate_i ∈ R^{d_expert × d_model}, W_down_i ∈ R^{d_model × d_expert}
  for i in selected:
      e_i = W_down_i @ (SwiGLU(W_gate_i @ h^t) ⊙ (W_up_i @ h^t))
  # SwiGLU: activation(W_gate @ h) ⊙ (W_up @ h), 其中 activation = SiLU

  # Step 3: Shared expert (if exists)
  if E^s > 0:
      e_shared = W_down^s @ (SwiGLU(W_gate^s @ h^t) ⊙ (W_up^s @ h^t))

  # Step 4: Weighted combination
  o^t = Σ_{i∈selected} g^t_i · e_i + e_shared  # g^t_i 是第 i 个 expert 的 gating score

  # FLOPs per token (forward):
  # M = L_attn · C_attn + L_dense · C_dense_ffn + L_moe · C_moe_ffn
  # C_moe_ffn ≈ 6 · d_model · (E^a · d_expert + d_shared)
  ```

  **Scaling Law 预测流程**：
  ```
  # 给定 A, G, C → 预测 EL
  def predict_el(activation_ratio, granularity, compute_budget):
      # 1. 计算饱和变换 Â
      inv_A_hat = 1/(activation_ratio + 1/(1/0.0163 - 1/5.28e16)) + 1/5.28e16
      A_hat = 1 / inv_A_hat
      # 2. 计算 compute-dependent exponent
      alpha = 1.23 - 0.0761 * log(compute_budget)
      # 3. 计算 granularity modulation
      log_G = log(granularity)
      gran_mod = 0.0167 * log_G^2 - 0.117 * log_G
      # 4. 联合 scaling law
      log_EL = (alpha + gran_mod) * log(A_hat)
      return exp(log_EL)
  ```
