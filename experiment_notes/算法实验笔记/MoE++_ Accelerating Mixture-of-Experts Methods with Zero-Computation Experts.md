## MoE++: Accelerating Mixture-of-Experts Methods with Zero-Computation Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：在标准 MoE 层中引入三种零计算专家（zero expert、copy expert、constant expert）与 FFN 专家混合，构建异构 MoE 框架 MoE++。Zero expert 输出零向量（丢弃），copy expert 输出输入本身（跳过），constant expert 用可训练向量替换输入。同时引入基于 gating residuals 的 pathway-aware router（将前一层路由分数通过可训练矩阵 W_g 融入当前层）和异构负载均衡损失（超参数 τ 控制零计算专家与 FFN 专家的 token 分配比例）及异构专家容量分配。
  实验比较：MoE++ vs. vanilla MoE（相同参数量级别的标准 Top-2 MoE），比较下游 benchmark 准确率和 expert forward throughput。

- 硬件平台是什么，配置是什么。
  训练：4 节点 32× NVIDIA A100 GPU 集群。7B 模型使用 8-way pipeline parallel（tensor parallel=1）。小模型（0.6B/1B/2B）不使用模型并行。

- 模型是什么。数据集和bench分别是什么。
  模型：MoE++ 0.6B/(8+4)E、1B/(16+4)E、2B/(32+8)E、7B/(16+4)E，其中每个模型包含 1 zero expert + 1 copy expert + n_const constant experts，Top-K=2。baseline 为相同参数量级别的 vanilla MoE（Top-2，纯 FFN 专家）。所有 FFN 层替换为 MoE/MoE++ 层。
  数据集：RedPajama、Dolma、Pile 按不同采样比例混合。Tokenizer：LLaMA2（65,536 vocab）。训练 budget：100B tokens（Tab.3 所有模型）或 1T tokens（7B MoE++ 大模型）。
  Benchmarks：SciQ、PIQA、WinoGrande、ARC-E (0-shot)、HellaSwag (10-shot)、LogiQA (0-shot)、BoolQ (32-shot)、LAMBADA (0-shot)、NQ (32-shot exact match)、ARC-C (25-shot)、MMLU (5-shot)。使用 lm-evaluation-harness 评估。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源：代码仓库 https://github.com/SkyworkAI/MoE-plus-plus（ICLR 2025，Apache 2.0）。当前仅发布推理代码和评估代码，训练代码待 Skywork-MoE 开源后一并发布。HuggingFace 上已发布 MoE++7B-Base 模型权重。
  算法 pipeline 伪代码（单个 MoE++ 层的 forward）：
  ```
  # 输入: x (shape: [B, S, D]), 前一层路由分数 G_prev (shape: [B, S, N])
  # N = N_FFN + N_ZC 总专家数, N_ZC = n_zero + n_copy + n_const

  # 1. Router with gating residuals
  logits = W @ x  # [B, S, N], W in R^{N x D}
  if layer_idx > 1:
      logits += W_g @ G_prev  # W_g in R^{N x N}, gating residuals
  G_curr = softmax(logits, dim=-1)  # 当前层路由分数

  # 2. Top-2 selection (考虑异构专家容量 C_i)
  # C_i = γ * τT / (τ*N_FFN + N_ZC) for FFN experts
  # C_i = γ * T / (τ*N_FFN + N_ZC) for zero-computation experts
  selected_indices, selected_probs = top_k_with_capacity(G_curr, k=2, capacities=C)

  # 3. Expert computation (异构专家)
  outputs = []
  for idx, prob in zip(selected_indices, selected_probs):
      if expert_type[idx] == FFN:
          out = FFN[idx](x)  # 标准 Feed-Forward
      elif expert_type[idx] == ZERO:
          out = 0  # 零输出
      elif expert_type[idx] == COPY:
          out = x  # 直通
      elif expert_type[idx] == CONST:
          alpha = softmax(W_c @ x)  # W_c in R^{2 x D}
          out = alpha[0] * x + alpha[1] * v  # v 是可训练向量
      outputs.append(prob * out)

  y = sum(outputs)  # 加权聚合
  # 异构负载均衡损失:
  L_b = sum_i η_i * f_i * P_i
  # η_i = 1 for FFN, τ for ZC expert
  # f_i = 选中频率, P_i = 平均 softmax 分数
  L_total = L_ce + 0.01 * L_b
  ```

  MoE++ 的计算复杂度仅为相同参数量 vanilla MoE 的 `τ*N_FFN / (τ*N_FFN + N_ZC)` 倍（Tab. 1）。典型 τ=0.75 时，MoE++ 0.6B/(8+4)E 的 expert forward throughput 从 535.3ms 降至 427.6ms（提升 25.2%），同时 average benchmark 从 44.3 提升至 45.6。τ=0.10 时 throughput 提升可达 164.5%。
