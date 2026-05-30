## Elastic Attention: Test-time Adaptive Sparsity Ratios for Efficient Transformers

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 Elastic Attention，一种 test-time 自适应稀疏注意力机制，通过轻量级 Attention Router 根据输入动态调整模型的整体稀疏度。核心设计：(1) **Attention Router**：每层引入一个由 Task MLP + Router MLP 组成的小型路由模块（仅 0.27M 参数/层），以 Key hidden states 为输入，通过 pooling 沿序列维压缩得到 task representation，经两阶段 MLP 输出 head-wise 的二值路由决策 r_{hard}^{(ℓ,h)} ∈ {0,1}，决定每个 KV head 使用 FA（r=0）还是 SA（r=1）计算模式；(2) **Gumbel-Softmax + STE 优化**：训练时使用 Gumbel-Sigmoid 连续松弛 + Straight-Through Estimator 解决 argmax 不可微问题，配合温度退火（τ=τ_init·exp(-r·p) 衰减至 τ_min），实现训练-推理一致性；(3) **Lagrangian 约束训练目标**：min-max 优化 max_{λ1,λ2} min L_language + λ1·L_diff + λ2·L_diff²，其中 L_diff = Ω_MSR - t，t 为任务相关的 target sparsity（sparsity-sensitive tasks t=0.7，sparsity-robust tasks t=1.0），λ 为可训练 Lagrange 乘子；(4) **backbone 冻结**：仅训练 Attention Router 参数（~0.27M/层），所有预训练模型权重冻结。采用 decoupled LR 策略（router LR=5e-4，regularization LR=1e-3）。实验比较 LongBench-E（14 tasks, 6 categories）、RULER（8K-256K 长度外推）和 LongBench-V2（long-form reasoning）三个长上下文 benchmark 上与 DuoAttention、PruLong、InfLLM-V2、MoBA、NSA、XAttention 的性能和 Ω_MSR sparsity。

- 硬件平台是什么，配置是什么。
  训练：8× NVIDIA A800 GPU，每轮训练 12 小时完成，BF16 精度，FSDP + Hybrid Sharding。推理评测：论文未明确说明推理用 GPU 型号（基于 LOOM-Eval 框架，使用单 GPU）。

- 模型是什么。数据集和bench分别是什么。
  模型：Qwen3-4B、Qwen3-8B（Yang et al., 2025）和 Llama-3.1-8B-Instruct（Grattafiori et al., 2024）。训练数据集：ChatQA2-Long-SFT-data、MuSiQue、CoLT-132K、GovReport、XSum 五源混合，覆盖 sparsity-sensitive（Single-Doc QA, Multihop QA）和 sparsity-robust（code completion, summarization, ICL）任务，序列长度 8K-64K，总约 0.74B tokens。Benchmark：LongBench-E（14 sub-tasks across 6 categories）、RULER（8K/16K/32K/64K/128K/256K）、LongBench-V2（Easy + Hard settings）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/LCM-Lab/Elastic-Attention。模型：https://modelscope.cn/collections/LCM_group/Elastic-Attention

  **算法 Pipeline 详细流程**：

  **Phase 1: Attention Router 前向（per layer ℓ）**
  ```
  Input: Key hidden states x_K ∈ R^{s×H×d'}（s=seq_len, H=KV heads, d'=head_dim）

  # Step 1: Pooling along sequence dimension → task representation
  # 默认 boundary-pooling: 仅聚合前 100 + 后 100 tokens
  x_K' = Pooling(x_K)  # [H, d']

  # Step 2: Task MLP → 提取 task-specific 特征
  z_task = MLP_task(x_K')  # [H, d_task]

  # Step 3: Router MLP → 输出 head-wise routing logits
  z = MLP_router(z_task)  # [H, 2]
  # z[h, 0] = FA score, z[h, 1] = SA score

  # Step 4: Gumbel-Sigmoid 连续松弛（训练时）
  g = -log(-log(u + ε) + ε)  # u ~ Uniform(0,1), Gumbel noise
  r_soft = σ((z + g) / τ)   # [H, 2], soft routing probabilities
  # 温度 τ 按 τ(t) = max(τ_min, τ_init · exp(-r·p)) 退火

  # Step 5: Hard routing via argmax
  r_hard[h] = argmax_c(r_soft[h, c])  # c ∈ {0,1}, 0=FA, 1=SA

  # Step 6: STE gradient
  r_hard[h] = r_hard[h] + (r_soft[h] - gradient_detach(r_soft[h]))
  ```

  **Phase 2: Hybrid Attention 计算（per layer ℓ）**
  ```
  # 按 head 类型分组
  I_full = {h | r_hard[h] = 0}   # FA heads
  I_sparse = {h | r_hard[h] = 1}  # SA heads

  # FA heads: 标准 causal attention
  O_r[h] = softmax(Q[h] @ K_full^T / √d') @ V_full  # h ∈ I_full

  # SA heads: Streaming Sparse Attention (SSA) 或 XAttention (XA)
  # SSA: 仅保留 sink tokens (128) + local window (2048)
  K_tilde = K[sink ∪ recent], V_tilde = V[sink ∪ recent]
  O_s[h] = softmax(Q[h] @ K_tilde^T / √d') @ V_tilde  # h ∈ I_sparse

  # Concat all heads
  O = concat([O_r[h for h in I_full], O_s[h for h in I_sparse]], dim=head_dim)
  ```

  **Phase 3: 训练目标（min-max optimization）**
  ```
  # 计算 model sparsity ratio
  Ω_MSR = (1/(H·L)) · Σ_l Σ_h I[r_hard[l,h] = SA]

  # 损失函数
  L_language = CE_Loss(y | f_θ(x))
  L_diff = Ω_MSR - t  # t: target sparsity
  Total = L_language + λ1·L_diff + λ2·L_diff²

  # Lagrange multipliers 更新（gradient ascent）
  λ1 += lr_λ · ∂Total/∂λ1
  λ2 += lr_λ · ∂Total/∂λ2
  ```

  **关键超参数**：
  - Sequence length: 65536 (training)
  - Global batch size: 48
  - Training steps: 300
  - Router LR: 5e-4, Reg LR: 1e-3
  - AdamW (β1=0.9, β2=0.95), weight decay=0.1
  - Cosine LR schedule, 20% warmup
  - Gumbel temperature: τ_init→τ_min via exp decay (r=0.6)
  - Sparsity targets: t_robust=1.0, t_sensitive=0.7
  - SA config: SSA (sink=128, local=2048) or XA (τ=0.9, default params)
  - Block-Sparse-Attention: block_size=64, chunk_size=16384, sink=128

  **算法核心创新**：将下游任务分为 sparsity-robust（summarization 等粗粒度任务）和 sparsity-sensitive（QA 等细粒度检索任务）两类，通过 Attention Router 在 test-time 自动判断任务类型并分配相应 sparsity level，无需 per-task 手动调参。实验表明 Elastic Attention 在长上下文 benchmark 上实现了与 full attention 可比甚至超越的性能（如 Llama-3.1-8B 在 LongBench-E avg 53.35 vs backbone 53.28），同时 Ω_MSR 达 0.69（FA-SSA）或 0.77（FA-XA），实现 efficient inference。
