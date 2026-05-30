## Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient

- 属于算法pipeline的实现是什么？实验比较什么？
  - 论文提出 **Joint MoE Scaling Laws**，将训练损失 L 表示为 active parameters N_act、训练 token 数 D 和 expert 数 E 的函数：
    L(N_act, D, Ê) = aÊ^δ · N_act^(α+γ·ln(Ê)) + bÊ^ω · D^(β+ζ·ln(Ê)) + c
    其中 Ê 是 E 的单调变换（见 Eq.4），c 为 dataset irreducible entropy。核心算法贡献：
    1. **联合形式推导**：从固定 E 的 Chinchilla 形式出发，引入 E 与 N_act、D 的交互项（通过 power-law exponent 中 ln(Ê) 项），统一描述 dense (E=1) 和 MoE (E≥2) 的 scaling behavior。
    2. **Compute Optimality 分析**：对固定 compute budget F=6·N_act·D，求解 argmin L(N_act,D,E)，得出 MoE 的 compute-optimal 配置：expert 越多 → 应减少 active parameters、增加 training tokens（Finding 1）。
    3. **Memory Optimality 分析**：引入 total parameter 约束 N_total ≤ M 和 KV-cache 约束，证明 MoE 可在 memory-constrained 场景下超越 dense 模型（Finding 2-3）。
    4. **Inference Optimality**：将 inference FLOPs (2·N_act·D_inf) 纳入 joint budget，给出训练+推理联合最优配置。
    5. **Learning Rate Scaling Law**：LR(N_act\e, E) = exp(8.39 - 0.81·ln(N_act\e) - 0.25·ln(E))，发现更多 expert 需要更低 LR（Finding 4）。
  - 实验比较：
    - 280+ 模型 runs，E ∈ {1,2,4,8,16,32}，N_act 最高 2.7B，N_total 最高 5B
    - 核心对比：同一 FLOPs budget 下不同 E 的 loss 曲线（IsoFLOP profiles, Fig.2）
    - Memory-matched 验证：1.1B 总参数 dense vs E={2,4} MoE，相同 FLOPs + memory budget 下 MoE 获得更低 loss（Fig.1b）
    - Scaling law fit quality：RMSE_v=0.0039 (validation), RMSE_t=0.0062 (training)；与独立 Chinchilla fit (RMSE_v=0.0041) 接近，验证联合公式的有效性
    - LR scaling law 验证：在 E={1,8} 上拟合，E=4 插值验证，E=32 外推验证（Fig.7）

- 硬件平台是什么，配置是什么。
  - 训练硬件：Polish HPC infrastructure PLGrid (ACK Cyfronet AGH)，以及 Writer.com 提供的计算资源
  - 论文未明确说明具体 GPU 型号和集群配置；在 memory constraint 分析中引用 H100 (80GB)、RTX 4090 (24GB)、8×H100 node (640GB) 作为典型 memory budget 场景
  - 论文未明确说明使用的 GPU 数量、节点互联、CPU 等具体配置

- 模型是什么。数据集和bench分别是什么。
  - **模型**：Decoder-only Transformer，Switch MoE (Fedus et al. 2022)，每 token 激活 1 个 expert。SwiGLU activation，hidden size = 3×d_model。RoPE position embeddings。GPT-2 tokenizer (vocab=50,257)。配置规则：N_blocks = N_heads = d_model/64。Mixed precision 训练（attention、RoPE、router 保持高精度）。Router z-loss=0.001, load balancing loss=0.01。Weight initialization: truncated normal (scale=0.1)。详细模型配置见 Appendix E（N_total 从 79M 到 5.0B，d_model 从 512 到 2304，E 从 1 到 32）。
  - **数据集**：FineWeb-Edu (Penedo et al. 2024)，高过滤质量 web 数据。训练 token 数从 500M 到 80B（随模型配置不同）。
  - **Benchmark/指标**：最终训练 loss（cross-entropy），无下游 NLP benchmark 评估。评估协议：基于 loss 的 scaling law fit quality（RMSE, Huber loss δ=0.01）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源状态**：论文声明"计划开源模型 checkpoint 和代码"（Footnote 2），但截至论文发布未提供开源链接。拟合方法基于开源 LBFGS (PyTorch/SciPy)。
  - **Scaling Law 拟合 pseudocode**：
    ```
    # === Joint MoE Scaling Law Fitting ===
    # Input: 实验数据 {(N_act_i, D_i, E_i, L_i)} for i = 1..280+
    # Output: 拟合系数 {a, α, δ, γ, b, β, ω, ζ, E_start, E_max, c}
    
    1. 计算 Ê_i via Eq.4:  1/Ê = 1/(E-1+(1/E_start-1/E_max)^(-1)) + 1/E_max
    2. 对每对 (N_act_i, D_i)，计算：
         pred_i = a·Ê^δ · N_act^(α+γ·ln(Ê)) + b·Ê^ω · D^(β+ζ·ln(Ê)) + c
    3. 优化目标：Huber loss (δ=0.01) over log-space predictions
         L_huber = Σ_i Huber(log(L_i) - log(pred_i), δ=0.01)
    4. 优化器：LBFGS，lr=1e-4, weight_decay=1e-5
    5. 初始化网格搜索：
         α ∈ {0.05, 0.25, 0.5}, β ∈ {0.05, 0.25, 0.5}
         a,b ∈ {30, 100, 300}, c ∈ {0.5, 1, 2}
         δ,γ,ω,ζ ∈ {-0.5, 0, 0.5}
    6. 选择 training RMSE + validation RMSE 之和最小的系数
    ```
  - **Compute-optimal 配置求解（给定 E 和 budget F）**：
    ```
    1. 从 joint formula 退化为固定 E 的 Chinchilla 形式：
         m(E) = a·Ê^δ,           μ(E) = α + γ·ln(Ê)
         n(E) = b·Ê^ω,           ν(E) = β + ζ·ln(Ê)
       → L(N_act, D|E) = m·N_act^μ + n·D^ν + c
    
    2. 给定 FLOPs budget F = 6·N_act·D，求解 compute-optimal:
         G = (μ·m / (ν·n))^(1/(μ+ν))
         N_act_opt = G · (F/6)^(ν/(μ+ν))
         D_opt = G^(-1) · (F/6)^(μ/(μ+ν))
    
    3. Memory-optimal 配置（约束 N_total ≤ M）：
         在 {N_act, D, E} 空间搜索，满足 6·N_act·D=F 且 N_total(E) ≤ M
         使 L 最小化的配置
    ```
  - **张量计算上下文**：scaling law 本身不涉及张量计算，其推导基于 Switch MoE 的标准计算范式。每一 token 前向通过 Router 选择 top-1 expert → expert FFN 计算 (W_o · SwiGLU(W_g·x, W_p·x)) → 输出。FLOPs 计数遵循 F_train = 6·N_act·D, F_infer = 2·N_act·D_inf。
