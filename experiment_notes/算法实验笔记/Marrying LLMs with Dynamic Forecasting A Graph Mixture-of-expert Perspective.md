## Marrying LLMs with Dynamic Forecasting A Graph Mixture-of-expert Perspective

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 LEGO（LLM Judge with Graph Mixture-of-expert）框架，核心是将 LLM 作为 context-aware routing function 融入 Graph MoE 框架，实现动态系统在环境变化下的鲁棒预测。具体实现包括三个组件：
    - **Hierarchical Prompt Engineering**：从 system level（系统参数/背景描述）、object level（各物体初始位置和速度向量，数值作为 token）、edge level（边连接关系文本化描述）三个层次提取环境上下文，编码为 LLM prompt。
    - **Graph Mixture-of-Expert（MoE）**：K 个同构 EGNN（E(n) Equivariant GNN）作为 graph experts，各自拥有独立参数 θ¹,...,θᴷ。每个 expert 通过迭代消息传递（Eq. 1-3：ϕ 学习交互 → AGG 聚合 → COM 组合 node/coordinate representation）生成隐藏表示 Hᴷ = f_{θᴷ}(G, X⁽⁰⁾)。最终预测通过 MoE routing function 加权组合：x̂ᵢ⁽ᵗ⁾ = Decoder(Σ ω(k)·hᵢᵏ)。
    - **LLM Judge for Context-Aware Routing**：将 LLM 作为 routing function（而非直接生成预测）。LLM 接收 hierarchical prompt 后评估 K 个 experts 的候选预测，选择最适应当前环境的一个。使用 one-hot routing + label smoothing（Eq. 7：选中 expert 权重 α，其余 (1-α)/(K-1)），配合 diversity-enhanced contrastive loss（Eq. 9-10）确保不同 expert 学习多样化动力学模式。
    - 优化采用交替更新（Algorithm 1）：每隔若干 epoch 更新 LLM 生成的 routing weights，内部循环通过梯度下降优化 graph expert 参数。
  - 实验比较：
    - Baseline：Linear、Dynamic（物理匀速模型）、GNN（Kipf & Welling 2017）、Radial Field（Köhler et al. 2019）、EGNN（Satorras et al. 2022）、EGNO（Xu et al. 2024）。
    - LEGO 变体组合：EGNN+LEGO、EGNO+LEGO、Radial Field+LEGO。
    - 消融实验（Ablation）：V1（仅 system level prompt）、V2（system + edge level prompt，无 object level）、V3（完整三层 prompt）。
    - LLM Judge vs LLM Forecasting 对比。
    - 敏感度分析：不同 LLM（规模对比）、不同 expert 数量 K ∈ {3,5,10,15,20}、不同 LLM temperature ∈ {0,0.25,0.5,0.75,1}。
    - Case Study：LLM Judge 的逐步推理过程分析。
    - 更多结果：ETH-UCY 上 vs Eq-Motion，MD17 上 vs Se3-Transformer/TFN，不同原子数分子间的迁移。

- 硬件平台是什么，配置是什么。
  - 训练硬件：论文未明确说明 GPU/CPU 具体型号和数量。
  - LLM：Llama 3.1 8B 版本作为 LLM Judge（推理用），论文未说明 LLM 推理所用的具体 GPU 配置。
  - 优化器：Adam，学习率 0.0005，batch size 100。

- 模型是什么。数据集和bench分别是什么。
  - 模型：
    - 基础 GNN expert：EGNN（Satorras et al. 2022）—— E(n) 等变图神经网络，消息传递包含 node representation h 和 coordinate x 的联合更新；EGNO（Xu et al. 2024）—— 等变图神经算子，结合 Fourier 神经算子；Radial Field（Köhler et al. 2019）—— 仅操作位置坐标的 E(n) 等变模型。
    - LEGO 框架可构建于任意基础 GNN 模型之上。
    - LLM Judge：Llama 3.1 8B（Dubey et al. 2024），用于 context-aware routing。
    - 默认 K=5 个 graph experts。
  - 数据集：
    - **Spring**（Satorras et al. 2022）：N-body 弹簧系统，粒子通过弹簧力相互作用（F=k·x）。5 个粒子，3D 空间。训练集 strength=1.0, start_state=30, end_state=40。Hard/Soft/Temporal Shift 三种环境变化。时间窗口 ΔT=10。3000/2000/2000 train/val/test。
    - **Charged**（Satorras et al. 2022）：N-body 电荷系统，粒子通过库仑力相互作用（F=k·q₁·q₂/r²）。5 个粒子，3D 空间。类似设置，含无环境变化、多种 strength 和 temporal shift 场景。
    - **MD17**（Chmiela et al. 2017）：分子动力学数据集。训练用 salicylic acid，测试用 naphthalene（不同分子 = OOD 环境变化）。去除氢原子。时间窗口 ΔT=50。500/2000/2000 train/val/test。
    - **Motion**（CMU 2003）：人体运动捕捉。训练用 Subject #35（Walk），测试用 Subject #9（Run）。关节点为边，关节点交点为节点。200 train / 240 val / 240 test 轨迹。时间窗口 ΔT=30。
    - **ETH-UCY**（Li et al. 2016）：行人轨迹预测（Appendix D.4），评估指标 ADE/FDE。
  - 评估指标：MSE（Mean Square Error）×10⁻²，ADE（Average Displacement Error），FDE（Final Displacement Error）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/jdp22/LEGO.git
  - 算法pipeline 伪代码：

```
输入: 初始状态 X⁽⁰⁾ ∈ R^{N×d}, 交互图 G=(V,E), 环境参数
      K 个 graph experts {θ¹,...,θᴷ}, 预训练 LLM, 时间 t
输出: 预测状态 X̂⁽ᵗ⁾

// 1. Hierarchical Prompt Extraction
prompt_system = f"System has {N} balls with spring constant k={coeff}"
prompt_object = f"Ball 0: pos=({x₀},{y₀},{z₀}), vel=({vx₀},{vy₀},{vz₀})..."
prompt_edge   = f"Ball 2 connects ball 0, ball 1, ball 3..."
hierarchical_prompt = [prompt_system, prompt_object, prompt_edge]

// 2. Graph Expert Forward Pass (per expert k)
for each expert k in {1..K}:
    h_i⁰ = MLP(x_i⁽⁰⁾)  // 初始 node embedding
    for l in {1..L}:     // L 层 EGNN
        for each edge (i,j):
            e_ij^l = ϕ(h_j^{l-1}, x_j^{l-1}, h_i^{l-1}, x_i^{l-1})  // Eq.1
        for each node i:
            h_i^l = COM^H(h_i^{l-1}, AGG({e_ij^l | j∈N(i)}))        // Eq.2
            x_i^l = COM^X(x_i^{l-1}, AGG({e_ij^l | j∈N(i)}))        // Eq.3
    H^k = [h_1^L, ..., h_N^L] = f_{θᴷ}(G, X⁽⁰⁾)

// 3. LLM Judge: Context-aware Routing
for each expert k:
    candidate_prediction X̂⁽ᵗ⁾,ᴷ = Decoder(h_i^k)  // Eq.6, one-hot routing
// 将 hierarchical_prompt + candidate_predictions 送入 LLM
LLM_input = hierarchical_prompt + descriptions of K candidate predictions
chosen_expert = LLM(LLM_input)  // LLM 选择最合适的 expert

// 4. Label Smoothing Routing Weights (Eq.7)
for k in {1..K}:
    ê^k(k) = α                if k == chosen_expert
    ê^k(j) = (1-α)/(K-1)      if j != chosen_expert

// 5. Final Prediction with Smoothed Weights (Eq.8)
for each node i:
    h_i_combined = Σ_{k=1}^K ê^k(k) · h_i^k
    x̂_i⁽ᵗ⁾ = Decoder(h_i_combined)  // Decoder: 另一层 EGNN

// 6. Loss Computation (Eq.11)
loss_mse = ||X⁽ᵗ⁾ - X̂⁽ᵗ⁾||²
// Diversity Loss (Eq.9-10): 同 expert 内的表征相近，不同 expert 间的表征远离
for each node i and expert k:
    S_i^k = {h_i^k from training data for expert k}  // activated representations
    ℓ_i^k = -1/C * Σ log(exp(h_i^k·h̃_i^k/τ) / Σ_{h∈S_i} exp(h_i^k·h/τ))
loss_div = (1/(K*N)) * Σ_k Σ_i ℓ_i^k
loss = loss_mse + loss_div

// 7. Alternative Optimization (Algorithm 1)
while not converged:
    更新 routing weights（通过 LLM 推理，每隔若干 epoch）
    for epochs in {1..E}:
        固定 routing weights
        通过梯度下降优化 {θ¹,...,θᴷ}（Adam, lr=0.0005）
```

  - 关键超参数：
    - K=5 个 graph experts（默认，来自参数敏感度实验）
    - α（label smoothing 系数）∈ (0,1)
    - τ（contrastive loss 温度系数）
    - LLM temperature = 0（推理阶段低 temperature 更优）
    - 交替更新间隔：每隔若干 epoch 更新一次 LLM routing weights
    - 优化器：Adam, lr=0.0005, batch_size=100
