## Skywork-MoE: A Deep Dive into Training Techniques for Mixture-of-Experts Language Models

- baseline方法是什么？
  Baseline 方法：(1) **Switch Transformer 标准 MoE 训练**：使用固定全局辅助损失系数 α（通常 1e-2 或 1e-3），原始门控层直接 softmax(Wx + b) 进行 top-k expert 选择，uniform pipeline parallelism + standard expert parallelism（EP 或 ETP）。门控输出可能退化为高熵分布（top-k 概率接近均匀），导致 expert 输出退化为简单平均而非加权平均，gating 失去区分能力。(2) **传统从 Dense 到 MoE 的 Upcycling**：直接复制 dense checkpoint 的 FFN 权重 n 次初始化 n 个 expert，所有 expert 完全相同（expert similarity = 1），依靠训练过程缓慢分化，初期 expert 多样性极差。(3) **传统 MoE 从头训练**：随机初始化所有 expert，不存在 expert 同质化问题，但需要较大训练预算才能达到与 upcycling 相当的性能。

  全栈执行例子（Baseline: Switch Transformer 标准 MoE，从 Skywork-13B dense upcycling，1536 A800 GPU）：
  - **算法Pipeline层**：Gate 直接 g = softmax(W_gate @ x + b_gate)，top-2 selection → expert FFN forward → weighted combine。辅助损失 L_total = L_ce + α · Σ L_aux^(l)，α 全局固定。门控概率可能退化为近似 1/16 均匀分布，Max1/Max2 ≈ 1, Max2/Max3 ≈ 1，expert 失去区分能力，输出 ≈ (E1(x) + E2(x))/2（简单平均）。
  - **系统框架层**：Megatron-LM 23.06 + Expert Parallelism (EP)，Size_EP = Size_DP * Size_TP。受 expert 数量限制（≤16），GPU 扩展性受约束。Pipeline parallelism 每 stage 均匀分配层数，最后 stage 因 loss calculation 成为 bottleneck。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：Uniform PP + 标准 EP/ETP，AllToAll 通信开销大（ETP 情况下随 TP 增大迅速增加），无通信-计算 overlap 优化。
  - **硬件架构层**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法 **Skywork-MoE** 通过三项核心技术创新解决 baseline 缺陷：

  **(1) Gating Logit Normalization —— 解决门控退化问题**
  Baseline 的 g = softmax(Wx + b) 在训练中容易退化为高熵分布。Skywork-MoE 在 softmax 前插入归一化：z̃ = λ · (z − μ)/σ，确保 logit 向量具有零均值和可控标准差 λ。λ 控制 softmax 输出的锐度——λ 越大分布越集中。实验证实 λ=1 时 Max1/Max2 和 Max2/Max3 比率远高于无归一化（后者退化到 1），token drop rate 大幅降低，训练 loss 改善。

  **(2) Adaptive Auxiliary Loss Coefficients —— 解决辅助损失与主任务冲突**
  Baseline 的全局固定 α 要么过度正则化（牺牲交叉熵优化）要么欠正则化（负载不均）。Skywork-MoE 为每层引入独立的 α^(l)，并根据实时 token drop rate d_i^(l) 自适应调整：α̂ = min(ξ·d, α_max)，α 通过移动平均平滑更新（β=0.99）。这使得负载均衡正则化仅在需要时增强，在负载均匀时自动减弱，优先确保交叉熵损失优化。

  **(3) Upcycling 预算决策框架 —— 提供何时 upcycle 何时从头训练的量化指导**
  通过控制实验（0.3B dense × 100B/300B MoE tokens）提出量化规则：C_MoE ≪ C_dense 时 upcycling 占优，C_MoE ≥ 2·C_dense 时从头训练占优。并发现 upcycling 过程中 expert similarity 从 1 逐渐下降（diversification 过程），可作训练监控指标。
  
  **(4) EDP + Unbalanced PP —— 提升训练效率**
  EDP（Size_EP = Size_TP）优化中等 expert 数量场景的 AllToAll 通信。Unbalanced PP（如 [5,5,5,5,4]）减少 pipeline bubble 10%。最终达到 38% MFU。

  全栈执行例子（Skywork-MoE，1536 A800 GPU，146B/16 experts）：
  - **算法Pipeline层**：Gate forward: z = W_gate @ x + b_gate → z̃ = (z-μ)/σ → g = softmax(z̃) → top-2 selection → expert FFN (SwiGLU) → y = (g1·E1(x)+g2·E2(x))/(g1+g2)。辅助损失：每层独立 α^(l)，每 iteration 根据 token drop rate 更新 α^(l) = 0.99·α_prev^(l) + 0.01·min(0.2·d_i^(l), 0.01)。总 loss = L_ce + Σ_{l=1}^{52} α^(l)·L_aux^(l)。对比 baseline 的门控退化（均匀概率）和固定 α，Skywork-MoE 的门控分布更尖锐（Max1/Max2 > 1），token drop rate 更低，辅助损失系数随层和训练阶段动态变化。
  - **系统框架层**：Skywork-Megatron（基于 Megatron-LM 23.06），EDP 策略（Attention TP mesh [PP, DP, 4] → Expert EP mesh [PP, DP, 4] 灵活切换），12-way unbalanced PP + 32-way DP + ZeRO-1。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：EDP 优化 AllToAll 通信（受限于 ≤64 experts 场景），unbalanced PP 减少 bubble 10%，kernel fusion + 通信-计算 overlap，38% MFU / 690 tok/GPU/s。
  - **硬件架构层**：论文未明确说明。
