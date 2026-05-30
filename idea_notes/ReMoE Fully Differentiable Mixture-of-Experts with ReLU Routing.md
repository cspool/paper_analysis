## ReMoE Fully Differentiable Mixture-of-Experts with ReLU Routing

- baseline方法是什么？
  Baseline 是 vanilla TopK-routed MoE（Token-choice dropless TopK, dMoE）。其路由函数为 R(x) = TopK(Softmax(x · W_l), k)，即先对 router logits 做 Softmax 归一化为概率分布，再通过 TopK 保留最大的 k 个值，其余强行置零。核心缺陷：TopK 操作在第 k 大值 x_{[k]} 处引入跳变不连续性——当 Softmax 输出从 (0.51, 0.49) 变为 (0.49, 0.51) 时，TopK 输出从 (0.51, 0) 跳变为 (0, 0.51)，导致训练目标函数非连续、非可微，限制了 router 的优化效果和模型的可扩展性。此外，TopK 使每个 token 被固定路由到恰好 k 个 expert，无法根据 token 难度动态分配计算资源。
  全栈执行例子（Baseline: dMoE, N=182M/E=8/k=1, LLaMA architecture, 8×A100 GPU, Megatron-LM）：
  - **算法层**：MoE layer 中 router 执行 Softmax(x · W_l) → TopK(·, k=1) → 每个 token 选择 1 个 expert → expert FFN (SwiGLU: W_down @ (SiLU(W_gate @ x) * W_up @ x)) → 加权求和。TopK 在第 k 大值处的跳变使 loss 在此处不可微，梯度估计不准确。训练需额外的 auxiliary load balancing loss (weight=0.01) 防止 routing collapse。
  - **系统框架层**：Megatron-LM 支持 Data/Expert/Tensor/Pipeline Parallelism。MoE layer 执行 all-to-all dispatch → per-expert FFN → all-to-all combine。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：标准 cuBLAS GEMM + NCCL all-to-all。无自定义 kernel。
  - **硬件架构层**：8×A100 GPU。TopK routing 的离散性不影响硬件执行效率，但限制了模型性能上限。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 ReMoE，用 ReLU 路由 R(x) = ReLU(x · W_l) 完全替代 TopK(Softmax(x · W_l), k) 路由，消除不连续性。配合自适应 L1 正则化精确控制稀疏度至目标水平 (1-k/E)，以及精炼负载均衡 L1 正则化天然防止 routing collapse。
  全栈执行例子（ReMoE, N=182M/E=8/k=1, LLaMA architecture, 8×A100 GPU, Megatron-LM）：
  - **算法层（解决 TopK 不连续性 + 固定激活数缺陷）**：
    - **ReLU 路由**：将 TopK 的断点 x_{[k]} 统一设为 0，即 ReLU(x)_e = x_e · 1{x_e ≥ 0}。当 x_e 在 0 附近平滑过渡时，输出连续变化（如 (0.01, 0) → (0, 0.01) 是连续的），消除了 TopK 的跳变不连续性（如 (0.51, 0) → (0, 0.51) 是不连续的）。训练 pipeline 因此完全可微，梯度流畅通。
    - **自适应 L1 正则化**：λ_{i+1} = λ_i · α^{sign((1-k/E)-S_i)}，当稀疏度 S_i < 目标时扩大 λ，反之缩小。正则项 L_reg = (1/LT) Σ||R(x)||_1 对所有非零 router output 加梯度偏置 λ_i/(LT)，驱动输出向零，使平均稀疏度稳定在 (1-k/E)，保证 FLOPs 与 TopK MoE 统计等价。
    - **负载均衡精炼**：f_{l,e} = (E/kT) Σ 1{R(x_t^l)_e > 0} 作为 per-expert 权重，使过载 expert 的 router output 受到 λ_i · f_{l,e}/(LT) 的更强梯度惩罚，自动均衡负载。与 TopK MoE 的 auxiliary load balancing loss 在数学形式上等价，但 ReLU 输出可任意小（无 Softmax 的和为 1 约束），因此需 λ_i 自适应更新以防止 routing collapse 至全零。
    - **动态 expert 分配**：每个 token 激活的 expert 数量可变——高频 token（如 "the", "\n"）激活较少 expert，低频 token（如特殊符号、罕见词）激活更多 expert，类似 Huffman 编码的自适应资源分配。Domain 级别也呈现差异化激活。
    - **自然三阶段训练**：Stage I (dense warm-up, ~100 steps)：λ 小，几乎所有 expert 被激活，从随机初始化中分化。Stage II (sparsifying)：L_reg 增强，expert 开始稀疏化。Stage III (stable sparse)：稀疏度稳定在目标值。
    - **对比 baseline**：baseline 的 TopK 在第 k 大值处不可微 → ReMoE 的 ReLU 在 0 处连续可微。Baseline 固定 k 个 expert/token → ReMoE 动态可变数量。Baseline auxiliary load balancing loss 需手动调权重 → ReMoE 通过自适应 λ_i 在单一 L1 正则化公式中统一稀疏度控制和负载均衡。
  - **系统框架层**：与 baseline 相同（Megatron-LM），ReLU routing 作为 drop-in replacement，仅需替换路由函数。支持 Data/Tensor/Pipeline/Expert Parallelism（全兼容）。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：与 baseline 相同。ReMoE 仅改变路由逻辑（ReLU 替代 Softmax+TopK），不修改 FFN 计算或通信。Training throughput 差异在 -2.29% ~ +3.89% 之间（近似等效）。
  - **硬件架构层**：同一 8×A100 硬件。Stage I/II（~100 steps, ~0.17% 总步数）有额外计算开销（激活更多 expert），但总体上 negligible。
  - 解决 Baseline 缺陷的方式总结：
    1. **针对 TopK 的不连续性**：ReLU 将断点统一设为零，从 (0.51,0)→(0,0.51) 的跳变变为 (0.01,0)→(0,0.01) 的连续过渡，训练目标完全可微，router 优化更稳定。实验证实：flip rate 在 E=16/32 时 ReMoE 比 MoE 低 2-3×，flip count 不随 E 增长（MoE 的 flip count 随 E 增大而增大）。
    2. **针对固定激活数**：ReLU routing 各 expert 独立决策，token 可激活 0~E 个 expert，实现动态计算分配——高频 token 激活少、低频 token 激活多。
    3. **针对负载均衡需额外 loss**：L1 正则化的精炼版 f_{l,e} 权重天然实现负载均衡，与稀疏度控制在单一公式中统一，无需额外 auxiliary loss。
    4. **Superior scalability**：ReMoE 随 E 增长的性能提升斜率比 MoE 更陡（从 E=4 到 E=128，ReMoE 的 loss 下降更大），且 fine-grained ReMoE (G=32/64) 达到理论上限 Dense×8 的性能而 FLOPs 显著更少。
