## Efficient Expert Pruning for Sparse Mixture-of-Experts Language Models: Enhancing Performance and Reducing Inference Costs

- baseline方法是什么？
  Baseline 包含三个层面：(1) **Full SMoE Model (Zero-Shot)**：Mixtral 8×7B-Instruct 使用 8 experts (Top-2)，每 token 激活 2/8 experts 进行推理，47B 总参数、13B 激活参数。Router 网络（single-layer perceptron）在高维 hidden space 中划分 expert 分配决策，但存在 expert activation imbalance 问题。(2) **Frequency-based Expert Pruning [37]**：对训练集统计每个 expert 的激活频次，剪枝掉激活频次最低的 experts。缺陷：不考虑 router weighting 的实际贡献，仅依赖频次统计，在低 expert budget 下容易 collapse。(3) **Soft Activation Pruning [37]**：累积 router weighting（soft activation value），剪枝累积值最低的 experts。缺陷：路由权重在后续层移位后剪枝效果不可靠。(4) **NAEE [34]**：逐层穷举所有剪枝方案，选择与 full model 输出差异最小的方案。缺陷：(a) 穷举计算量随 expert 数组合爆炸（Qwen 的 60+ experts 无法穷举，只能随机采样 5000/2000 种方案，实际性能接近 random）；(b) 仅基于 output discrepancy 做选择，不考虑下游任务实际表现；(c) 剪枝后不做 knowledge recovery，高 sparsity 下性能下降显著。

  **Baseline 全栈执行例子（以 Mixtral 8×7B-Instruct, 2×A100, 单个 token 推理为例）**：
  - **算法层**: SMoE with 8 experts (SwiGLU FFN), Top-2 gating。每个 expert θ_i = {W_{1i}, W_{2i}, W_{3i}}。Router 计算 G = softmax(ZW_G) ∈ R^{n×E}，TopK(G_j) 选择 2 个 expert 激活。H_j = Σ_{i∈TopK} G_{ji} · FFN_i(Z_j)。
  - **系统框架层**: 标准 HuggingFace Transformers + PyTorch。加载 8 experts 全部参数 (45B out of 47B 为 expert 参数)，batch inference 时全部 experts 驻留显存。
  - **编译框架层**: 论文未明确说明（标准 PyTorch + CUDA graph 路径）。
  - **Kernel调度层**: 标准 grouped-GEMM for expert computation。所有 8 个 experts 的 W₁, W₂, W₃ 完整加载并计算，每个 activated expert 的 FFN = SwiGLU(Z_sub, W₁i, W₃i) · W₂i。Router top-k softmax 为标准核函数。
  - **硬件架构层**: 2× NVIDIA A100 GPU。8 experts → 显存占用 ~88.6 GB (FP16 约 94 GB 参数 + activations)。全部 experts 进入 HBM，prefill/decode 阶段完整执行 2 个 expert 的 FFN 计算。

  **Baseline 核心痛点**：
  1. **Expert 冗余性未被充分利用**：单个 expert 即可维持合理的推理性能（仅微小下降），但 8 个 experts 全部保留在显存中，造成 ~72% 参数量能被剪枝而性能不降的浪费。现有的剪枝方法（Frequency/Soft Activation/NAEE）在 expert 粒度选择上精度不足，高 sparsity 下 collapse 严重。
  2. **Router 网络在高维空间分配不精确**：Router 作为 single-layer perceptron 难以精确划分高维 hidden space，导致 expert activation imbalance 和 sub-optimal routing。但剪枝改变了 routing 行为——剩余 experts 的 routing weights re-normalize，提供了 routing 优化的机会。
  3. **梯度式 fine-tuning 资源需求过高**：传统剪枝范式中剪枝后需用 SGD fine-tuning 恢复性能，需要大量 GPU 显存和计算时间。权重复用只能通过 "select subset by importance criteria" 或 "distillation" 两种范式，缺乏高效的 weight merging 范式。
  4. **剪枝后 expert knowledge 丢失**：直接丢弃 pruned experts 导致知识丢失，现有方法只做 selection 不做 recovery。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **EEP (Efficient Expert Pruning)**，一种无梯度进化策略，通过设计 Router Mapping (WRM) 和 Expert Merging (WEM) 两个参数空间矩阵，在两阶段进化搜索中完成 expert 剪枝和知识合并。

  **(1) Expert Pruning Phase — 进化搜索发现最优剪枝模式（解决 pain point #1 "粗糙剪枝"）**：
  Baseline 的 Frequency/Soft Activation 使用固定的单一统计量指标选择 expert，无法适应不同下游任务的 expert 贡献分布。EEP 通过进化策略在巨大搜索空间（每层从 C(E, E') 种组合中选择）中搜索最优剪枝模式。WRM 和 WEM 初始化为 one-hot rows，且约束 WRM = WEM，只选择保留 expert 并保持离散性。每次迭代中个体按累积 F 分数排名，Top M_CP 作为 candidate parents，通过 Crossover（沿 expert 维度组合）和 Mutation（随机替换 pruned experts）产生后代。搜索 40 轮找到最优子集。
  结果：EEP (Prune Only) 在 4/8 expert 保留时大幅超越 baselines（Avg. 70.3 vs NAEE 60.5 vs Frequency 45.8），在 2/8 expert 保留时仍维持有效性能（Avg. 59.7）。

  **(2) Expert Merging Phase — 权重合并恢复知识（解决 pain point #3 "缺乏高效 fine-tuning" 和 #4 "知识丢失"）**：
  Baseline 范式中剪枝后 knowledge recovery 要么不需要（selection-only 方法精度低），要么需要梯度-based fine-tuning（资源要求高）。EEP 引入一种第三范式：**Weight Merging**。WRM 和 WEM 解耦后元素从离散 0/1 过渡到连续值，通过 block-wise weighted sum 将 pruned experts 的知识合并到 retained experts 中：
  - θ'_j = {Σ_i ω_ji W₁i, Σ_i ω_ji W₂i, Σ_i ω_ji W₃i}
  - 其中 ω_ji 为连续值，可包含负值（负系数说明某些 expert 的知识对下游任务无益）
  - 进化搜索 160 轮完成 continuous optimization
  结果：Merging 后在几乎所有数据集上实现 5%-7% 的额外提升（如 WIC 57.8→65.0, CB 69.6→75.0, SQuAD 75.2→80.6）。且整个过程无梯度计算，可在仅支持推理的设备上运行。

  **(3) Router 重新聚焦（解决 pain point #2 "Router 不精确"）**：
  Baseline 中 Router 需要在 8 个 experts 间分配高维 hidden space，剪枝后仅剩 4 或 2 个 experts，router re-normalization 使决策空间大幅缩小。实验显示剪枝后 expert 的 accumulated activation times、accumulated routing weights 和 activation correlation 发生明显变化，路由更加聚焦，部分数据集上即使不做 parameter update 也能超越 full model（如 SQuAD: 4 experts 下从 53.4%→75.2%）。

  **(4) 双重使用场景（解决 pain point #3 in broader deployment）**：
  - Use case 1 (减少 total experts): 8→4→2，节省 47%-71% GPU 显存
  - Use case 2 (减少 active experts): Top-2→Top-1，实现 prefill 1.63× 加速，decode 1.34× 加速
  - 组合使用: 4 total + 1 active → 47% 显存节省 + 1.41× 推理加速

  **EEP 方法全栈执行例子（以 Mixtral 8×7B, Top-2→Top-2, 8→4 experts per block, 搜索 SQuAD 训练子集为例）**：
  - **算法pipeline层**: Input Z ∈ R^{n×d}。
    阶段 I (Pruning): WRM = WEM ∈ R^{4×8} (one-hot rows)。G' = WRM · softmax(ZW_G)（路由降维）。θ'_j = WEM row_j 选择的原始 expert 权重。进化搜索 40 iterations, population size=|P|, 每天代评估 F(W·Θ)（下游任务 accuracy 作为 fitness）。Crossover 沿 expert dimension 交换；Mutation 随机替换 one-hot 位置。
    阶段 II (Merging): WRM, WEM ∈ R^{4×8} (continuous)。θ'_j = {Σ_i ω_ji W₁i, Σ_i ω_ji W₂i, Σ_i ω_ji W₃i}。进化搜索 160 iterations, Mutation=Gaussian noise。Expert weights 按深度分为 4 groups（或 32 groups per dataset），组内共享 merging coefficients 减少优化参数。
    最终 model: 4 experts per MoE block（参数量从 45B→~12.8B experts），权重为 merged form。
  - **系统框架层**: 无特定 serving framework（标准 PyTorch/HuggingFace 推理）。EEP search 过程在 inference-only 环境完成。搜索完成后将 merged model weights 导出为标准 HF format，在目标部署平台使用。
  - **编译框架层**: 论文未明确说明。
  - **Kernel调度层**: 论文未明确说明 KERNEL 级别实现。标准 grouped-GEMM kernel for expert computation（4 experts vs 8 experts 减少了 GEMM 调用次数和权重加载量）。Active expert 减少（2→1）后 decode 阶段 FP compute 和 HBM→SRAM weight fetching 减半。
  - **硬件架构层**: 2× NVIDIA A100 GPU。显存占用从 88.6GB→46.6GB（4 experts）→25.6GB（2 experts）。Prefill speedup 1.11×(4E)/1.18×(2E)/1.63×(1 active)/1.75×(4E+1 active)，Decode speedup 1.29-1.60×。Merging phase 在精度恢复阶段的贡献与显存/计算节省关系：merging 增加 ~floating-point operations in post-training 但不增加 inference cost。
