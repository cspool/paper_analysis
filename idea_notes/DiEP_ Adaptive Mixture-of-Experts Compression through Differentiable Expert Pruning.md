## DiEP: Adaptive Mixture-of-Experts Compression through Differentiable Expert Pruning

- baseline方法是什么？
  现有 MoE 专家剪枝方法分为两类：(1) **Feature Statistics 类**（M-SMoE、Expert Trimming）：统计每个专家的 activation frequency 或 feature similarity，在每层独立地删除频率最低或合并相似的专家。M-SMoE 将低频专家合并到高频专家，但在层内做 activation count normalization 后跨层信息被抹除，隐含假设各层冗余程度相同。(2) **Greedy Search 类**（NAEE、S-SMoE）：NAEE 在每个 MoE 层内枚举所有 k-expert 组合，通过最小化 reconstruction loss 选出最优子集；S-SMoE 基于相似度做 pruning+merging。两类方法的核心缺陷是**所有层使用统一的剪枝比例**，忽略了不同 MoE 层之间专家冗余程度的显著差异（如 CKA 可视化所示，浅层 1-15 的 intra-layer similarity 模式与深层 16-32 明显不同）。对于 64 experts/layer 的模型，仅 12.5% sparsity 就需要评估 C(64,8) ≈ 4×10^8 种组合，使全局 exhaustive search 在计算上不可行。

  **Baseline 全栈执行例子（以 NAEE on Mixtral 8×7B, 50% sparsity 即每层 8→4 experts, 推理一个 token 为例）**：
  - **算法pipeline层**: 在每层 MoE 内枚举 C(8,4)=70 种专家组合，计算每种组合的 reconstruction loss，选出 loss 最小的子集作为保留专家。32 层 × 70 组合 = 2240 次评估。每层独立选择，无跨层信息传递。
  - **系统框架层**: HuggingFace Transformers 加载模型 → 逐层执行 expert combination search → 输出 pruned checkpoint。
  - **编译框架层**: 论文未明确说明。
  - **Kernel调度层**: 标准 PyTorch FFN forward，无 custom kernel。
  - **硬件架构层**: 4× NVIDIA A800 GPU。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **DiEP (Differentiable Expert Pruning)**，将离散专家选择重新表述为连续优化问题。核心设计：(1) **Intra-layer + Inter-layer 双层次重要性学习**：定义 intra-layer 重要性 α_i^(l)（每层内专家相对重要性）和 inter-layer 重要性 β^(l)（层对全局模型的贡献），通过 softmax 归一化实现连续松弛，将全局离散搜索空间（指数级）转换为连续可微空间，直接解决 NAEE 等 exhaustive search 的计算不可行问题。(2) **交替梯度优化**：以 α:β = 3:1 的比例交替更新，解耦两个参数组的梯度路径避免优化冲突（与 DiffPruning 的单调梯度下降形成对比），目标函数组合 CE loss + Reconstruction Regularization (∥F' − F∥_F)，无需 validation set。(3) **全局统一排序剪枝**：s_i^(l) = α_i^(l) · β^(l)，全局排序所有 L×N 个专家后按 ratio r 统一删除 bottom-K。这种 cross-layer global ranking 解决了 baseline 中"每层统一比例"导致的浅层/深层冗余差异被忽视问题——浅层自动保留更多专家（因为 β 和 α 学到的浅层重要性更高，符合 CKA 可视化结果）。(4) **Adaptive Expert Skipping 在线推理加速**：γ = γ1（routing weight ratio 中位数）× γ2（CKA similarity ratio），当 token 的次要专家 routing weight 低于 γ 倍的主要专家时跳过，消除冗余专家计算。额外获得 1.2−1.3× 推理加速。

  **DiEP 方法全栈执行例子（以 Mixtral 8×7B, 50% sparsity, 推理一个 token 为例）**：
  - **算法pipeline层**: α_i^(l) (32×8=256 参数量) + β^(l) (32 参数) → 仅约 0.01% 额外参数。Calibration: 128 C4 samples, 10 epochs, lr=5e-3 cosine schedule, batch=16。前向计算 y'^(l+1) = β^(l) · Σ_i softmax(α_i^(l)) · FFN_i(x) → 交替优化 3:1 更新 α 和 β → 收敛后全局排序 s_i^(l) → 删除 128 个最不重要专家 → 保留约 92% 原模型性能（MMLU avg 57.9 vs full 67.9, 50% sparsity）。
  - **系统推理层**: 加载 pruned checkpoint → 对每个 MoE layer 的 Top-2 激活专家计算 γ 阈值 → 若 w_e1 < γ·w_e0 跳过 e1 的 FFN 计算 → 减少 GPU 计算量和 memory access。
  - **编译框架层**: 论文未明确说明。
  - **Kernel调度层**: 标准 PyTorch FFN forward，adaptive skipping 通过条件判断跳过冗余专家计算，无 custom kernel。
  - **硬件架构层**: 4× NVIDIA A800 GPU。50% sparsity + skipping 下 GPU memory 降至 0.52× 原模型，token generation speedup 1.28×。
