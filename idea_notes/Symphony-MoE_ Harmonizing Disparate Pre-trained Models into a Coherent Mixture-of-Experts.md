## Symphony-MoE: Harmonizing Disparate Pre-trained Models into a Coherent Mixture-of-Experts

- baseline方法是什么？
  Baseline 方法包括三类 upcycling 方案：(1) **BTX (Branch-Train-Mix)**——将各 dense model 的 FFN 权重直接复用为 expert，共享 backbone 由所有模型权重的简单线性平均构成。(2) **BAM**——复用 FFN 权重和部分 attention 权重 (W^q, W^o) 作为 expert，其余权重线性平均构成 shared backbone。(3) **Drop-Upcycling**——复用 FFN 权重并对随机选择的参数施加 Gaussian perturbation 防止 expert 同质化，其余权重平均复用为 shared backbone。三者的共同缺陷：要么来自单一 dense checkpoint（expert 多样性受限），要么在融合多个不同训练历史的模型时使用粗糙的线性平均，无法解决 parameter space misalignment 问题——各 source model 的神经元在数值和语义参数空间中占据互不兼容的位置，直接合并导致 catastrophic interference。

  全栈执行例子（Baseline: BTX, Qwen 1.5B×4, 24× V100, upcycling + post-training on 5B tokens）：
  - **算法Pipeline层**：4 个 source dense models 的 FFN 层直接作为 4 个 experts（无 alignment）→ 共享 backbone = linear average(W_1, W_2, W_3, W_4) → 初始化随机 router → post-training 6 epochs with load balancing loss。问题：无 alignment 时，各 expert 的内部神经元排序不同（同一功能的神经元在不同 model 中的索引位置不同），linear averaging 产生"功能交叠"——每个 expert 的独特能力被互相稀释的共享 backbone 混淆，CKA 分数高达 0.65-0.75（专家功能崩溃为冗余子空间）。
  - **Serving/系统框架层**：LLaMA-Factory 框架训练 dense models，post-training 使用标准 transformer 前向传播。MoE forward 时 router 分配 token 到 4 个 experts。无 expert 间的功能区分，router 难以学习有意义的 dispatching。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：标准 PyTorch GEMM/CUDA kernel 执行 expert FFN。无 expert-level 的 kernel 调度优化。
  - **硬件架构层**：NVIDIA V100（32 GB HBM2），24 卡。各 expert 的 FFN 参数驻留 GPU 内存。无跨卡 expert parallelism（单卡容纳全部 experts）。
  Baseline 的核心缺陷：(a) **Parameter space misalignment**——不同训练历史的模型占据互不兼容的参数空间，简单拼接/平均导致功能崩溃；(b) **Expert diversity loss**——naive merging 使 expert 的独特功能指纹被模糊化，测量为高 CKA 值；(c) **Router 失效**——在功能不一致的 experts 上，router 无法学习有意义的 token 到 expert 的映射。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **Symphony-MoE**，核心是 **training-free functional alignment + post-training coordination** 两阶段框架。这直接解决 baseline 的三大缺陷：(1) **Parameter space misalignment**——通过 activation-based Hungarian permutation alignment（min_P ||A_1 - A_i P||²_F）将各 model 的 FFN 神经元在 training-free 下重排到 anchor 的功能空间，使得不同 model 中功能等价的神经元被映射到相同的空间位置，消除参数空间的"坐标系差异"。(2) **Expert diversity loss**——alignment 在消除 misalignment 的同时保留各 model 的独特参数值（permutation 仅改变神经元顺序，不改变数值），使 experts 功能兼容但不相同，CKA 分数恢复到接近原始 unmerged experts 的水平。(3) **Router 失效**——alignment 后的 experts 共享一致的坐标空间，router 可以基于 token-level 的语义内容学习有意义的 dispatching，实现真正的 expert 专化激活。

  全栈执行例子（论文方法：Symphony-MoE, Qwen 1.5B×4, 24× V100, Stage 1 alignment + Stage 2 post-training）：
  - **算法Pipeline层**：**Stage 1 (Training-free)**：
    (a) 共享 backbone 构建：对 self-attention 的 Q/K/V/O 矩阵用 SLERP 融合（preserving geometric integrity，sphere 上的最短测地线插值，公式为 SLERP(W1,W2,t) = (sin((1-t)Ω)/sin(Ω))·W1 + (sin(tΩ)/sin(Ω))·W2，其中 Ω = arccos(tr(W1·W2)/||W1||·||W2||)）；对 embedding 用 MergeKit selective linear（共享 vocabulary token → 线性平均，独有 token → 保留原 embedding）；对 LayerNorm 用简单平均。
    (b) Expert FFN alignment：以 M1 (General) 为 anchor → 从 General/Code/Math/Science 四个 domain 等量采样构建 D_cal (10.4M tokens) → 对每层每对 (M1, Mi) 提取 FFN 输出激活 → 用 Hungarian 算法求解 min_P ||A_1 - A_i P||²_F（O(d_ff³)) → 应用 P 重排 W_up, W_gate, W_down。结果：4 个 experts 功能对齐但参数各异。
    **Stage 2 (Post-training)**：随机初始化 router W_g ∈ R^{d_model × 4} → top-2 routing → 在扩展 D_cal (5B tokens) 上训练 6 epochs，AdamW lr=5e-5 → L_total = L_lm + 0.01·L_bal → 专家协作学习。最终：MMLU 58.91 vs BTX 45.12 (+13.8%)，HumanEval 42.39 vs BTX 29.08 (+13.3%)，MedCQA 35.26 vs BTX 26.92 (+8.3%)。

  - **Serving/系统框架层**：LLaMA-Factory 用于 dense model instruction tuning。MoE 架构为标准 decoder-only transformer，每 L 层替换一个 dense FFN 为 N 个 expert FFN + top-2 router。Forward 时为每个 token 激活 2/4 experts，FLOPs per token 为 dense 模型的 2×。无 serving framework 修改。

  - **编译框架层**：论文未明确说明。

  - **Kernel调度层**：标准 PyTorch 和 CUDA kernel 执行 expert FFN forward/backward。Align 阶段使用 Hungarian 算法（scipy.optimize.linear_sum_assignment）。Post-training 使用标准 transformer 训练 pipeline。无自定义 kernel。

  - **硬件架构层**：NVIDIA V100 (32 GB HBM2)，24 卡。Stage 1 alignment 是单卡计算（仅需 calibration data 前向 + Hungarian + weight remap）。Post-training 在 24 卡上进行数据并行训练。所有 4 个 experts 的完整参数驻留 GPU 内存（1.5B×4 架构，约 6B 总参数但每个 token 仅激活 2 experts ≈ 3B 参数量，与 dense 1.5B 相比 FLOPs per token 近似翻倍）。

  方法 vs Baseline 对比核心差异：(a) Training-free functional alignment vs naive averaging——用 permutation 消除 misalignment 而无需训练，保留专家多样性；(b) Layer-aware backbone vs uniform averaging——SLERP 保护 attention 的几何结构，MergeKit 处理 embedding 的 vocabulary mismatch，避免功能退化；(c) 两阶段协调 vs end-to-end 训练——先解决参数空间 misalignment（Stage 1），再学习 router routing（Stage 2），分离了结构融合和功能协调两个目标。
