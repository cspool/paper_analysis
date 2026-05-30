## MoLA: MoE LoRA with Layer-wise Expert Allocation

- baseline方法是什么？
  Baseline 是 **标准 LoRA-MoE（即 MoLA-□ / MoLA Rectangle）**——将 LoRA 与 MoE 结合，在 Transformer 每层使用**相同数量**的 LoRA expert，通过 router 做 top-K 路由选择。典型代表如 MoELoRA (Liu et al., 2023)、LoRA-MoE (Dou et al., 2023)、MoLORA (Zadouri et al., 2023)。

  核心缺陷：
  1. **不考虑层级差异**：所有 Transformer 层分配相同数量的 expert。但不同层处理的信息粒度不同——底层处理 token-level 特征（词义、语法），中层学习有效表示，高层处理抽象推理。等量分配忽略了这一层级差异，导致底层 expert 冗余、中高层 expert 不足。
  2. **底层 expert 高度冗余**：底层的 LoRA expert 学习的低秩矩阵彼此非常相似（Frobenius Norm 小），多个 expert 产生重叠表示，浪费参数预算。
  3. **中高层 expert 能力受限**：中高层需要处理多样化的抽象特征和任务特定模式，但固定的 expert 数量限制了其 fitting 能力，无法充分学习 fine-grained task-specific 模式。

  全栈执行例子（以 LLaMA-2-7B MoLA-□(5555) 微调为例）：
  - **算法pipeline**：输入 token 序列 x → 第 j 层（j=1..32）self-attention：对每个 token，W_q/W_k/W_v/W_o 各创建 5 个 LoRA expert（A_i,B_i 低秩对, r=8），router W_r 计算 5-dim softmax → top-2 选择 → 两个 expert 的 A_iB_i x 加权求和加到原始 W_0 x 上 → MLP 同理（W_gate/W_down/W_up 各 5 expert）。第 1 层和第 32 层用同样 5 个 expert → 底层 expert 间 Frobenius Norm 差异小（冗余大），高层 expert 间差异大但 expert 数不足。
  - **系统框架**：Hugging Face Transformers 训练循环，PyTorch 数据并行在 8×A100-40G 上，AdamW 优化器，仅训练 LoRA expert 和 router 参数，预训练权重 W_0 冻结。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：标准 PyTorch 矩阵运算，无自定义 kernel。各 linear module 的 LoRA expert 计算为 batch 低秩矩阵乘加操作。
  - **硬件架构**：A100-40G GPU + A6000 GPU，无定制硬件。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **MoLA（MoE-LoRA with Layer-wise Expert Allocation）**，核心创新是**为不同 Transformer 层分配不同数量的 LoRA expert**。关键设计：
  1. **层级别灵活 expert 分配**：每层 j 分配 N_j 个 expert，ΣN_j 为总 expert 预算。不是所有层用相同数量，而是根据层级重要性差异化分配。
  2. **五种层级别配置假设**：MoLA-△（底层多）、MoLA-▽(2468, 高层多）、MoLA-▷◁(8228, hourglass，两端多）、MoLA-✸(2882, diamond，中层多）、MoLA-□(5555, rectangle，等量 baseline）。
  3. **全 dense weight 覆盖**：与 LoRA-MoE (Dou et al., 2023) 仅对 FFN 应用不同，MoLA 对 attention 的 W_q/W_k/W_v/W_o 和 MLP 的 W_gate/W_down/W_up 全部应用 LoRA expert。
  4. **专家冗余的定量分析**：通过 Frobenius Norm 量化各层 expert 间相似度，发现底层 expert 差异更小（更冗余），高层 expert 差异更大。

  对应缺陷的解决：
  - **缺陷 1（忽略层级差异）→ 层级别灵活分配**：不再强制所有层相同 expert 数量。实验证明 MoLA-▽ (2468) 和 MoLA-✸ (2882) ——即中层/高层分配更多 expert ——在三个 base model（LLaMA-2、Mistral、Gemma）上均优于等量分配，即使总参数量相同甚至更少。
  - **缺陷 2（底层冗余）→ 减少底层 expert**：Frobenius Norm 分析证实底层 expert 最相似。极端配置实验（10-2-2-2 vs 2-2-2-10）显示底层过多 expert 平均性能最低。MoLA-▽ 将底层 expert 从 5 减至 2，将节省的参数分配给中高层，性能反而提升。
  - **缺陷 3（中高层能力受限）→ 增加中高层 expert**：中高层分配更多 expert 可增强 fitting 能力。MoLA-▽ (2468) 以仅 62.5% 的参数量（vs MoLA-□ 8888）在部分 benchmark 上取得更好或相当性能，证明了参数效率。

  全栈执行例子（以 LLaMA-2-7B MoLA-▽ (2468) 微调为例）：
  - **算法pipeline**：输入 token 序列 x → 第 1-8 层（底层）：每层仅 2 个 LoRA expert（总 16 expert），router 选择 top-2 → 第 9-16 层：每层 4 expert（总 32） → 第 17-24 层：每层 6 expert（总 48） → 第 25-32 层（高层）：每层 8 expert（总 64）。总 expert 数 = 8×(2+4+6+8) = 160，与 MoLA-□(5555) 的 32×5=160 总 expert 数相同但性能更优。router 的 top-2 选择 + load balancing loss 确保所有 expert 被充分训练。Frobenius Norm 分析显示：底层 expert 间差异 ~0.1，高层 ~0.6（避免冗余最大化利用度）。
  - **系统框架**：Hugging Face Transformers + PyTorch，与 baseline 相同训练框架，仅在模型结构中为每层配置不同 expert 数量。训练循环无额外计算开销（每 token 的激活 expert 数 = K×7 个 linear module = 2×7 = 14 个 LoRA 前向，与 MoLA-□ 相同）。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：无特殊 kernel 优化，标准 PyTorch 矩阵运算。每层 expert 数的差异不影响 kernel 执行路径——所有 expert 的 A_i B_i x 均为独立低秩矩阵乘法。
  - **硬件架构**：A100-40G / A6000 GPU，与 baseline 相同硬件，无需定制硬件支持。

  关键效果：MoLA-▽ (2468) 在 LLaMA-2 上以 105.6M 可训练参数（1.5% of 7B）超越 LoRA（159.9M）和 MoLA-□(8888)（169M），在 CommonsenseQA 上达 78.95%（vs LoRA 75.51%, MoLA-□5555 78.13%）。Continuous Learning 中 MoLA-▽ 的 performance drop 仅 -0.47%（vs LoRA -2.17%），显示优越的抗遗忘能力。
