## 2DQuant Low-bit Post-Training Quantization for Image Super-Resolution

- baseline方法是什么？
  Baseline 是 DBDC+Pac（CVPR 2023），当时 SR 领域的 SOTA PTQ 方法。DBDC 阶段需手动指定 clipping ratio 来截断长尾分布数据，Pac 阶段以极低学习率对量化器参数进行微调（导致收敛缓慢）。该方法主要针对 CNN-based 模型（EDSR、SRResNet）设计，无法有效适配 Transformer-based SR 模型（如 SwinIR），因为 SwinIR 的权重和激活分布呈现"对称+非对称共存+长尾"的独特特征。使用对称量化会浪费至少一半候选值；长尾效应会导致绝大多数浮点数被压缩到 1-2 个候选值中。
  
  Baseline 全栈执行例子（SwinIR ×2 4-bit）：
  - 算法pipeline：DBDC 对每个张量用统一的手动指定 clipping ratio 截断 → 对权重用对称量化、激活用不对称量化（未区分分布类型）→ Pac 阶段以极低 lr 微调 → 量化后 Linear/BMM 转为 INT4 算术。
  - 系统框架：论文未明确说明。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明。
  - 硬件架构：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 2DQuant，一个 coarse-to-fine 的两阶段 PTQ 方法，针对 Transformer-based SR 模型的"对称+非对称共存+长尾"分布特征进行专门优化：
  (1) **DOBI 替代 DBDC**：不再需要手动指定 clipping ratio，而是基于张量分布类型自动选择搜索策略——对称钟形分布用双界同时收缩搜索，非对称指数分布用固定下界+仅收缩上界搜索，以 MSE 最小化为目标自动找到最优 (l, u)。
  (2) **DQC 替代 Pac**：使用知识蒸馏方式（教师=FP 模型，学生=量化模型），联合优化输出 L1 loss 和中间层特征归一化 L2 loss，学习率从极低值提升至 1e-2，大幅加速收敛并避免局部最优。
  (3) **两阶段组合**：DOBI 的 search-based 方法天然避免局部最优，为 DQC 提供良好初始化；DQC 的 optimization-based 方法进一步向任务目标 fine-tune，形成互补。

  论文方法全栈执行例子（SwinIR ×2 4-bit）：
  - 算法pipeline：加载预训练 FP32 SwinIR-light 权重 → Stage1 DOBI：对每个 Linear 层权重（对称钟形分布，双界同时搜索 K=100 步）和每个激活张量（Attention Map/FC2 输入为非对称指数分布固定下界仅收上界；V/FC1 输入为对称钟形双界同搜）执行 MSE 最小化搜索，得到粗粒度 (l_i, u_i) → Stage2 DQC：用 DF2K 校准集，以输出 L1 loss + 特征归一化 L2 loss 联合蒸馏，Adam lr=1e-2，CosineAnnealing 3000 iter，batch=32，STE 近似梯度反向传播 fine-tune 每个量化器的 (l_i, u_i) → 量化后权重存为 INT4，推理时 Linear/BMM 转为 INT 算术，speedup 3.99×。
  - 系统框架：论文未明确说明。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明。
  - 硬件架构：论文未明确说明。
