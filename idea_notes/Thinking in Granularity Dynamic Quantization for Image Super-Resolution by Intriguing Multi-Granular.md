## Thinking in Granularity Dynamic Quantization for Image Super-Resolution by Intriguing Multi-Granularity Clues

- baseline方法是什么？
  CADyQ (Content-Aware Dynamic Quantization) 在 SR 模型每一层嵌入可训练的 bit selector，同时对 layer 和 patch 进行 bit-width 自适应分配，依据特征梯度大小测量量化敏感度。全栈执行示例：给定一张 LR 输入图像，CADyQ 将其分割为 patch 送入 SR 模型 → 每个卷积层前，bit selector 根据该层特征梯度 magnitude 为该层+该 patch 选择 bit-width（4/6/8 bit）→ 对应层用选中的 bit 量化激活值（weight 固定 8-bit）→ 逐层累积计算 → 输出 SR 重建。缺陷：(1) 每层的 bit selector 引入额外计算开销，深层网络尤为严重；(2) 逐层独立调整 bit 会打乱原始模型层间关系，降低量化模型的表示能力（t-SNE 可视化证实 CADyQ 量化后层特征分布与原始全精度模型显著偏离）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Granular-DQ 完全放弃 layer sensitivity 考量，转为 patch-wise + layer-invariant 的动态量化范式，基于图像内在的多粒度线索和熵统计。全栈执行示例：给定一张 LR 输入图像 X：
  - **算法pipeline层**：GBC 编码器对 X 提取 D 层多粒度特征（Z_1 细粒度纹理→Z_D 粗粒度结构），经 GroupNorm+AvgPool+Concat+GAP 融合为全局描述 S。线性层 + Gumbel-Softmax 为每个 patch 采样门控分数 p_i，映射到候选 bit {4,6,8}。E2B 在训练集上预计算所有 patch 的像素熵分布 H（Gaussian 核密度估计），插入分位数阈值（0.5, 0.9）将 H 切分为 3 个子区间对应 [4,5,8] bits。对 GBC 分配高 bit 的 patch，根据其熵值落入区间确定最终适配 bit。ATC 用 EMA(γ=0.9997) 动态校准阈值。量化使用 QuantSR 方案，weight 固定 8-bit。
  与 baseline 的核心差异：所有层对同一 patch 使用相同 bit-width，保持了层间关系的一致性（t-SNE 验证 Granular-DQ 量化后特征分布更接近全精度模型）；不需要每层插入 bit selector，仅输入端一个 GBC，计算开销可忽略。
  - **系统框架层**：论文未明确说明（基于 PyTorch 实现，不涉及 serving/编译框架/kernel/硬件层次修改）。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：论文未明确说明。
  - **硬件架构层**：论文未明确说明。
