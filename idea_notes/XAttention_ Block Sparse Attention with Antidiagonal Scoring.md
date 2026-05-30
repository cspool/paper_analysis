## XAttention: Block Sparse Attention with Antidiagonal Scoring

- baseline方法是什么？
  Baseline 方法为基于 FlashInfer 的 FlashAttention（dense full attention），以及其他 training-free 稀疏注意力方法 MInference 和 FlexPrefill。

  **Baseline（MInference/FlexPrefill）的执行流程**：
  - **算法 Pipeline**：MInference 和 FlexPrefill 使用 "Vertical-Slash" 稀疏模式——通过分析输入序列末端的 query 来识别重要的"垂直列"和"斜线"注意力模式索引，然后用这些索引构建稀疏 mask 并执行 sparse attention。这种方法依赖 pooling（mean/sum pooling）来估计 block 重要性，但 pooling 在 block 内仅有少量显著垂直/斜线模式时会严重低估重要性。
  - **Serving 调度**：论文未明确说明。
  - **编译框架**：论文未明确说明。
  - **Kernel 调度**：MInference 使用 Vertical-Slash Sparse Index kernel（基于 point-range two-way merge 算法，在 GPU 上并行构建 per-row block indices）和 Vertical-Slash FlashAttention kernel（混合 block-sparse + column-sparse 计算）。Pattern selection 的 index search 开销巨大，尤其在短上下文时（pattern selection overhead 占比高）。
  - **硬件架构**：论文未明确说明。
  - **芯片设计**：论文未明确说明。

  Baseline 的核心缺陷：
  1. Pattern selection 计算开销大——MInference 的 vertical-slash index search 和 FlexPrefill 的复杂超参数搜索在短上下文时反而成为瓶颈。
  2. Pooling 方法不准确——mean/sum pooling 无法有效检测 block 内的稀疏但关键的垂直/斜线模式。
  3. 固定稀疏度策略（Top-K 或 Top-Ratio）无法适应不同序列长度和输入内容。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  XAttention 提出使用注意力矩阵的**反对角线值之和**（antidiagonal sum）作为 block 重要性的轻量级代理。核心洞察：反对角线天然交叉 block 内所有可能的垂直和斜线注意力模式，因此反对角线和能高效、准确地检测这些关键模式。

  **XAttention 的执行流程**（全栈对比 Baseline）：
  - **算法 Pipeline**：三步流程——(1) Strided Antidiagonal Scoring：按步长 S 沿反对角线采样并求和；(2) Threshold Block Selection：基于累积 softmax 概率选择 block，实现动态稀疏度（不同头、不同输入稀疏度自适应）；(3) Minimum Threshold Prediction：通过动态规划为每个注意力头离线搜索最优阈值 τ_h，进一步优化稀疏度-准确率平衡。

  - **Serving 调度**：论文未明确说明。

  - **编译框架**：论文未明确说明。

  - **Kernel 调度**：基于 FlashInfer 框架实现。反对角线 scoring 的计算复杂度仅 O(L×d/S²)（vs MInference 的 O(L×k_v×k_s)），通过简单的 Q/K reshape + 小矩阵乘法完成近似注意力分数计算。Block selection 使用贪心累积阈值算法，无需复杂的 index search。Sparse attention 直接调用 FlashInfer 的 block-sparse kernel，仅计算选中 block 对的注意力。

  - **硬件架构**：论文未明确说明。

  - **芯片设计**：论文未明确说明。

  **对比 Baseline 的关键改进**：
  1. **反对角线 scoring 解决 Pooling 不准确问题**：反对角线交叉每个 block 内所有垂直和斜线模式（Figure 2），保证了信息完整性——每个 token 至少贡献一条反对角线的值。消融实验显示 antidiagonal pattern 在同等计算量下密度最低且准确率最高（S=8: antidiagonal average 88.47 vs random 82.48 vs diagonal 81.06）。
  2. **极低 scoring 开销解决 Selection 瓶颈**：反对角线 scoring 仅需 reshape Q/K + 小矩阵乘法，计算量仅为完整注意力的 1/S²。实测 pattern selection 比 MInference 快 24.9×、比 FlexPrefill 快 5.9×。
  3. **动态阈值解决固定稀疏度问题**：Threshold block selection 按累积概率 τ 自适应决定稀疏度——短序列（注意力密集）自然保留更多 block，长序列（注意力稀疏）自动提高稀疏度。128k 时密度仅 6.89%（S=8），4k 时密度 52.16%。DP-based per-head threshold 进一步优化。
  4. **训练自由且即插即用**：与需要 costly pretraining 的 SeerAttention 不同，XAttention 无需任何训练，可直接替换任意 Transformer 模型的注意力模块。
