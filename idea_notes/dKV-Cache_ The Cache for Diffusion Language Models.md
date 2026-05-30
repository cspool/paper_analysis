## dKV-Cache: The Cache for Diffusion Language Models

- baseline方法是什么？
  Baseline 是标准的 diffusion language model（DLM）推理，具体为 LLaDA-8B-Instruct 和 Dream-Base-7B 在无 KV-Cache 下的全序列去噪推理。DLM 推理中，每个去噪步需要完整编码长度为 L 的全部 token（双向注意力），生成 L 个 token 需 T 个去噪步，复杂度为 O(L³)（AR 模型带 KV-Cache 仅为 O(L²)）。加速 baseline 为 Few-Steps/Half-Steps（减少去噪步数到 50-62.5%），但以生成质量为代价。

  全栈执行例子（LLaDA-8B-Instruct 标准推理，A6000 GPU）：
  **算法pipeline**：输入全 [MASK] 序列 x^{1:L}_{c(T)}，每步 t ∈ [T, 1]：调用 p_θ 预测全部 L 个位置的 x_0，根据 confidence/random 策略选择部分位置 remask，其余位置保持已解码 token 不变。每步需对全部 L 个 token 计算双向 self-attention（QKV ∈ R^{L×d}），softmax 在全部 L 个位置上归一化。L=256, T=256 时总计算量 = 256 步 × O(256²) = O(256³)，与 L=256 的 AR（256 步 × O(256²) cumulative = O(256³)）理论相当，但因缺少 KV-Cache 导致每步都从头计算全部 token 的 K/V。
  **系统框架**：HuggingFace Transformers + PyTorch，LLaDA-Model 加载 8B 参数，使用标准 Transformer forward（无 caching），每步完整计算所有 hidden states。
  **编译框架**：论文未明确说明。
  **kernel调度**：FlashAttention-2 kernel，每步执行 O(L²d) 的完整双向注意力。无 KV-Cache 的情况下每次都需要从 HBM 读取完整的 K/V 矩阵，memory-bound 严重。
  **硬件架构**：论文未明确说明。
  **芯片设计**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出 delated KV-Cache (dKV-Cache) —— 首个用于 DLMs 的 KV-Cache 机制。核心思想源于对 DLM 去噪过程中 token 表征动态的实证观察（Figure 2）：(1) 已解码 token 的 K/V 表征在后续步趋于稳定，而 [MASK] token 持续波动；(2) 相邻步间 K/V 相似度整体较高。据此设计延迟缓存策略：(a) **延迟缓存（delayed caching）**：仅缓存已解码 token 的 K/V 并跨步复用，掩码 token 每步重新计算；(b) **一步延迟（one-step delayed caching）**：使用上一步 M_{t-1} 而非当前步 M_t 决定缓存集合，避免刚解码 token 在表征剧变时被过早缓存，这是保证质量的关键设计（Figure 3 证明无延迟时性能崩溃）；(c) **缓存刷新（cache refreshing）**：每 N 步清空缓存重新计算全序列 K/V，避免长时间累积的缓存误差导致质量退化。两种变体覆盖不同场景：dKV-Cache-Decode（近乎无损，refresh=4-8）和 dKV-Cache-Greedy（O(L²) 复杂度，refresh=2，加局部窗口 w≤6）。

  全栈执行例子（dKV-Cache-Decode on LLaDA-8B-Instruct, A6000 GPU）：
  **算法pipeline**：每步 t，根据上一步掩码集 M_{t-1} 确定已解码 token（缓存复用）和仍在掩码的 token（重新计算）。重排序列：将缓存 token 置于左侧（连续块），掩码 token 置于右侧，同时调整位置编码。Transformer 仅计算掩码 token 的 Q/K/V（|M_{t-1}| 个 token），已解码 token 的 K/V 从缓存直接拼接（concat），完整 K^I / V^I 参与注意力计算。注意力输出 scatter 回原始位置。每 8 步刷新缓存。与 Baseline 对比：每步计算量从 O(L) 降至 O(|M_t|)，cache ratio 从 0 提升到逐渐接近 1。|M_t| 从 L 递减到 0，累计加速约 2-3.5×。
  **系统框架**：PyTorch + HuggingFace Transformers，修改 LLaDA 模型的 forward 函数增加 concat_reorder 逻辑。concat_reorder 实现将索引操作从 K/V 矩阵层级（[B,L,D]）转移到 token 层级（[B,L]），大幅减少内存碎片。Generation 脚本修改为 step-by-step 调用并管理缓存状态。
  **编译框架**：论文未明确说明。
  **kernel调度**：使用标准 FlashAttention。concat_reorder 通过重排使缓存 token 连续，从而可用简单 concat 和 slice 操作替代高开销的 gather/scatter 索引。位置编码重排仅每步一次、跨层共享，开销可忽略。数据移动仍为关键瓶颈，batch size=1 时 memory-bound 导致加速有限甚至退步。
  **硬件架构**：论文未明确说明。
  **芯片设计**：论文未明确说明。

  dKV-Cache-Greedy 的额外设计：将缓存集合从 I \ M_{t-1} 激进缩减为 M_t = {D_t} ∪ {D_{t-1}} ∪ W(D_{t-1})（3 个组件共最多 8 个 token），将每步计算量固定为 O(w·L)（w 为窗口大小 ≤6），复杂度从 O(L³) 降至 O(L²)，以轻微性能下降换取更大加速（1.51-1.73× vs baseline speed 即 1.63-1.70× speedup on LLaDA）。
