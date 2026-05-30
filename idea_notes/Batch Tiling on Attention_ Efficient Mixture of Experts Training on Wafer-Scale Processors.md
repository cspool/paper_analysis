## Batch Tiling on Attention: Efficient Mixture of Experts Training on Wafer-Scale Processors

- baseline方法是什么？
  - Baseline 是 Conventional Uniform Batching（G=1），即 MoE 训练中 attention 层和 expert MLP 层使用相同的全局 batch size，不进行 batch tiling。
  - 全栈执行例子（Baseline, G=1）：
    - **算法层**：输入 X ∈ ℝ^(B̃×S×H)，Attention 在完整 B̃ 上执行，产生 O(S²) 级别的 KV cache 和 softmax 中间激活，当增大 B̃ 来提升 expert 计算密度时，attention 层的激活内存超出 on-chip SRAM 限制（WSE-2 为 40 GB），导致 OOM 或强制降低 B̃。
    - **系统框架层**：论文未明确说明。Cerebras CS-2 使用其专有软件栈执行训练。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：Attention kernel 在 B̃ 上执行 matmul/softmax，expert MLP kernel 在 B̃ 上按 router 分配后的 token 子集执行。B̃ 增大使 attention 内存溢出，B̃ 减小使 expert 计算密度不足——无法同时满足两者。
    - **硬件架构层**：WSE-2 的 850,000 核心和 40 GB on-chip SRAM。Attention 的激活中间结果（KV/softmax）占用大量 SRAM，限制了 B̃ 的上限；而 expert MLP 需要大的有效 batch 来利用 20 PB/s 带宽和大量核心。G=1 时两阶段使用相同 B̃，形成"batch interface conflict"。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：BTA (Batch Tiling on Attention)，通过 attention 层的 batch 维度 tiling 解耦 attention 和 expert 的 batch size 需求。
  - 全栈执行例子（BTA, G > 1）：
    - **算法层**：输入 X ∈ ℝ^(G×B×S×H)。Attention 以 per-tile batch B 执行 G 次循环（B = B̃/G），每次 attention 的 KV/softmax 中间激活仅对应 B 个序列，降低到激活内存安全范围。G 次 attention 的输出拼接为 B̃ 张量，送入 expert MLP。Router 在 B̃ 张量上执行，expert 以 B̃ 为有效 batch size 执行大 matmul，填满计算核心。
    - **系统框架层**：论文未明确说明。BTA 修改训练循环中 attention 前置的 batch reshaping/tiling 逻辑。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：Attention kernel 在 B 大小的 tile 上执行，每次 tile 的激活中间结果可放入 SRAM；G 个 attention tile 的 kernel 可串行或流水线执行。Expert MLP kernel 在 B̃ = G·B 上执行，获得足够的计算密度。关键区别：attention 和 expert 使用不同的有效 batch size，解决了 G=1 时的 trade-off。
    - **硬件架构层**：同一 WSE-2 硬件。BTA 通过算法层面的 batch tiling 避免了 attention 层因 B̃ 过大导致的 SRAM 溢出，同时保证了 expert 层因 B̃ 足够大而获得的高核心利用率。对比 baseline：G=1 时 128 experts/top_k=1 的 throughput 为 7,091 tokens/s，而 BTA (G=64) 为 49,335 tokens/s，提升约 7×。
  - 核心洞察：BTA 不是通过通信优化（如 FlashAttention、expert parallelism）或 placement 调整来解决批处理冲突，而是通过改变每个阶段处理的 token 数量（attention 少、expert 多）来直接解决 attention 内存 vs expert 计算密度的矛盾。Baseline 缺陷的根因是"attention 的峰值激活内存限制了全局 batch size 的上限，从而限制了 expert 的计算密度"，BTA 通过解耦两阶段的 batch size 直接消除了这一约束。
