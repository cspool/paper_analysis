## MoEs Are Stronger than You Think: Hyper-Parallel Inference Scaling with RoE

- baseline方法是什么？
  Baseline 是标准确定性 Top-K 路由的 MoE 推理（greedy decoding）。每层 MoE 中，每个 token 仅激活 k 个 expert（router logits → softmax → top-k → FFN），其余 E−k 个 expert 处于闲置状态。全栈执行例子：
  - **算法 Pipeline**：输入 token → embedding → 逐层 Attention + MoE FFN。MoE 层中 router 计算 $\mathbf{R} \in \mathbb{R}^E$ → softmax → TopK 选择 k 个 expert → 各 expert SiLU(W_gate·h) ⊙ (W_up·h) → W_down 投影 → 加权求和 → 残差连接 → 最终 lm_head logits → argmax 取下一 token。整个推理过程完全确定性，每 token 只走一条内部计算路径。
  - **系统框架/Serving调度**：论文未明确说明 baseline 使用的 serving 框架。实验使用标准 PyTorch（HuggingFace Transformers）单 batch 推理，无特殊调度优化。
  - **编译框架/Kernel调度/硬件架构/芯片设计**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？

  Baseline 的核心缺陷：标准 MoE 推理每 token 仅激活 k 个 expert，大量已训练好的 expert（E−k 个）在推理时闲置，模型内部知识未被充分利用。增加 top-k 的 active expert 数量不能奏效，因为模型训练时只见过 k-expert 聚合模式，直接增加 k 会导致训练-推理不匹配。这限制了 MoE 模型的"潜力天花板"——模型参数远多于每 token 实际使用的参数。

  RoE 通过三项设计解决：

  **对应缺陷 1（Expert 利用率低）→ Gumbel-Top-K 随机路由**
  - 在 router logits 上注入可控 Gumbel 噪声：$\text{Indices} = \text{TopK}(\mathbf{R} + \tau \cdot \mathbf{G}, k)$
  - Gumbel-Max 性质保证这是一个从 router 定义的 categorical 分布中无放回采样的过程——高 logit 的 expert 仍更可能被选中，但低 logit 的 expert 也有机会被激活。
  - 每个 token 运行 n 次独立采样 → n 条不同的内部计算路径 → n 个候选 logits → 概率平均聚合。
  - 效果：以概率方式探索了训练期间见过的各种 expert 组合，充分释放 MoE 的"潜力"。

  **对应缺陷 2（计算开销）→ Batched Inference + Clean Cache**
  - 将 n 次独立 forward 合并为单次 batched call，利用 GPU 的 sub-linear batch scaling 减少 wall-clock 时间。
  - Clean Cache：batch 中第一个样本使用确定性路由（τ=0）产生共享 KV-cache，其余样本复用此 cache，无需维护 n 份 KV-cache。内存开销与单样本完全相同。

  全栈执行例子（对比 baseline）：
  - **算法 Pipeline**：输入 token → embedding → 逐层 Attention（共享 KV-cache，sample 0 计算一次，其余复用）→ MoE 层中 router 计算 R ∈ ℝ^E → 对 batch 中 n 个样本分别采样 Gumbel 噪声（sample 0 用 τ=0 确定性路由，sample 1..n-1 用调优后的 τ_l）→ TopK 选择不同的 expert 组合 → 并行 FFN 计算 → 各样本独立残差连接 → 最终 n 组 lm_head logits → softmax 后概率平均 → argmax 下一 token。
  - **系统框架**：论文实现了 custom batched inference（HuggingFace Transformers 级别），非标准 serving 框架修改。
  - **温度搜索**：Optuna TPE 逐任务搜索逐层温度 τ_l，搜索空间受两个启发式约束：仅中段层参与搜索（首尾层 τ=0），温度上限 0.5。
  - **编译框架/Kernel调度/硬件架构/芯片设计**：论文未明确说明。

  **关键量化结果**：OLMoE-7B + RoE (K=32) 达到 10.5B 标准 MoE 的性能水平，同时内存开销减少 25%，每 token 延迟减少 30%。使用 64 samples 时 GPU 内存仅增加 12%，功耗增加 20%。
