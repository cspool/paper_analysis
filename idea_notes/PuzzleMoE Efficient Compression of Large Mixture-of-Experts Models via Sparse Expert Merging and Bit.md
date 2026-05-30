## PuzzleMoE Efficient Compression of Large Mixture-of-Experts Models via Sparse Expert Merging and Bit-packed Inference

- baseline方法是什么？
  - Baseline 方法分为两类：(1) **Expert Dropping**（如 NAEE, STUN）——在 calibration dataset 上评估各 expert 重要性，直接移除被认为不重要的整组 expert 参数。但不同下游任务需不同 calibration data——NAEE 对 commonsense benchmarks 用 C4 校准，对 math tasks 需换 MATH 数据集，且校准数据选择严重影响模型精度。(2) **Expert Merging**（如 HC-SMoE, D2, Sub-MoE）——通过 hierarchical clustering 或多阶段合并（先 clustering 再低秩近似）合并相似 expert。但它们采用 coarse-grained 合并，将整组 expert 权重聚合，破坏了 expert 间的关键区分。D2 和 Sub-MoE 还需要 SVD 分解等重计算操作。
  - 全栈执行例子（Baseline: HC-SMoE, Mixtral-8x7B, 50% sparsity, 2×A100-80GB）：
    - **算法层**：HC-SMoE 基于 expert 输出相似度进行 hierarchical clustering：(1) 在 calibration data 上收集各 expert 的 output activation；(2) 计算 expert 间的 cosine similarity 构建距离矩阵；(3) 使用 agglomerative clustering 将 expert 按相似度层次合并；(4) 每个 cluster 内对 expert 权重做直接平均 W_merged = (W_i + W_j)/2 生成 merged expert。这种方式不区分 shared knowledge（共享权重）和 expert-specific knowledge（专有权重），coarse-grained 平均化导致 MMLU 从 67.9% 骤降至 49.0%（-18.9 points）。
    - **系统框架层**：论文未明确说明 HC-SMoE 使用的推理框架。压缩后加载 merged model checkpoint 进行标准 autoregressive decoding。无 specialized inference kernel——使用标准 PyTorch dense GEMM。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：标准 cuBLAS GEMM kernel。Expert weights 以 Bfloat16 格式存储。50% sparsity（压缩后 expert 数量减半）的 merge 操作完成后，inference 使用标准 dense GEMM 计算，不涉及 sparse 计算或自定义 kernel。但 Baselines 的 mask/sign 存储引入额外 metadata 开销——Lasby et al. (2025) 指出 CSR 格式存 50% unstructured sparsity 无 memory savings。
    - **硬件架构层**：2×A100-80GB（tensor parallelism 分片 expert 权重到两卡）。压缩后仍需 2 GPUs（仅减少 expert 数量的一半 weight，而非减少 attention/embedding 部分）。每个 GPU 上的 GEMM 计算仍为 dense matmul。
  - Baseline 核心缺陷根因：(1) **粗粒度合并**——整组 expert 权重平均化，无法区分共享知识和专家特殊化参数，导致 -18.7% MMLU；(2) **任务依赖**——calibration data 改变显著影响精度（NAEE C4 校准在 GSM8K 仅 41.5%，换 MATH 校准升至 48.7%）；(3) **高压缩成本**——SVD 分解（D2 需 55min）、exhaustive search（NAEE 对 64-expert Deepseek-MoE 需 10^18 次 forward pass 不可行）；(4) **mask 存储开销**——binary mask 的 metadata 存储抵消压缩收益。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：PuzzleMoE 通过两个核心创新解决：(1) **Pairwise Dual-Mask Sparse Expert Merging**——元素级（entry-wise）而非整组 expert 的合并，利用 similarity mask 保留共享知识、saliency mask 保留专家特化参数；(2) **Bit-packed Encoding + Custom CUDA GEMV Kernel**——利用 Bfloat16 的 underutilized exponent bits 嵌入 mask/sign，消除 metadata 存储开销并实现高效推理。
  - 全栈执行例子（PuzzleMoE, Mixtral-8x7B, 50% sparsity, 1×A100-80GB）：
    - **算法层（解决"粗粒度合并"和"任务依赖"的缺陷）**：
      - **Fine-grained Entry-wise Merging**：对 W^i, W^j ∈ R^{d×h} 的每个元素位置 [p,q] 独立决定是平均化还是保留更重要的 expert 权重，而非对整个矩阵做统一操作。
      - **Similarity Mask M^sim**：Δ[p,q] = | |W_i[p,q]| - |W_j[p,q]| | / (|W_i[p,q]| + |W_j[p,q]|)，M^sim[p,q] = 1 若 Δ[p,q] ≤ 0.4。在数值上与 Wanda (Sun et al., 2024) 的 pruning 不同——M^sim 识别的是"两个 expert 都共识认为重要的位置"，而非"对单一 expert 重要的位置"。理论分析（Appendix B.2）证明元素级相似性源于 MoE weights 的 Gaussian 分布特性。
      - **Saliency Mask M^sal**：A_i = |W_i| ⊙ ||X_i||_2（activation-aware importance），M_i^sal[p,q] = 1 若 A_i[p,q] ≥ A_j[p,q]。仅需一次 forward pass 完成校准，且 C4 与 MATH 校准结果等价（GSM8K: 51.7 vs 51.7; Avg Acc: 72.6 vs 72.5），证明 task-agnostic。
      - **对比 baseline**：HC-SMoE 的 coarse-grained averaging 将 MMLU 从 67.9% 降至 49.0%（-18.9pts, 50% sparsity）；PuzzleMoE 降至 65.7%（-2.2pts）。在更难的 reasoning benchmarks（Qwen3-MoE），HC-SMoE 25% sparsity 时 Math-500 从 97.2 降至 24.6、AIME24 从 83.3 降至 0.0；PuzzleMoE 分别保持 96.2 和 71.1。
    - **系统框架层（解决"高压缩成本"的缺陷）**：
      - PuzzleMoE 的压缩流程为 linear time：前向 pass 计算 saliency（O(N_layers × d×h)）→ 元素级 mask 构造（O(d×h) per expert pair）→ merging（O(d×h) per pair）。无需 SVD、无需 exhaustive search。
      - Mixtral-8x7B 压缩仅 2 分钟 vs D2 的 55 分钟 vs NAEE 的不可行。Deepseek-MoE（64 experts）仅 10 分钟。
      - Pairwise grouping 采用随机策略（与 search-based 差异 <0.3% Avg Acc），进一步降低压缩复杂度。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层（解决"mask 存储开销"的缺陷）**：
      - **Bit-packed Encoding**：观察到 Bfloat16 exponent 集中在 [112, 128]（仅需 5 bits 编码 32 个值），通过减去 112 的 shift 操作释放出 3 bits。释放的 bits 用于存储 2 bits mask（M_i, M_j）+ 1 bit sign（S_i 或 S_j，取决于 expert position）。Packed 后 Perplexity 无变化（Mixtral-8x7B: before=4.37, after=4.37）。所有 data 仍在 Bfloat16 格式内，无需额外 metadata。
      - **Custom CUDA GEMV Kernel**：在 data-loading path 上融合 decoding——每个 weight 在从 global memory 加载后、FMA 计算前执行 Algorithm 1 的 bit-level decoding（3-4 条 bit ops）。Decoding 延迟远小于 global memory 读取延迟（~200 cycles vs ~1 cycle for bit ops），因此完全被访存隐藏。
      - **对比 baseline**：baseline 用 CSR 存 50% sparse matrix 无 net memory savings（Lasby et al., 2025）；PuzzleMoE 的 bit-packed 形式无额外 metadata——Mixtral-8x7B 从 2 GPUs → 1 GPU 部署。
    - **硬件架构层**：PuzzleMoE 的推理在 A100 GPU 上运行——压缩后的模型仅需单卡 A100-80GB（从原需 2 卡降至 1 卡），1.28× speedup。Qwen3-MoE 从 2×A100-40GB 降至 1×A100-40GB，1.19× speedup。与 quantization 结合（50% sparsity + 3-bit group quantization）：4.8× 总压缩比，Mixtral-8x7B 仅 -1.7% accuracy drop vs full model。
  - 解决 Baseline 缺陷的方式总结：
    1. **针对"粗粒度合并"**：元素级 dual-mask——M^sim 保留 expert 间共享的 consensus 参数，M^sal 保留各 expert 的独特重要参数。在 -50% experts 时 MMLU loss 仅 -2.2pts（HC-SMoE -18.9pts）。
    2. **针对"任务依赖"**：activation saliency 指标 A_i = |W_i| ⊙ ||X_i||_2 对 calibration data 不敏感——C4 和 MATH 校准结果几乎相同，简化部署无需领域特化。
    3. **针对"高压缩成本"**：single-pass forward（1 次）→ O(d×h) merging，无 SVD/exhaustive search。Mixtral-8x7B 压缩 2min vs D2 55min，Deepseek-MoE 10min。
    4. **针对"mask 存储开销"**：Bit-packed encoding 将 mask/sign 嵌入 Bfloat16 exponent bits——zero metadata overhead。Custom CUDA GEMV kernel 在 data-load path 上 decode，解码延迟被访存完全隐藏。
