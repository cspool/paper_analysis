## MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention

- baseline方法是什么？
  Baseline 是现有的免训练稀疏注意力方法和 full attention。全栈执行例子（以 LLaMA-3-8B-262K, 128K context, 单 A100 推理为例）：

  - **算法层（Full Attention baseline）**：标准 dense self-attention，pre-filling 阶段计算完整的 $A = \text{Softmax}(QK^T/\sqrt{d}) V$。对于 128K context 的输入，$Q, K, V \in \mathbb{R}^{131072 \times 128}$，$QK^T$ 矩阵大小为 $131072 \times 131072$，FLOPs 量级为 $O(S^2 d_h) = O(2.2 \times 10^{11})$。attention 计算占 pre-filling 总延迟的 >90%。对于 1M token 的 prompt，仅在单 A100 上 pre-filling 就需要 30 分钟。

  - **算法层（StreamingLLM baseline）**：固定保留 attention sink（初始 1K tokens）+ 滑动局部窗口（最近 4K tokens），丢弃中间的绝大部分 token。模式等同于论文中的 A-shape pattern（仅 static structured spatial distribution）。核心缺陷：(a) 无法处理需要在中间位置检索的动态信息（如 KV retrieval 任务中 PassKey 在 token 10000-50000 范围时完全失效，Retr.KV 得分从 Full Attention 的 14.4 降至 0.8）；(b) 对 Vertical-Slash 和 Block-Sparse 模式的注意力分配无法覆盖——例如 vertical lines（特定 token 被广泛 attend）和 slash lines（固定间隔的周期性注意力）不在 local window 范围内时会被丢弃。

  - **算法层（InfLLM baseline）**：使用 memory unit 处理流式长序列，128 global tokens + 8K local windows。与 StreamingLLM 类似，在 KV retrieval 等需要非局部信息的任务中表现不佳（Retr.KV=1.2 for LLaMA-3-262K），且有效 context window 仅 4K-8K。

  - **算法层（Ours w/ static baseline）**：在 Vertical-Slash 和 Block-Sparse heads 使用静态稀疏索引。在动态任务（如 KV retrieval）中性能崩溃（Retr.KV 接近 0），证明了 sparse indices 必须动态适配不同输入的必要性。

  - **kernel调度层**：FlashAttention（Triton 实现）——标准的 tiled dense attention kernel。对 $S \times S$ attention matrix 执行完整计算，无稀疏优化。虽然 FlashAttention 通过 tiling 和 recomputation 优化了 HBM 访问，但仍执行 O(S²) 的 FLOPs。

  - **系统框架层**：PyTorch + HuggingFace Transformers 标准推理 pipeline。原始 PyTorch LLaMA 实现在 prompt >50K tokens 时即触发单 A100 OOM。

  - **编译框架层/硬件架构层**：论文未明确说明。

  Baseline 核心缺陷总结：
  1. **静态模式失效**：StreamingLLM 的固定 A-shape pattern 无法覆盖 Vertical-Slash 和 Block-Sparse 模式的注意力分布，在 retrieval、multi-hop 等需要全局上下文的任务中准确率崩溃
  2. **动态性的双重挑战**：(a) attention 分布高度 dynamic（同一位置在不同 prompt 下 attend 的 token 完全不同，top-4K 列在另一 prompt 上 recall 从 96.8% 降至 83.7%）；(b) 但 attention pattern（A-shape/VS/BS 类型）在同一 head 上跨 prompt 保持 consistent
  3. **Top-K 方法 GPU 不友好**：直接 top-K 选择（fine-grained dynamic）在 GPU 上 latency 高，因为非结构化的稀疏索引导致不规则内存访问和低 tensor core 利用率
  4. **在线估计开销过大**：现有动态稀疏方法（如 SparQ Attention）使用 low-rank hidden states 估计注意力模式，开销过大，不适用于长上下文场景

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MInference 的核心洞察：attention heads 的稀疏性虽然动态变化，但其**空间分布模式（pattern type）在跨 prompt 时保持一致**——即每个 head 在不同输入下都表现为同一类模式（A-shape / Vertical-Slash / Block-Sparse），但具体哪些 token 被选中是动态变化的。基于此，MInference 将"模式识别"离线完成，"稀疏索引构建"在线完成，实现了低开销的动态稀疏注意力。

  MInference 的全栈执行例子（以 LLaMA-3-8B-262K, 128K context, 单 A100）：

  - **算法层（核心创新——三步动态稀疏注意力）**：
    1. **离线 Kernel-Aware Sparse Pattern Search → 解决 Baseline 缺陷 1（静态模式）和 缺陷 3（GPU 不友好）**：
       不是使用单一静态稀疏模式，而是对每个 attention head 在 kernel-aware 搜索空间中搜索最优模式（A-shape/VS/BS 之一）及其参数。关键设计：(a) "kernel-aware"——搜索空间中的 FLOPs 以真实 GPU kernel FLOPs 为准（非概念估计），确保搜索结果在 GPU 上有实际的加速效果；(b) 以 attention output recall（$\min |y_i - y|$，$y_i$ 是稀疏 attention 输出，$y$ 是 dense attention 输出）为优化目标，而非 attention score recall；(c) 仅需一条 30K tokens 的 reference sample，15 分钟即可完成搜索，且同一模型的不同 context 长度版本可复用。搜索结果显示 >90% 的 heads 被分配为 Vertical-Slash 模式。

    2. **在线动态稀疏索引近似 → 解决 Baseline 缺陷 2（动态性挑战）**：
       - **Vertical-Slash heads**：仅使用最后 64 个 query 向量（$Q_{[-64:]}$）与完整 K 矩阵计算近似注意力 $\hat{A}$，然后沿垂直和斜线方向求和，取 top-k 垂直列和斜线索引。开销极小（仅 $64 \times S$ 的 matmul，占 5-15% 时间），但准确估计了全 attention matrix 的垂直和斜线分布。
       - **Block-Sparse heads**：对 Q 和 K 做 block_size=64 的 mean pooling，然后在 block 级别计算注意力并取 top-k blocks。利用了 mean pooling 和 matmul 的交换性（$\text{MeanPool}(Q) \cdot \text{MeanPool}(K)^T \approx \text{MeanPool}(QK^T)$），以极少开销近似 block-level 注意力分布。
       - **A-shape heads**：静态稀疏 mask，零开销。

    3. **三种结构化的 GPU 友好稀疏模式 → 解决 Baseline 缺陷 3（GPU 不友好）**：
       放弃 fine-grained top-K 选择（GPU latency 高），改用三种结构化稀疏模式：(a) A-shape——block-level 的 structured static 模式，直接利用 FlashAttention block tiling；(b) Vertical-Slash——混合 block-level（斜线用 $64\times64$ blocks）+ column-level（垂直线用 $1\times64$ blocks），通过 point-range two-way merge 算法高效构建索引；(c) Block-Sparse——$64\times64$ block-level top-K 选择，Block-Sparse FlashAttention kernel 延迟与 block 数量线性相关。

  - **kernel调度层 → 解决 Baseline 缺陷 3（GPU 不友好）和 缺陷 4（在线估计开销）**：
    实现了三个高度优化的 GPU kernel（Triton + PIT + FlashAttention）：
    - **Block-Sparse FlashAttention kernel**：每个 thread block 循环遍历每行的 top-K blocks（64×64），延迟与 block 数量线性相关。1M context 下仅需计算 ~1% 的原始 FLOPs，kernel 级加速 30×。
    - **Vertical-Slash FlashAttention kernel**：混合 kernel——第一部分使用 Block-Sparse FlashAttention 处理斜线 blocks，第二部分使用 PIT（Permutation Invariant Transformation）sparse attention 处理垂直列。PIT 将非连续 column data 通过排列不变变换加载到 dense compute blocks，最大化 tensor core 利用率。
    - **A-shape kernel**：静态结构，使用 FlashAttention 仅计算固定区域的 attention。
    - 1M context 下 kernel 实际稀疏度 >95%（Block-Sparse 和 VS），理论加速 >15×。动态索引构建开销 <25%（大部分被稀疏计算节省的 FLOPs 所覆盖）。

  - **系统框架层**：
    基于 PyTorch + HuggingFace Transformers。做了三项单 A100 优化以支持 1M token 推理：(a) Tensor Splitting——按 head 拆分 Attention、按 sequence 拆分 MLP，保持 GPU 利用率 100%；(b) 消除中间变量——mask logic 在 kernel 内实现 causal mask；(c) 仅计算最后 token 的 LM Head logits（pre-filling 阶段只需最后 token 的 logits）。与 KV cache 压缩方法（SnapKV）兼容，可叠加使用。

  - **编译框架层**：PIT（Permutation Invariant Transformation）——动态稀疏编译器，将稀疏数据加载到 dense compute blocks。论文通过 PIT 实现 column-level sparse attention 的高效计算。论文未明确说明修改了哪些编译框架。

  - **硬件架构层**：NVIDIA A100 80GB GPU。论文未修改硬件架构。

  方法 vs Baseline 缺陷的对应关系：
  1. **静态模式失效** → 三种模式（A-shape/VS/BS）可覆盖所有 attention head 的稀疏分布特征，Pattern Search 确保每个 head 分配到最优模式
  2. **动态性挑战** → 在线动态稀疏索引近似（VS: last_64 query estimation, BS: mean pooling approximation），既捕获了动态性，又保持了低开销
  3. **GPU 不友好** → 结构化稀疏模式（block-level/column-level 而非 fine-grained per-token），三个定制 kernel 在 A100 上实现显著 speedup
  4. **在线估计开销** → 极简估计方法（VS 仅用 64 个 query，BS 仅用 mean pooling），总开销 5-25%，随 context 增长占比下降
