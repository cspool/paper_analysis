## MoBA: Mixture of Block Attention for Long-Context LLMs

- baseline方法是什么？
  Baseline 是标准 Full Attention（Transformer self-attention）以及静态稀疏注意力方法（Sliding Window Attention / Attention Sink / 各类 static sparse patterns）和线性注意力方法（Mamba / RWKV / RetNet）。

  全栈执行例子（以 Full Attention baseline, Llama 8B, 1M context, single GPU prefill）：
  
  - **算法层（Full Attention）**：标准 scaled dot-product attention: O = Softmax(QK^T/√d)V。对于 1M context, Q,K,V ∈ R^{1M×128×8}（per head），QK^T ∈ R^{1M×1M}，FLOPs = O(N²d) ≈ 2×10^12 per head。prefill 1M tokens 需 ~30 分钟（FlashAttention applied）。KV cache = 2×L×N×d×h_kv ≈ 2×32×1M×128×8 = 64GB (BF16)。复杂度 O(N²) 在长 context 下计算和内存开销 prohibitive。

  - **算法层（Sliding Window Attention baseline）**：每个 query 仅关注最近 W 个 token（如 W=4096）。可视为 MoBA 特例：gating network 固定选择最近 blocks。缺陷：(a) task-specific——对需要跨长距离检索的任务（如 Needle in a Haystack）性能崩溃；(b) 丢失中间 context 信息，模型无法利用非局部的 key information；(c) 需要对 W 外的 token 做因果关系的信息路由，但缺乏有效机制。

  - **算法层（Attention Sink baseline）**：每个 query 关注初始 token (sink) + 最近 token。可视为 MoBA 特例：gating 固定选择首尾 blocks。缺陷：(a) 同样 task-specific；(b) 丢弃中间 token 可能包含关键检索信息；(c) "为什么初始 token 重要" 缺乏充分理论基础。

  - **算法层（线性注意力 baseline: Mamba/RWKV）**：将 Softmax attention 替换为线性近似 O_t = Q_t Σ_{i=1}^t K_i^T V_i。缺陷：(a) 与现有 Transformer 预训练模型不兼容——转换成本高（H. Liu et al. 2023）或需从头训练（A. Li et al. 2025）；(b) 复杂推理任务上的有效性缺乏充分证据。

  - **kernel调度层（FlashAttention baseline）**：tiled online softmax attention kernel，O(N²) FLOPs 但 O(N) memory。长 context 下仍受限于 O(N²) 计算量。1M prefill 需 30 分钟。
  
  - **系统框架层**：PyTorch + FlashAttention + HuggingFace Transformers。
  
  - **编译框架层/硬件架构层/芯片设计层**：论文未明确说明。

  Baseline 核心缺陷总结：
  1. Full Attention O(N²) 计算复杂度导致长 context 训练和推理成本 prohibitive
  2. 静态稀疏 attention（SWA/Attention Sink）task-specific，缺乏内容感知能力，无法自适应不同输入
  3. 线性注意力与现有 Transformer 生态不兼容，复杂推理能力未充分验证
  4. 现有动态稀疏方法（Quest, MInference）仅优化推理而非训练，无法降低长 context 训练成本

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MoBA 将 MoE 的 "专家路由" 原理从 FFN 层迁移到 attention 层，通过 block-level top-k gating 实现内容感知的动态稀疏注意力，同时保持与 full attention 的参数等价性（0 参数增量），支持训练和推理阶段的无缝切换。

  MoBA 全栈执行例子（以 Llama-8B-1M-MoBA, 1M context, block size=4096, top-k=12 prefill）：

  - **算法层（核心创新——Block Partitioning + MoE-style Routing）**：
    1. **Block Partitioning**：将 1M context 划分为 n = 1M/4096 ≈ 244 个 block，每 block 4096 tokens。K, V 按 block 划分后 mean_pool 得到 block-level key representation K̄ ∈ R^{n×h×d}（每 block 的 4096 个 K vectors 的均值）。
    
    2. **Gating Score**：每个 query q 与 K̄ 中 n 个 block representation 做内积 s_i = ⟨q, mean_pool(K[I_i])⟩，得到 query-to-block affinity scores S ∈ R^{N×h×n}。计算量为 O(N·n·d) vs Full Attention O(N²·d)，n=N/B ≪ N。

    3. **Top-k Gating + Causality**：在 S 上施加 causal mask（future blocks = -∞）后取 top-k（k=12），每个 query 仅关注 (k+1) 个 blocks（k 个历史 + 1 个当前）。Sparsity = 1 - 4096×13/1M = 94.7%。

    4. **Hybrid Design**：MoBA 与 full attention 参数等价（无参数增减），支持：
       - 两阶段训练：90% tokens MoBA + 10% tokens Full Attention → 接近 full attention 的 loss
       - Layer-wise hybrid：最后 3 层 full attention + 其余 MoBA → SFT 性能显著恢复
       - 推理切换：prefill 用 MoBA（快速处理长 prompt），generation 用 full attention（保证生成质量）

    5. **Fine-Grained Block Segmentation**：类似 MoE 的 fine-grained expert segmentation，将 32K context 从 8 blocks 细分至 128 blocks（维持 sparsity 75%），性能提升 ~0.01 LM loss。

  - **kernel调度层（FlashAttention + MoE 融合）**：
    1. **Block-based query grouping**：根据 top-k gating 结果将 queries 按分配的 KV blocks 重排分组（类似 MoE 的 token dispatch）
    2. **Varlen FlashAttention**：对每个 (query_group, kv_block) 对使用 FlashAttention varlen 分别计算 block-wise attention
    3. **Online Softmax Combining**：将 self-attention output（当前 block）和 MoBA output（历史 blocks）用 online softmax tiling 合并
    4. **Tensor Parallelism for Extreme Length**：将 K/V broadcast 到不同 query heads 解决 10M context 显存限制
    
    Speedup: 1M context → 6.5× (vs FlashAttention), 10M context → 16× speedup. 复杂度 sub-quadratic.

  - **系统框架层**：
    基于 PyTorch + FlashAttention + DeepSpeed-MoE。MoBA layer 可直接替换标准 attention layer，无需修改模型其他部分。训练和推理使用同一套代码，MoBA/full attention 动态切换。

  - **编译框架层/硬件架构层/芯片设计层**：论文未明确说明。

  对应解决 Baseline 缺陷的设计-缺陷映射：

  1. **O(N²)→O(k·B·N) → 解决缺陷 1（Full Attention 计算量）**：通过 block-level top-k routing，每个 query 仅关注 (k+1)B tokens 而非 N tokens。例如 1M context, k=12, B=4096 → 仅关注 ~53K tokens（5.3%），计算量降低 ~20×。

  2. **Content-Aware Gating → 解决缺陷 2（静态稀疏 task-specific）**：gating score s_i = ⟨q, mean_pool(K[I_i])⟩ 是 query-dependent 且 content-dependent 的——不同 query 根据自身语义选择不同的历史 blocks。SWA 和 Attention Sink 被证明是 MoBA 的特例（gating 固定选择最近/首尾 blocks），MoBA 表达力更强且可自适应学习。

  3. **参数等价 + 无缝切换 → 解决缺陷 3（线性注意力不兼容）**：MoBA 不改变 Transformer 架构，不引入或删除参数。现有预训练模型可无痛转换（全 attention→MoBA），训练中可动态切换。已部署于 Kimi 长 context 请求服务。

  4. **训练+推理双重加速 → 解决缺陷 4（仅推理优化）**：与 Quest/MInference 等仅推理优化的方法不同，MoBA 同时降低训练计算量——scaling law 实验（5 个模型规模, Chinchilla scaling）证明 MoBA 的训练 loss 与 full attention 高度一致（差值 < 1e-3），但训练 FLOPs 大幅降低。

  5. **"Less Structure" 原则**：MoBA 让模型自己学习 attention pattern，而非预设固定结构（SWA/Sink/Strided）。这符合论文的核心理念：attention 稀疏性应由数据驱动而非人工设计。
