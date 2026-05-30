## HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention

- baseline方法是什么？
  Baseline 是 DeepSeek Sparse Attention (DSA) 的 flat token scan indexer，用于 DeepSeek-V3.2 和 GLM-5 的 token-level 稀疏注意力。DSA 包含两个组件：(1) **Token-wise Indexer**：对每层每个 query，用轻量 indexing heads（含 gating weights）对全前缀 L 个 token 逐一打分 I_{t,s} = Σ_j w_{t,j}^I · ReLU(q_{t,j}^I · k_s^I)，选出 top-k 个最高分 token 的索引 T_t；(2) **Sparse MLA**：仅在 T_t 中的 k 个 token 上执行 attention 计算。DSA 的设计使得下游 attention 是稀疏且廉价的（O(Lk)），但 indexer 本身需要扫描全前缀（per-query O(L)，per-layer O(L²)），在超长上下文（128K-1M tokens）下 indexer 从可忽略开销转变为主导瓶颈。block-sparse 方法（如 MoBA、NSA）虽然硬件友好，但 block 级别的粗粒度选择会丢失 token 级别的重要度差异——block 内所有 token 必须整体保留或丢弃，浪费预算在不重要的 token 上且可能遗漏关键 token。

  全栈执行例子（Baseline / DSA on DeepSeek-V3.2）：
  - **算法pipeline**：全前缀 token scan → indexer 逐 token 计算 relevance score I_{t,s} → TopK(k) token selection → Sparse MLA (MQA mode, 单 KV latent entry 共享于所有 query heads) → 输出。复杂度 per-layer O(L²)。
  - **系统框架**：vLLM online serving framework，FP8 精度，支持 continuous batching
  - **编译框架**：论文未明确说明
  - **kernel调度**：TileLang kernel（https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32）实现 DSA indexer。在 A100 上 64K context 时 indexer ~5.6 ms，Sparse MLA ~1.6 ms——瓶颈在 indexer
  - **硬件架构**：NVIDIA A100 GPU

- 论文方法是什么？如何对应解决Baseline的缺陷？
  HISA 将 DSA 的 flat token scan 替换为两阶段层级搜索（coarse-to-fine），通过"先粗筛、后精排"的策略，在保持 token-level 细粒度选择的同时大幅降低索引复杂度。

  **1. Block-level 粗过滤 → 解决 indexer O(L²) 复杂度瓶颈**：
  将前缀划分为大小为 B 的连续 block，每个 block 用 mean pooling 生成一个代表向量 k̃_b^I。query 仅需对 M = ⌈L/B⌉ 个 block 代表打分（而非 L 个 token），选出 top-m blocks。这步复杂度 O(L/B)，将搜索空间从 L 压缩到最多 mB（m ≪ M）。由于 block pooling 信息可增量维护在 KV cache 旁边，额外开销可忽略。

  **2. Token-level 精筛 → 解决 block-sparse 方法的粗粒度缺陷**：
  在粗过滤选出的候选 block 内，使用与原始 DSA 完全相同的 token-level scoring 机制，从 mB 个候选 token 中选出最终 k 个 token。这一步保留了 DSA 的 token-level 细粒度——block 内的 token 不再"全留或全弃"，而是逐 token 竞争。block-sparse baseline（仅 Stage 1 无 Stage 2）在 NIAH（needle 在中间位置）和 LongBench（Synthetic 任务）上显著退化，而 HISA 的 token 精筛弥补了这一差距。

  **3. 即插即用、免训练 → 解决工程落地门槛**：
  HISA 输出与 DSA indexer 完全相同的数据结构（每个 query 的 k 个 token 索引集），下游 Sparse MLA 完全不变，KV cache 布局不变，模型权重不变。可直接替换 DeepSeek-V3.2 和 GLM-5 的 indexer 模块，无需任何 fine-tuning。

  全栈执行例子（HISA on DeepSeek-V3.2, B=128, m=64, k=2048）：
  - **算法pipeline**：
    1. Block 划分: L 个 token → M = ⌈L/B⌉ 个 block，每 block 128 tokens
    2. Block pooling: MeanPool(k_s^I, s ∈ B_b) → k̃_b^I（增量维护）
    3. Stage 1 (Block Filter): query 对所有 block 代表打分 → J_{t,b} → TopK(m) blocks + 强制首尾 → 候选集 Ω_t（≤8192 tokens）
    4. Stage 2 (Token Refine): 在 Ω_t 上用 DSA 机制逐 token 打分 → I_{t,s} → TopK(k=2048) → T_t
    5. Sparse MLA: Attn(h_t, {c_s | s ∈ T_t})（与 DSA 完全相同）
    复杂度 per-layer: O(L²/B + LmB) = O(L²/128 + 8192L) vs DSA O(L²)
  - **系统框架**：vLLM + indexer 模块替换，FP8 精度，无需改 Sparse MLA 和 KV cache
  - **编译框架**：论文未明确说明
  - **kernel调度**：TileLang 实现两阶段 kernel：(a) Block filtering kernel: M × d matmul → TopK；(b) Token refine kernel: mB × d matmul（仅候选 token）→ TopK。在 fixed 8K budget 下第二阶段计算量恒定，更易优化
  - **硬件架构**：NVIDIA A100 GPU

  **对比 baseline 的关键差异**：
  - DSA indexer 扫描全前缀 O(L²) → HISA O(L²/B + LmB)，64K 时加速 2.16×-3.75×
  - Block-sparse (MoBA/NSA) 仅 block 级粗选 → HISA block 粗选 + token 精排，NIAH 和 LongBench 上接近 DSA 质量
  - HISA 的 token 精筛使 block 内低质 token 被剔除（而 block-sparse 全留），关键 token 即使所在 block 排名不高也能通过在候选池中的 token 级竞争被选中
  - 首尾 block 强制保留策略处理 attention sink 和局部上下文，避免关键信息丢失
  - 候选池 mB=8192 远大于输出 k=2048，提供 4:1 过采样率以保证精筛质量
  - block size B 与 top-m 的 trade-off: B 越大粗过滤越快但 block 代理越粗糙，m 越小效率越高但遗漏风险增

  核心创新：HISA 发现 DSA 的瓶颈不在 Sparse MLA 而在 indexer，并通过层级索引将 indexer 的搜索路径从 flat scan 改写为 coarse-to-fine——保留了 token-level 的细粒度稀疏模式，同时将搜索成本从 O(L²) 降至亚二次方。这一设计使 HISA 能够作为 DSA 的免训练 drop-in replacement，直接在 DeepSeek-V3.2 和 GLM-5 上使用。
