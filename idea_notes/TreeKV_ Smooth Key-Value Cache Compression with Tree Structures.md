## TreeKV: Smooth Key-Value Cache Compression with Tree Structures

- baseline方法是什么？
  Baseline 是三类 KV cache 压缩方法：(1) **位置驱动方法（StreamingLLM, LM-Infinite）**：仅保留 initial tokens（attention sinks）和 recent tokens（sliding window），丢弃所有中间 token。缺陷：可能漏掉"预定义区域"之外的重要信息，如位置靠前但与当前生成高度相关的上下文。(2) **全局重要性方法（H2O, TOVA, Scissorhands）**：基于 attention weights 的全局排序（H2O 用累积 attention、TOVA 用最后 token 的 attention、Scissorhands 二值化）选择高分 token。缺陷：产生强烈的**区域偏差（regional bias）**——图 1 显示 H2O 和 TOVA 在特定位置区域集中选择 token，无法覆盖序列全局，导致 KV cache 失去全局视图，损害需要完整上下文的复杂任务。(3) **Prefilling-only 方法（SnapKV, PyramidKV）**：仅优化 prefill 阶段，通过 funnel-like 跨层策略选择关键 token。缺陷：只覆盖一个阶段，decoding 阶段仍需额外策略。

  全栈执行例子（H2O on Llama-2-7B decoding stage, RTX 4090）：
  **算法pipeline**：对生成 step t，标准 QKV projection 后追加新 KV pair。H2O 在每个 head 独立计算 token i 的累积 attention score = Σ_j a_j[i]（a_j 是 step j 的 attention weights vector），对所有历史 token 的累积 attention 做全局排序，保留 top-k。复杂度：每 step 需 O(t) 空间存储 per-head attention scores + O(t log t) 排序。缺陷：由于每次都在全局范围贪心选高分 token，attention scores 相关性强 → 被选 token 在位置空间中集聚在少数区域（如高注意力密集段），未被选区域的好 token 永久丢失。Llama-2-7B 32 layers × 32 heads → 共 1024 个独立缓存策略（因每个 head 独立选择）。
  **系统框架**：HuggingFace Transformers 原生实现，每次 decode step 对所有层同步追加新 KV 并执行独立淘汰。
  **编译框架**：论文未明确说明。
  **kernel调度**：标准 FlashAttention kernel 计算 QK^T attention weights。H2O 的额外开销为 per-head attention score 累积与排序，常驻内存随序列长度线性增长。
  **硬件架构**：论文未明确说明。
  **芯片设计**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  TreeKV 通过三个核心设计解决 baseline 缺陷：

  **(1) Wavelet 分析发现"平滑递增"规律 → 树形淘汰结构解决 Regional Bias**
  对 attention-weighted values 信号做 multi-level Haar wavelet 分解发现：从远到近，token 的信息贡献平滑递增，且与邻居的差异性递增（高频分量增长尤为显著）。这表明上下文存在 smooth transition from coarse-grain (distant) to fine-grain (nearby)。基于此设计"左疏右密"的 tree structure：eviction scope 在 cache 中从远（idx=1）到近（idx=c）循环移动，每次仅在相邻两个 token {idx, idx+1} 中淘汰较低的重要性分者。这保证淘汰均匀分布在整个序列上 → 与 H2O/TOVA 的"在某个区域集中淘汰"形成对比，图 1(c) 的 token distribution map 证实 TreeKV 分布更均匀。

  **(2) 循环 eviction scope 而非全局贪心 → 解决全局排序的计算负担**
  不同于 H2O 每次对所有 token 做全局排序（O(t log t)），TreeKV 每次仅比较相邻两个 token 的重要性（O(1)），且 idx 循环递增保证每轮每个 token 都有被评估的机会。这本质上在 token 间建立了二叉树竞争关系：左子树 vs 右子树逐级向上淘汰低分 token → tree structure 平滑保留各层级的"胜出者"，而非贪心取全局 top。

  **(3) Block-level prefill + observation window query → 统一双阶段**
  大多数方法（H2O 除外）只覆盖 decoding 或只覆盖 prefilling。TreeKV 在 prefill 将 prompt 切分为 blocks，用最后一个 block 做 observation window query 得到各 block importance，再在 block 级别复用 decoding 的树形淘汰。所有 blocks 并行计算。position encoding re-assignment 保证淘汰后位置编码语义连续性。

  **(4) Ablation 证实树结构才是核心，非 attention weight**
  TreeKV_Select_Left_Token 变体（每次固定淘汰左侧 token，完全不用 attention weight）在 PG19 65k token 书上与完整 TreeKV 的 perplexity 差距极小（Figure 5），而两者均远超 H2O → 树结构本身而非 attention-weight-based selection 才是性能来源。

  全栈执行例子（TreeKV on Llama-2-7B decoding, RTX 4090）：
  **算法pipeline**：每 layer 每 head 维护独立的 importance scores（S: 累积 attention、C: 计数）。step t 时：(a) 标准 QKV projection → append cache；(b) 计算 attention a = softmax(qK^T/√d)；(c) 更新 S += a, C += 1；(d) 若 |cache| > c：比较 S_avg[idx] vs S_avg[idx+1]，淘汰较低者，idx = (idx+1) mod c + 1；(e) re-assign position IDs。关键差异 vs H2O：每 step 淘汰 O(1) 比较 vs O(t log t) 全局排序，且淘汰均匀分布在序列 → 长序列下 cache 保留 coarse-to-fine 的信息层次。复杂度：额外 O(c) 存储 per head → 与 H2O 相同量级，但每 step 计算量为 O(1)。
  **系统框架**：HuggingFace Transformers + PyTorch，使用 HuggingFace 原生 LlamaForCausalLM 加载模型，在前向传播的 attention 层中插入 TreeKV cache management 逻辑（无需修改模型权重）。
  **编译框架**：论文未明确说明。
  **kernel调度**：标准 FlashAttention 计算 attention weights。TreeKV 增加的 O(1) per-step overhead（一次比较 + 一次 index 更新）可忽略。无自定义 CUDA kernel。
  **硬件架构**：论文未明确说明。
  **芯片设计**：论文未明确说明。
