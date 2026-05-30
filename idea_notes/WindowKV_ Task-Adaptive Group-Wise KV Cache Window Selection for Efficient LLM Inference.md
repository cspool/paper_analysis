## WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference

- baseline方法是什么？
  Baseline 是三类 state-of-the-art KV cache 压缩方法：(1) **StreamingLLM (SLM)**：基于 attention sink 现象，保留最前 b-α 个 token 和最后 α 个 token 的 KV cache，所有层使用统一 cache size，但丢弃了中间大量可能包含关键语义的 token；(2) **H2O**：基于 Heavy Hitter 观察（少量 token 贡献大部分 attention score），动态保留最近 token + 历史中累积 attention scores 最高的 token，所有层统一 cache size，但 token 级选择破坏语义连贯性；(3) **PyramidKV (PKV)**：观察到底层 dense attention、顶层 sparse attention 的金字塔效应，按等差数列跨层分配不同 KV cache size（底层多、顶层少），但仍然是 token 级离散选择。三种方法共同缺陷：(a) 逐个 token 选择导致 context 语义碎片化，破坏了人类阅读的窗口级信息处理模式；(b) 对所有任务使用统一压缩策略，没有考虑不同任务（信息定位 vs 信息聚合）对语义上下文的不同需求。

  全栈执行例子（LLaMA3-8B-Instruct + H2O/PyramidKV on A100 40G，KV cache size=2048, context=7950 tokens）：
  **算法pipeline**：LLaMA3-8B-Instruct（32 层 Transformer，GQA，8 KV heads/32 Q heads）。H2O 在每层计算 attention scores 后，对每个 head 维护最近 w 个 token + top-(b-w) 个累积 attention 最高的历史 token，其余 evict。Attention 计算时仅加载保留 token 的 KV → softmax 分布在离散 token 上，相邻 token 的语义关联被打断。PyramidKV 额外根据层编号分配不同预算（底层 1024 tokens，顶层 256 tokens），但仍选离散 token。复杂度：O(n·b) per layer。
  **Serving框架**：HuggingFace Transformers 加载模型，自定义 KV cache manager 在每层 decode 后动态 evict/retain KV entries。论文未修改 serving 框架本身。
  **编译框架**：论文未明确说明。
  **kernel调度**：标准 FlashAttention 或 PyTorch SDPA，无自定义 CUDA kernel。KV cache 通过 tensor indexing 动态裁剪，无 kernel 级优化。
  **硬件架构**：论文未明确说明。
  **芯片设计**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  WindowKV 提出三个关键设计解决 baseline 缺陷：

  (1) **Task-Adaptive Window Selection → 解决语义碎片化和任务无关性**：不再逐 token 选择，而是将 context 切分为固定大小的 review windows（ω tokens/window），以 window 为单位做保留/evict 决策。Window 级别的选择天然保留了 consecutive tokens 的语义连贯性。同时训练 bert-base-cased 分类器将输入任务分为 Information Localization（QA 类，p=ω，保留窗口中所有 token 以理解完整语义）和 Information Aggregation（摘要类，p<ω，从每个窗口中提取 top-p 高注意力 token）。这种任务自适应机制确保不同任务场景下 KV cache 的分配策略与任务语义需求匹配。

  (2) **Observation Window 驱动的注意力打分 → 解决选择标准单调性**：使用最后 α 个 token（observation window）作为 query 端，计算其对 review context 中各 token 的累积注意力 t_j = Σ A_ij。这不同于 H2O 的全 query 平均注意力（容易被 outliers 主导），也不同于 PyramidKV 仅用 instruction tokens 作为 query（可能遗漏 context 中的重要长程依赖）。Observation window 紧邻生成位置，天然携带当前生成阶段最相关的上下文需求。

  (3) **Intra-Group Layer KV Cache Indices Sharing + Dynamic Budget Allocation → 解决计算效率**：利用相邻层 attention 分布的相似性（Jaccard similarity 实验验证），将 m 层分为 H=m/γ 组，仅每组首层执行完整的 window selection。预算按等差数列跨组分配（底层组多、顶层组少），继承 PyramidKV 的金字塔结构优势但应用于 window 级别。实验中 γ=7（Qwen2.5）或 γ=8（LLaMA3），window selection 的计算开销降至原来的 1/γ。

  全栈执行例子（WindowKV on LLaMA3-8B-Instruct on A100 40G，KV cache size=2048, context=7950 tokens）：
  **算法pipeline**：LLaMA3-8B-Instruct 32 层分为 4 组（γ=8）。输入 7950 tokens → 分类器判定任务类型 → 选取 (ω=8, α=16, p=8) for localization 或 (ω=16, α=32, p<16) for aggregation。仅 groups 首层 (layer 0/8/16/24) 计算 full attention A [7950,7950] → observation window [7934:7950] 累积 attention → token scores t_j → window scores s_k → 按 dynamic budget (b^0=704, b^1=576, b^2=448, b^3=320) 选择 top-n windows。组内其余 28 层直接复用首层 indices。结果：prefill 后仅保留 ~2048 tokens 的 KV cache（12% of 原 7950 tokens），LongBench 平均分 41.35 vs FKV 41.51（差距 0.16），Needle-in-a-Haystack 超越所有 baseline。
  **Serving框架**：HuggingFace Transformers + PyTorch，自定义 KV cache pruning module 在 prefill stage 后执行 window selection。Throughput test 显示 Vanilla+WindowKV+Classifier 吞吐 881 token/s vs Vanilla 764 token/s（+15%），延迟 1.14 ms/token vs 1.31 ms/token（-13%）。
  **编译框架**：论文未明确说明。
  **kernel调度**：标准 PyTorch tensor operations（attention score computation + gather/scatter for KV cache selection）。无自定义 CUDA kernel。Window selection 的额外 overhead 主要来自组首层的 full attention computation 和 top-k ranking，但因仅 4/32 层执行，overhead 可控（Classifer overhead 在 throughput test 中仅 ~13 token/s 下降）。
  **硬件架构**：论文未明确说明。
  **芯片设计**：论文未明确说明。
