## FastTree Optimizing Attention Kernel and Runtime for Tree-Structured LLM Inference

- baseline方法是什么？
  **SGLang 的 radix tree KV cache 共享 + 传统 per-query 分离 attention kernel**：SGLang 用 radix tree 组织全局 KV cache 实现 multi-level prefix sharing，减少 GPU 内存占用以服务更多并发请求。但在 computation 层面，SGLang 仍使用传统 attention kernel（FlashAttention/FlashInfer），将每个 query 的 attention 计算分配到独立的 GPU thread block，各 query 间无数据复用。

  全栈执行例子（Llama-2-7B, batch=128, multi-level system prompt + few-shot learning, H100 GPU）：
  - **模型推理算法层**：标准 scaled dot-product attention。Decoder 每步处理 batch 中所有 query 的最后一个 token（Q 为 vector），attention weight = softmax(QK^T/√d)V。各 query 独立计算，无交互。论文未明确说明算法层修改。
  - **系统框架层**：SGLang v0.2.13，radix tree 组织 multi-level shared KV cache。Key 创新在 memory layout，computational part 只是将其分派到 FlashAttention/FlashInfer 等标准 kernel。论文未明确说明框架层额外修改。
  - **编译框架层**：论文未明确说明。Triton/CUDA kernel 通过 PyTorch 调用链执行。
  - **kernel调度层**：FlashAttention decode kernel：每个 query 单独分配 thread block → 每个 block 独立从 HBM 加载 KV cache → Q·K^T 为 GEMV（matrix-vector）→ softmax → P·V（GEMV）。问题：(i) 共享 KV cache 被不同 query 的 thread block 从 HBM 重复加载，shared memory 无法跨 block 复用（HBM bandwidth ≈ 1/10 shared memory bandwidth）；(ii) decode 阶段 Q 为 vector，attention 退化为 GEMV，无法有效使用 tensor core（FlashAttention 仅 <1% effective computation after padding）；(iii) 每个 query 单独 launch 增加 kernel launch overhead。
  - **硬件架构层**：NVIDIA H100 GPU，无自定义硬件修改。Shared memory 在 SM 内各 thread block 间不共享，导致 KV 冗余加载。

  Baseline 缺陷：
  - (a) **Memory-computation gap**：radix tree 优化了 memory layout（内存复用），但 computation 仍 per-query 分离，无法利用 tree 结构隐式的 query-context 共享关系聚合计算。
  - (b) **Redundant HBM loads**：共享 KV cache 被每个 query 从 HBM 独立加载，浪费 memory bandwidth。
  - (c) **Tensor core underutilization**：decode 阶段 GEMV 无法填满 tensor core 的最小输入 shape 要求（需 padding → wasted computation）。
  - (d) **无 group 级优化**：不同 queries 共享不同长度的 prefix，如何分组才能在 padding overhead、intermediate result overhead、parallelism 之间取得平衡——baseline 不考虑此问题。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **FastTree：tree-structured attention kernel + tree structure-adaptive runtime optimization**。核心设计：(i) greedy heuristic 将 radix tree 转换为最优 context-queries grouping plan——把 KV cache 以 tree 结构为 guid 分区为 shared contexts，将共享同一 context prefix 的 queries 聚合为一个 group；(ii) tree-structured attention kernel 以 Flash-Attn 风格 tile-by-tile 处理各 group——Q tile 在 query dim 并行化（跨 block）、KV tile 在 context dim 串行迭代（block 内），Q tile 聚合后 GEMV→GEMM 使能 tensor core，KV tile 在 shared memory 中被 group 内所有 query 共享复用（消除 HBM 重复加载）；(iii) multi-phase tiling 根据 node 在 tree 中的位置（near-root vs near-leaf）自适应选择 tile size；(iv) long context splitting 在 GPU SM 欠饱和时 split 超长 context 提升并行度。

  全栈执行例子（同样 batch=128, multi-level system prompt, H100 GPU）：
  - **模型推理算法层**：attention 计算逻辑不变（scaled dot-product），但 execution 从 per-query 独立计算变为 group-aggregated 计算。Group 内各 query 共享同一 attention context prefix 部分的 K/V tile，仅在 unique suffix 部分各自计算。论文未明确说明算法层修改。
  - **系统框架层**：FastTree 作为 SGLang plugin——读取 SGLang 维护的 radix tree → runtime 生成 grouping plan → 替换 attention backend。每次 radix tree 结构变化时（新请求 arrival/completion）重新执行 runtime search。CPU preprocessing overhead 被 SGLang 的多步 continuous decoding 摊销。论文未明确说明框架层额外修改。
  - **编译框架层**：论文未明确说明。FastTree 的 attention kernel 用 Triton 实现（Python DSL），无需编译框架修改。
  - **kernel调度层**：核心理念——**从 memory layout-guided computation optimization**。
    Step 1 - Runtime greedy heuristic：BFS 遍历 radix tree → 对每条 parent→child 边比较 SplitKVCost（分离：padding cost + intermediate result cost）和 SplitQCost（拼接：padding cost）→ 贪心选开销更小的边赋值 → 生成 virtual tree → node-centric query aggregation 得出 (context, {queries}) grouping plan。
    Step 2 - Tree-structured attention kernel launch：单 kernel 处理所有 groups。每个 group 内：Q 矩阵 tiles 沿 query dim 分派到不同 block（并行）→ 每个 block 循环 KV tiles（串行，因 softmax 跨 context dim 有 inter-tile dependency）→ BMM1(Q_tile·K_tile^T) on tensor core (GEMM) → online softmax (shared memory) → BMM2(P·V_tile) on tensor core (GEMM) → 写 partial O 和 L 到 HBM。
    Step 3 - Reduce kernel：利用 LogSumExp vectors rescale 各 group 的 partial O_i 后累加得到 final output。
    Step 4 - 优化：靠近 root 的 node query 多 → 大 tile size（如 64）最大化 KV 复用；靠近 leaf query 少 → 小 tile size（如 16）避免 shared memory 浪费；long context 若导致 SM 欠饱和 → split context 增加 parallelism。
  - **硬件架构层**：同一 NVIDIA H100 GPU，无自定义硬件。FastTree 通过 query aggregation + shared memory KV reuse + GEMM tensor core utilization，不再受制于 HBM bandwidth 和 GEMV 的低效。未使用 H100-specific features（TMA 等），可移植到其他 GPU。

  关键设计选择与 baseline 缺陷的对应：
  - **defect (a): Memory-computation gap** → 方案：tree structure-adaptive runtime 将 radix tree 的 memory layout 作为 grouping plan 的输入，使 computation 直接受益于 tree 结构的共享关系。radix tree 边 → binary assignment → virtual tree → grouping plan，memory layout 和 computation 统一在同一 tree representation 下。
  - **defect (b): Redundant HBM loads** → 方案：query aggregation 后，同一 group 内的 K/V tile 只需从 HBM 加载一次到 shared memory，被 Q tile 内所有 query 复用。shared memory bandwidth >> HBM bandwidth，大幅减少 memory transaction。特别在 root node（聚合 query 最多）处效果最显著。
  - **defect (c): Tensor core underutilization** → 方案：query aggregation 使 Q 从 vector 变为 matrix（batch of queries），attention 从 GEMV 变为 GEMM，满足 tensor core 的最小 tile shape 要求。无需 padding 或仅少量 padding。FlashAttention 在 decode 阶段 <1% effective computation → FastTree 显著提升 tensor core utilization。
  - **defect (d): 无 group 级优化** → 方案：greedy heuristic 的 cost model 同时考虑 padding overhead（C_P,q + C_P,c）和 intermediate result overhead（SplitKVCost_R），在 query splitting（更多聚合/更大 tile → 可能更多 padding）和 context concatenation（更少 groups/更少 intermediate results → 可能 query splitting 导致 padding 恶化）之间做 trade-off。实验结果：复杂 deep tree 中 greedy heuristic 比 direct aggregation 快 up to 2.2×。
  - **额外设计：multi-phase tiling** → 根据 node 在 tree 中的层级自适应 tile size，解决 uniform tile size 在 heterogeneous tree（不同层级 node 的 query 数差异大）下的 shared memory waste 问题。DeFT（concurrent work）使用 fixed tile size 导致大量 shared memory 浪费，FastTree 在这方面明显胜出。
