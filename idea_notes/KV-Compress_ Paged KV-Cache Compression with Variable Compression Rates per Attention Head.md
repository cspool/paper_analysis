## KV-Compress: Paged KV-Cache Compression with Variable Compression Rates per Attention Head

- baseline方法是什么？
  Baseline 是现有 KV cache eviction 方法（H2O、SnapKV、PyramidKV、Ada-SnapKV、Ada-PyramidKV），其全栈执行例子如下：
  - **算法层**：H2O 基于全部过去 queries 的累积 attention score（A2S）识别 "heavy-hitter" KVs 并 evict 低分 KVs，uniform eviction rate 跨所有 heads；SnapKV 使用有限 observation window（w=8）内的 queries 聚合 attention + max-pooling（p=7），保留 top-N high-attention KVs，同样 uniform eviction across heads；PyramidKV 使用 SnapKV 的 observation window 但按金字塔形分配各层 eviction rate（浅层少 evict、深层多 evict）；Ada-SnapKV/Ada-PyramidKV 允许每 head 可变 eviction rate（cross-head eviction），但该方案在现有推理框架中仅增加 cache 碎片化而无法实际减少物理内存占用。所有 baseline 的共同缺陷：(a) 针对 MHA 模型设计，在 GQA 模型实现中先将 KV cache repeat 到 query head 数量再进行压缩，导致 cache 中 3/4 (Mistral/Llama-3 gqa ratio r=4) 的 KVs 为重复数据，压缩效率低下——需压缩率超过 r 才能改善已有 GQA 提供的压缩效果；(b) Uniform eviction rate across heads 忽略不同 attention head 对 KV cache 的异质性需求，限制了理论压缩率；(c) Ada-SnapKV 虽提出 variable-head-rate 但缺乏使碎片化被 PagedAttention 管理的实现。
  - **系统框架层**：HuggingFace Transformers 或 PyramidKV 实现（https://github.com/IsaacRe/PyramidKV），在标准 attention 计算后执行 KV selection/index-gather。非 paged attention，KV cache 为 contiguous tensor，uniform eviction 通过缩小 L 维度实现内存节省。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：FlashAttention-2 加速标准 attention。Eviction 操作（sort, top-k, gather）在 GPU 上通过 PyTorch 算子执行。
  - **硬件架构层**：NVIDIA L4 / H100 GPU（baseline 评测）；NVIDIA A100（Ada-SnapKV/PyramidKV 原论文评测）。长上下文下 KV cache 内存为 throughput bottleneck。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  KV-Compress 通过三个核心设计解决 baseline 缺陷：

  **1. Query-Group Compression + Non-repeated GQA Cache → 解决缺陷(a)**：
  将 KV eviction metric 的聚合范围从 "所有已 repeat 的 query heads" 改为 "key 所属 query group 内的 queries"：M_{h_k,j} = Σ_{h∈H_k} Σ_i (A_{h,i,j})^2，其中 H_k = {h: r·h_k ≤ h < r·(h_k+1)}。直接在非 repeat 的 GQA KV cache（shape H_kv × L × d）上执行压缩，而非先 repeat 到 H_q × L × d。对于 Mistral/Llama-3 (r=4)，同样 max-cache-size C 下 KV-Compress 持有 1/4 的 KVs——即实现 4x 额外有效压缩率，LongBench 上以 1/4 KVs 达到 state-of-the-art。

  **2. Paged Block Eviction + Variable-Head-Rate 实现 → 解决缺陷(b)(c)**：
  PagedAttention 扩展：将 block 从 "每个 block 存储所有 layer×all heads 的 KVs" 改为 "每个 block 仅存储单 head 的 KVs"，block table 扩展为 B×l×H×L_max/b。这使得不同 head 可以有不同数量的 allocated blocks——variable-head-rate eviction 可以实际释放 evicted blocks 的物理内存。MoveCache 算法重排物理 cache 使得被 evicted 的 blocks 在物理上连续可释放。Block eviction 选择：跨 head 排序候选 block eviction（按每 block 最大 eviction metric），选择总 metric 最低的 E_s blocks 进行 eviction。Variable per-layer compression rate 通过相同机制实现。

  **3. Squared Attention (L2) Metric → 改善 eviction 质量**：
  使用 Σ(A_hij)² 替代 ΣA_hij 作为 eviction metric——等价于最小化未来 attention 的 L2 error 而非 L1 error。在 LongBench 所有 max-cache-size 和所有变体（KVC-w, KVC-full）中 L2 一致优于 L1。

  全栈执行例子（KV-Compress on Llama-3.1-8B-Instruct, vLLM modified, compression rate 32x, L4 GPU）：
  - **算法层（核心创新）**：
    (a) Prefill 阶段：正常计算 attention → 对 observation window w=8 的 queries 计算 squared attention 聚合（GQA group-wise）→ max-pooling p=7 得到 M_{h_k,j} → 跨 (head, seq_len) 排序 → 按 head 分组 reshape 为 [N, b] → 每 head 各 block 的最大 eviction metric m(h,e) → 跨 head 排序候选 eviction blocks → 选择 E_s blocks → MoveCache 重排物理 cache → 释放 E_s 个 blocks。
    (b) Decoding 阶段：每 step 累积新 token 的 squared attention → 按需（preemption 即将发生时）触发基于 updated metric 的再次压缩。
    (c) 调度：prefill 后 + preemption 前压缩，实现 dynamic cache size management。
  - **系统框架层**：vLLM v0.6.0 修改版（开源 https://github.com/IsaacRe/vllm-kvcompress），GPU 端 block manager 替代 CPU scheduler（消除 l×H 倍 block 增长的 CPU 调度开销），block table 扩展支持 per-head per-layer 索引，unified physical KV cache K_u, V_u ∈ R^{N×16×d}。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：PyTorch sort API 用于 metric 排序（主要 overhead，额外内存 ~8× sorted tensor 大小）。PagedAttention kernel 通过修改后的 block table 索引读取 per-head KVs。Eager mode（no CUDA graph）。
  - **硬件架构层**：NVIDIA L4 24GB 和 H100 80GB。GPU block manager 利用 SIMT parallelism 实现 block 分配和释放的并行化。

  **对比 baseline 的关键差异**：
  - Baseline 所有 head uniform eviction → KV-Compress variable per-head per-layer eviction，实际释放内存（非仅碎片）
  - Baseline GQA 先 repeat KV 再压缩 → KV-Compress query-group compression 在非 repeat cache 上压缩（4x 额外效率）
  - Baseline L1 attention aggregation → KV-Compress L2 squared attention aggregation（LongBench 平均分 KVC-w8-L2 vs KVC-w8-L1: +1.3 pp for Llama-8B C=128）
  - KV-Compress 8B C=128: avg score 46.26（state-of-the-art）vs SnapKV 45.93, PyramidKV 45.97（且仅使用 1/4 KVs）
  - Llama-3.1-70B-FP8 64x compression: 多数 non-summarization 任务保持 >90% full-cache 性能
  - Throughput on L4, L_c=6000: 4.93x (32x) / 5.18x (64x) over vanilla vLLM
  - Summarization 任务（GovReport, QMSum）对压缩最敏感，低压缩率即显著退化
