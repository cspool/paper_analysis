## PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling

- baseline方法是什么？
  Baseline 是所有层使用固定相同 KV cache size 的压缩方法：H2O（基于累积 attention score 动态淘汰，保留 recent + heavy hitter tokens）、SnapKV（基于 instruction token attention score 选择/clustering 重要 KV positions）、StreamingLLM（仅保留初始 sink tokens + 局部 window）。核心缺陷：(a) **跨层 uniform cache budget**：所有 baseline 对每一层分配相同 KV cache size，忽略底层 attention 分散（需要更多 cache 覆盖全局信息）和高层 attention 集中（少量 cache 即可）的差异；(b) **底层信息丢失**：底层 uniform small budget 下，分散的 attention 中许多关键 token 被错误淘汰，高层虽然 attention 已集中但 uniform budget 仍保留大量不重要 token，资源浪费。

  全栈执行例子（H2O baseline, LLaMa-3-8B-Instruct, 8K context, A100, KV size=64）：
  - **算法层**：prefill 时对所有 32 layers 每层每 head 保留最后 α 个 instruction token + 选出 top-64 个 heavy hitter tokens（基于累积 attention score）。底层 layer 0 仅保留 64 tokens（占 8K 的 0.8%），大量分散在全局的 attention 信息被丢弃。顶层 layer 31 同样保留 64 tokens，但此时 attention 已集中在极少数关键 token 上，保留的 64 tokens 中包含许多不必要的 token。总计 KV cache memory = 64 × 32 layers × 2 (K,V) × d_model × 2 bytes(fp16)。
  - **系统框架层**：HuggingFace Transformers 推理 pipeline 或 vLLM paged attention。KV cache 为 contiguous tensor，每层 uniform eviction。与 FlashAttention-2 兼容。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：FlashAttention-2 加速标准 attention 计算。Eviction 操作（sort, top-k, gather）在 GPU 上通过 PyTorch 算子执行（torch.topk, torch.gather），无自定义 kernel。
  - **硬件架构层**：NVIDIA A100 GPU，fp16 精度。8K context 下 Full KV cache 占用 ~6848M，64 KV size 下仅 ~428M（6.3%）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  PyramidKV 通过观察 LLM 中跨层的"Pyramidal Information Funneling"注意力模式（底层→均匀分布/广播模式，中层→局部聚拢，顶层→massive attention 集中于少量关键 token），提出跨层不均匀 KV cache budget 分配和 attention score-based token 选择。

  **(1) 算法层——跨层 Pyramid-Shaped Budget Allocation 解决 uniform budget 缺陷**：
  核心思想：底层 attention 分散 → 分配更多 cache budget；顶层 attention 集中 → 分配更少 cache budget，形成金字塔形分配。

  Budget 分配公式（arithmetic sequence）：
  ```
  k^{m-1} = k^{total} / (β·m)         # 顶层（最少）
  k^0 = 2·k^{total}/m - k^{m-1}       # 底层（最多）
  k^l = k^0 - (k^0 - k^{m-1})/(m-1) × l  # 中间层 linear decay
  ```
  超参 β=20 控制顶层陡峭程度，α=8 为各层固定保留的 instruction token 数。**Token 选择**：每层每 head 保留 instruction tokens + top-(k^l - α) 个按 attention score（来自 instruction tokens 的 attention sum）排序的最高分 token。

  **对比 Baseline 的改进**：
  - vs H2O/SnapKV/StreamingLLM (uniform budget)：PyramidKV 底层多分配 cache（layer 0: ~2× 平均 budget），充分保留分散注意力所需的信息；顶层少分配 cache（layer 31: ~0.1× 平均 budget），仅保留 massive attention 集中的关键 token。LongBench avg: KV size=64 时 PyramidKV 34.76 vs H2O 33.89 vs SnapKV 33.05 (LLaMa-3-8B)；KV size=128 时 PyramidKV 37.25 vs H2O 35.37 vs SnapKV 35.50。
  - 极端压缩下优势更显著：TREC 任务 KV size=64 时 PyramidKV 58.00 vs H2O 38.00/SnapKV 38.50（+20.5/19.5 pp）。
  - Needle-in-a-Haystack：LLaMa-3-70B KV size=128 时 PyramidKV 100.0 Acc = FullKV 100.0，vs H2O 82.3 / SnapKV 98.6。

  **(2) Arithmetic Sequence 选择优于其他衰退策略**：
  论文 ablation 比较了线性(arithmetic)、几何(geometric)、指数(exponential)衰退以及自适应分配策略（entropy-based, Gini coefficient-based）。线性策略 LongBench avg=34.76 优于几何(34.36)和指数(34.23)，且远优于 entropy(32.71)和 Gini(32.58)。论文认为线性衰退与观察到的注意力模式的自然渐进收窄更吻合，计算开销最小（budget 一次性预计算，无在线 entropy/Gini 计算开销）。

  **(3) vLLM 集成——Per-Layer Block Table 解决 cache fragmentation**：
  Naive vLLM 实现中不同层不同 budget 导致小 chunk 内存分配/释放/移动/访问的碎片化和低效。PyramidKV 将每个 sequence 的 block table 扩展为 per-layer block table，使得每层独立 page-out KV cache，避免固定内存偏移限制。Throughput 显示 compression 下相对 throughput 随 input context length 增加而降低（因新 sequence 需等 decoding batch 加入），需进一步优化。

  全栈执行例子（PyramidKV on LLaMa-3-8B-Instruct, 8K context, A100, KV size=64 avg）：
  - **算法层**：(a) 预计算 32 layers 的 budget：底层 layer 0 ~100 tokens，layer 31 ~10 tokens（instruction tokens 不算在内）；(b) Prefill 阶段计算 attention scores，各层选 top-k^l tokens + 8 个 instruction tokens；(c) torch.gather 执行非 in-place eviction，释放原 tensor。总计 KV cache memory 与 baseline uniform 64 相同（平均 budget 相等），但分配更匹配信息流。
  - **系统框架层**：HuggingFace Transformers 即插即用（无需 training/fine-tuning）。论文开源实现：https://github.com/Zefan-Cai/PyramidKV。vLLM 集成通过 per-layer block table 实现。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：FlashAttention-2 标准 attention。Eviction 使用 torch.gather（非 in-place，需额外临时 tensor），allocation time < 0.000006s（占总 inference time 可忽略），selection time ~0.013s。PyramidKV 延迟与 H2O/SnapKV/StreamingLLM 可比（e.g., prompt 4096 + gen 4096: PyramidKV 138.87s vs H2O 139.87s vs SnapKV 138.57s）。
  - **硬件架构层**：NVIDIA A100 GPU，fp16。KV cache size=2048 时 memory=1712M（25% of Full 6848M），performance match/exceed FullKV（LongBench avg 41.49 vs FullKV 41.46 on LLaMa-3-8B）。
