## CompressKV: Semantic Retrieval Heads Know What Tokens are Not Important Before Generation

- baseline方法是什么？
  Baseline 是 SnapKV（代表性 KV cache eviction 方法），其全栈执行例子如下：
  - **算法层**：SnapKV 在 prefill 阶段计算所有 attention head 的 attention scores，使用末尾 observation window（默认 8 tokens）内的 attention scores 通过 clustering（per-head 或 per-GQA-group 的 max-mean pooling）来评估每个 token 的重要性。选择 top-N 高 attention 的 token 保留其 KV cache，其余 evict。所有 head 同等对待——对 GQA group 内的多头 attention scores 求和后统一判断。问题：(a) 当 GQA group 内 Streaming Head 占主导时，仅保留首尾 token 的 KV cache，evict 中间关键 token；(b) 每层使用相同的固定 cache budget，不考虑层间差异。
  - **系统框架层**：基于 PyTorch/HuggingFace Transformers 推理 pipeline。Prefill 阶段计算 QKV → 所有 KV 存入 cache → 计算 observation window 的 attention → voting/select top-N → evict 非重要 KV。Decoding 阶段使用压缩后 cache 进行 attention，新 KV pair 追加。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：使用 FlashAttention-2 加速 attention 计算，eviction 操作在 GPU 上实现为索引选择和内存拷贝。
  - **硬件架构层**：运行于 NVIDIA A100 GPU。

  对于 PyramidKV 和 CAKE（扩展 baseline）：在 SnapKV 的 token 选择基础上增加了层级自适应 cache 分配。PyramidKV 按金字塔形分配（浅层少、深层多），CAKE 使用 attention entropy 和 variance 在线计算层级重要性。问题：(a) 依赖 attention 分布的统计量（entropy/variance），计算开销大且跨模型泛化性差；(b) 仍使用所有 head 的 attention scores 做 token 选择，Streaming Head 主导问题未解决。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  CompressKV 通过两个核心设计解决 baseline 缺陷：(1) SRH 驱动的 token 选择解决"Streaming Head 主导 eviction"问题；(2) Error-Aware 层级自适应分配解决"层级无差异化/依赖 attention 统计量"问题。

  全栈执行例子（CompressKV）：
  - **算法层（核心创新）**：
    (a) **SRH 识别**：不要求 head 的 top-1 attention 精确落在正确答案 token 上（传统 Retrieval Head 标准），而是将 head 在整个 answer span 上的 attention scores 求和作为评估指标。公式：SemanticRetrievalScore(h) = Σ_{t} I[y_t∈A] Σ_{j∈A} a_{t,j}^h。这能捕捉到对 "sandwich" 周边语义相关 token（如 "eat", "a thing"）有高 attention 的 head——这些 head 即使 top-1 attention 不在 "sandwich" 上，仍然具有语义检索能力。
    (b) **SRH 驱动的 Token 选择**：每层仅使用 top-4 SRH（而非全部 head）来判断 token 重要性——对这些 SRH 的 attention scores 在 observation window 上求和、1D average pooling（kernel=5）、取平均后选出 top-N token。因为 SRH 不太受首尾 token 的 "attention sink" 影响，所以选出的 token 更均衡地覆盖了文本中间的语义关键信息，避免了 Streaming Head 主导导致的仅保留首尾 token 的问题。
    (c) **Error-Aware 层级分配**：离线在 LongBench 上模拟极端压缩（每层仅保留 32 tokens，约 0.3%），计算每层 attention output 的 Frobenius norm 重建误差 e^(l) = Σ_t ||O_comp,t^l - O_full,t^l||_F / ||O_full,t^l||_F。跨数据集归一化平均后得到层级重要性分数 ẽ^(l)。在线推理时按 ẽ^(l) 比例分配 cache budget，设置 per-layer 上下界 [m=32, M=3×B_per-layer]。与 CAKE/PyramidKV 不同，该分数离线计算、无需在线 attention 统计量计算、且基于真实压缩误差而非注意力分布统计量——因此泛化性更好。
  - **系统框架层**：基于 PyTorch/HuggingFace Transformers 推理 pipeline，与 SnapKV 的集成方式相同。额外步骤：(a) 推理前加载预计算的 SRH 索引和 ẽ 层级分数；(b) Prefill 阶段 token 选择时，仅聚合 top-4 SRH 的 attention scores 而非全部 head；(c) 各层的 cache budget B_i 由 ẽ 分数和 Algorithm 1 确定，而非均分。与 GQA 和 FlashAttention-2 完全兼容。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：包含自定义 CUDA kernel（`adakv`，位于 `methods/adakv/`），需要单独编译。使用 FlashAttention-2 加速标准 attention 计算。eviction 操作在 GPU 上完成。
  - **硬件架构层**：运行于 NVIDIA A100 GPU，实验覆盖 4K-128K context length。

  **对比 baseline 的关键差异**：
  - Baseline (SnapKV) 使用全部 head 的 attention sum 做 token 选择，Streaming Head 的影响导致仅保留首尾 token → CompressKV 仅使用 top-4 SRH，避免 Streaming Head 主导，更均衡地保留中间关键 token。消融实验：在 SnapKV 基础上加入 SRH Selection → LongBench 准确率从 43.76% 提升至 44.96%（+1.20 pp）。
  - Baseline (SnapKV) 每层固定相同 cache budget → CompressKV 使用 error-aware 层级自适应分配。消融实验：SRH Selection + Layer Allocation → 准确率从 44.96% 进一步提升至 45.43%（+0.47 pp）。
  - Baseline (PyramidKV/CAKE) 使用 attention 统计量（entropy/variance）做层级分配，需在线计算且模型泛化性差 → CompressKV 使用 offline 计算的 Frobenius norm 重建误差，无在线开销，基于真实压缩效果而非代理统计量，跨模型泛化性更好。
  - 极端压缩下优势更明显：LongBench 上 128 KV cache budget 时，CompressKV 领先 CAKE 0.26 pp（Llama-3.1-8B）和 0.52 pp（Mistral-7B）；NIAH 上 256 KV entries（0.07% 容量）达到 90% full-cache 准确率。
