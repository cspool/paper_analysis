## Less Is More: Fast and Accurate Reasoning with Cross-Head Unified Sparse Attention

- baseline方法是什么？
  Baseline 是现有的免训练稀疏注意力方法（TidalDecode、Quest、StreamingLLM）和需训练方法（SeerAttention-r）。这些方法的全栈执行例子如下：
  - **算法层**：TidalDecode 采用 per-head 独立 token 选择——每个 attention head 基于各自的 attention scores 独立选择 top-k token 子集，在不同层间进行周期性重新选择，各 head 维护独立的 token 索引。Quest 使用 hybrid attention layers 和 chunk-based block size（16/32），在所有层应用稀疏 attention。StreamingLLM 保留 attention sink（初始 token）+ 固定大小的 sliding window（最近 token），丢弃中间 token。这些方法的核心缺陷：(a) per-head 独立选择假设 attention heads 需要完全不同的 token 子集，但推理中 token 重要性实质上高度跨 head 重叠（cross-head spatial locality），导致选择效率低下和 selection error 的 head 间不一致传播；(b) 局部选择策略（per-head / per-layer / per-step）在成千上万个 decoding step 中产生累积误差——attention recall 从 ~95% 逐步退化至 ~65%（图 1a），导致推理链逻辑不一致和生成长度膨胀（表 2：TidalDecode 2K budget 生成 17.4K vs Full Attention 14.8K）；(c) StreamingLLM 的固定大小滑动窗口不支持按 token budget 比例适配不同近邻需求。
  - **系统框架层**：基于 HuggingFace Transformers 推理 pipeline，采用 FlashInfer attention kernel。TidalDecode 需要 per-head token 索引管理，Quest 需要 block-based sparse attention mask。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：FlashInfer attention kernel。TidalDecode/Quest 在 GQA 模型下需要 per-head 独立的 KV cache 子集加载——同一 KV group 的不同 query head 选择不同 token 集合时，需加载更多 token 的 KV，导致 global-to-shared memory 传输冗余（TidalDecode: 2.34MB vs LessIsMore: 1.04MB, Table 4）。
  - **硬件架构层**：NVIDIA A100 80GB / A5000 GPU，无专用硬件修改。

  Baseline 核心缺陷总结：
  1. **假设错误**：假设 token 重要性是 head-local 属性，忽略了推理中跨 head 空间局部性和时间近邻局部性
  2. **误差累积**：局部最优的 per-head/per-layer 选择在长程 decoding 中误差循环累积，导致 attention recall 退化、推理链不一致和生成长度膨胀
  3. **KV loading 冗余**：GQA 下 per-head 独立选择导致 KV cache 加载冗余

- 论文方法是什么？如何对应解决Baseline的缺陷？
  LessIsMore 基于一个核心洞察重新设计稀疏注意力：推理模型中 token 重要性是全局属性而非 head-local 属性，由此直接推导出两个设计需求——(a) token 选择必须跨 head 全局一致，(b) token 选择必须跨层稳定且显式保留近邻上下文。

  LessIsMore 的全栈执行例子：
  - **算法层（核心创新——Cross-Head Unified Sparse Attention, CUSA）**：
    1. **跨 head 统一 token 选择 → 解决 Baseline 缺陷 1（head-local 假设）**：
       各 attention head 独立提案 top-k 候选 token（基于精确 attention score P = q·C.K^T），但通过 UnionFlatten 将所有 head 的候选聚合为统一候选集，全局排名后取 top K·(1-r)。关键设计：(a) 不假设 head 功能完全一致，而是利用观察到的 token 重要性跨 head 重叠；(b) 聚合步骤通过多数 head 赞同的方式消除个别 head 的噪声选择，降低 selection variance；(c) 同一 KV group 内所有 query head 共享最终 token 索引 ρ，消除 per-head 独立选择带来的 KV loading 冗余。

    2. **稳定近邻保留 → 解决 Baseline 缺陷 1 的 recency 部分**：
       固定比例 r=0.25 的 token budget 分配给最近 K·r 个 token（而非 baseline 的固定大小 sliding window）。此设计直接源于观察：近邻 token 占总关键 token 的比例在 decoding 全程保持稳定，因此比例性分配比固定窗口更好地适应不同 token budget 和序列长度。

    3. **低频 token 重选 → 解决 Baseline 缺陷 2（误差累积）**：
       Token 选择仅在一层（token selection layer，如 Layer 12）执行，产生的统一索引 ρ 跨后续所有 Sparse Attention Layers 复用。图 4 验证：CUSA 仅在 Layer 2 选择 vs 每层都选择，attention recall 几乎无差异（~95% vs ~96%），而 per-head 方法从 ~96% 降至 ~65%。原因：CUSA 的全局 token 重要性跨层高度稳定（由 cross-head spatial locality + temporal recency locality 共同保证），局部方法的高频重选反而引入更多 noise。
  - **系统框架层**：集成到 SGLang + FlashInfer。三种 layer 类型（Full Attention / Token Selection / Sparse Attention）分层执行，token selection layer 更新 ρ，sparse attention layers 复用，无需 per-head 独立 token 索引管理。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：定制化 FlashInfer kernel 利用统一 token 索引实现更高效的 GQA 稀疏 attention。全 query heads 共享 K[ρ]/V[ρ]（仅 K 个 token），单次 global-to-shared memory 加载，减少 G2S 传输 55%（1.04MB vs 2.34MB for TidalDecode），kernel latency 从 32.1µs 降至 20.1µs（1.6× speedup）。
  - **硬件架构层**：NVIDIA A100 80GB / A5000 GPU，无专用硬件修改。

  **对比 baseline 的关键差异**：
  - **Baseline**（TidalDecode）：per-head 独立选择 → 每个 head 维护独立的 token 索引 → GQA 下同一 KV group 的不同 query head 选择不同 token → KV loading 冗余（2.34MB G2S）→ attention recall 退化（~75% at 32K）→ 生成长度膨胀（17.4K vs 14.8K）→ kernel latency 32.1µs
  - **LessIsMore**：跨 head 统一选择 → 全局统一 token 索引 ρ → GQA 下所有 query head 共享 ρ → KV loading 最优（1.04MB G2S）→ attention recall 稳定（~90% at 32K vs ~75% TidalDecode, 图 1a）→ 生成长度几乎无膨胀（15.8K vs 14.8K, 表 2）→ kernel latency 20.1µs

  **效果量化**：
  - AIME-24 on Qwen3-8B: 2K budget 达 73.8% 准确率（vs Full Attention 74.5%, TidalDecode 53.3%, Quest 18.2%）
  - 87.5% sparsity 零精度损失
  - 端到端 decode speedup up to 1.6×（64K context, Figure 6a）
  - Kernel 级 sparse attention speedup up to 1.72× vs TidalDecode（Figure 6b）
  - LongBench 4K token budget avg F1 44.78（vs Full Attention 44.08, TidalDecode 44.56, Quest 42.62）
  - MHA 模型 LongChat-7B-32k on NIAH: 256-token budget 达 100%（vs TidalDecode 99%, Quest 99%, H2O 1%, StreamingLLM 3%）
