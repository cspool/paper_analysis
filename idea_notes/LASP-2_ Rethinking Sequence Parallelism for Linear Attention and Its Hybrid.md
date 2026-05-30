## LASP-2: Rethinking Sequence Parallelism for Linear Attention and Its Hybrid

- baseline方法是什么？
  Baseline 包括 Ring Attention、Megatron-SP（针对标准 attention 的 SP）和 LASP-1（专门针对线性注意力的 SP）。

  Ring Attention / Megatron-SP / LASP-1 的全栈执行例子：
  - **算法层**：Ring Attention 将序列切分后使用 ring-style P2P 通信逐设备传递 K、V blocks，每个设备在收到相邻设备的 KV block 后计算局部 attention 输出。Megatron-SP 在标准 attention 上实现类似的 ring-style 通信重叠。LASP-1 针对线性注意力做了定制化：使用 ring-style P2P 通信在各设备间顺序传递 memory state M_t（d×d 大小），每步（共 W-1 步）执行一次 send & receive → 计算 O_{t,inter} → 更新 M_t 的顺序操作。
  - **系统框架层**：Ring Attention / LASP-1 基于 Megatron-Core，使用 NCCL P2P send/recv 原语，ring 拓扑按 rank 排列。
  - **编译框架层**：论文未明确说明。
  - **Kernel 调度层**：Ring Attention 使用 FlashAttention-2 kernel 做局部 attention；LASP-1 使用 Triton 加速线性注意力计算。P2P send/recv 通信需逐个 launch 大量小算子。
  - **硬件架构层**：DGX-A100 集群（NVSwitch 600 GB/s 互联），无专用硬件修改。

  Baseline 核心缺陷：
  1. **Ring-style 通信导致计算并行度低**：LASP-1 的 ring-style P2P 通信需要按顺序从 rank i-1 接收 M_{t-1} → 计算 O_t → 更新 M_t → 发送 M_t 到 rank i+1，这 W-1 步完全串行，导致后续设备大量空闲等待。
  2. **通信-计算 overlap 困难**：大量细粒度 P2P send/recv 算子使得通信与计算的重叠调度复杂且低效，实际 overlap 程度远低于理论值。
  3. **通信步骤过多**：每 iteration 共 2(W-1) 个通信步骤（forward W-1 + backward W-1），随设备数线性增长，在大规模集群中通信开销显著。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  LASP-2 通过重新设计通信-计算工作流，用单次 AllGather 集合通信替代 ring-style P2P。

  LASP-2 的全栈执行例子：
  - **算法层**：将序列切分后各设备并行计算 Q_t, K_t, V_t 和 local M_t = K_t^T V_t，然后通过**单次 AllGather 集体通信**将所有 M_t 同步到所有设备（而非依次传递）。各设备本地累加 M_{1:T} = Sum([M_t]_1^T)，再本地计算 O_t = Q_t M_{1:T}。通信量仍为 BHd^2，但通信步骤从 2(W-1) 降至 2（forward 1 + backward 1）。对有 causal mask 的情况，采用计算分解：intra-chunk 保持 quadratic 左乘，inter-chunk 用线性右乘，且 AllGather 与 intra-chunk 计算可通过不同 CUDA stream overlap。
  - **系统框架层**：基于 Megatron-Core 0.9.0，使用 NCCL AllGather 集体通信原语替代 P2P send/recv，通信组基于 SP group。支持与 Tensor Parallelism (TP)、Pipeline Parallelism (PP)、Data Parallelism (DP/ZeRO/FSDP) 混合使用。LASP-2H 对混合模型的 standard attention 层使用 AllGather K_t, V_t 的 Context Parallelism。
  - **编译框架层**：论文未明确说明。
  - **Kernel 调度层**：Triton 2.3.1 加速 GPU 上的线性注意力计算（chunked intra-chunk attention）。FlashAttention-2 用于标准 attention。AllGather 为 NCCL 高度优化的集体通信算子，相比 P2P 更易于与计算 overlap（单次大粒度通信 vs 多次小粒度通信）。
  - **硬件架构层**：DGX-A100 集群，无专用硬件修改。

  **对应解决 Baseline 缺陷的具体设计**：

  1. **单次 AllGather 替代 Ring P2P → 解决串行依赖**：LASP-1 需要 rank i 等待 rank i-1 完成并发送 M_{t-1} 后才能开始计算，形成严格的串行链。LASP-2 通过 AllGather 一次性地将所有设备的 M_t 并发同步到所有设备，消除了逐设备传递的串行依赖。通信步骤从 2(W-1) 降至 2，计算并行度从"逐个设备串行"变为"全部设备并行"。

  2. **通信粒度从细粒度 P2P 变为单次大粒度集体通信 → 解决 overlap 困难**：LASP-1 需要 launch W-1 次小粒度的 send/recv 算子对，调度复杂且有大量 kernel launch 开销。LASP-2 仅需 1 次 AllGather，通信粒度大、调度简单，在有 mask 的场景下可直接与 intra-chunk 计算 overlap（不同 CUDA stream 并发执行 line 7 AllGather 和 line 8 intra-chunk 计算）。

  3. **Memory state 通信量与序列长度无关 → 长序列场景优势放大**：M_t ∈ R^{d×d} 的大小仅取决于 hidden dim，与 chunk/sequence 长度无关。在序列长度 2048K 时，通信数据量不变，但计算量（intra-chunk quadratic 部分）随 chunk 大小增长，因此通信-计算比进一步降低，LASP-2 的优势更加显著。

  4. **实际效果**：在 64 A100 GPU、序列长度 2048K 上，LASP-2 比 Ring Attention 快 36.6%，比 LASP-1 快 15.2%。序列长度 ≥64K 时优势开始显现，序列越长越显著。LASP-2 支持线性扩展：每 GPU 内存使用恒定（~25.6-57.8 GB）下，增加 GPU 数量即可扩展支持更长序列（如 128 GPU 支持 2048K）。

- baseline方法是什么？
  Baseline 是 query-aware KV cache 淘汰方法，包括 SnapKV、PyramidKV、H2O。
  
  SnapKV / PyramidKV / H2O 的全栈执行例子：
  - **算法层**：在 prefill 阶段利用 trailing context window 中的 query token 计算 attention-based 重要性分数（SnapKV: max pooling over observation window; PyramidKV: pyramidal layer-budget; H2O: cumulative attention scores during prefill），选择性地保留与当前 query 相关的 KV pairs。核心假设：对当前 query 重要的 KV pairs 对后续也重要。
  - **系统框架层**：论文未明确说明特定 serving 框架，方法可集成到任意支持 KV cache 的推理框架（HuggingFace Transformers、vLLM 等）。使用 FlashAttention-2 加速注意力计算。
  - **编译框架层**：论文未明确说明。
  - **Kernel 调度层**：使用 FlashAttention-2 kernel。SnapKV 使用 max pooling kernel_size=7 对注意力分数平滑。论文未明确说明其他自定义 kernel。
  - **硬件架构层**：在 NVIDIA A100 80GB GPU 上运行，无专用硬件修改。
  
  Baseline 核心缺陷：
  1. **Query 过拟合**：在 prefill 时依赖当前 query 信息决定 KV pair 保留策略，压缩后的 KV cache 对初始 query 过拟合。在多查询场景下，复用该压缩 cache 处理不同 query 时性能显著下降（Figure 2：SnapKV 在 SQuAD multi-QA 中复用压缩 cache，准确率大幅衰减）。
  2. **重复 prefill 开销**：若每个 query 独立执行 prefill + evict（Figure 1a），则每个 query 都需要完整 prefill 计算，总开销随查询数量线性增长。
  3. **Self-attention 稀疏性不匹配**：H2O 使用 prefill 阶段的 self-attention scores 作为重要性指标，但 prefill 阶段的 self-attention 模式比 cross-attention 更密集（Figure 5），且与下游任务 attention 模式重叠度低（Figure 13e），导致无法有效识别冗余 KV pairs。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  KVzip 提出 query-agnostic KV cache 淘汰策略，通过上下文重建（context reconstruction）评估 KV pair 重要性。
  
  KVzip 的全栈执行例子：
  - **算法层**：将 "Repeat the previous context:" prompt + 原始 context chunk 拼接后通过 LLM forward pass，模拟 teacher-forced decoding 重建上下文。对每个 KV pair 取其在重建过程中收到的最大 cross-attention score 作为重要性分数 S ∈ R^{L×H×n_c}。保留 top r% 高分 KV pairs，淘汰其余。核心 insight：(1) Transformer 天然作为 encoder-decoder——将 context 编码进 KV pairs（类比 Zip 压缩）；(2) 重建上下文所需的关键 KV pairs，恰好也是 QA、摘要、推理等多种下游任务所需（Figure 6 2D histogram 显示下三角区域集中，即重建高分 KV pairs 在各任务中也高分）；(3) 基于重建的自监督 proxy task 能泛化到多种下游任务（类似 BERT/MAE 范式）。Chunked scoring 将复杂度从 O(n_c²) 降至 O(m·n_c)，m=2K 固定。
  - **系统框架层**：与 FlashAttention-2 集成。non-uniform head-budget allocation（跨所有 head 取 top r% 而非 per-head 均匀分配）。支持两种模式：(a) context-dependent eviction——per-context 压缩，高压缩比（可低至 30% budget）但有一次 ~2× prefill 的压缩开销；(b) context-independent eviction——预计算 head-level score S_head ∈ R^{L×H}（单次 88K-token 样本），部署时零开销，应用 DuoAttention 的 head-level KV eviction 策略，显著优于 DuoAttention 原生 head-score（KVzip 用 1 GPU 一分钟 vs DuoAttention 需 8 GPU 数小时优化）。
  - **编译框架层**：论文未明确说明。
  - **Kernel 调度层**：主要使用 FlashAttention-2。chunked scoring 在 FlashAttention 中引入 key subsampling（仅取当前 chunk 对应的 keys）。附录 C.3 提出 softmax-free 变体，通过定制 Triton-based CUDA kernel 将评分嵌入 fused attention kernel，消除 ~10% 评分开销（代价是压缩比下降 ~10%）。与 QServe W8A8KV4 量化无缝集成。
  - **硬件架构层**：在 NVIDIA A100 80GB GPU 上运行，无专用硬件修改。
  
  **对应解决 Baseline 缺陷的具体设计**：
  
  1. **Query-agnostic 评分 → 解决 Query 过拟合**：重要性评分不依赖任何 query，仅基于 context 自身的重建能力。压缩后的 KV cache 可跨任意 query 复用（Figure 1c），无需重复 prefill。实验证明（Figure 2），KVzip 在单次 prefill + 多 query 场景下性能稳定，而 SnapKV 复用压缩 cache 时性能显著退化。
  
  2. **Cross-attention 稀疏性 → 解决 H2O self-attention 密度问题**：上下文重建过程中的 cross-attention 比 prefill self-attention 显著稀疏（Figure 5 直方图对比），因为模型可以高效利用 KV_c 中的高层表示 + 自身权重中的知识，减少不必要的注意力查找。这种稀疏性使 KVzip 能更精准地识别可淘汰的冗余 KV pairs。
  
  3. **重建驱动的评分原理 → 保证多任务泛化**：实验证明重建所需的 KV pairs 与 QA、摘要、推理等下游任务的注意力模式高度重叠（Figure 6 前三张 2D histogram 的 lower-right triangular region），而不同 QA 任务之间的注意力模式却呈现 query-specific 差异（第四张 heatmap 沿 x/y 轴分散）。这表明重建作为一个通用的 proxy task，能够捕获跨任务的通用关键信息。
  
  4. **实际效果**：Baseline 方法在 90% cache budget（仅淘汰 10%）时即出现性能退化，KVzip 在 30% budget（淘汰 70%）下仍保持接近无损性能。FlashAttention 解码延迟降低约 2×，KV cache 大小减少 3-4×。结合 4-bit 量化后，16-bit 124K-token KV cache 从 16.3GB 降至 1.2GB。

- baseline方法是什么？
  Baseline 是 Full Attention（完整 KV cache）+ 现有 KV cache 压缩方法（H2O、TOVA、StreamingLLM、FastGen），其全栈执行例子如下：
  - **算法层**：Full Attention 对所有 token pair 计算 O(n²) attention，KV cache 随序列长度线性增长（BF16 下 Llama-3-8B 处理 1M tokens 需要 ~137 GB KV cache 仅此项就超出单卡 80GB 容量）。H2O 基于累积 attention scores 识别 heavy-hitter token 保留在 KV cache 中，TOVA 基于 attention scores 贪心 evict 不重要的 token，StreamingLLM 保留初始 token（attention sink）+ 最近 token 的 sliding window。这些方法的核心缺陷：(a) 不分 head 类型差异——对所有 attention head 使用相同的压缩策略，抹杀了不同 head 的功能异质性（retrieval vs streaming）；(b) 仅依赖 attention scores 做逐 token eviction，忽略了 value states 的影响和跨层跨 head 的 attention 分布差异；(c) 在长上下文 benchmark（NIAH/LongBench）上严重退化——H2O/TOVA/StreamingLLM 在 NIAH 上几乎完全失败（无法在不同序列深度正确检索），因为它们在 pre-filling 阶段需要 materialized attention scores 但 FlashAttention 不物化这些 scores，导致 pre-filling 阶段无法 evict tokens、造成 OOM；(d) 无法降低 pre-filling 的计算和内存开销——这些方法仅在 decoding 阶段减少 KV cache，pre-filling 仍是 full computation。
  - **系统框架层**：基于 HuggingFace Transformers 推理 pipeline，在 prefill 后/decoding 中执行 KV cache eviction。H2O/TOVA 需修改以兼容 FlashAttention（prefilling 用 exact attention，仅 decoding 阶段 evict）。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：FlashAttention-2 加速标准 attention。H2O/TOVA token eviction 在 GPU 上执行 TopK + index gather 操作。
  - **硬件架构层**：NVIDIA A100 GPU（80GB）。长上下文（≥128K）时 baseline 方法由于 KV cache 爆炸导致 OOM。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  DuoAttention 通过三个核心设计解决 baseline 缺陷：

  **1. Head 功能二分（Retrieval vs Streaming Heads）→ 解决缺陷(a)**：
  观察到 attention heads 呈现两种不同功能模式：Retrieval Heads（少数）关注跨长距离的语义相关 token，对长上下文处理至关重要；Streaming Heads（多数）主要关注 attention sink（首 token）和最近 token，不依赖完整历史。利用这一观察，为两类 head 分配不同的 KV cache 策略，而非 baseline 的统一处理。

  **2. 基于优化的 Retrieval Head 识别（优化-based + 合成数据）→ 解决缺陷(b)**：
  不再依赖 attention scores profiling（如 FastGen、RazorAttention 所用），而是直接测量输出偏差——当 KV cache 压缩为仅 sink+recent 时导致输出偏差显著增大的 head 即为 retrieval head。用可训练 gate value α_{i,j} 混合 full 和 streaming attention 输出，在合成 passkey retrieval 数据集上以 L2 distillation loss + L1 regularization 端到端优化。合成数据确保每个监督信号都与最终压缩策略相关（passkey recall 需要长上下文能力），优于 natural language modeling（自然文本中跨长距离的监督信号稀疏）。与 attention profiling 相比：直接测量 output deviation 能捕捉 attention scores 上看不到的 retrieval heads、考虑 value states 的影响、以及跨层跨 head 的分布差异。

  **3. Chunked Pre-filling 中的 streaming head 优化 → 解决缺陷(d)**：
  Streaming heads 的 pre-filling 计算中，每个 chunk 的 KV 计算完毕后立即 prune 仅保留 sink+recent tokens，下一 chunk 仅需 attend 到 constant number 的历史 token。Pre-filling 复杂度从 O(L²) 降至 O(LK)，memory 从 O(L) 降至 O(K)。

  **全栈执行例子（DuoAttention on Llama-2-7B-32K-Instruct, 25% retrieval ratio, 1×A100）**：
  - **算法层（核心创新）**：
    (a) Offline Phase：8×A100 上 2,000 steps gate value 训练（仅数千参数，模型权重冻结）→ synthetic passkey dataset（BookSum + 10×32-word passkeys）→ L2 distillation loss on last hidden states + L1 regularization (λ=0.05) → AdamW (lr=0.02 warmup→decay)。
    (b) Binarization：按 sparsity quantile τ 将 α_{i,j} 二值化为 {retrieval, streaming}，head 重排 Q/K/V 权重使两类连续。
    (c) Decoding：retrieval heads → full KV cache (all tokens) + FlashAttention；streaming heads → constant KV cache (64 sink + 256 recent) + streaming mask attention。
    (d) Chunked Pre-filling：chunk_size=32K，每 chunk 后 streaming head 的 KV cache 立即 prune → 下一 chunk 仅 attend 到 O(K) 而非 O(L) tokens。
  - **系统框架层**：基于 PyTorch + FlashInfer (RoPE/RMSNorm kernels) + FlashAttention-2。支持 chunked pre-filling，与 GQA 完全兼容（per KV group gate value）。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：FlashAttention-2 执行 retrieval heads 的 full attention 和 streaming heads 的 constant-length attention。Chunked pre-filling 中 streaming heads 的 attention 利用标准 FlashAttention kernel（仅 mask 改变为 Λ-like pattern），无需特殊 kernel。与 FlashInfer 的 RoPE/RMSNorm kernel 配合使用。
  - **硬件架构层**：单 NVIDIA A100-80G GPU。DuoAttention + QServe (W8A8KV4 quantization) → Llama-3-8B 容纳 3.3M contextual tokens（6.4× capacity vs full attention BF16）。

  **对比 baseline 的关键差异**：
  - Baseline 统一处理所有 head → DuoAttention 区分 retrieval/streaming，仅 retrieval heads 保留 full KV cache，retrieval ratio 25%（MHA）/ 50%（GQA）即保持 accuracy，其余 memory 大幅减少
  - Baseline 依赖 attention scores eviction（H2O/TOVA/FastGen）; DuoAttention 用优化-based 输出偏差方法识别，更准确（ablation 图 13(1) 证明优于 attention profiling 和 language modeling）
  - Baseline 在 NIAH 上完全失败（Figure 6: H2O/TOVA/StreamingLLM 在不同深度无法检索）; DuoAttention 在所有深度保持接近 full attention 的性能（因 retrieval heads 保留完整 KV cache）
  - Baseline pre-filling 无优化 → DuoAttention streaming heads pre-filling O(LK) 时间 + O(K) 内存（vs baseline O(L²) + O(L)）
  - MHA 模型 memory reduction up to 2.55×, latency reduction up to 2.18×（decoding）和 1.73×（pre-filling）
  - GQA 模型 memory reduction up to 1.67×, latency reduction up to 1.50×（decoding）和 1.63×（pre-filling）
  - GQA 模型的 retrieval head ratio (50%) 高于 MHA (25%)，因为 GQA 中 per-group gate value 绑定多个 query head，必须保守压缩; MHA 中每个 head 独立 gate，压缩更激进
