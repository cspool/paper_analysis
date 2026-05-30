## Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference

- baseline方法是什么？
  Baseline 包含两类：(A) Full Attention（FlashInfer 实现）：decode 阶段每步加载完整 KV cache，对 Llama2-7B @ 32K context，KV cache 16GB，内存加载占 decode 阶段 53% 以上时间；(B) KV Cache 驱逐算法：H2O（基于累积历史 attention score 裁减 token）、TOVA（基于当前 query 的 attention score 决定丢弃哪些 token）、StreamingLLM（仅保留 attention sink + 滑动窗口）。核心缺陷：

  | 缺陷 | 具体表现 |
  |------|----------|
  | Full Attention 内存瓶颈 | 长 context 下 KV cache 加载量随 seq_len 线性增长，decode 阶段严重 memory-bound，32K context 下 KV 加载耗时 >50% |
  | 历史信息驱逐不可逆 | H2O/TOVA 丢弃的 token 可能对未来 query 关键（如 passkey 在 question 之前），导致 passkey retrieval 准确率近乎 0%（Tab. 1, 10K/100K tests） |
  | 静态窗口无法覆盖长依赖 | StreamingLLM 仅关注最近 window，passkey 在 window 之外的 100K 测试完全失败 |
  | Query-agnostic 假设错误 | H2O 假设历史 attention score 高的 token 对将来也关键，但 Fig. 2 证明同一 token 对不同 query 的关键性差异巨大 |

  全栈执行例子（Full Attention baseline, Llama2-7B, 32K context, RTX 4090）：
  - **算法层**：输入 prompt tokens 已编码为 KV cache（32 layers × 32 heads × 32768 tokens × 128 dim × FP16 = 16GB）。每步 decode：取最后一个 token 的 Q（1×128），加载全部 32768 个 K 向量，计算 S = QK^T/√128 ∈ R^{1×32768}，softmax 后乘全部 V 向量得 O ∈ R^{1×128}。attention 计算量 O(seq_len × d_head) = O(32768×128) ≈ 4.2M FLOPs，但 KV cache 加载量 2 × 32768 × 128 × 2 bytes ≈ 16.8MB/head/layer（memory-bound）。
  - **系统框架层**：FlashInfer 作为 attention kernel 库。单 batch decode，所有 KV cache 驻留 GPU HBM。FlashInfer 使用 FlashAttention 的 tiling 策略（分 tile 加载 K,V 到 SRAM 做 online softmax rescaling），但 tile 数量随 seq_len 线性增长。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：FlashInfer CUDA kernels。Decode 阶段：加载 Q（registers），循环加载 K/V tiles 从 HBM → SRAM（每个 tile B_r × d_head），compute QK^T（Tensor Cores），online softmax update（CUDA Cores），accumulate P×V（Tensor Cores），write O 到 HBM。Memory-bound：arithmetic intensity ≈ 4.2M FLOPs / (2×32768×128×2 bytes) ≈ 0.25 FLOPs/byte，远低于 RTX 4090 的 ~200 FLOPs/byte 拐点。
  - **硬件架构层**：NVIDIA RTX 4090（24GB GDDR6X, 1.0 TB/s memory bandwidth, 82.6 TFLOPS FP16）。Decode 阶段仅利用 ~2.5% peak FLOPs（因 memory-bound）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Quest 提出 **query-aware KV cache sparsity**：不丢弃任何 KV cache token，而是在每步 decode 时基于当前 Query 动态估计 token 关键性，仅加载 Top-K 关键 page 的 K、V 参与 attention。核心设计：

  **(1) Page 粒度元数据（1/PageSize 存储开销）**：对每 page（默认 16 tokens），维护 per-channel Key 向量的最小值 m_i 和最大值 M_i。元数据大小 = 2 × d_head / page_size × KV cache size ≈ 12.5% KV cache for page_size=16。插入新 token 时 M_i = max(M_i, k_i), m_i = min(m_i, k_i)，O(d_head) overhead。

  **(2) Query-Aware Upper-Bound Criticality Estimation**：给定当前 Query Q，每 page 的关键性上界 s_p = Σ_{i=1}^{d} max(Q_i · m_i^p, Q_i · M_i^p)。该上界保证 s_p ≥ Q_i · K_i^{(t)} 对 page 内任意 token t 成立，因此选 score 最高的 K 个 page 不会遗漏高 attention 的 token。这与 query-agnostic 方法（H2O/TOVA）根本不同——H2O 基于历史 attention 裁减可能错误丢弃对未来 query 关键的 token。

  **(3) Two-Stage Attention Execution（内存减载的核心）**：
  - Stage 1：加载 metadata（2 × num_pages × d_head，而非完整 KV cache），计算 per-page criticality scores → Top-K page indices
  - Stage 2：仅加载 Top-K pages 的完整 K、V 到 SRAM → 标准 FlashAttention
  - 内存加载减少：1/PageSize + K/PageNum of total KV cache。如 page_size=16, 64K context (4096 pages), K=256 → 加载量仅为完整 KV cache 的 ~12.5%，约 8× 减少。

  **(4) 前两层豁免**：观察到前两层 attention sparsity < 10%（Fig. 3），对前两层保持 full attention，其余层使用 Quest。

  全栈执行例子（Quest, Llama2-7B, 32K context, 2048 token budget, RTX 4090）：
  - **算法层**：与 baseline 相同的 KV cache 存储（不丢弃任何 token）。每步 decode：(a) 加载 page metadata（2 × 2048 pages × 128 dim × FP16 ≈ 1MB，vs baseline 16MB K cache）→ 计算 per-page criticality upper-bound scores；(b) Top-K 选 128 pages（2048 tokens = token budget）；(c) 加载 128 pages × 16 tokens × 128 dim × FP16 ≈ 512KB K + 512KB V → FlashAttention → 输出 O。总内存加载 ~2MB vs baseline ~32MB/layer，16× 减少。
  - **系统框架层**：基于 FlashInfer 实现。PageAttention 兼容性使 sparse page loading 可直接通过 FlashInfer page table indirection 实现，无需额外数据重组。单 batch decode 评估。论文未修改 serving 框架的多请求调度逻辑。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：三个 CUDA kernel 在 FlashInfer 中实现：(a) Criticality estimation kernel —— element-wise max + reduce-sum, memory-bound on metadata；(b) RAFT batched Top-K —— 5-10 µs latency, compute-bound；(c) Approximate attention —— FlashInfer PageAttention with sparse page indices。32K seq_len, 2048 budget → self-attention 7.03× speedup（Fig. 9）。
  - **硬件架构层**：NVIDIA RTX 4090，无专用硬件修改。利用现有 GPU 的 HBM 带宽和 Tensor Cores。

  效果量化：
  - Passkey Retrieval (10K)：Quest 64-token budget 达 100%（Tab. 1），H2O 256-budget 仅 1%
  - Passkey Retrieval (100K)：Quest 1024-token budget 达 100%，H2O 4096-budget 仅 4%
  - LongBench 六数据集：Quest 1K budget 达 full cache 可比性能，H2O/TOVA/StreamingLLM 即使在更大 budget 下仍有明显差距
  - Self-attention speedup：7.03× @ 32K, 2048 budget vs FlashInfer（Fig. 9）
  - End-to-end speedup：2.23× @ 32K, 2048 budget, 4-bit weight quantization（Fig. 10）
  - 同等精度约束下：Quest 比 TOVA 减少 2.6-7.7× latency（Fig. 11b）
