## Rectified Sparse Attention

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  ReSA 实现了一个基于 Flash Decoding split-execution 策略的自定义 CUDA kernel，用于加速 GBSA（Group Block Sparse Attention）的 decoding 阶段。核心设计：
  (1) **GQA-aware SM 分配**：每个 GQA group 分配到一个独立的 streaming multiprocessor (SM)，确保高效利用 GPU 资源和最小化 SM 间通信。
  (2) **Block-level workload splitting**：decoding workload（batch_size × num_kv_heads）按 block indices 维度拆分到所有可用 SM。每个 step 激活 k 个 memory block，这些 block 被均匀分到约 k/split 个 block per SM——每个 SM 独立 fetch 对应 KV 并执行 sparse attention。
  (3) **Shared KV fetching**：intra-GQA key 共享机制，同一 GQA group 内的 query heads 复用加载的 KV cache block，减少 HBM 访问。
  (4) 使用 TileLang 库实现（论文 Acknowledgments 致谢 TileLang 团队）。

  实验比较：
  (a) Kernel-level latency breakdown（Section 3.5.1）：16K/64K/256K context 下 ReSA vs Dense FlashAttention 的 CUDA kernel 执行时间对比，延迟分解为 sparse estimation、attention computation、rectification 三部分。
  (b) End-to-end throughput（Section 3.5.2）：FP16 和 INT4（Marlin kernel）下，4K/16K/64K/256K context 的 end-to-end speedup。结果表明 256K 下 ReSA 达 FP16 2.28×、INT4 2.44× speedup。

- 后端平台是什么，配置是什么。
  NVIDIA A100-80G GPU。kernel 参考 Flash Decoding（crfm.stanford.edu/2023/10/12/flashdecoding.html）的 split-execution 策略实现。INT4 实验使用 Marlin kernel 进行 low-bit matmul（group-wise scaling, group size=128）。Custom decoding kernel 使用 TileLang 库实现。Batch size 默认为 8。评测时仅报告 CUDA kernel 时间，排除 CPU overhead。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 PyTorch 框架和自定义 CUDA kernel（通过 TileLang 实现）。修改内容：
  (a) 在 Flash Decoding kernel 基础上增加 block-sparse attention 支持：在 split-execution pipeline 中，外层循环不再遍历全部 KV blocks，而是只遍历由 GBSA block selection 选中的 top-n block indices（算法见 Appendix A）。
  (b) GQA group SM 绑定：每个 SM 固定处理一个 GQA group 的 query，减少 SM 间 KV 传输。
  (c) Block key cache 在线更新：每个 decode step 后增量更新新 token 所在 block 的 min/max 描述符，无需从头计算。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  代码开源在 https://aka.ms/ReSA-LM。kernel 与 Flash Decoding 的集成见论文 Appendix A 的 Algorithm 2。

  **评估原理**：测量 CUDA kernel 执行时间（latency）和 token 生成吞吐量（throughput）。延迟排除 CPU scheduling overhead，仅计 kernel 实际执行。Throughput 测量包括完整 decode step（sparse + rectification 摊销）。

  **Kernel 输入**：Queries Q ∈ R^{h×g×n×d}，Keys K ∈ R^{h×n×d}，Values V ∈ R^{h×n×d}，block_indices（由 GBSA block selection 产生），num_splits（SM 拆分粒度）。

  **Kernel 执行流程**（Flash Decoding + Block-Sparse）：
  ```
  for each (num_splits, num_kv_heads, batch_size) in grid:
      1. Load query vectors q in a GQA group
      2. Compute partial_block_indices = block_indices[split_slice]
         # 只取当前 split 负责的 block 子集
      3. Initialize accumulators: mi ← -∞, li ← 1.0, acc ← 0
      4. for each block_id in partial_block_indices:
           Load k, v from KV cache at block block_id
           qk = q @ k^T * sm_scale
           qk[invalid_positions] = -1e6
           Update mi, li, acc (online softmax)
      5. Store partial logsum and attention outputs
  end for
  6. Combine(logsum_partial, out_partial) → final output O
  ```

  **性能结果**：
  - 256K context：FP16 2.28×，INT4 2.42× end-to-end speedup vs dense FlashAttention
  - Rectification overhead：256K 时占 attention 总延迟的 32.7%，64K 时 28.9%
  - Sparse estimation 和 attention computation 耗时相当（均 ≈ mem(KV cache) × p ≈ mem(KV cache) × 1/b）
