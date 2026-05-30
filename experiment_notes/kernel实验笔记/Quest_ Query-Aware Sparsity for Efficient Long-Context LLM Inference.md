## Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  Quest 在 FlashInfer kernel 库（Ye et al., 2024）基础上实现了三个专用 CUDA kernel：(1) **Criticality Estimation kernel**：将每 page 的 channel-wise min/max Key 元数据（2 个 float16/通道/page）与当前 Query 向量做 element-wise 乘法和 max 选择，求和得到 per-page criticality score；(2) **Top-K Filtering kernel**：使用 RAFT（RAPIDS, Zhang et al. 2023a）的 batched Top-K CUDA operator，从所有 page score 中选出 Top-K 索引，延迟仅 5-10 µs（seq_len < 128K）；(3) **Approximate Attention kernel**：基于 PageAttention 实现，将 Top-K page 索引传入 FlashInfer 的 sparse page 加载接口，仅对选中 page 执行标准 FlashAttention。

  实验比较：(a) Per-kernel micro-benchmark（NVBench）：criticality estimation 延迟 vs FlashInfer full attention，不同 seq_len 和 page_size；(b) Approximate attention 延迟 vs FlashInfer，不同 token budget 和 seq_len，page_size=16；(c) Combined kernel time breakdown（PyTorch Profiler）：Quest attention（estimation + Top-K + approximate）vs FlashInfer full attention，Llama2-7B 配置，seq_len=8K/16K/32K，token budget=2048。

- 后端平台是什么，配置是什么。
  GPU：NVIDIA RTX 4090（kernel 级 micro-benchmark），NVIDIA Ada 6000 48GB（端到端长 context 评估）。CUDA 12.2。模型配置：Llama2-7B（32 layers, 32 heads, d_head=128, FP16）。FlashInfer 作为 attention kernel 库。量化：支持 4-bit 权重量化（端到端场景）。

- 评估性能的软件/脚本是什么。修改了什么。
  软件：FlashInfer（kernel library for LLM serving），RAFT（GPU-accelerated vector search library，提供 batched Top-K CUDA kernel），NVBench（NVIDIA GPU kernel micro-benchmarking tool），PyTorch Profiler（端到端 kernel 时间分解）。修改内容：在 FlashInfer 中新增三个 CUDA kernel——criticality estimation（读取 page metadata 替代完整 KV cache 加载），Top-K filtering（调用 RAFT），approximate attention（基于 PageAttention sparse page loading）。同时维护 per-page metadata buffer（min/max Key values, 2 × d_head × page_size 个 FP16）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  Quest 开源：https://github.com/mit-han-lab/Quest。基于 FlashInfer（https://flashinfer.ai）和 RAFT（https://github.com/rapidsai/raft）。

  **Kernel 评估流程（NVBench，Fig. 8）**：

  (a) **Criticality Estimation kernel 评估**：
  ```
  输入：Query Q ∈ R^{B×d_head}, Metadata M,m ∈ R^{num_pages×d_head}
  原理：测量 element-wise max(Q*m, Q*M) + reduce-sum 的 GPU 延迟
  对比 FlashInfer full attention（加载完整 K cache + QK^T matmul）
  评估参数：seq_len ∈ {4K, 8K, 16K, 32K, 64K}, page_size ∈ {16, 32, 64}
  输出：latency (µs)，随 seq_len 增长 latency 相对 FlashInfer 比例趋近 1/PageSize
  ```

  (b) **Approximate Attention kernel 评估**：
  ```
  输入：Q ∈ R^{B×d_head}, 选中 page 的完整 K,V ∈ R^{K×S×d_head}
  原理：PageAttention sparse loading + FlashAttention 计算
  对比：FlashInfer 标准 attention at seq_len=B (token budget)
  输出：给定 token budget B，latency 为常数（与总 seq_len 无关），接近 FlashInfer@seq_len=B
  ```

  (c) **Combined Quest Attention 评估（PyTorch Profiler, Fig. 9）**：
  ```
  输入：完整推理的单步 decode（Llama2-7B, single batch）
  测量：Criticality estimation + Top-K filtering + Approximate attention 的合计 wall-clock 时间
  对比：FlashInfer full attention（加载完整 KV cache + GQA attention）
  结果：seq_len=32K, token budget=2048 → 7.03× self-attention speedup
  时间分解：estimation + Top-K ≪ approximate attention ≪ full attention KV loading
  ```

  **Kernel 数据流**：
  ```
  GPU HBM:
    [Full KV Cache: 2 × seq_len × d_head × FP16]
    [Page Metadata: 2 × num_pages × d_head × FP16]  ← Quest 额外存储

  Step 1: Criticality Estimation (memory-bound, O(num_pages × d_head))
    HBM → SM registers: Q (d_head), M_p (d_head), m_p (d_head)
    Compute: per-channel Q_i * max(m_i^p, M_i^p), sum → scalar score_p
    Output: scores ∈ R^{num_pages} (SRAM)

  Step 2: Top-K Filtering (compute-bound, O(num_pages × log K))
    scores → RAFT batched Top-K → top_k_indices ∈ Z^K
    Latency: 5-10 µs (negligible vs KV loading)

  Step 3: Approximate Attention (memory-bound, O(K × S × d_head))
    HBM → SM: Q, K[top_k_indices], V[top_k_indices] (via PageAttention sparse load)
    Compute: S = QK^T/√d_head, softmax, O = SV (FlashAttention tiling)
    Output: O ∈ R^{B×d_head}
  ```

  Memory load reduction: Quest 加载 1/PageSize + K/PageNum of total KV cache。如 page_size=16, 64K context (4096 pages), K=256 pages → ~1/16 + 256/4096 ≈ 12.5%，约 8× 减少。
