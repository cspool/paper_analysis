## TailorKV: A Hybrid Framework for Long-Context Inference via Tailored KV Cache Optimization

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现两个 CUDA kernel 和异步数据传输管线用以支持混合 KV Cache 压缩：(1) **1-bit 量化 kernel**：对 quantization-friendly 层的 KV cache 执行 per-channel key + per-token value 的 1-bit 静态量化，group size=64，zero-point 和 scaler 存 FP16；(2) **FP16×INT1 GEMV kernel**：实现 FP16 输入矩阵与 INT1 权重矩阵的矩阵乘 kernel，支持高压缩率下的高效解码，通过直接使用 FP16×INT1 矩阵乘法替代 FP16×FP16，减少计算量并充分利用内存带宽；(3) **DGL-based 直接行传输**：使用 Deep Graph Library (DGL, Wang et al., 2019) 的稀疏 tensor API 直接从 CPU tensor 按行索引传输选中的 KV cache rows 到 GPU 设备，避免先 gather 到 CPU 连续内存再传输的中间步骤（PQCache 的做法）；(4) **异步 double buffering 预取管线**：在 decoding 阶段，layer l-1 计算时异步预取 layer l 的 critical key cache（基于 inter-layer query 预估），使用读写双缓冲区实现 computation 与 CPU-GPU communication 的 overlap。

  实验比较：(1) Latency breakdown 对比（Figure 8）：在 A100 上单 Transformer block 16k 序列长度下，TailorKV 相比 PQCache 降低 retrieval latency 27.8% (GQA) / 40.5% (MHA)，降低 data transfer latency 83.5% (GQA) / 82.2% (MHA)；(2) End-to-end decoding latency 对比（Table 4, Table 13）：TailorKV 延迟接近 Full Cache (FlashAttention-2)，相比 OffloadCache 加速数十倍；(3) 不同 critical channel 数量 (2/4/8/12) 的性能/latency trade-off（Figure 9c）。

- 后端平台是什么，配置是什么。
  NVIDIA RTX 3090 (24GB, PCIe 1.0, 4GB/s) + Intel Xeon Gold 6240；NVIDIA A100 (80GB, PCIe 4.0, 32GB/s) + Intel Xeon Platinum 8369B。使用 PyTorch 2.4.0 + FlashAttention-2.6.3 + DGL。多线程执行异步任务，实现 GPU 计算与 CPU→GPU 数据传输的 overlap。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 HuggingFace Transformers 4.46.1 的推理脚本修改。主要修改在 Transformer 的 forward pass 中：
  1. 在 decoder layer 的 attention 模块中插入混合压缩逻辑——根据离线确定的层类型 C(l) 分支：quantization-friendly 层执行 per-layer 量化 + FP16×INT1 GEMV；sparsity-friendly 层执行动态检索（critical channel selection + CPU-GPU 异步预取 + Top-K token fetch）
  2. 使用 DGL 的稀疏 tensor row-wise transfer API（dgl.sparse.from_coo / dgl.ops.gspmm）将 CPU tensor 中指定行索引的数据直接传输到 GPU
  3. 实现 CUDA kernel 的 1-bit quantize/dequantize GEMV，用于 quantization-friendly 层的 attention 计算
  4. prefilling 阶段每层 KV cache 逐层 offload 到 CPU memory

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源代码：https://github.com/ydyhello/TailorKV（ACL 2025 Findings，代码已发布）。

  **评估原理**：使用 PyTorch 模型推理 + `torch.cuda.Event` / `time.perf_counter()` 测量 per-layer 和端到端 latency。peak memory 通过 `torch.cuda.max_memory_allocated()` 记录。latency breakdown 在 A100 上单测每个 Transformer block 各操作的耗时分布（attention compute vs CPU-GPU transfer vs retrieval computation）。

  **Kernel 执行流程**（sparsity-friendly 层的动态检索）：
  ```
  // === Stage 1 (at layer l-1, on GPU) ===
  h_l_minus_1 = hidden_state  // shape: (1, d)
  q_hat = W_q[l] @ h_l_minus_1  // 预估算当前层 query, shape: (d_h,)
  // 计算 channel-wise importance
  s = abs(q_hat) * max(abs(K_cpu), dim=seq)  // shape: (d_h,)
  critical_ch = argsort(s)[-d_s:]  // 选 d_s 个 critical channels
  // 启动异步预取: CPU→GPU 传输 critical key cache
  // K_critical shape: (d_s, n), 使用 double buffering

  // === Stage 2 (at layer l, on GPU) ===
  q = W_q[l] @ h_l  // 真实 query
  q_crit = q[critical_ch]  // shape: (d_s,)
  // K_critical 已预取完成（double buffer read）
  a_approx = q_crit @ K_critical.T  // 近似 attention, shape: (1, n)
  topk_idx = TopK(a_approx, k=n_topk)  // GPU 端排序选 Top-K

  // Fetch Top-K tokens: CPU→GPU, 传输完整 K[topk_idx], V[topk_idx]
  // 唯一不可 overlap 的操作——取决于当前层 query

  // 完整 attention: concat(local_KV, fetched_KV) → FlashAttention
  output = flash_attn(q, cat(local_K, fetched_K), cat(local_V, fetched_V))
  ```
