## ShadowKV__KV_Cache_in_Shadows_for_High-Throughput_Long-Context_LLM_Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  ShadowKV 的实现中包含多个自定义 CUDA kernel 和运行时调度优化：(a) **Attention approximation kernel**：基于 landmarks（chunk 均值）的近似注意力分数计算，包括 chunk-level MatMul(Q, L^T)、Softmax、聚合、ArgTopK 的 fused kernel；(b) **Key cache low-rank reconstruction kernel**：GPU 端从低秩投影 A、B 和 chunk indices I 通过 MatMul(Gather(A, I), B) 重建稀疏 key cache 的 fused gather+matmul kernel；(c) **Value cache fetching kernel**：CPU→GPU 的 value cache 批量 gather 操作，通过 PCIe 传输调度；(d) **Cache mechanism kernel**：利用 KV cache temporal locality 的 index scan kernel，比较相邻 decoding step 的 chunk 选择索引，检测 miss chunks 以跳过命中部分的重复重建和 PCIe 传输；(e) **Multi-stream overlap 调度**：CUDA multi-stream 将 key 低秩重建（GPU compute-bound）与 value CPU 抓取（PCIe I/O-bound）放入不同 stream 并发执行，实现计算与数据传输重叠。实验比较不同 chunk size 下的 batch size 上限和命中率，不同 context 长度下 decoding 的逐操作延迟分解（GEMM+Softmax、Max、TopK、Recompute K (Overlapped)、Fetch V、Attention、FFN、QKV），以及不同 chunk size / rank / sparse budget 下的准确率消融。

- 后端平台是什么，配置是什么。
  NVIDIA A100 80GB PCIe GPU，CUDA 多流并发，GPU 内存带宽 2 TB/s，PCIe 带宽 31.5 GB/s。

- 评估性能的软件/脚本是什么。修改了什么。
  评估基于 PyTorch + FlashAttention（FlashAttention-2）+ FlashInfer（fused layer norm 等）+ CUTLASS + vLLM（PagedAttention）的集成框架。修改/新增的 kernel 和运行时组件：
  1. **SVD kernel**：prefilling 阶段对 pre-RoPE key cache 的在线截断 SVD 计算
  2. **Landmark 构建 kernel**：post-RoPE key 按 chunk 规约求均值 + cosine similarity 检测 outlier
  3. **Fused attention approximation kernel**：Q 与 landmarks L 的 MatMul + Softmax + 聚合 + TopK，减少 kernel launch 和中间数据搬运
  4. **Fused key reconstruction kernel**：Gather(A, I) + MatMul(gathered_A, B) 的 fused kernel，减少显存搬运
  5. **Cache index scan kernel**：相邻 decoding step chunk indices 的 miss detection
  6. **Multi-stream 编排**：将 key 低秩重建与 value CPU 抓取分配到不同 CUDA stream 实现 overlap

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源地址：https://github.com/ByteDance-Seed/ShadowKV（Apache 2.0，CUDA 内核占代码量 17.4%）。Kernel 运行时执行流程（以 decoding 阶段单 Transformer block 为例）：

  **输入**：Q ∈ R^{b×h_q×1×d}（batch=24, 8 query heads, 1 token, 128 dim），landmarks L ∈ R^{b×h_kv×n_c×d}（n_c=16K chunks for 128K context/chunk_size=8），低秩投影 A ∈ R^{b×128K×160}、B ∈ R^{b×h_kv×160×128}，V_CPU ∈ CPU memory，K_outlier、V_outlier ∈ GPU

  **Stage 1 — Approx Attention（Fused Kernel）**：
  ```
  // CUDA kernel launch 1: fused Gemm-Softmax-AggTopK
  P[h, nc] = sum_d(Q[h, d] * L[h, nc, d])     // MatMul, compute-bound
  S[h, nc] = softmax(P / sqrt(d))               // online softmax
  S_agg[nc] = sum_h(S[h, nc])                   // aggregate across query heads
  S_kv[nc] = max over kv_group(S_agg)            // GQA mapping
  I = TopK(S_kv, k=256)                          // select top-256 chunk indices
  I_miss = CompareAndMiss(I, I_prev)             // cache mechanism scan
  // 输出：I_miss ∈ R^{h_kv × k'}（k' ≈ k × (1-0.6) ≈ 102）
  ```

  **Stage 2 — Key Reconstruct + Value Fetch（Multi-Stream Overlap）**：
  ```
  Stream 1 [GPU compute]:
    A_selected = Gather(A, I_miss)               // index gather
    K_lowrank = MatMul(A_selected, B)            // reconstruct keys
    K_recon = RoPE(K_lowrank)                    // apply RoPE
    // 耗时 ~1.36ms for batch=24×128K

  Stream 2 [PCIe→GPU]:
    V_selected = cudaMemcpyAsync(V_CPU[I_miss], H2D, Stream2)
    // PCIe 传输 (256+48) * 8 * 1024 bytes = ~2.4MB per head
    // 总传输量取决于 batch × h_kv
    // 耗时 ~1.66ms for batch=24×128K

  cudaStreamSynchronize(both)
  // 总延迟 = max(1.36ms, 1.66ms) ≈ 1.84ms（含同步开销）
  // 若无 overlap：1.36 + 1.66 = 3.02ms
  ```

  **Stage 3 — Sparse Attention**：
  ```
  K_final = Concat(K_outlier, K_recon, K_new_token)
  V_final = Concat(V_outlier, V_selected, V_new_token)
  O = FlashAttention(Q, K_final, V_final)
  // 仅对 (48+256)*8 ≈ 2432 tokens 做 attention，而非完整 128K tokens
  // 耗时 ~0.21ms vs Full Attention ~0.5ms+
  ```

  **性能评估原理**：通过延迟分解（Table 13）测量每操作的 wall-clock time：GEMM+Softmax 0.58ms、Max+TopK 0.22ms、Recompute K(overlapped) 1.25ms、Fetch V 1.84ms、Attention 0.21ms、FFN 0.29ms、QKV 0.05ms，总计约 4.44ms/decoding step/Transformer block。对比 Full Attention 约 10ms+，加速主要来自：(1) sparse attention 只计算 1.56% tokens；(2) multi-stream overlap 隐藏 key 重建延迟；(3) cache mechanism 减少 60% 重复操作；(4) fused kernel 减少中间数据搬运和 launch overhead。chunk size 对性能的影响：chunk size 越大，landmark 数量越少（n_c=S/C 越小），近似注意力计算更快但精度下降，论文选择 C=8 作为准确率-效率平衡点。
