## Accelerating Mixture-of-Experts Inference by Hiding Offloading Latency with Speculative Decoding

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是 **CPU Chunked Attention Verification Kernel**——专为 speculative decoding 在 offloading 场景下设计的 CPU 端 chunked attention 算子，以及 GPU-CPU 异构 kernel 调度流水线。核心内容包括：
  1. **CPU Chunked Attention Kernel**：处理 speculative decoding verification 中的 Q∈R^{n×d}, K∈R^{(l+n)×d}, V∈R^{(l+n)×d} 的 attention 计算，其中 n 为 draft tokens 数，l 为历史 tokens 数。针对 n>1（chunked）而非 n=1（单 token decode）优化。
  2. **Intel MKL 加速**：利用 Intel oneAPI Math Kernel Library 进行高效矩阵乘法，充分利用 CPU SIMD 和 MIMD 能力（区别于 GPU 的 SIMT 架构 + 手动 managed shared memory）。
  3. **Mask 内存压缩**：attention mask M∈{-∞,1}^{n×(l+n)} 中 draft-to-prefix 部分固定为 1，仅存储 draft-token 间相关部分（M 的右下角 n×n 子区域），大幅减少内存占用。

  实验比较：
  - 间接体现在端到端系统性能中——CPU Attention 在 iteration time 中占主导（Table 3: 4.29s actual vs 3.88s estimated）
  - Iteration breakdown (Figure 13): 随 draft length 增加 CPU Attention 时间占比增长，逐渐成为 target model 的瓶颈
  - Profiling estimator 精度: CPU Attention estimation error 10.6%

- 后端平台是什么，配置是什么。
  A30 环境: Intel Xeon Gold 6426Y CPU (2.43 TFLOPS, 357 GB/s memory bandwidth)。4090D 环境: Intel Xeon Gold 5418Y CPU (1.45 TFLOPS, 197 GB/s)。

- 评估性能的软件/脚本是什么。修改了什么。
  评估工具: PyTorch + Intel MKL。使用 APPS 和 CNN/DailyMail 数据集测量不同配置下各算子的实际执行时间（Table 3: Actual vs Estimated 对比）。

  **修改/优化内容**:
  1. **CPU Chunked Attention 实现**（替代 GPU chunked attention 和 naive CPU decode attention）：
     - GPU chunked attention 的替代方案：避免将 KV cache 从 CPU DRAM 反复传输到 GPU HBM
     - Naive CPU decode attention (n=1) 的替代方案：避免对每个 draft token 单独执行 attention（减少重复 KV cache 访问）
     - PyTorch CPU prefill attention 的替代方案：避免对已在 KV cache 中的 token 重复计算

  2. **Intel MKL GEMM 加速**: chunked attention 的计算核心为两个矩阵乘法——Q@K^T (attention scores) 和 softmax(scores)@V (weighted sum)——均通过 Intel MKL 的优化 GEMM 实现

  3. **Mask 压缩存储**：传统方案存储完整 n×(l+n) mask 矩阵，其中 l 远大于 n（如 l=512, n=5 时 mask 为 5×517）；SpecMoEOff 只存储 draft-to-draft 子区域 (n×n) 和 draft-to-prefix 的固定值（全 1），节省约 (l+n)^2 - n^2 的内存

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文未公开独立开源仓库。CPU Chunked Attention kernel 基于 Intel MKL 实现于 SpecMoEOff 系统内部。

  **CPU Chunked Attention Kernel 全过程（以单 request, n=5 draft tokens, l=512 prefix tokens, d=4096 为例）**：

  1. **输入准备**：
     - Q ∈ R^{5×4096}: 5 个 draft tokens 通过 target model Q projection 得到
     - K ∈ R^{517×4096}: 512 prefix + 5 draft tokens 的 key，从 CPU DRAM KV cache 读取
     - V ∈ R^{517×4096}: 同上 value
     - M_draft ∈ {0,1}^{5×5}: 仅存储 draft tokens 间的 causal mask（下三角=1）
     - M_prefix: draft→prefix 全 1（不实际存储，计算时直接忽略 mask 加项）

  2. **Step 1 - Attention Scores (Intel MKL SGEMM)**：
     ```
     # Q@K^T: [5, 4096] @ [4096, 517] → [5, 517]
     scores_full = mkl_sgemm(Q, K.T) / sqrt(4096)  # [5, 517]
     # scores_full[:, :512]: Q vs prefix K (无 mask, 全 1)
     # scores_full[:, 512:]: Q vs draft K (需 causal mask)
     ```

  3. **Step 2 - Mask Application（仅处理 draft 部分）**：
     ```
     scores_full[:, 512:] += causal_mask(M_draft)  # M_draft 下三角=0, 上三角=-inf
     # prefix 部分无需 mask（所有 draft token 都可 attend 到所有 prefix token）
     ```

  4. **Step 3 - Softmax + Weighted Sum (Intel MKL SGEMM)**：
     ```
     attn_weights = softmax(scores_full, dim=-1)    # [5, 517]
     # attn_weights @ V: [5, 517] @ [517, 4096] → [5, 4096]
     output = mkl_sgemm(attn_weights, V)            # [5, 4096]
     ```

  5. **Batch 扩展**：对 b 个 requests，每个 request 的上述计算并行化（multi-threaded CPU parallelism），各 request 独立执行

  **与 Baseline 的对比**：

  | Kernel 方案 | Q@K^T 次数 | KV Cache 访问 | CPU-GPU 传输 | Mask 内存 |
  |------------|-----------|-------------|-------------|----------|
  | GPU chunked attention | 1 次 (b×n size) | ~0 (KV在GPU) | 需传输全部KV cache | 完整 n×(l+n) |
  | Naive CPU decode (repeat n times) | n 次 (b×1) | n× 重复读取 | 0 | 无 (per-token) |
  | PyTorch CPU prefill attention | 1 次 (全量) | 全部重读 + 重复计算 | 0 | 完整 n×(l+n) |
  | **SpecMoEOff CPU Chunked** | **1 次** | **1 次，无重复** | **0** | **仅 n×n** |

  核心优势：1 次 Q@K^T 覆盖全部 draft tokens，无 KV cache 重复访问，无 CPU-GPU 传输，mask 内存 O(n²) 而非 O(n·(l+n))。

  **GPU-CPU Kernel 调度流水线**：
  - GPU stream 1 (comp): GPU Other1 kernel → CPU Attention trigger → GPU Other2 kernel → GPU MoE kernel
  - GPU stream 2 (load): 异步 HtoD transfer 下一层 expert weights
  - CUDA Event synchronization: CPU Attention 完成 → record event → GPU Other2 等待 event
  - 两个 micro-batch 的 CPU Attention 和 GPU MoE 交错执行实现重叠
