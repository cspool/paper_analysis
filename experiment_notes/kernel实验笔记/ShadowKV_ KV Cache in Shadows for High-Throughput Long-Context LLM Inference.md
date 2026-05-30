## ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现多个自定义 CUDA kernel 用于 ShadowKV 的 sparse attention 解码流程：(1) **Landmark Attention Approximation Kernel**：将 Q 与 landmarks L 的矩阵乘法和 softmax 融合为单个 kernel，使用 chunk-level 粒度减少计算量（从 O(S×d) 降至 O(S/c×d)），输出 top-k chunk indices；(2) **Key Cache Low-Rank Reconstruction Kernel**：基于预存储的低秩投影 A、B 和选中的 chunk indices，通过 Gather(A[I]) × B 的 GEMM 重建 sparse K cache，利用 Tensor Core 加速；(3) **Value Cache Fetching + Overlap Kernel**：使用 CUDA multiple streams 将 CPU→GPU 的 value 数据传输（cudaMemcpy H2D via PCIe）与 key 重建 GEMM 重叠执行，通过 CUDA event 同步；(4) **Temporal Locality Cache Kernel**：维护 chunk index 的环形缓冲区，通过 index scan 检测相邻解码步的 chunk 选中重复，跳过已驻留 GPU 的 KV 对的取回和重建；(5) **融合 Kernel**：将 attention approximation、key cache reconstruction、value cache fetching、cache mechanism 等操作用 CUTLASS 融合以减少 memory movement 和 kernel launch overhead。

  实验比较：(1) 吞吐量实验：A100 上多模型多上下文下的 tokens/s 对比；(2) 延迟分解实验：48×64K / 24×128K / 12×256K / 6×512K 下各操作（GEMM+Softmax, Max, TopK, Recompute K, Fetch V, Attention, FFN, QKV）的延迟分解，显示重叠后 key 重建延迟被 value 取回延迟所掩盖；(3) Pre-filling 延迟分解：64K-512K 上下文下 SVD、Reduce、CosineSimilarity、TopK、Gather 的 overhead 占比分析。

- 后端平台是什么，配置是什么。
  NVIDIA A100 GPU (80GB)。GPU 内存带宽 2 TB/s (HBM2e)，Tensor Core 支持 FP16/BF16/FP8 等。PCIe 4.0 x16 连接 CPU，理论带宽 31.5 GB/s。配套 CPU 端为 x86 服务器平台，用于存放 offloaded value cache 和执行异步数据传输。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 PyTorch 框架，使用 CUTLASS 编写自定义 CUDA kernel，集成 FlashAttention (v2) 和 FlashInfer 的高效 attention 和 layer norm kernel。评估脚本：自定义的 latency/throughput benchmark 脚本，测量每层的 attention/FFN/SVD 等操作延迟（ms），以及整体生成吞吐（tokens/s）。

  修改的 kernel 操作：
  - Pre-filling 新增 kernel：SVD (cuSOLVER gesvd)、Reduce（chunk-mean 归约）、CosineSimilarity（chunk 内 cosine similarity + TopK argmin）、Gather（按 index 选取 outlier）
  - Decoding 替换/新增 kernel：Q×L^T 的 chunk-level attention approximation（融合 GEMM+Softmax+TopK）、Gather(A[I])×B 的低秩 key 重建 GEMM、CPU→GPU value 取回的异步 cudaMemcpy、Temporal cache index scan + merge

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源地址：https://github.com/ByteDance-Seed/ShadowKV 。Kernel 评估原理和全流程：

  **评估原理**：
  评估脚本测量每个 Transformer block 中各操作在给定 batch size × context length 下的延迟（ms），使用 CUDA event (cudaEventRecord) 精确计时。生成吞吐通过端到端运行固定数量的 decode steps，统计总 tokens 数 / 总耗时得到 tokens/s。

  **Kernel 输入到性能输出全过程**（以 decoding 为例）：

  输入状态：
  - GPU memory: A (low-rank key 左奇异向量), B (右奇异向量), L (landmarks), K_outlier, V_outlier
  - CPU pinned memory: V_CPU (全量 value cache)
  - 输入 Q ∈ R^{b×h_q×1×d}

  Step 1 — Attention Approximation Kernel：
  - 输入：Q [b, h_q, 1, d], L [b, h_kv, n_c, d]
  - 计算：P = Q @ L^T → [b, h_q, 1, n_c]; S = softmax(P/√d); 沿 h_q 聚合（GQA）→ [b, h_kv, n_c]
  - 输出：I = ArgTopK(S, k) → [b, h_kv, k]（selected chunk indices）
  - 性能：CUTLASS GEMM + custom softmax/topk fusion，延迟 ~0.56ms (48×64K)

  Step 2 — Key Reconstruction + Value Fetching (Overlapped)：
  - Stream 1 (GPU compute): A_selected = Gather(A, I) → [b, k*c, r]; K_sparse = A_selected @ B → [b, h_kv, k*c, d]
  - Stream 2 (PCIe H2D): V_sparse = cudaMemcpy(V_CPU[I], H2D) → [b, h_kv, k*c, d]
  - CUDA event 同步确保两者完成后继续
  - 性能：Recompute K ~1.25ms（overlapped），Fetch V ~1.84ms（overlapped 后 net latency 即 max(1.25, 1.84) 而非 sum）

  Step 3 — Attention Compute：
  - 输入：Q, K_combined = [K_outlier; RoPE(K_sparse); K_new], V_combined = [V_outlier; V_sparse; V_new]
  - 计算：FlashAttention (exact) on selected subset → O [b, h_q, 1, d]
  - 性能：~0.23ms (48×64K, selected chunk 占比 1.56%)

  Step 4 — Temporal Cache Mechanism：
  - 维护 prev_indices ring buffer，index_scan 检测 I 与 prev_indices 的交集
  - 跳过命中的 chunk，仅对未命中 chunk 执行 Step 2
  - Hit rate ~60%，有效减少 60% 的重建和取回开销

  SVD Overhead 分析（pre-filling）：64K→128K→256K→512K 上下文下，SVD 延迟分别为 17.19/26.62/50.56/108.38ms，占总 pre-filling 时间 6.65%/3.25%/1.75%/0.97%，随序列长度增加占比递减（因为 attention 计算为 O(n²) 而 SVD 为 O(n×d×r)）。
