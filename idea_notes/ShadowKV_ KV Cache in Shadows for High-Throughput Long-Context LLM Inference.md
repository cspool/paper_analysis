## ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference

- baseline方法是什么？
  长上下文 LLM 推理中，baseline 是 Full Attention（保留完整 KV cache 在 GPU 显存中进行标准 attention 计算）。全栈执行过程：

  **算法pipeline**：输入 prompt tokens → Token Embedding → 逐层 Transformer Block。每层：X → QKV Projection (W_qkv) → Q, K, V。K 经 RoPE 旋转位置编码。QK^T 计算 attention scores（O(n²)），Softmax 归一化，P×V 加权求和得到 attention output。残差连接后经 FFN。输出 hidden states 送入下一层。最后一层输出经 LM Head → logits → 采样下一个 token。

  **系统框架**：vLLM 系统框架，使用 PagedAttention 管理 KV cache 内存。Prefill 阶段：所有 prompt tokens 并行计算，KV cache 写入 GPU HBM。Decoding 阶段：每次生成一个 token，新 token 的 Q 与完整 KV cache 计算 attention，新 KV 对追加写入 GPU HBM。Batch 请求共享 GPU，Continuous Batching 动态调度。

  **编译框架**：论文未明确说明。PyTorch eager mode 或 torch.compile 自动图捕获。

  **kernel调度**：FlashAttention (v2) CUDA kernel，基于 tiling 和 online softmax，利用 GPU shared memory 减少 HBM 访问，将 attention 计算的内存复杂度从 O(n²) 降至 O(n)（IO 层面）。QKV Projection 使用 cuBLAS GEMM 在 Tensor Core 上执行。FFN 使用 cuBLAS GEMM。

  **硬件架构**：NVIDIA A100 GPU (80GB HBM2e)。Tensor Core 用于 GEMM 计算（312 TFLOPS FP16），HBM2e 带宽 2 TB/s。CPU-GPU 通过 PCIe 4.0 x16 连接（31.5 GB/s）。GPU 上 KV cache 占用 2×b×h_kv×s×d×sizeof(dtype) bytes。以 Llama-3.1-8B 在 128K 上下文为例：s=128K, h_kv=8, d=128, dtype=BF16 → 单层 KV cache = 2×8×128K×128×2 = 512 MB，32 层总计约 16 GB（单 batch），batch=4 即 64 GB，batch=8 即 128 GB 超出 80 GB 显存。

  Baseline 痛点：
  1. GPU 显存瓶颈：KV cache 随序列长度线性增长，长上下文（128K-1M）下即使小 batch size 也会 OOM
  2. 解码延迟高：每步需对所有 s 个 token 计算 attention，计算量 O(s×d)，访存量 O(s×d)
  3. Batch size 受限：小 batch 导致 GPU 计算资源利用率低，吞吐低下
  4. CPU offloading 朴素方案延迟大：将完整 KV cache 移至 CPU 并每次取回稀疏 KV 对，PCIe 传输成为瓶颈

- 论文方法是什么？如何对应解决Baseline的缺陷？
  ShadowKV 提出 GPU-CPU 异构 KV cache 存储 + 准确的稀疏 attention 选择策略，全栈执行过程：

  **算法pipeline**（核心创新）：
  - **发现**：pre-RoPE key cache 具有极低秩特性（奇异值衰减最快），同一序列内 key 的低秩子空间高度共享（内序列相似度高），不同序列间低秩子空间不同（跨序列相似度低）。因此 online SVD 比 data-independent weight decomposition 更精确。
  - **低秩 Key 存储**：pre-filling 时对 pre-RoPE K ∈ R^{s×d} 执行 SVD 保留 rank r=160 的截断分解：A ∈ R^{s×r}, B ∈ R^{h_kv×r×d}。K 通过 K ≈ A @ B（忽略奇异值对角矩阵）重建。压缩比 = d/r ≈ 128/160 不合适？实际上低秩分解将存储从 s×d 降至 s×r + h_kv×r×d。以 Llama-3.1-8B (d=128, r=160, h_kv=8) 为例：原始存储 = s×128，低秩存储 = s×160 + 8×160×128 = 160s + 163840。对 s=128K：原始 = 16.4M floats，低秩 = 20.5M + 0.16M（实际上论文称约 6× 压缩，涉及 batch 累计效应和 landmarks 替代完整 K cache 存储）。
  - **Landmark Approximate Attention**：将 post-RoPE K 分为 c=8 token 的 chunk，每 chunk 均值作为 landmark L。解码时 Q×L^T（O(n_c×d) 替代 O(s×d)）近似注意力选择 top-k chunk。
  - **Outlier 缓存**：检测 chunk 内 cosine similarity 最低的 o=48 chunk（0.3%）作为 outlier，完整保留其 KV 对在 GPU。
  - **Temporal Locality Cache**：利用相邻解码步 attention 模式高重复性（>60% chunk hit rate），跳过已缓存 chunk 的取回和重建。

  **系统框架**（Serving调度）：
  - GPU 存储：低秩 key 投影 A/B、landmarks L、outlier KV 对（总计 ~1/6 原始 KV cache 大小）
  - CPU 存储：完整 value cache V_CPU（pinned memory，快速 H2D 传输）
  - 解码调度：Q → Landmark Attention（选择 top-k chunk）→ 并行执行 key 重建（GPU GEMM）与 value 取回（PCIe H2D）→ Sparse FlashAttention → 输出
  - Temporal cache 减少 60% 重复操作

  **编译框架**：论文未明确说明。

  **kernel调度**（解决解码延迟问题）：
  - 自定义 CUDA kernel：注意力近似融合 kernel（GEMM+Softmax+TopK）、低秩 key 重建 GEMM、异步 value 取回
  - CUDA multi-stream overlap：Stream 1 执行 K_sparse = A_selected @ B（GPU Tensor Core），Stream 2 执行 V_sparse = cudaMemcpy(V_CPU[I], H2D)（PCIe）。两者通过 CUDA event 同步，net latency = max(compute, transfer) 而非 sum
  - 理论等效带宽 7.2 TB/s = 3.6× A100 原生带宽，超越假设无限显存条件下的吞吐

  **硬件架构**：NVIDIA A100 GPU + x86 CPU，PCIe 4.0 x16。无硬件修改。

  Baseline 缺陷 → ShadowKV 解决方案对照：
  1. **显存瓶颈** → 低秩 key + CPU value offloading，GPU KV cache 占用降至 1/6，batch size 提升 6×
  2. **解码延迟高** → Landmark 近似 attention 降至 O(s/c×d)，仅对 1.56% 的 sparse chunk 做精确 attention
  3. **Batch 受限** → 显存节省后 batch 从 2-8 升至 12-48，吞吐提升 2.23-3.04×
  4. **CPU offloading 延迟大** → 仅取回 value（非完整 KV），与 key 重建重叠执行，等效带宽 7.2 TB/s
