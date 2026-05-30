## ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference

- 属于Serving调度的实现是什么？实验比较什么？
  实现 ShadowKV 系统，通过 GPU-CPU 异构 KV cache 管理策略大幅提升长上下文 LLM 推理吞吐。核心调度优化：(1) **KV Cache 分层存储**：pre-filling 阶段对 pre-RoPE key cache 执行 SVD 后仅保留低秩投影（rank=160）在 GPU，value cache 全量下放 CPU，仅 outlier chunk 的完整 KV 对保留 GPU 作为 static cache；(2) **Sparse Attention 解码调度**：解码时用 landmarks + Q 近似注意力选择 top-k chunk，通过 CUDA multi-stream 将 CPU→GPU 的 value 取回与 GPU 上低秩 key 重建重叠执行，掩盖 PCIe 传输延迟；(3) **Temporal Locality Cache**：利用相邻解码步 KV 选择高重复率（>60%），通过 index scan 仅重建缺失 chunk 的 KV 对，减少 60% 计算和数据搬运；(4) **大 Batch 支持**：通过 GPU 内存节省（KV cache 占用降至 1/6），将最大 batch size 从 2-8 提升至 12-48，超越假设无限显存下的吞吐。

  实验比较：(1) 吞吐量实验：A100 上测量 Llama-3-8B-1M、Llama-3.1-8B、GLM-4-9B-1M、Yi-9B-200K 在 60K/122K/244K 上下文下的生成吞吐（tokens/s），对比 Full Attention baseline 最大 batch size 和 Infinite batch size（理论极限）；(2) Batch size 扩展性实验：Llama-3-8B-1M 在 60K/122K/244K/488K 上下文、batch size 2-48 下的吞吐矩阵；(3) 延迟分解实验；(4) 与 Quest 在 1M 上下文下的效率对比。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 GPU (80GB)，GPU 内存带宽 2 TB/s，PCIe 4.0 x16 带宽 31.5 GB/s，搭配 CPU 大内存（用于存放 offloaded value cache）。CPU-GPU 通过 PCIe 连接。

- 开源Serving框架是什么。修改了什么。
  ShadowKV 基于 PyTorch 实现，与 vLLM 的 PagedAttention 机制兼容，集成 FlashInfer 的高效融合 kernel（如 layer norm）。使用 FlashAttention (v2) 作为注意力计算后端，利用 CUTLASS 编写自定义 CUDA kernel。修改包括：
  (1) KV cache 管理策略：不再使用 vLLM 标准的全量 GPU KV cache，而是在 pre-filling 后将 value 移至 CPU pinned memory，GPU 仅保留低秩 key 投影、landmarks 和 outliers；
  (2) Attention 计算流程：解码时不使用标准 FlashAttention 的完整 QK^T 计算，而是先通过 Q×L^T 的 landmark 近似 attention 选择 top-k chunk，再对选中的 chunk 执行精确 attention；
  (3) 数据搬运调度：使用 CUDA multiple streams 实现 key 重建（GPU 计算）与 value 取回（PCIe 传输）的 overlap，等效带宽可达 7.2 TB/s（理论分析）；
  (4) Temporal cache：维护 chunk index 的最近访问记录，检测相邻步的 chunk 重复，跳过已缓存 KV 对的取回和重建。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源地址：https://github.com/ByteDance-Seed/ShadowKV 。框架输入到硬件执行全过程：

  **输入**：用户请求到达，包含 prompt tokens 和生成参数（max_tokens, temperature 等）。

  **Pre-filling 阶段**（每层重复）：
  1. 输入 tokens → PyTorch Embedding → Transformer Block
  2. QKV Projection：X @ W_qkv → Q, K, V（GPU 计算，Tensor Core）
  3. 对 pre-RoPE K 执行 SVD → 存储 A, B 在 GPU（cuSOLVER/cuBLAS）
  4. K 经 RoPE → K_RoPE，分 chunk 计算 landmark 和 outlier
  5. V（非 outlier 部分）→ CPU pinned memory（cudaMemcpy，PCIe）
  6. Landmarks L、outlier KV 对保留 GPU
  7. Q, K_RoPE(完整), V(完整) → FlashAttention 计算 prefill attention（所有 token 对其他 token）
  8. FFN 计算

  **Decoding 阶段**（每层每步重复）：
  1. 新生成 token → Q 向量
  2. Q × L^T → 近似 chunk attention scores（自定义 CUDA kernel，GPU 计算）
  3. Softmax + TopK → 选择 k 个 chunk indices
  4. 并行执行（CUDA multi-stream）：
     - Stream 1: Gather(A, I) × B → 重建 sparse K cache（GPU，Tensor Core）
     - Stream 2: cudaMemcpy(V_CPU[I], V_sparse, H2D)（PCIe 传输）
  5. Temporal cache index scan → 跳过已缓存 chunk
  6. 拼接 K = [K_outlier; RoPE(K_sparse); K_new], V = [V_outlier; V_sparse; V_new]
  7. Q, K, V → FlashAttention（仅对选中 chunk + outliers 计算）
  8. FFN → 输出 logits → 采样下一个 token

  **关键调度决策**：batch size 由可用 GPU 内存决定 — Full Attention 在 60K context 仅能容纳 batch=8，ShadowKV 可容纳 batch=48（6× 提升）。Sparse budget k=256 对应 1.56% 的 128K 序列。等效带宽分析公式：
  $$\widetilde{B} = \frac{2SB_{\mathrm{GPU}}}{S/C + 2(K+O)C + (1-\alpha)KCB_{\mathrm{GPU}}/B_{\mathrm{PCIe}}}$$
  对 S=128K, C=8, K=256, O=48, α=0.6 计算得等效带宽 7.2 TB/s。
