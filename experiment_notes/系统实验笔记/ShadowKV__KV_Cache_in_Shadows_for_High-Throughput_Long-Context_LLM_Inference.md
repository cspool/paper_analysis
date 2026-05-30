## ShadowKV__KV_Cache_in_Shadows_for_High-Throughput_Long-Context_LLM_Inference

- 属于Serving调度的实现是什么？实验比较什么？
  ShadowKV 是一个面向长上下文 LLM 的高吞吐推理 serving 系统。其核心服务于调度层实现为：(a) **GPU 显存管理**：prefilling 阶段对每层 pre-RoPE key cache 做在线 SVD 低秩压缩（rank=160），仅保留低秩投影矩阵 A 和 B 在 GPU，value cache 全量 offload 至 CPU，仅保留检测到的 outlier chunk（0.3%）的 KV 对在 GPU，将 GPU KV cache 显存占用降低 >6×。(b) **请求调度与 batch 扩容**：由于 GPU KV cache 显存大幅减少，相同 GPU 可容纳更大 batch size（从 2-8 扩至 12-48），支持从 60K 到 488K 更长上下文的高吞吐服务。(c) **CPU-GPU 数据传输调度**：decoding 阶段使用 CUDA multi-stream 将 key cache 低秩重建（GPU 计算）与 value cache CPU 抓取（PCIe 传输）重叠，隐藏 PCIe 延迟，降低 sparse attention 的 decoding overhead。实验比较 Full Attention（GPU 显存内完整 KV cache）在不同 batch size 下的吞吐，以及 Quest、Loki、InfiniGen 在相同 sparse budget 下的效率，展示 ShadowKV 可支持 6× larger batch size，吞吐提升最高 3.04×。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 80GB PCIe GPU，GPU 内存带宽 2 TB/s，PCIe 带宽 31.5 GB/s。

- 开源Serving框架是什么。修改了什么。
  基于 PyTorch + HuggingFace Transformers 构建 serving 框架，集成了 FlashAttention（FlashAttention-2）、FlashInfer（fused kernels、layer norm）、vLLM（PagedAttention）中的高效 kernel，以及 CUTLASS。ShadowKV 在此基础上修改/新增了以下 serving 层组件：
  1. **Prefilling 阶段 SVD 计算**：对每层 pre-RoPE key cache 调用 SVD 分解，保留 rank-r 低秩投影
  2. **KV cache 存储管理**：替换原有 GPU-side KV cache 存储策略，GPU 端保留 A、B、landmarks L、outlier KV；CPU 端存储完整 value cache V_CPU
  3. **Decoding 阶段 KV 选择与重建调度**：用 landmarks 近似注意力选出 top-k chunks → 并发调度 key 重建（GPU）与 value 抓取（CPU→GPU）→ 多流重叠
  4. **Cache mechanism**：利用相邻 decoding step 间 KV 选择的高命中率（~60%），维护命中缓存减少重复计算和 PCIe 传输

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源地址：https://github.com/ByteDance-Seed/ShadowKV（Apache 2.0）。Serving 框架全链路执行过程：
  
  **请求到达与 Batch 组建**：
  1. 用户提交推理请求（prompt + max_new_tokens），框架将多条请求按到达顺序组建成 batch
  2. 由于单个请求的 GPU KV cache 仅占原有的 ~1/7（低秩键 + landmark + outlier vs 完整 KV），batch size 可从 Full Attention 的 2-8 扩至 12-48
  
  **Prefilling 阶段**（以一条 prompt 128K tokens、batch=24 为例）：
  3. 输入 tokens 经过 embedding → 对每个 Transformer layer：
     a. QKV 投影：Q, K, V = W_Q·X, W_K·X, W_V·X
     b. Pre-RoPE K 在线 SVD：对 K_{pre-RoPE} ∈ R^{128K × 128} 做 truncated SVD，得 A ∈ R^{128K×160}、B ∈ R^{h_kv×160×128} 存 GPU
     c. Post-RoPE K 分 chunk（chunk_size=8，得 16K chunks），每个 chunk 算均值作为 landmark L ∈ R^{h_kv×16K×128} 存 GPU
     d. Cosine similarity 计算每 chunk 内各 token 与均值的相似度，选出 48 个 outlier chunk，其 KV 对存 GPU static cache
     e. 其余 value cache V offload 到 CPU（通过 PCIe），对应 landmark L 保留在 GPU
     f. FlashAttention 完成 prefill attention（本文保留完整 prefill attention）
  
  **Decoding 阶段**（autoregressive generation，每步生成一个 token）：
  4. 新 token 经 embedding + QKV 投影得 query Q ∈ R^{24 × h_q × 1 × 128}
  5. **Landmark-based KV 选择**：
     - P ← MatMul(Q, L^T) → Softmax(P/√d) → sum over query heads → max over kv_group
     - ArgTopK 选出 top-k=256 个 chunk indices I
  6. **缓存命中检查**：对比上一步选择的 chunk indices 与当前步，命中部分（~60%）跳过
  7. **并发执行（CUDA multi-stream）**：
     - Stream 1 (GPU compute)：K_sparse ← MatMul(Gather(A, I_miss), B)，RoPE(K_sparse)
     - Stream 2 (PCIe→GPU)：V_sparse ← Gather(V_CPU, I_miss) 从 CPU 内存通过 PCIe 读取
     - 两个操作时间接近，重叠后总延迟 ≈ max(PCIe 传输, 低秩重建) 而非二者之和
  8. **Attention 计算**：K ← [K_outlier; K_sparse; K_new]，V ← [V_outlier; V_sparse; V_new]，FlashAttention(Q, K, V)
  9. FFN → 输出投影 → next token
  10. 新 token 的 pre-RoPE key 投影到同一低秩空间（K_new × Ψ，其中 Ψ 为 prefilling SVD 的右奇异矩阵），追加到低秩状态
  
  **作用**：通过将 GPU KV cache 显存降低 6-7×，支持 6× larger batch size，吞吐从 Full Attn 的 80.78 tokens/s（batch=4, 122K context, Llama-3.1-8B）提升至 245.90 tokens/s（batch=24），加速 3.04×，甚至超过无限 GPU 显存假设下的吞吐（134.30 tokens/s, batch=Inf）。
