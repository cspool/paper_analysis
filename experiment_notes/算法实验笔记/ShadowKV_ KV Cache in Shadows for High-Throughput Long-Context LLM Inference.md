## ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 ShadowKV，一种面向长上下文 LLM 推理的高吞吐系统，核心算法包括三部分：(1) **低秩 Key Cache 压缩**：对 pre-RoPE key cache 执行在线 SVD 分解（rank r=160），仅存储低秩投影矩阵 A∈R^{b×s×r} 和 B∈R^{b×h_kv×r×d}，压缩比约 6× 且无精度损失；(2) **Landmark 近似稀疏 Attention**：将 post-RoPE key cache 按 chunk_size=8 分块，每块计算均值作为 compressed landmark L∈R^{b×h_kv×s/c×d}，解码时用 Q 与 L 计算近似注意力分数选择 top-k chunk；(3) **Outlier 静态缓存**：通过 chunk 内 cosine similarity 检测近似最差的 outlier chunk（仅 0.2-0.3%），将其完整 KV 对作为 static cache 保留在 GPU 上以保证精度。总体 sparse budget 仅为 1.56%（k=256, o=48, S=128K 下）。

  实验比较：(1) Accuracy 实验：在 RULER (128K)、LongBench (>4K tokens)、Needle In A Haystack (16K-1M) 上对比 Full Attention、Quest、Loki、InfiniGen，ShadowKV 以 1.56% sparse budget 保持与 Full Attention 一致的精度；(2) 消融实验：sparse budget 变化、chunk size 选择 (4/8/16/32)、pre-RoPE key rank 选择 (32/64/96/128/160/192/256)、outlier 数量影响；(3) 兼容性实验：与 MInference 结合加速 prefill 阶段；(4) 多轮对话实验：Multi-turn NIAH 对比 SnapKV、StreamingLLM；(5) 精度敏感性实验：FP8 精度下与 baseline 对比。

  伪代码核心流程：

  ```
  # Pre-filling 阶段 (Algorithm 1):
  K = X @ W_k^T                    # pre-RoPE key, shape: [b, h_kv, s, d]
  A, B = SVD(K, rank=r)            # A: [b, s, r], B: [b, h_kv, r, d]
  K_RoPE = RoPE(K)                 # post-RoPE key
  C = Reduce(K_RoPE, chunk_size=c) # chunk mean landmarks [b, h_kv, s/c, d]
  S = CosineSimilarity(C, K_RoPE)  # 每 chunk 内 cosine similarity
  I = ArgTopK(-Min(S, dim=-1), o)  # 选择 o 个最差近似的 chunk 作为 outlier
  K_outlier, V_outlier = Gather(K_RoPE, V, I)
  V_CPU = V \ V_outlier            # 其余 value 下放 CPU
  L = C \ Gather(C, I)             # 非 outlier chunk 的 landmarks 保留 GPU

  # Decoding 阶段 (Algorithm 2):
  P = MatMul(Q, L^T)               # 用 landmarks 近似注意力分数 [b, h_q, s_q, n_c]
  S = Softmax(P / sqrt(d))
  S1 = sum(S, dim=-2)              # 沿 query 维度求和
  S2 = max_kv_group(S1)            # GQA 情况下聚合到 KV heads
  I = ArgTopK(S2, k)               # 选择 top-k chunk
  V_sparse = Gather(V_CPU, I)      # 从 CPU 取回 value（PCIe 传输）
  K_sparse = MatMul(Gather(A, I), B) # 从低秩投影重建 key（与上面重叠执行）
  K = [K_outlier; RoPE(K_sparse); K_new]  # 拼接 outlier + 重建 + 新 token
  V = [V_outlier; V_sparse; V_new]
  O = FlashAttention(Q, K, V)      # 标准 attention 计算
  ```

- 硬件平台是什么，配置是什么。
  NVIDIA A100 GPU (80GB)。GPU 内存带宽 2 TB/s，PCIe 带宽 31.5 GB/s (PCIe 4.0 x16)。部分实验在 8×A100 上进行。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-3.1-8B、Llama-3-8B-1M (Gradient extended)、GLM-4-9B-1M、Yi-9B-200K、Phi-3-Mini-128K、Qwen2-7B-128K、Llama-3-70B-1M。所有模型均使用 GQA (Grouped Query Attention)。
  数据集/Benchmark：RULER（13 个子任务，含 NIAH 变体、multi-key、multi-query、multi-value、variable tracking 等）、LongBench（6 大类 21 任务：单/多文档 QA、摘要、代码补全、信息检索等）、Needle In A Haystack（上下文窗口 16K-1M tokens）、InfiniteBench（10 任务，平均长度 214K）。RULER 128K 上下文为主，LongBench 仅使用超过 4K tokens 的样本。部分测试使用 PG-19 数据集分析低秩特性。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/ByteDance-Seed/ShadowKV 。
