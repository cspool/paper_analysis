## ShadowKV__KV_Cache_in_Shadows_for_High-Throughput_Long-Context_LLM_Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  ShadowKV 提出两阶段算法：(1) **Prefilling**：对 pre-RoPE key cache 做 SVD 低秩分解（rank r=160），仅保留低秩投影矩阵 A∈ℝ^{b×s×r} 和 B∈ℝ^{b×h_kv×r×d} 在 GPU 上，将 value cache 全部 offload 到 CPU；同时对 post-RoPE key cache 按 chunk size=8 切分并计算 chunk 均值作为 landmarks L∈ℝ^{b×h_kv×s/c×d}；通过 chunk 内 cosine similarity 检测 outlier chunks（约占 0.3%，即 48/128K），outlier 的 KV 对保留在 GPU 作为 static cache。(2) **Decoding**：对每个 query，先用 landmarks 计算 chunk attention score P = MatMul(Q, L^T)，Softmax 后按 KV head 聚合选出 top-k chunks（k=256，1.56% sparse budget=256×8/128K），从 CPU fetch 对应 value cache，同时从低秩投影 MatMul(Gather(A, I), B) 重建 key cache，利用 CUDA multi-stream 重叠二者；额外维护 cache mechanism 利用 KV cache temporal locality（相邻 decoding step 命中率约 60%），通过 index scan 检测 miss chunks 减少重复计算和数据搬运。实验比较 Full Attention、Quest、Loki、InfiniGen 等 baseline，在 RULER（128K-1M）、LongBench（>4K）、Needle In A Haystack（16K-1M）、InfiniteBench 上评估准确率，在 A100 上评估吞吐（tokens/s）和 batch size。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 80GB PCIe GPU（GPU 显存带宽 2 TB/s，PCIe 带宽 31.5 GB/s）。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-3.1-8B（8 KV heads）、Llama-3-8B-1M（8 KV heads）、GLM-4-9B-1M（4 KV heads）、Yi-9B-200K（4 KV heads）、Phi-3-Mini-128K、Qwen2-7B-128K、Llama-3-70B-1M。Benchmarks：RULER（13 tasks，包括 N-S1、N-S2、N-MK1、N-MK2、N-MQ、N-MV、QA-1、QA-2、VT、FWE）、LongBench（NarratQA、MultiFQA、HotpotQA、MuSiQue、DuRead、GovRep、SAMSum、PassRetr、LCC）、Needle In A Haystack（16K-1M）、InfiniteBench。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/ByteDance-Seed/ShadowKV（Apache 2.0，301 stars，Python 35.1% + C++ 25.7% + CUDA 17.4%）。
  
  **Prefilling 算法（Algorithm 1）**：
  ```
  Input: K, K_RoPE, V ∈ R^{b × h_kv × s × d}, rank r, chunk size c, outlier num o
  // Step 1: SVD on pre-RoPE key cache
  A ∈ R^{b × s × r}, B ∈ R^{b × h_kv × r × d} ← SVD(K, rank=r)
  // Step 2: Segment post-RoPE keys into chunks, compute mean as landmarks
  C ∈ R^{b × h_kv × s/c × d} ← Reduce(K_RoPE, chunk_size=c)
  // Step 3: Compute cosine similarity within each chunk to find outliers
  S ← CosineSimilarity(C, K_RoPE)  // each chunk's tokens vs chunk mean
  I ← ArgTopK(-Min(S, dim=-1), o)  // lowest cosine similarity = outliers
  K_outlier, V_outlier ← Gather(K_RoPE, V, I)
  // Step 4: Offload values to CPU, keep non-outlier landmarks on GPU
  V_CPU ← V \ V_outlier
  L ← C \ Gather(C, I)
  // GPU retains: A, B, L, K_outlier, V_outlier
  ```
  
  **Decoding 算法（Algorithm 2）**：
  ```
  Input: A, B, L, V_CPU, Q ∈ R^{b × h_q × s_q × d}, K_outlier, V_outlier
  // Step 1: Compute approximate attention scores via landmarks
  P ∈ R^{b × h_q × s_q × n_c} ← MatMul(Q, L^T)
  S ∈ R^{b × h_q × s_q × n_c} ← Softmax(P / sqrt(d))
  S1 ∈ R^{b × h_q × n_c} ← sum(S, dim=-2)        // aggregate over query tokens
  S2 ∈ R^{b × h_kv × n_c} ← max_kv_group(S1)      // max over GQA group
  I ∈ R^{b × h_kv × k} ← ArgTopK(S2, k)           // top-k chunk indices
  // Step 2: Overlapped ops (multi-stream)
  V_sparse ← Gather(V_CPU, I)                       // PCIe fetch from CPU
  K_sparse ← MatMul(Gather(A, I), B)                // reconstruct from low-rank
  K ← [K_outlier; RoPE(K_sparse); K]                // concat outlier + sparse + new token
  V ← [V_outlier; V_sparse; V]
  // Step 3: Standard FlashAttention on sparse KV pairs
  Output ← FlashAttention(Q, K, V)
  ```
  
  核心设计：pre-RoPE keys 极低秩（rank 160 即可 6× 压缩无精度损失）→ 存低秩投影；post-RoPE keys 空间局部性 + outlier 稀少（0.3%）→ chunk 均值做 landmarks + outlier 静态缓存；temporal locality（60% hit rate）→ cache mechanism 减少重复操作；CUDA multi-stream → overlap key 重建与 value 抓取。理论等效带宽 7.2 TB/s（A100 原生带宽 2 TB/s 的 3.6×）。
