## KV-Compress: Paged KV-Cache Compression with Variable Compression Rates per Attention Head

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 KV-Compress，一种基于 PagedAttention 的 KV cache 压缩方法，支持可变压缩率（variable-head-rate compression）和分块级 eviction。核心算法设计：(1) Query-Group Compression：针对 GQA 模型，将每个 key 的 eviction 指标在所有属于该 key 的 query group 的 queries 上聚合（Sum_{h in H_k}），而非先 repeat KV cache 到 query head 数量再压缩，使得同样 max-cache-size C 下 KV-Compress 实际持有 1/r 的 KVs（r 为 query head 与 KV head 之比，Llama-3/Mistral 中 r=4），实现 4x 额外压缩；(2) Paged KV Block Eviction（MoveCache 算法）：在 block size b 的 paged KV cache 中，先确定每 head 的 (be)^{th} 最小 metric 值 m(h,e)，再跨 head 排序候选 block eviction，按 budget E_s 选出总 metric 最低的 blocks，通过 MoveCache 重排物理 cache 使 evicted blocks 可被释放；(3) Squared Attention Metric (L2)：使用 Σ(A_hij)² 替代标准 ΣA_hij 作为 eviction metric，在 LongBench 上各变体一致优于 L1 聚合；(4) 两个变体——KVC-full：聚合全部过去 queries 的 squared attention（排除 local window v=10）；KVC-w：仅聚 observation window w=8 的 squared attention + max-pooling p=7；(5) Continual Compression：在 decoding 阶段持续累积新生成 token 的 squared attention 到已有 metrics 中；(6) Variable per-head and per-layer compression rates。实验比较 LongBench 16 子集上与 H2O、SnapKV、PyramidKV、Ada-SnapKV、Ada-PyramidKV 的性能（Mistral-7B-Instruct-v0.2, Llama-3.1-8B-Instruct），以及不同压缩率 (1x-64x) 下 continual compression 的 LongBench 性能百分比（Llama-3.1-8B 和 Llama-3.1-70B-FP8）。

- 硬件平台是什么，配置是什么。
  NVIDIA L4 GPU（Llama-3.1-8B throughput 和 LongBench 实验），gpu_memory_utilization=0.9，max-model-length=19,000；NVIDIA H100 GPU（Llama-3.1-70B-FP8 throughput 和 LongBench 实验），gpu_memory_utilization=0.96，max-model-length=33,000。KVC-full 在 H100 上需 gpu_memory_utilization=0.6 以预留 metric 计算空间，逐 query block 计算（block size=1024）。默认 block size b=16，vLLM v0.6.0 eager mode。

- 模型是什么。数据集和bench分别是什么。
  模型：Mistral-7B-Instruct-v0.2（32 层，GQA r=32/8=4）、Llama-3.1-8B-Instruct（GQA r=32/8=4）、Llama-3.1-70B-Instruct-FP8（FP8 量化，GQA r=64/8=8）。Benchmark：LongBench（16 子集，6 类别——Single-Doc QA: NarrativeQA, Qasper, MultiFieldQA-en；Multi-Doc QA: HotpotQA, 2WikiMultihopQA, MuSiQue；Summarization: GovReport, QMSum, MultiNews；Few-shot Learning: TREC, TriviaQA, SAMSum；Synthetic: PassageCount, PassageRetrieval-en；Code: LCC, RepoBench-P）。评测 max-cache-size C={128, 256, 512, 1024}，baseline 方法 C 定义保留 C×H×H 个 KVs（repeat 后），KV-Compress 保留 C×H×H/r 个 KVs（非 repeat cache）。Throughput benchmark：256 prompts，fixed output=500 tokens，varied input lengths {500, 1000, 2000, 4000, 6000, 8000, 10000, 12000}。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/IsaacRe/vllm-kvcompress/tree/main（vLLM 集成 fork from v0.6.0）。PyramidKV baseline 实现：https://github.com/IsaacRe/PyramidKV。算法 pipeline 如下：

  **Prefill 阶段（Eviction Metric 计算）**：
  ```
  输入：input sequence length L, model with l layers, H kv heads (GQA group count), r query heads per group
  超参：observation window w=8, pooling p=7, excluded local query window v=10

  for each layer m in 1..l and each KV head h_k in 1..H:
      # 计算完整 attention 矩阵 A ∈ R^{r × L × L}（causal masked）
      # GQA: 该 key head 的 query group H_k = {h: r·h_k ≤ h < r·(h_k+1)}

      # KVC-w: 有限 observation window + squared attention + max-pooling
      for each query head h in H_k:
          for query i = L-w to L:           # 仅最后 w 个 queries
              for key j = 1 to i:            # causal range
                  M_{h_k,j} += (A_{h,i,j})^2
      for each key j:                         # max-pooling over keys
          M_{h_k,j} = max(M_{h_k, t}) for t in [j-p/2, j+p/2]

      # KVC-full: 全部 queries（排除 v 个 local queries）
      for each h in H_k:
          for key j = 1 to L:
              for query i = j+v to L:         # 排除 key j 之后的 v 个 local queries
                  M_{h_k,j} += (A_{h,i,j})^2
  ```

  **Block-level Eviction（MoveCache, Algorithm 1）**：
  ```
  输入：physical K_u, V_u ∈ R^{N×b×d}, metrics M ∈ R^{N×b}, logical indices P, evict count E_s blocks

  # Step 1: 排序 metrics 以获得 per-head per-eviction-block 的最大 metric
  M1 = M.view(-1)  # [Nb], 每个元素对应一个 KV
  M2 = sort(M1, by=(head_id, metric))  # 按 head 分组
  M3 = reshape(M2, [N, b])  # 每行: 该 head 的 b 个最低 metric
  # m(h,e) = M3[o_h+e-1, b-1]  # head h 的 e-th eviction block 的最大 metric

  # Step 2: 跨 head 排序候选 block evictions
  M4 = sort(M3, by=M3[:, b-1])  # 按每 block 的最大 metric 排序

  # Step 3: 为序列 s 选择 E_s 个 block eviction
  W = zeros(N, b)
  for the first E_s blocks (offset O_s for sequence s):
      mark all KVs in these blocks for eviction in W

  # Step 4: 重排物理 cache（MoveCache）
  eviction_range = [end - E_s*b, end]
  i = end, j = end - 1
  while i > eviction_range_start:
      while W[i] == 0: i -= 1        # 找到 eviction range 内的非 evicted KV
      while W[j] == 1: j -= 1        # 找 eviction range 外的 evicted KV
      # swap: 将 range 外的 evicted KV 移到 range 内
      swap(K_u[P[i]], K_u[P[j]]), swap(V_u[P[i]], V_u[P[j]]), swap(M[P[i]], M[P[j]])
      i -= 1, j -= 1
  # 释放 eviction range 内连续的 E_s 个 block
  ```

  **压缩调度策略**（Section 4.2.3）：
  - 方案 1: 每 c 次 model iteration 压缩一次
  - 方案 2: total uncompressed tokens 超过阈值时压缩
  - 方案 3: 当有序列被新 prefill 时压缩（选中）
  - 方案 4: 当 preemption 即将发生时压缩（选中）
  - 最终使用方案 3+4 组合。
  - Sort 操作 overhead：额外内存约 8× sorted tensor 大小，runtime 在 1.7e8 元素后线性增长。限制每次压缩 iteration 中总 KV 数不超过阈值。

  **Continual Compression（Equation 20）**：
  M_{h_k,j}^{(cc)} = M_{h_k,j}^{(pool)} + Σ_{i=L_c}^{L_c+t} Σ_{h∈H_k} (A_{h,i,j})^2
  其中 M^{(pool)} 为 prefill 阶段 metric，L_c 为 input context 长度，t 为当前 decoding step。

  **关键性能**：
  - Llama-3.1-8B, LongBench C=128: KVC avg 46.26 vs PyramidKV 45.97, SnapKV 45.93（同时 KV-Compress 仅使用 1/4 KVs）
  - Mistral-7B, LongBench C=128: KVC-w8-L2 avg 37.64 vs Ada-SnapKV 36.71（同时仅使用 1/4 KVs）
  - 8B@6000 tokens, compression rate 32x: 4.93x throughput over vanilla vLLM on L4
  - 70B-FP8@6000 tokens, compression rate 64x: 2.14x throughput over vanilla vLLM on H100
  - 8B compression rates 8x-64x 保持 negligible impact，Summarization 任务最敏感
  - 70B 模型对压缩更不敏感，多数 non-summarization 任务 64x 压缩保持 >90% 性能
