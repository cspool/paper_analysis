## PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling

- 属于算法pipeline的实现是什么？实验比较什么？
  PyramidKV 是一种动态 KV cache 压缩算法，基于"Pyramidal Information Funneling"现象（底层注意力分散在全局上下文、中层注意力逐步收窄到局部区域、顶层注意力集中在少量关键 token 上），提出：(1) 跨层不均匀 KV cache budget 分配：底层分配更多 cache、顶层分配更少 cache，按算术序列递减；(2) 基于 attention score 的 token 选择：保留最后 α 个 token（instruction tokens）的 KV cache，然后根据这些 token 对其他 token 的 attention score（sum over instruction tokens）选择 top-k^l tokens 保留。公式：k^l = k^0 - (k^0 - k^{m-1})/(m-1) × l，其中 k^{m-1} = k^{total}/(β·m)，k^0 = 2·k^{total}/m - k^{m-1}。超参 β=20（控制金字塔形状）、α=8（instruction tokens 数）。

  实验比较：(a) PyramidKV vs H2O / SnapKV / StreamingLLM / FullKV 在 LongBench 17 个数据集上的性能（KV cache size 64/96/128/256/2048）；(b) Needle-in-a-Haystack 长上下文检索实验（Mistral-7B @32k, LLaMa-3-8B @8k, LLaMa-3-70B @8k, cache size 64/96/128）；(c) KV cache 内存节省实验（LLaMa-3-8B, seq_len=8192, batch=1, fp16）；(d) 推理速度对比（H2O/SnapKV/StreamingLLM/PyramidKV latency）；(e) 额外开销分析（allocation time vs selection time vs total inference time）；(f) PyramidKV+MInference 混合方法；(g) 与 PyramidInfer 对比（arithmetic vs geometric decay, token re-evaluation vs discard）；(h) Ablation: 算术/几何/指数/熵/Gini 分配策略对比，α 和 β 超参数敏感性；(i) 128K context 扩展实验（Llama-3-8B-Instruct-Gradient-1048k）；(j) vLLM 集成 throughput 实验。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 GPU。评估模型：LLaMa-3-8B-Instruct、Mistral-7B-Instruct、LLaMa-3-70B-Instruct。推理精度：fp16。batch size=1（内存消耗实验），greedy decoding（性能评估）。延迟测量实验 prompt length 512/1024/2048/4096，generation length 512/1024/2048/4096。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMa-3-8B-Instruct、LLaMa-3-70B-Instruct、Mistral-7B-Instruct（Jiang et al., 2023）。数据集和 Bench：(a) LongBench（Bai et al., 2023），17 个数据集覆盖 6 类任务——Single-Document QA (NarrativeQA, Qasper, MultiFieldQA-en)、Multi-Document QA (HotpotQA, 2WikiMultihopQA, MuSiQue)、Summarization (GovReport, QMSum, MultiNews)、Few-shot Learning (TREC, TriviaQA, SAMSum)、Synthetic (PassageCount, PassageRetrieval-en)、Code Completion (LCC, RepoBench-P)。平均输入长度 1,235-18,409 tokens。Metrics: F1 (QA), Rouge-L (Summarization), Accuracy (Synthetic), Edit Sim (Code)；(b) Needle-in-a-Haystack（Fact Retrieval Across Context Lengths），最多 32K context (Mistral-7B)、8K (LLaMa-3)；(c) 128K context 实验使用 Llama-3-8B-Instruct-Gradient-1048k。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/Zefan-Cai/PyramidKV。算法 pipeline 如下：

  **Step 1 - Budget Allocation（Pre-computed，在推理前一次性计算）**：
  ```python
  # m = 总层数, k_total = 总 KV cache budget, beta = 20 (超参)
  k_top = k_total / (beta * m)                    # 顶层 budget
  k_bottom = 2 * k_total / m - k_top              # 底层 budget
  # 按算术序列分配各层 budget
  for l in range(m):
      k_l = k_bottom - (k_bottom - k_top) / (m - 1) * l  # k^l = k^0 - Δ · l
      budgets[l] = int(k_l)
  # 最后 α 个 token 的 KV 在所有层均保留（instruction tokens）
  ```

  **Step 2 - Attention Score Calculation（Prefill 阶段）**：
  ```python
  # Q, K: [batch, heads, seq_len, d_k]
  # α = 8 (instruction tokens)
  A = softmax(Q @ K.T / sqrt(d_k))                # [heads, seq_len, seq_len]
  # 对每层每个 head 计算 token 重要性分数
  for h in range(num_heads):
      s_h = A_h[-α:, :].sum(dim=0)                 # sum attention from instruction tokens
      # s_h[i] = Σ_{j ∈ [n-α, n]} A_{ij}^h
  ```

  **Step 3 - KV Cache Selection**：
  ```python
  # 对每层 l，每个 head h，选 top-k^l tokens
  for l in range(m):
      k_l = budgets[l]
      for h in range(num_heads):
          _, top_indices = torch.topk(s_h, k_l)    # 选最高分的 k^l 个 token
          K_selected[l, h] = K[l, h, top_indices]
          V_selected[l, h] = V[l, h, top_indices]
      # 使用 torch.gather 执行 eviction（非 in-place 操作）
  ```

  **Step 4 - 推理时使用压缩后的 KV cache**：
  ```python
  # Decoding 阶段仅使用 K_selected, V_selected
  # 位置编码保持原始位置不变（不滚动 position）
  # 各层独立维护其 compressed KV cache
  output = attention(Q_new, K_selected[l], V_selected[l])
  ```

  **Step 5 - vLLM 集成（附录 R）**：
  每个 sequence 的 block table 扩展为 per-layer block table，使得每层可以独立检索其 KV cache，而非使用固定内存偏移。解决 naive 实现中不同层不同 budget 导致的 cache fragmentation 问题。Throughput 结果：PyramidKV 在 compression 下 throughput 随 input context length 增长而降低（因小 chunk 的内存分配/释放/移动/访问导致碎片化），需 per-layer page-out 解决。
