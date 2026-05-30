## SentenceKV: Efficient LLM Inference via Sentence-Level Semantic KV Caching

- 属于算法pipeline的实现是什么？实验比较什么？
  SentenceKV 是一种句子级语义 KV 缓存管理方法。在 prefilling 阶段，将输入按标点符号分割成句子桶，用最后 N 个 token（observation window，默认 N=32）对前面所有 token 计算注意力分数并求和得到重要性 α_i，保留 top ⌊r·τ⌋ 个 token（r 为 semantic keeping factor，默认 2-3；τ 为 token budget，默认 1024），对每个句子桶计算平均 key 向量 $\bar{k}_{s,h}$ 存于 GPU，完整 KV 对 offload 到 CPU。在 decoding 阶段，维护 sentence cache $Q_s$ 累积当前生成句子的 query 向量，一旦遇到句子边界就计算平均 query $\bar{q}$，与所有句子桶的 $\bar{k}_{s,h}$ 做内积相似度排序，按相似度从高到低选取句子桶直到 token 数达到 τ，将对应 KV 对从 CPU 加载回 GPU 计算 attention。实验比较 Full KV、H2O、SnapKV、Quest、InfLLM、ShadowKV 等方法在 LongBench（单文档/多文档 QA、few-shot、合成、代码任务）、PG-19（PPL）、NIAH（检索准确率）、RULER（8 个子任务检索准确率）上的表现，以及 GPU 内存和推理延迟的消融。

- 硬件平台：2 块 NVIDIA H100 80GB GPU + 2 块 NVIDIA H100 NVL 96GB GPU。

- 模型：Llama-3-8B、Llama-3.1-8B-Instruct、Llama-3.2-3B-Instruct、Longchat-v1.5-7B。

- 数据集和 benchmark：PG-19（语言建模 PPL，最长 32k tokens）、LongBench（单文档 QA/Multi-doc QA/few-shot/合成/代码）、NIAH（Needle-In-A-Haystack，最长 8000 tokens）、RULER（8 个子任务，最长 64k tokens）。

- 开源情况：代码开源 https://github.com/zzbright1998/SentenceKV。

  算法 pipeline 使用例子（伪代码）：
  ```
  # === Prefilling Phase ===
  sentences = split_by_punctuation(prompt)  # 按标点分句
  obs_window = prompt[-N:]                  # 最后 N=32 个 token
  
  for token_i in prompt[:-N]:
      alpha[token_i] = sum over heads, obs_tokens of attn(obs_t, token_i)
  
  selected_tokens = top_k(alpha, k=int(r * tau))  # r=2-3, tau=1024
  discard rest
  
  for sentence_s in sentences:
      retained_in_s = selected_tokens ∩ sentence_s
      for head_h in range(H):
          k_bar[s][h] = mean(k[x][h] for x in retained_in_s)  # Eq.1
          # k_bar 存 GPU 作为语义向量
  
  offload_to_CPU(K_selected, V_selected)  # KV 对存 CPU
  
  # === Decoding Phase ===
  Q_s = []  # sentence cache
  for t in range(max_new_tokens):
      x_t, q_t = generate_next_token()
      Q_s.append(q_t)
      q_bar = mean(Q_s)  # Eq.2
      
      scores = []
      for sentence_s in sentences:
          for head_h in range(H):
              S = dot(q_bar, k_bar[s][h])  # 内积相似度
          scores.append((s, aggregate_similarity))
      
      sorted_buckets = sort_by_similarity(scores, descending=True)
      retrieved = []
      for bucket in sorted_buckets:
          if len(retrieved) + len(bucket.retained_tokens) <= tau:
              retrieved.extend(bucket.retained_tokens)
          else:
              break
      
      load_from_CPU_to_GPU(K[retrieved], V[retrieved])  # shape: (tau, H, d)
      O_t = softmax(q_t @ K_retrieved.T / sqrt(d)) @ V_retrieved  # Eq.3
      
      if is_sentence_boundary(x_t):  # 遇到句号/问号等
          Q_s = []  # 重置 sentence cache
  ```

  关键参数：τ=1024（约为 32k 上下文全量 KV 的 3%），r∈{2,3}，observation window N=32。在 256k tokens 时 GPU 内存从 Full KV 的 89.71GB 降至 52.71GB，延迟从 84.9ms 降至 17.8ms。NIAH 上检索准确率 97.5%（τ=128），远超 SnapKV 的 78.2%。
