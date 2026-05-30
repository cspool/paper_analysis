## Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs

- 属于算法pipeline的实现是什么？实验比较什么？
  提出两套KV cache压缩算法：(1) **Chunked Eviction**：将post-fill eviction方法（PyramidKV、SnapKV）适配到chunked pre-filling场景，使KV在pre-filling的每个chunk后可被提前evict，包括Naive Chunked Eviction（直接对每个chunk独立执行eviction heuristic）和Patched Chunked Eviction（在每个chunk末尾拼接最后k个prompt token作为query计算重要性分数）；(2) **PruLong**：一种end-to-end的attention head specialization方法，将attention heads二分类为retrieval heads（保留完整KV cache）和streaming heads（仅保留local window + attention sinks），通过hard concrete重参数化和next-token prediction loss端到端学习head类型，支持精确target sparsity正则化。实验比较：(1) PruLong vs DuoAttention vs PyramidKV（naive/patched）在HELMET和LongProc共8类task category上的critical KV footprint（保持≥90% full attention性能的最小KV footprint）；(2) 不同training data（Pre-training Mix vs BookSum Passkey vs Context Synthesis）和training stage（pre-SFT vs post-SFT）的消融；(3) chunk size sensitivity（8K vs 32K pre-filling chunk size）。

- 硬件平台是什么，配置是什么。
  PruLong训练：论文未明确说明具体GPU型号。评估：论文未明确说明具体GPU型号（使用PyTorch推理）。附录F提供了hardware metrics（throughput和peak memory），在装有特定GPU（未明确型号）的机器上测量，DuoAttention/PruLong peak memory约16-29 GiB，PyramidKV/SnapKV约17-47 GiB（因task而异）。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-3.1-8B-Instruct（主要）；ProLong-8B-Base（ablations for SFT stage analysis）。
  训练数据：
  - PruLong默认：Gao et al. (2025)的stage-II continued pre-training mix（长度512K，截断至128K），short:long data ratio 40%:60%
  - 对比数据：BookSum Passkey（DuoAttention原始数据）、Context Synthesis（Zhu et al., 2025）
  Benchmark：
  - HELMET（128K context setting，long input→short output）：Recall（JSON KV, RULER MK Needle/UUID, RULER MV）、RAG（NQ, TriviaQA, PopQA, HotpotQA）、Re-ranking（MS MARCO NDCG@10）、Many-shot ICL（TREC Coarse/Fine, NLU, BANKING77, CLINC150）、Long-document QA（NarrativeQA, ∞QA）、Summarization（∞Sum, Multi-LexSum）
  - LongProc（short/long input→long output）：HTML→TSV（structured prediction, 12K-38K input, 1K-10K output），Travel Planning（multi-city itinerary generation, 6K→3K）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/princeton-pli/PruLong

  **PruLong算法核心流程**：

  **Phase 1: Attention Head Type Classification**
  ```
  // 每层每头 i∈[1,L], j∈[1,H] 学习二值mask z_{i,j} ∈ {0,1}
  // z=1 → retrieval head (full attention), z=0 → streaming head (local window + sinks)

  // 混合attention机制
  Attn_{i,j}(Q,K,V) = z_{i,j} × Attn_full(Q,K,V) + (1-z_{i,j}) × Attn_streaming(Q,K,V)

  // Attn_full: 标准causal attention，key序列为所有历史token
  // Attn_streaming: 仅attend到local window (W=1024 token) + attention sinks (S=128 token)
  ```

  **Phase 2: Mask Learning via Hard Concrete Reparameterization**
  ```
  // 将z_{i,j}建模为Bernoulli随机变量，参数π_{i,j}
  // 使用hard concrete distribution [Louizos et al., 2018] 重参数化

  // 前向采样
  u ~ Uniform(0, 1)  // truncated at (1e-6, 1-1e-6)
  s = σ( (1/τ) × log(u/(1-u)) + log α )  // τ = 3/2, Gumbel reparameterization
  g̃ = l + s × (r - l)  // stretch to [-0.1, 1.1]
  z̃ = min(1, max(0, g̃))  // hard sigmoid → support {0,1}

  // 参数log α_{i,j}可训练，通过梯度下降优化
  ```

  **Phase 3: End-to-End Training Objective**
  ```
  // 公式(2): min-max optimization

  max_{λ1,λ2} min_{π} E_{z~Bern(π)} [
      1/N Σ_{n=0}^{N-1} log p_θ(x_{n+1} | x_{:n}; z)
  ] + λ1(s(π) - t) + λ2(s(π) - t)²

  // 第一项: next-token prediction loss（LM loss）
  // 第二项: Lagrangian penalty，约束sparsity s(π) → target t
  // λ1, λ2: 可训练Lagrange乘子（gradient ascent优化）
  // s(π) = 1 - 1/(LH) Σ σ(α_{i,j} - log(-l/r))  // 闭式期望L0 sparsity
  ```

  **Phase 4: Sparsity Warmup & Discretization**
  ```
  // 训练配置
  Target sparsity t: 从0 warmup到t_∞（如0.7），over 800/1000 steps
  LR (log α): 1.0
  LR (λ1, λ2): 1.0
  LR schedule: linear warmup first 10% → linear decay to 1% peak
  Batch size (tokens): 1,048,576
  Sequence length: 131,072
  Training steps: 1,000
  Model weights: frozen（ablation中unfrozen时LR=1e-5）
  Adam (β1, β2): (0.9, 0.95)
  Window size: 1024, Sink size: 128

  // 训练后离散化
  // 对任意target sparsity k%，将top k%的log α设为+∞ (z=1)，其余设为-∞ (z=0)
  ```

  **Phase 5: Inference**
  ```
  // 对每个attention head (i,j):
  if z_{i,j} == 1:  // retrieval head
      attn_i_j = FlashAttention(Q, K_full_history, V_full_history)
  else:  // streaming head
      K_local = K[-1024:, :]  // 最近1024个token
      K_sinks = K[:128, :]     // 前128个attention sink token
      attn_i_j = FlashAttention(Q, concat([K_sinks, K_local]), concat([V_sinks, V_local]))
      // evict非local非sink的KV → memory saving
  ```

  **Chunked Eviction (PyramidKV/SnapKV) 核心流程**：
  ```
  // Naive Chunked Eviction
  // 在chunked pre-filling的每个chunk后立即执行eviction
  for each chunk c of size C:
      X_c = tokens[c*C : (c+1)*C]
      K_c, V_c = forward(X_c)  // 计算当前chunk的KV
      scores = attention_score(K_c[-k:], V_c[-k:])  // 最后k=64个token的attention收分
      smoothed_scores = moving_average(scores)  // 平滑
      keep_indices = top-p%(smoothed_scores)  // 保留p%最重要KV
      evict(KV[keep_indices之外])  // 逐出其余KV

  // Patched Chunked Eviction
  // 关键区别：每个chunk末尾拼接prompt的最后k个token作为query
  for each chunk c of size C:
      X_patched = concat([X_c, prompt_tail_k])  // 拼接最后k个prompt token
      K_c, V_c = forward(X_patched)
      scores = attention_score(K_c[-k:], V_c[-k:])  // 用拼接的prompt tail计算重要性
      // 仅保留X_c对应的KV（丢弃补丁token的KV，除非是最后一个chunk）
      keep_indices = top-p%(smoothed_scores)
      evict(KV[keep_indices之外])

  // PyramidKV: 按pyramidal结构分配各层KV budget
  // 浅层budget多，深层budget少（后续层压缩率更高）
  budget[l] = base_budget × (1 - l/L)^γ  // γ控制pyramid陡峭程度

  // KV Group优化（GQA场景）
  // 对每个KV group内的多个query head取attention mean后再做eviction
  // 避免为每个query head独立选择KV → 减少8x内存（Llama-3.1-8B GQA g=4, 2 query groups/key）
  ```

  **全栈执行流程（PruLong inference on Llama-3.1-8B-Instruct）**：
  1. Pre-filling阶段（chunked, chunk_size=32K）：每个chunk中，各attention head根据z_{i,j}决定使用full attention还是streaming attention；streaming heads evict非local/sink的KV，retrieval heads保留全部KV
  2. Decoding阶段：对每个新生成的token，所有heads重新计算attention；streaming heads维持fixed-size KV cache（window + sinks），retrieval heads的KV cache线性增长
  3. KV Footprint计算：聚合所有timestep的un-evicted KV entries数量，归一化至full causal attention

  **关键性能数据（Table 2 - Critical KV Footprint %）**：
  | Task      | DuoAttention | PruLong | PyramidKV(Naive) | PyramidKV(Patched) |
  |-----------|-------------|---------|------------------|--------------------|
  | Recall    | 58.0        | 46.0    | >93.0            | 64.0               |
  | RAG       | 49.0        | 37.0    | 44.0             | <34.0              |
  | Re-Rank   | 69.0        | 61.0    | >94.0            | 94.0               |
  | ICL       | 49.0        | 38.0    | 42.0             | <36.0              |
  | LongQA    | 60.0        | 49.0    | 62.0             | <35.0              |
  | Summ      | 63.0        | 59.0    | 53.0             | 49.0               |
  | HTML      | 87.0        | 83.0    | 97.0             | 97.0               |
  | Travel    | 91.0        | 93.0    | >98.0            | >98.0              |

  **PruLong vs DuoAttention 消融（Table 3, 70% sparsity）**：
  - Pre-training Mix data: PruLong Recall 91.4 vs DuoAttention 38.6（+52.8, key differentiator）
  - PruLong在natural long-context data上表现优异，DuoAttention依赖synthetic passkey data
  - PruLong不更新model weights，保持instruction-following能力

  **Real Hardware Metrics（Appendix F, 70% sparsity）**：
  - DuoAttention: throughput 10.0×10⁻² req/s (Recall), peak memory 26.6 GiB
  - PruLong: throughput 10.8×10⁻² req/s (Recall), peak memory 26.3 GiB
  - PyramidKV+P+C: throughput 8.0×10⁻² req/s, peak memory 33.7 GiB
  - PruLong consistently achieves highest throughput and lowest peak memory across tasks
