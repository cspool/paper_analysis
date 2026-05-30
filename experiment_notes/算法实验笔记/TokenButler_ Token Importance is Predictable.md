## TokenButler: Token Importance is Predictable

- 属于算法pipeline的实现是什么？实验比较什么？
  TokenButler 提出一个轻量级预测器（<1% 参数开销），通过 attention distillation 学习预测 token 重要性，实现 query-aware 的细粒度 KV-cache 稀疏访问。核心实现：在固定的深度间隔 G（producer_frequency=4）处放置 producer layer，用二层 MLP（hidden=512）从 hidden states 预测低维 importance queries（d'=16），与经过学习投影矩阵 W_K 降维后的真实 KV-cache keys 做点积得到 token 重要性分数，在固定 budget 下筛选 top-k token 送入 attention kernel。训练时冻结 LLM，仅训练预测器，用 cross-entropy loss 蒸馏 masked causal attention distribution。推理时将 KV-cache 划分为 Sink Buffer（前 S 个 token）、Local Window Buffer（最近 N 个 token 循环缓冲区）和 Important Buffer（TokenButler 动态填充），保证 attention kernel 访存连续。引入 prediction interval（每 N 步运行一次预测器）和 neighbor fetching（对选中 token 扩展空间邻居）以摊销预测开销。
  实验比较：vs StreamingLLM（静态 recency）、H2O（注意力分数驱逐）、SnapKV（滑动窗口池化注意力驱逐）、Quest（逐页 min-max token 幅度）、PyramidKV、KIVI（KV 量化）、SingleSVD、xKV、MiniCache、KVzip、TokenSelect。评估指标：synthetic co-reference accuracy/coverage、RULER 各项子任务得分、LongBench 各项子任务得分、AIME24 accuracy、per-token decode latency、throughput。

- 硬件平台是什么，配置是什么。
  Nvidia A6000（主要评测平台，用于 latency 测量和训练），Nvidia H100（throughput 评测，Figure 8）。训练在单张 A6000 上完成。CPU offloading 场景的 latency 在 >=256K context 下评测。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-3.1-8B-Instruct、Qwen2.5-7B-Instruct-1M、Llama-3.2-1B/3B、DeepSeek-R1-Distill-Llama-8B。
  训练数据：C4 (RealNewsLike, 90k)、FineWeb-Edu (sample-10BT, 90k)、CodeParrot-Clean (90k)、BABILong (context {2k,4k,8k,16k})，训练序列长度 1024（预测器通过 key-cache 投影直接泛化到 64K context）。
  Benchmark：synthetic co-reference resolution（100 个虚构地点名，100^4 组合空间）、RULER（64K context）、LongBench（64K context）、AIME24（reasoning）、WikiText2（perplexity/recall 评测）。

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/abdelfattah-lab/TokenButler
  算法 pipeline 伪代码：
  ```
  # 训练阶段（冻结 LLM）
  for each producer layer p in {0, G, 2G, ...}:
      H = hidden_states_at_layer_p          # [B, L, E]
      Q_imp = MLP(LayerNorm(H))             # [B*H, G, L, d']
      for consumer layer l in [p+1, p+G]:
          slot = (l-1) % G
          K_proj = K_cache[l] @ W_K[l]      # [B, H_kv, L, d']
          A_pred = Q_imp[:,slot] @ K_proj.T # [B*H, L, L]
          A_true = teacher_attention[l]      # 冻结模型产出
          loss += CrossEntropy(softmax(A_true), softmax(A_pred))
  
  # 推理阶段（decode step t）
  # 每 i 步运行一次预测器（prediction interval）
  if t % i == 0:
      for each producer layer p:
          H = hidden_states_at_layer_p
          Q_imp = MLP(LayerNorm(H))
          for consumer layer l:
              K_proj = K_cache[l] @ W_K[l]
              scores = Q_imp[slot] @ K_proj.T  # [H, L]
              # 排除 sink + local_window 中的 token
              scores[candidates] = top_k(scores[candidates], B)
              # 将选中的 KV pairs 迁移到 Important Buffer
              # neighbor fetching: 为每个选中 token 扩展空间邻居
              selected_tokens = expand_with_neighbors(selected_tokens)
  # Attention Kernel: 拼接 [Sink | Important | Local_Window]
  attn_output = FlashAttention(Q, K[selected], V[selected])
  
  # 延迟投影优化：新 token 在 local window 中停留 N 步后，
  # 才批量投影其 key 到 d' 维空间加入预测器搜索空间
  if token_evicted_from_window:
      K_proj_batch = K[recent_N] @ W_K  # 批量投影，利用 HBM 带宽
  ```
  关键张量维度：Q_imp ∈ R^{(B*H) x G x L x d'}，K_proj ∈ R^{B x H_kv x L x d'}，d'=16 << D=128/head_dim。预测器参数占比：Llama-3.1-8B 为 29.4M (0.368%)，Qwen2.5-7B 为 20.9M (0.299%)。
