## Communication-Efficient Sparsely-Activated Model Training via Sequence Migration and Token Condensation

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是 **LUFFY**，一个通信高效的分布式 MoE 训练系统，包含两个核心算法级技术：
  
  **Token Condensation（令牌凝聚）**：利用 MoE 训练中被路由到同一 expert 的 token 之间存在高相似度（例如 MoE-TransformerXL 中约 62% 的 token 对被路由到同一 expert 且高度相似）的观察，提出令牌凝聚算法来消除冗余 token 传输。
  - **快速相似度测量（Fast Similarity Measurement）**：将 token 和相似度关系建模为全连接图。三步策略：(1) 被路由到不同 expert 的 token 直接标记为不相似（边权重=0），因为不同 expert 设计为处理不同输入类型；(2) 极端相似/不相似的 token 对根据历史相似度信息（前一 block 的相似度值）直接分配边权重为 1 或 0，跳过计算；(3) 仅对剩余高度不确定的 token 对进行真实 cosine 相似度计算。
  - **自适应凝聚策略（Adaptive Token Condensation）**：动态调整相似度阈值 $h_t = 1/(1+\exp(l_{norm}))$，其中 $l_{norm} = (l_{ini} - l_{t-1})/l_{ini}$。训练早期 $l_{norm}$ 小 → 阈值大 → 保留更多 token 保证收敛；训练后期 $l_{norm}$ 大 → 阈值降低 → 凝聚更多 token 减少通信。

  实验比较：
  - **End-to-End Performance (Fig. 8)**：LUFFY vs Vanilla (DeepSpeed expert parallelism)、EXT (Janus expert transfer)、HYT (FasterMoE hybrid)，在 MoE-TransformerXL/MoE-BERT-Large/MoE-GPT2 三种模型上，expert 数量 2/4/8/16 下的 batch training time speedup
  - **Performance Breakdown (Table III)**：Computation time vs Communication time 详细分解
  - **Ablation Study (Fig. 9)**：Token Condensation only vs Sequence Migration only vs Both，分析各组件对不同模型的贡献
  - **Convergence Evaluation (Table IV)**：MoE-TransformerXL on WikiText-103 (PPL)、MoE-BERT-Large on SQuAD (F1)、MoE-GPT2 on SAMSum (ROUGE-1)，对比 static threshold (h=0.3, h=0.8) vs adaptive LUFFY vs Vanilla
  - **Sensitivity Analysis (Fig. 10)**：候选 GPU 数 q、cost model 精度、fast similarity measurement 参数 S₁/S₂ 的敏感性

- 硬件平台是什么，配置是什么。
  16× NVIDIA V100 GPU (16GB HBM)，PCIe 互联。Ubuntu 20.04 (kernel 5.15)，NVIDIA driver 525.85，CUDA 11.7，cuDNN 8.6.0。

- 模型是什么。数据集和bench分别是什么。
  模型：
  - **MoE-TransformerXL**：18-block decoder，d_model=1024，d_hidden=4096/4090，参数 0.44B-2.55B
  - **MoE-BERT-Large**：24-block encoder，d_model=768/708，d_hidden=3072，参数 0.54B-3.36B
  - **MoE-GPT2**：12-block decoder，d_model=768，d_hidden=3072，参数 0.18B-0.97B
  - Expert 数量配置：2/4/8/16 per MoE layer，top-2 gating，batch size=64
  
  数据集与 Benchmarks：
  - MoE-TransformerXL: WikiText-103 → Perplexity (PPL↓)
  - MoE-BERT-Large: SQuAD → F1 (F1↑)
  - MoE-GPT2: SAMSum → ROUGE-1 (ROUGE-1↑)

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源情况：论文未提供公开开源代码仓库。LUFFY 基于 PyTorch 实现，约 4.5K 行 Python 代码，以 plug-and-play 插件形式提供。未搜索到 GitHub 仓库链接。

  **LUFFY Token Condensation 算法 Pipeline**：

  ```
  # === 符号说明 ===
  # T: tokens, E: experts, d: hidden dimension
  # N_gpu: number of GPUs, B: number of sequences

  # === Token Condensation (Dispatch & Combine Phase) ===
  def token_condensation(tokens, gate_output, block_idx, prev_similarity, loss_prev):
      """
      tokens: [num_tokens, d] - token embeddings after attention
      gate_output: {token_idx -> expert_idx} - gate routing result
      block_idx: current MoE block index
      prev_similarity: historical similarity matrix from block (b-1)
      loss_prev: loss value from previous training iteration
      """
      
      # Step 1: Build token similarity graph
      graph = build_token_graph(tokens, gate_output)
      
      # Step 2: Fast similarity measurement
      for (token_i, token_j) in graph.edges:
          # 2a: Different experts → dissimilar (weight = 0)
          if gate_output[token_i] != gate_output[token_j]:
              graph[token_i][token_j].weight = 0
              continue
          
          # 2b: Historical similarity check
          s_prev = prev_similarity.get((token_i, token_j), None)
          if s_prev is not None:
              if s_prev > S1:  # extremely similar
                  graph[token_i][token_j].weight = 1
                  continue
              if s_prev < S2:  # extremely dissimilar
                  graph[token_i][token_j].weight = 0
                  continue
          
          # 2c: Compute real cosine similarity for uncertain pairs
          sim = cosine_similarity(tokens[token_i], tokens[token_j])
          graph[token_i][token_j].weight = sim
      
      # Step 3: Adaptive condensation threshold
      l_norm = (loss_ini - loss_prev) / loss_ini
      h_t = 1.0 / (1 + exp(l_norm))
      
      # Step 4: Condense similar tokens
      # Remove edges with weight < h_t → sparse graph with subgraphs
      for subgraph in connected_components(graph, threshold=h_t):
          # Keep token with highest degree, condense others
          representative = argmax(degree(subgraph))
          for token in subgraph \ {representative}:
              token_to_token[token] = representative  # mapping table
      
      # Step 5: Dispatch — only send representative tokens
      for expert_idx in unique_experts:
          tokens_to_send = [t for t in tokens 
                           if gate_output[t] == expert_idx 
                           and t in representatives]
          all_to_all_send(tokens_to_send, target_gpu=expert_owner[expert_idx])
      
      # Step 6: Expert computation (fewer tokens → less computation)
      for expert_idx, received_tokens in received_tokens_by_expert:
          expert_output = expert_ffn(received_tokens)  # Fused MoE kernel
      
      # Step 7: Combine — expand condensed tokens using representative output
      for token in all_tokens:
          if token in token_to_token:
              # Use representative's expert output
              token_output[token] = expert_output[token_to_token[token]]
          else:
              token_output[token] = expert_output[token]
      
      return token_output
  ```

  **关键张量计算与通信量变化**：
  
  以 MoE-TransformerXL，4 experts，batch=8 为例：
  - Baseline (Vanilla Expert Parallelism): All-to-All 通信量 = 3.19 GB/batch，通信时间 327ms (18.1%)
  - LUFFY Token Condensation: 凝聚约 62% 相似 token 后，通信量显著减少
  - LUFFY 总体: Communication speedup 1.76×-3.72× vs Vanilla

  **Token Condensation 的核心计算复杂度**：
  - Naive pairwise: O(T²·d) 对所有 token 对 → 不可行
  - Fast measurement: 大部分 token 对通过 expert activation (O(T)) 和历史相似度 (O(1) lookup) 直接判断 → 仅剩余少量不确定对需 O(d) 余弦计算
  - 图凝聚: O(T log T) 通过连通分量分析
