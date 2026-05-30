## Communication-Efficient Sparsely-Activated Model Training via Sequence Migration and Token Condensation

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是 LUFFY 的 **Sequence Migration Controller** 和 **Token Condensation Scheduler** 两个运行时调度模块：
  
  1. **Sequence Migration Controller**：在 combine phase 中，一个指定的 controller 机器收集所有 GPU 的 token 分布信息（token_to_sequence、token_to_gpu、sequence_to_gpu 三张哈希表），运行迁移算法决定每个 sequence 应在哪个 GPU 重构以最小化跨 GPU token 拉取流量，并优化后续 attention 计算的效率（通过将相似长度 sequence 聚集到同一 GPU 减少 padding zeros）。迁移决策通过 `torch.distributed.rpc` API 指导 GPU 间的 token 交换。
  
  2. **Token Condensation Scheduler**：每个 GPU 维护一个独立 CUDA stream 运行 token condensation scheduler。使用 DGL (Deep Graph Library) API 构建 token 图，通过 edge-wise 函数 `edge_sim_calculation` 并行计算 token 间相似度，维护 `token_to_token` 哈希表记录凝聚映射。

  实验比较（与其他模块混合评估）：
  - **Ablation Study (Fig. 9)**：Token Condensation Only vs Sequence Migration Only vs Both，在三种 MoE 模型上分析各调度模块的独立贡献
  - **Sensitivity Analysis (Fig. 10a, 10b)**：候选 GPU 数 q 对 traffic 和 computation time 的影响；cost model 估计精度（平均误差 ~5%）
  - **Performance Breakdown (Table III)**：Computation time vs Communication time 分解，验证调度优化的效果

- 后端平台是什么，配置是什么。
  16× NVIDIA V100 GPU (16GB HBM)，PCIe 互联，无 NVLink。Ubuntu 20.04 (kernel 5.15)，NVIDIA driver 525.85，CUDA 11.7，cuDNN 8.6.0。底层通信使用 PyTorch distributed (NCCL backend)。

- 评估性能的软件/脚本是什么。修改了什么。
  **评估工具**：PyTorch Profiler 采集各 phase 的 computation/communication time。端到端训练迭代时间（iteration time）作为主要性能指标，所有方法在相同配置下归一化到 Vanilla 计算 speedup。
  
  **LUFFY 对 PyTorch 的修改（~4.5K 行 Python 代码）**：
  1. **Sequence Migration Controller**：新增集中式 controller 模块
     - 收集分布式信息：在 expert running 期间并行收集 token_to_sequence、token_to_gpu、sequence_to_gpu 映射
     - 迁移算法：对每个 sequence i，估算迁移到各 GPU j 的 token 拉取流量 f_{i,j} → 选择 top-q 候选 GPU H^i → 通过 cost model T_att(B, L) 评估 attention 计算时间增长 → 选择使计算成本增长最小的 GPU j*
     - 通过 `torch.distributed.rpc` 更新 sequence_to_gpu 哈希表，指导 combine phase 的 token 路由
  2. **Token Condensation Scheduler**：每 GPU 独立 CUDA stream
     - DGL 图构建：node = token (features: expert index + token embedding)，edge = token pair
     - 三步快速相似度测量：expert activation filtering → historical similarity lookup → cosine similarity 计算
     - 图剪枝 + 连通分量：根据自适应阈值 h_t 删除低相似度边 → 每子图保留 degree 最高的 token，其余凝聚
     - 维护 token_to_token 哈希表指导 dispatch/combine 的 token 替换
  3. **Cost Model**：$T_{att}(B, L) = (3BLd² + 2BL²d) / P$，其中 P 通过 profiling attention 层多次获得
  
  评估原理：每个 training iteration 各 phase 的时间通过 `torch.cuda.Event` 精确测量 → batch training time = attention compute + token condensation + dispatch all-to-all + expert compute + sequence migration + combine all-to-all → speedup = Vanilla_time / LUFFY_time

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源情况：论文未提供公开开源代码仓库。LUFFY 以 PyTorch plug-and-play 插件形式实现，依赖 DGL (Deep Graph Library) 和 PyTorch distributed。

  **Sequence Migration 调度全流程（以 4 GPU, MoE-TransformerXL, 1 个 training batch 为例）**：

  ```
  === 输入状态 ===
  - GPU 0-3 各持有 1 个 expert 和完整的 attention 参数
  - Batch 包含 B=8 个 sequences，长度不等
  - Expert parallelism: top-2 gating

  === Phase 1: Attention Computation (所有 GPU 本地) ===
  每个 GPU 处理其分配的 sequences:
    for seq in local_sequences:
        Q, K, V = Linear_QKV(seq)  # [L, d] → [L, 3d]
        attn_out = Softmax(QK^T/√d) V  # FlashAttention
    → 输出: token embeddings for all local sequences

  === Phase 2: Token Dispatch (All-to-All) ===
  Router 计算 gate → token 路由到 expert 所在 GPU:
    GPU 0 → tokens for Expert 1,2,3,4 → all-to-all scatter
    → GPU 0 收到: tokens routed to Expert 0 (local) + others' tokens

  === Phase 3: Expert Computation (与 Sequence Migration 并行) ===
  GPU 0 计算 Expert 0 的 FFN:
    out = expert_ffn(received_tokens)  # W_gate → SiLU → W_up → × → W_down
  同时，Controller 执行迁移算法:
    
    # 收集分布信息（与 expert running 并行）
    for each GPU g:
        gather(token_to_gpu[g])  # 每个 token 在哪个 GPU 被 expert 处理
    
    # Algorithm 1: Sequence Migration
    for each sequence i (i = 1..B):
        # Step 1: 估算迁移到各 GPU 的 combine 流量
        for each GPU j:
            f_{i,j} = count({token t in seq i | token_to_gpu[t] != j})
        H^i = top_q(argmin(f_{i,j}))  # 候选 GPU 集合
        
        # Step 2: 选择最小 attention cost 增长的 GPU
        for each GPU j in H^i:
            B_{j←i} = current_sequences_on_gpu(j) + [seq i]
            L_{j←i} = max_length(B_{j←i})
            s_{i,j} = T_att(B_{j←i}, L_{j←i}) - T_att(B_j, L_j)
        
        j* = argmax(s_{i,j})  # min cost growth
        
        # 检查容量: GPU 可容纳更多短序列，但有限的长序列
        if GPU j* has capacity:
            sequence_to_gpu[i] = j*
  
  === Phase 4: Sequence-Aware Combine (基于迁移决策) ===
  原始 Vanilla: 所有 token 拉回原 GPU → inter-GPU traffic 大
  LUFFY: Controller 广播 sequence_to_gpu 映射 →
    GPU 0 sequences: [seq_0, seq_3] (migrated here)
    → tokens of seq_3 pulled from GPU 1,2,3 to GPU 0
    → token pulling traffic 大幅减少
    → sequences 在迁移目标 GPU 上重构

  === Phase 5: Next Block Attention (优化的 batch) ===
  GPU 0 收到 seq_0 (len=250) + seq_3 (len=230):
    → 相似长度 → 仅需 padding 20 个 zeros
    → vs Vanilla: 混合长短序列 → padding 浪费大
  
  === 性能输出 ===
  - Communication time reduction: 1.76×-3.72× vs Vanilla
  - Computation time reduction: 1.16×-1.57× vs Vanilla  
  - Overall speedup: 1.51×-2.73× vs Vanilla (16 experts)
  ```

  **Token Condensation Scheduler 执行流程（单 GPU, CUDA Stream）**：

  ```
  === 输入 (Attention 输出后) ===
  - tokens: N 个 token embeddings [N, d]
  - gate_output: {token_idx → expert_idx}
  - historical_similarity: 来自 block (b-1) 的相似度缓存
  - loss_prev: 上一 iteration 的 loss 值

  === Step 1: DGL Graph Construction (CUDA Stream) ===
  g = dgl.graph((src_nodes, dst_nodes))
  g.ndata['expert'] = gate_output  # token → expert mapping
  g.ndata['embedding'] = tokens    # token embeddings

  === Step 2: Fast Edge Weight Computation ===
  g.apply_edges(edge_sim_calculation):
      for each edge (u, v):
          # 2a: Expert activation filter
          if g.ndata['expert'][u] != g.ndata['expert'][v]:
              return {'weight': 0.0}
          
          # 2b: Historical similarity lookup (O(1))
          s_prev = historical_cache.get((u, v))
          if s_prev is not None:
              if s_prev > S1: return {'weight': 1.0}
              if s_prev < S2: return {'weight': 0.0}
          
          # 2c: Real cosine similarity (O(d))
          emb_u = g.ndata['embedding'][u]
          emb_v = g.ndata['embedding'][v]
          sim = dot(emb_u, emb_v) / (norm(emb_u) * norm(emb_v))
          return {'weight': sim}

  === Step 3: Adaptive Threshold ===
  l_norm = (loss_ini - loss_prev) / loss_ini
  h_t = 1.0 / (1.0 + exp(l_norm))
  # 若 loss 下降大 → l_norm 大 → h_t 小 → 凝聚更多 token

  === Step 4: Graph Pruning + Component Selection ===
  # 移除 weight < h_t 的边
  g_sparse = g.edge_subgraph(g.edges()[g.edata['weight'] >= h_t])
  
  # 连通分量分析
  components = dgl.connected_components(g_sparse)
  
  # 每个分量保留 degree 最大的 token
  for comp in components:
      degrees = g_sparse.in_degrees(comp)
      rep = comp[argmax(degrees)]
      for token in comp:
          if token != rep:
              token_to_token[token] = rep

  === Step 5: Dispatch with Condensation ===
  for each expert_idx:
      # 只发送 representative tokens
      tokens_to_send = filter(representatives, expert_routing)
      all_to_all_send(tokens_to_send, expert_owner[expert_idx])

  === Step 6: Expert Computation ===
  # 更少的 token → 更少的计算
  for expert_idx, tokens_received:
      expert_out = expert_ffn(tokens_received)

  === Step 7: Combine with Expansion ===
  for each token:
      if token in token_to_token:
          out[token] = expert_out[token_to_token[token]]
      else:
          out[token] = expert_out[token]
  # token similarity preserved after expert: ~95% pairs change < 0.2
  ```

  **与 Baseline 的通信模式对比**：
  - Vanilla: dispatch all-to-all + combine all-to-all = 2 × all-to-all × 全量 tokens
  - LUFFY: dispatch (condensed tokens) + combine (migrated sequences) = 显著减少的 all-to-all 流量
  - 核心差异: LUFFY 不移动 expert 参数（保持最大 expert parallelism），而是通过 sequence migration 改变 combine 的 token 路由目标 + token condensation 减少 dispatch 的 token 数量
