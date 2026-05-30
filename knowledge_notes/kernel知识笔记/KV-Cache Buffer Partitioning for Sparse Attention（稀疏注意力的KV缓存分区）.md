## KV-Cache Buffer Partitioning for Sparse Attention（稀疏注意力的KV缓存分区）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
KV-Cache Buffer Partitioning 是一种将 KV-cache 在 GPU 显存中组织为三个连续缓冲区的设计策略，用于支持稀疏注意力中的高效 token 选择：**Sink Buffer**（前 S 个 attention sink token，始终保留）、**Local Window Buffer**（循环队列存储最近 W 个 token，每步更新）、**Important Buffer**（由 token 重要性预测器动态填充的稀疏选中 token）。三个 buffer 在物理内存中连续排列，使得 attention kernel 可以一次性访问连续内存块，避免因 token 选择导致的碎片化内存 gather/scatter 操作。TokenButler 使用 S=128 sink tokens + W=256 local window tokens + B~8K important tokens 的配置。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
GPU Kernel 视角的 buffer 组织和调度：

```
GPU Memory Layout:
  ┌─────────────────────────────────────────────────────────┐
  │ Sink Buffer  │ Important Buffer  │ Local Window Buffer  │
  │  (S tokens)  │    (B tokens)     │    (W tokens)        │
  │  contiguous  │    contiguous     │    circular           │
  └─────────────────────────────────────────────────────────┘
  Attention Kernel 输入：单个连续指针 → [S + B + W] tokens

Token Lifecycle in Buffers:
  Step t: 生成新 token → 写入 Local Window Buffer[t % W]
  Step t+N: token 从 window 驱逐 → 批量子投影 K_proj = K[N] @ W_K
            → predictor 评估重要性 → 可能进入 Important Buffer
  Important Buffer: 每 prediction_interval 步完全刷新

Kernel Pseudocode (Importance-Guided Gather):
  // 每 prediction_interval 步执行
  for each consumer_layer l:
      // 1. 计算所有候选 token 的重要性分数（低维）
      scores = Q_imp[slot] @ K_proj[l].T      // GEMM: [H, d'] × [d', L_kv]
      
      // 2. 构造 selection mask
      candidate_mask = ones(L_kv)               // 所有已投影 token
      candidate_mask[0:S] = 0                   // 排除 Sink（已隐式保留）
      candidate_mask[-W:] = 0                   // 排除 Window（已隐式保留）
      
      // 3. Top-B selection + neighbor expansion
      selected = topk(scores * candidate_mask, B)
      selected = cluster_aware_neighbor_expand(selected, B)  // → 2B
      
      // 4. Gather KV pairs → Important Buffer（连续写入）
      K_important[l] = gather(K_cache[l], selected)  // contiguous write
      V_important[l] = gather(V_cache[l], selected)

  // Attention Kernel（每步执行）
  K_dense = concat(K_sink, K_important, K_window)   // 连续内存
  V_dense = concat(V_sink, V_important, V_window)
  output = FlashAttention(Q, K_dense, V_dense)       // 标准 kernel
```

Timing Breakdown (Llama-3.1-8B, A6000, 128K context, budget=8K):
- Attention Kernel: ~恒定（因 budget 固定 = S+2B+W ≈ 8K+ tokens）
- Importance Score Computation: O(L_kv · d'), 随 context 增长但斜率低
- KV Gather: O(B), 与 sparse budget 成正比
- Query Prediction (MLP): ~恒定

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：
1. **连续内存分配**：三个 buffer 预分配为一块连续 GPU 内存，通过偏移量指针访问各段，避免运行时内存碎片。
2. **Important Buffer 刷新**：每 prediction_interval 步覆盖写入，无需保留旧 selection。使用 cudaMemcpy 或 custom gather kernel 将选中 KV pairs 复制到 Important Buffer。
3. **延迟 Key 投影**：token 在 Local Window 期间自然被 dense attention 覆盖，无需投影；仅在驱逐时批量投影（N 个 token × D 维 GEMM），利用 cuBLAS 批量 GEMM 的 HBM 带宽优势。投影结果追加到 K_proj 搜索空间。
4. **与 FlashAttention 兼容**：三 buffer 拼接后的连续内存可直接作为 FlashAttention 的 K/V 输入，无需修改 attention kernel。
5. **Sink Buffer 为静态**：prefill 后即固定，无需运行时更新。

涉及论文标题：
- TokenButler: Token Importance is Predictable
