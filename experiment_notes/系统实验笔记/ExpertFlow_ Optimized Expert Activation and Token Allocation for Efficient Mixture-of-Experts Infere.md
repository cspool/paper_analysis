## ExpertFlow: Optimized Expert Activation and Token Allocation for Efficient Mixture-of-Experts Inference

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 **ExpertFlow**，一个面向单 GPU 内存受限场景的 MoE 推理系统，包含三个协同组件：
  (1) **Routing Path Predictor (RPP)**：T5-style encoder-decoder 架构，在单次前向传播中预测所有 token 在所有 MoE 层的 expert 激活路径，输出形状为 (B, S, L, E) 的激活概率矩阵。训练使用 binary cross-entropy 的多标签分类任务，预测器大小 7.21 MB（FFN dim=2048, hidden size=32）。在多数 in-domain 场景下达到 >90% 预测准确率，跨域仅下降 5-10%。
  (2) **Token Scheduler (TS)**：基于 K-means 聚类将具有相似路由路径的 token 重新分组到同一 batch 中。以两个相邻 batch 的 2T 个 tokens 为输入，构造 routing path 相似度矩阵，通过最小化 batch 级 expert 激活数（公式: min Σ(R1 + R2)）将 token 重新分配到两个等大小 batch 中，减少 active expert 数并提高 per-expert token 负载。包含自适应 KV-Cache 管理（Merge + Reindex）和 Dual-Batch Inference Pipeline 以隐藏 overhead。
  (3) **Expert Cache Engine (ECE)**：包含 Predictive Locality-aware Expert Caching (PLEC) 和 Real-time Correction。PLEC 基于 RPP 预测自适应分配各层 cache slot，预取预测需要的 expert；运行时检测误预测并执行优先交换，与 compute 重叠以减少 I/O 等待。

  实验比较：(a) in-domain throughput vs Cache-MoE/SE-MoE/Pregated-MoE（Switch 系列在 WMT16, Mixtral-8 在 XSUM, Qwen1.5 在 Alpaca, Deepseek-MoE 在 AIME2024）；(b) cross-domain throughput（Qwen1.5 在 WMT16/XSUM，RPP 用 Alpaca 训练）；(c) 峰值 GPU memory vs All-in-GPU；(d) RPP 预测准确率 vs TLP/SLP baselines；(e) Cache hit ratio PLEC vs LRU；(f) TS 对 throughput 的 ablation。

- 硬件平台是什么，配置是什么。
  单卡 NVIDIA A40 GPU (48 GB memory)，CPU 为 Intel(R) Xeon(R) Gold 6338 @ 2.00GHz。

- 开源Serving框架是什么。修改了什么。
  ExpertFlow 是独立设计的 MoE 推理系统，从零构建，而非基于现有开源 Serving 框架（如 vLLM、SGLang）修改。其核心组件（RPP、TS、ECE）均为全新实现。RPP 使用 T5-style encoder-decoder 架构；TS 使用 K-means 聚类进行 token rebatching；ECE 实现 PLEC + real-time correction 的预测驱动缓存策略。论文未明确说明基于哪个 Serving 框架集成。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  **未开源**（论文未提供代码链接，GitHub 搜索无公开仓库）。

  以下基于论文描述给出 ExpertFlow 从输入到硬件执行的全过程：

  ```
  === 离线阶段：RPP 训练 ===
  对每个 (task, MoE model) 组合:
    采样 10,000 个输入序列
    每个序列运行 MoE 模型 3 次，收集 30,000 个 (input, output, routing_path) triple
    每个 routing path 编码为 r ∈ {0,1}^{L×E} 的 binary matrix
    训练 RPP (T5 encoder-decoder, FFN=2048, hidden=32, 7.21MB):
      Loss = BCE(r, p)  where p = RPP(input)
  
  === 在线推理：Dual-Batch Pipeline ===
  Input: batch_0, batch_1 (各 B 个 sequence, 每个 S 个 tokens)
  
  Step 1 - RPP 预测 (与上一 scheduling unit 的 MoE 执行并行):
    # T5 encoder 编码全输入序列
    encoder_output = T5_encoder.encode(concat(batch_0_inputs, batch_1_inputs))
    # T5 decoder + L 个 light-weight heads (每层一个) 一次性输出
    p = RPP_decoder(encoder_output)  # shape: (2B, S, L, E)
    # p[l][e] = 预测 expert e 在 layer l 被激活的概率
    activation_matrix = (p > threshold)  # 二值化

  Step 2 - TS Token 重新分组 (CPU, <10ms):
    # 2T 个 tokens 的 routing path: r_i ∈ {0,1}^{L×E}
    # 计算 Hamming distance 相似度矩阵 S ∈ R^{2T×2T}
    S[i][j] = 1 - Hamming(r_i, r_j) / (L*E)
    # K-means 聚类为两个等大小 batch
    while not converged and iter < max_iter:
      分配每个 token 到最近的 cluster centroid
      更新 centroid 为 intra-cluster 平均相似度最高的 token
    yield (T1, T2)
    # Merge + Reindex KV cache 以保持 attention 语义

  Step 3 - ECE Expert 预取与缓存:
    # PLEC: 基于预测分配各层 cache slot
    预测需求: layer_1 需要 3 experts, layer_2 需要 2 experts
    GPU cache capacity: 4 experts
    allocation = [3 slots for layer_1, 1 slot for layer_2]
    # 预取最可能需要的 4 个 experts: CPU→GPU copy
    prefetch([e_12, e_13, e_14, e_22])

  Step 4 - MoE 模型执行:
    for layer in layers:
      # Gating (GPU 上执行)
      gate_scores = softmax(x @ W_gate)
      top_k_experts = topk(gate_scores)
      
      # 检查 expert 是否已在 GPU cache 中
      for expert in top_k_experts:
        if expert not in gpu_cache:
          # Real-time Correction: 异步 CPU→GPU 加载缺失 expert
          # 与当前 running expert 的 compute 并行 (overlap)
          async_load(expert)
        # 执行 expert FFN
        out += gate_score * expert_ffn(x)
      
      # 释放已完成 early-layer expert，加载下一层 expert
      free_completed_experts()
      prefetch_next_layer_experts()

  Step 5 - Token 输出:
    # 与 baseline offloading 相同
    logits = lm_head(hidden) → sample → next_token
    
  === GPU Memory 管理 (NVIDIA A40 48GB) ===
  GPU 常驻: attention weights + gate weights + RPP (7.21MB)
  GPU 动态: expert cache (大小由 cache_size 参数控制, 如 4/8/16 experts)
  CPU 常驻: 全部 expert 参数 (Mixtral-8×7B: 45.1B params in experts)
  ```

  关键性能收益：(a) Switch-128 在 CS=4, BS=32 下达 9.99× throughput vs SE-MoE；(b) GPU memory 最大降低 93.72%（Switch-128: 15.26GB → 1.03GB）；(c) Mixtral-8×7B 在 AIG 下 OOM 但 ExpertFlow 下仅需 15.99GB；(d) PLEC cache hit ratio 91.90%（CS=16, BS=4），比 LRU 高 15-36%；(e) TS 在 Switch-128 上额外提升 1.17× throughput；(f) RPP 跨域 accuracy 仅下降 5-10%，Qwen1.5 上达 >95%。
