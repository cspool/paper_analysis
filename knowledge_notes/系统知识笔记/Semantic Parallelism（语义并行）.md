## Semantic Parallelism（语义并行）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Semantic Parallelism（语义并行）是 Speculative MoE 论文提出的新型 MoE 推理并行范式，核心思想是通过 **model-data collaborative scheduling（模型-数据协同调度）** 最大化 MoE 推理中 token 与 expert 的本地共置率，从而最小化 Expert Parallelism (EP) 的 all-to-all 跨设备通信开销。与传统的 expert parallelism（仅关注 expert 的设备分布）和纯粹的 data scheduling（仅关注请求分配）不同，Semantic Parallelism 同时优化 expert placement（模型维度）和 token/request routing（数据维度），利用 token-expert 之间的语义亲和力（semantic affinity）实现协同调度。其核心洞察是：MoE 模型中 token 对 expert 的激活模式具有强 **context-independent correlation**——即 token 倾向于稳定地路由到一组固定的 expert，这一现象跨越不同上下文保持稳定（median cumulative hotness of top-k experts: 0.833-0.976）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Semantic Parallelism 在 Sem-MoE 系统（基于 SGLang）中的三层调度架构：

```
[Offline Phase — Profiling & Co-Clustering]
1. Profile MoE model on representative dataset (e.g., ShareGPT, 20% data)
   → Record per-token expert activation frequencies across all layers
2. Build token-to-expert confidence table C_p ∈ R^{t×N}
   (t = vocabulary size, N = experts per layer)
3. Solve ILP co-clustering problem (alternating optimization, Algorithm 1):
   Objective: min θ·load_imbalance + (1-θ)·remote_activation_volume
   Output: E (expert-to-device table), T (token-to-device table), A (inter-layer table)
4. Apply E to reconfigure expert placement across layers
5. Shuffle gate matrix columns for transparent expert redistribution

[Online Phase — Attention-DP (Inter-Request Scheduling)]
For each incoming request r:
  1. Extract token IDs from r
  2. Lookup T[token_id] → device assignment per token
  3. Aggregate: S_r = argmax_j Σ_{token_i∈r} R_{ij}  (most-affine device)
  4. Workload-aware round-robin:
     - Batch E requests at a time (E = EP degree)
     - Allocate one request per device per round
     - Reset device mask after each complete round
     → Prevents decoding-phase load skew while maintaining affinity
  5. Assign request r to DP rank = S_r

[Online Phase — Attention-TP (Intra-Request Scheduling)]
For each MoE layer L:
  1. After attention: receive hidden states X ∈ R^{B×H}
  2. Query scheduling tables:
     - T[token_ids] → token-level device prediction
     - A[(prev_dev1, prev_dev2)] → 2-gram Markov device prediction
     - Select prediction with higher confidence score C_p vs A_p
  3. Device_ids D = argmax(confidence)
  4. shuffle_indices = argsort(D)
  5. Shuffled-Reduce-Scatter (SRS): shuffle X by indices + reduce-scatter
  6. MoE computation (gate + expert FFN) on local token shards
  7. Shuffled-AllGather (SAG): allgather + reverse-shuffle to restore order
```

关键指标：Local Activation Rate (LAR) = (#tokens computed on local device) / (#total tokens)。Baseline EP 下 LAR≈25%（EP8），Sem-MoE 将 LAR 提升至 62%（DeepSeek-V2-Lite）和 68%（Qwen3-30B-A3B），对应 all-to-all 通信量减少和 41.8%/46.6% expert layer latency reduction。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Sem-MoE 以插件形式集成于 SGLang，约 5000 行 Python + 自定义 Triton kernel。使用流程：(1) 离线阶段：用 20% 目标数据 profile → 运行 co-clustering solver → 生成调度表 E/T/A；(2) 部署时：按 E 重排 expert placement + shuffle gate matrix；(3) 在线阶段：Attention-DP 场景扩展 SGLang request scheduler 加入 affinity-aware 逻辑；Attention-TP 场景用 SRS/SAG 替换标准 allreduce。调度表常驻 GPU memory（<12 MB for DeepSeek-V2, int16）。Cross-dataset 零样本迁移能力：ShareGPT 训练的预测器在 lmsys-chat-1m 上 LAR 达 41.25%（vs baseline 25%），为 in-domain 最优的 87%。

涉及论文标题：
- Speculative MoE: Communication Efficient Parallel MoE Inference with Speculative Token and Expert Pre-scheduling
