## Expert-Tensor Parallelism (ETP)

术语解释
Expert-Tensor Parallelism (ETP) 是在 MoE 分布式部署中同时使用 Expert Parallelism (EP) 和 Tensor Parallelism (TP) 的并行策略，将 expert 权重沿 TP 维度切分以缓解单 GPU 内存压力。ETP 的通信为两阶段 "AlltoAll + AllGather" 或 "ReduceScatter + AlltoAll"，多轮 collective 增加 kernel launch 和 sync overhead。DualSparse-MoE 提出 S-ETP 从算法层面替代 ETP，将通信简化为单 AlltoAll。

术语是什么？
在大规模 MoE 推理中，单个 expert 参数量可能超出单 GPU 内存。ETP 结合 EP (expert-wise 分布) 和 TP (weight-wise 切分)：例如 EP=4, TP=2 on 8 GPUs → 4 EP groups × 2 TP each。通信流程：AlltoAll (EP token dispatch) → AllGather 或 ReduceScatter (TP partial results) → FFN on TP-sliced weights → AllGather/ReduceScatter (reverse TP) → AlltoAll (result return)。共 3+ collectives。vs S-ETP (2×AlltoAll only)：real H20 bandwidth 提升 3.0-29.9%，NVL72 模拟提升 10.2-80.4%。

从系统架构角度拆解术语：
```
=== ETP Communication (EP=4, TP=2, 8 GPUs) ===
Layer execution:
  AlltoAll → dispatch tokens to 4 EP groups
  AllGather → within TP group (GPU0↔GPU1)
  FFN on TP-sliced weights → local computation
  ReduceScatter → reduce within TP group
  AlltoAll → return results

Total: 2×AlltoAll + AllGather + ReduceScatter
S-ETP: 2×AlltoAll only (algorithmic partition replaces TP slicing)
```

术语一般如何实现？如何使用？
- Megatron-Core/SGLang: moe_ep_size + moe_tp_size config；TP 切分对 attention + expert weights 均支持
- 适用场景：大 expert (d_ffn≥14336, Mixtral-8×7B) 需 TP 满足 GPU memory；全互联高带宽系统 (NVL72) 通信负担轻
- DualSparse-MoE 替代方案：Partial transformation expert partition 实现等价 TP 效果 → S-ETP (single AlltoAll)
- 与 load-aware thresholding 协同：EP load balance 涉及 EP + TP 双维度调度

涉及论文标题：
- DualSparse-MoE: Coordinating Tensor/Neuron-Level Sparsity with Expert Partition and Reconstruction
- EPS-MoE: Expert Pipeline Scheduler for Cost-Efficient MoE Inference
