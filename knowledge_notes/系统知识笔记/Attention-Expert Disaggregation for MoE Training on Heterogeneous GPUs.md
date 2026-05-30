## Attention-Expert Disaggregation for MoE Training on Heterogeneous GPUs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Attention-Expert Disaggregation 是 HeterMoE 的核心系统设计原则：将 MoE transformer layer 的 attention blocks 和 expert blocks 分配到不同代 GPU 上——新 GPU（A40/L40S）仅执行 attention + gate，旧 GPU（V100/T4）仅执行 expert FFN。该设计的动机源于关键观察：旧 GPU 在 expert 计算上仍高效（V100 ≈ 80% A40 expert 性能，因 expert 主要是 GEMM，CUDA core 高度优化），但在 attention 上性能严重退化（V100 不支持 FlashAttention，64K 序列时仅 A40 attention 的 27%）。

关键不变量：由于 MoE 训练本就使用 EP 的 all-to-all 在不同 GPU 间交换 token，将 attention 和 expert 分离到不同 GPU 不引入额外通信——dispatch/combine 的数据总量不变，仅通信方向从 "attention GPU↔attention GPU" 变为 "attention GPU↔expert GPU"。此外，将 bulky expert 权重 offload 到旧 GPU，缓解了新 GPU 的稀缺内存压力。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

```
┌── Heterogeneous GPU Cluster Configuration ──────────────────┐
│ Attention GPUs (newer gen, e.g. A40 48GB):                   │
│   - 复制: attention blocks, MoE gate, embedding              │
│   - 可选通过 Asym-EA: 持有部分 experts                       │
│   - 使用 FlashAttention v2/v3 加速 attention                  │
│                                                              │
│ Expert GPUs (older gen, e.g. V100 16GB):                    │
│   - 分布: expert FFN 模块（gate_proj, up_proj, down_proj）  │
│   - 仅需执行 GEMM（无需 FlashAttention 支持）                │
│                                                              │
│ ZP group = M attention GPUs + N expert GPUs                  │
│ 例: O1 setup = 6×A40 + 6×V100                                │
└──────────────────────────────────────────────────────────────┘

┌── Communication Pattern (Bipartite) ────────────────────────┐
│ Forward dispatch: M attention GPUs → N expert GPUs           │
│ Forward combine:  N expert GPUs    → M attention GPUs        │
│ Backward: reverse direction                                   │
│                                                              │
│ 与标准 EP 对比:                                              │
│   EP:  M+N GPUs 互相 dispatch/combine（full all-to-all）     │
│   ZP:  M ⇄ N bipartite dispatch/combine                     │
│   总通信数据量相同（相同的 global batch × seqlen × d_model） │
└──────────────────────────────────────────────────────────────┘
```

实验结果：HeterMoE 使用 2×A40+2×V100 达到 4×A40（全量新 GPU）平均 95% 的 throughput；序列越长效果越显著——32K 时比 EP 快 1.89× on average。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 基于 PyTorch v2.2 + DeepSpeed v0.14 实现
- ZP engine 在初始化时将模型在 ZP group 内 split
- 所有 GPU 共享 ZP group，创建分离的 NCCL dispatch/combine group
- 可选择性地与 data parallelism 组合
- 当旧 GPU 内存不足时必须通过 Asym-EA 将部分 experts 迁回 attention GPU（下限 n_min）；当 attention GPU 内存不足时 offload 上限为 n_max

涉及论文标题：
- HeterMoE: Efficient Training of Mixture-of-Experts Models on Heterogeneous GPUs
