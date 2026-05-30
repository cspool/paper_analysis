## Batch-Aware Expert Selection / Expert Remapping in MoE Serving

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Batch-Aware Expert Selection 是 LYNX 提出的 MoE 推理优化方法，在 batch 级别（而非 per-token 级别）动态减少活跃专家数量。核心思想：在每次 decode iteration 中，不同 token 的 top-k expert 选择存在重叠和冗余——由训练时 load-balancing loss 强制的 "forced diversification" 导致许多 token 被分配到 confidence 较低的 expert。LYNX 通过识别低置信度 token-expert assignment 并将其重映射到 batch 中已被其他 token 激活的高置信度 expert 上，减少 batch 级别活跃专家的并集大小，直接降低从 HBM 加载的 expert 权重数据量。

与已有方法的区别：
- **静态 expert pruning/merging**：永久修改模型，需离线校准，不灵活
- **Per-token dynamic k reduction**：减少每个 token 的 expert 数，但 batch 级 expert 利用率仍随 batch size 增长
- **LYNX**：保持 per-token k 不变，通过 batch 内 remapping 减少并集大小。不修改模型，不需校准数据

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

LYNX Expert Remapping 在 vLLM serving 系统中的集成流程：

```
┌── LYNX Batch-Aware Expert Selection Pipeline ──────────────┐
│  (per decode iteration, per MoE layer)                      │
│                                                              │
│  1. [vLLM Scheduler]                                        │
│     Continuous batching → 组装 batch B                       │
│     Phase-Aware Optimizer: 检查是否为 memory-bound decode   │
│     → 若是: flag=ENABLE_LYNX                                │
│                                                              │
│  2. [MoE Router (标准)]                                     │
│     z = W_gate @ h            # [B, N] logits               │
│     p = softmax(z)            # router probabilities         │
│     topk_idx, topk_prob = topk(p, k)                         │
│                                                              │
│  3. [LYNX Confidence Analyzer]  ← Kernel 1 (Triton)        │
│     对每个 token: log_ratio = z[e] - z[top1] for e in top-k │
│     AffinityBinning: bin[e] = clamp(floor(log_ratio*α),-β,0)│
│                                                              │
│  4. [LYNX Adaptive Expert Scorer] ← Kernel 2-3 (Triton)    │
│     对每个 expert e:                                         │
│       score[e] = Σ_t B^{bin[t][e]}  (batch_size 指数加权)   │
│     动态确定 active expert set (基于 score distribution)     │
│                                                              │
│  5. [LYNX Expert Remapper] ← Kernel 4 (Triton)             │
│     对 low-confidence token:                                 │
│       lower-ranked experts → remap 到 active expert set 内替代│
│     对 high-confidence token:                                │
│       保留 top-1 expert 不变                                  │
│     Compaction: 重排 token-to-expert 映射为连续索引          │
│     Renormalize: 重新计算 softmax → 最终 dispatch weights    │
│                                                              │
│  6. [Expert Computation (vLLM fused MoE kernel)]            │
│     以 reduced active expert set 启动 grouped GEMM           │
│     从 HBM 加载更少的 expert 权重                             │
│     → decode latency 降低                                     │
└──────────────────────────────────────────────────────────────┘
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

LYNX 实现为 vLLM v0.10.1 插件，通过 CLI flag 启用。4 个 fused Triton kernel（confidence analyzer, adaptive scorer x2, remapper）嵌入在每层 MoE router 和 expert computation kernel 之间。Kernel overhead <4% 总延迟，远小于 expert weight 加载减少带来的收益（median TPOT 降低 1.09-1.30x）。LYNX 的 batch-size-adaptive 指数加权方案确保 scoring 自然适应不同 batch composition——batch 越大，竞争越激烈，expert 保留阈值越高。

涉及论文标题：
- LYNX: Enabling Efficient MoE Inference Through Dynamic Batch-Aware Expert Selection
