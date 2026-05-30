## Phase-Aware Optimizer for MoE Inference

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Phase-Aware Optimizer 是 LYNX 系统中负责判断当前 inference iteration 是否应启用 expert remapping 的组件。其核心功能是区分 compute-bound iteration（如 prefill）和 memory-bandwidth-bound iteration（如 decode），仅在后一种情况下启用 LYNX 的 full pipeline，避免在 compute-bound 阶段引入不必要的 routing overhead。

Phase-Aware Optimizer 的设计基于 LYNX 的核心观察：MoE 的 expert remapping 减少的是 memory bandwidth 消耗（HBM 加载 expert 权重），而非 computation。因此它仅在 memory-bandwidth-bound 的 decode iteration 中有用。此外，Prefill 和 decode 对 expert fidelity 的敏感度也存在根本性不对称——prefill 需要严格的 expert fidelity，decode 则高度容错。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Phase-Aware Optimizer 在三种 serving 部署模式下的行为：

```
┌── Phase-Aware Optimizer ────────────────────────────────────┐
│                                                              │
│  模式 1: Co-located Prefill/Decode                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Scheduler 接收 mixed prefill+decode requests          │   │
│  │ → 分析 batch composition:                              │   │
│  │   - 若 batch 含 prefill tokens → compute-bound        │   │
│  │     → flag=SKIP, 绕过 LYNX, 直接标准 MoE forward      │   │
│  │   - 若 batch 仅含 decode tokens → memory-bound        │   │
│  │     → flag=ENABLE, 启用 LYNX pipeline                 │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  模式 2: Disaggregated Prefill/Decode                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Prefill 节点和 Decode 节点物理分离                      │   │
│  │ → Decode 节点仅处理 decode tokens                       │   │
│  │   → 始终 memory-bound                                   │   │
│  │   → flag=ENABLE (constant), 无需 per-batch 判断        │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  模式 3: Chunked Prefill                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Batch 可能同时包含 prefill chunks 和 decode tokens     │   │
│  │ → - 仅含 decode tokens → flag=ENABLE                   │   │
│  │   - 含 prefill chunk (即使 small) → flag=SKIP         │   │
│  │     (edge case: small prefill chunk 也可能 memory-bound │   │
│  │      但 remapping 可能影响 TTFT，留给 future work)     │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Phase-Aware Optimizer 在 vLLM batch scheduler 内实现，作为一个轻量级 pre-dispatch check。对于 decode-only iterations，设置一个 boolean flag 传递给下游 LYNX 组件。该 flag 在 CUDA Graph 捕获时被 bake 进静态执行图（compute-bound path 直接跳过 LYNX kernels）。Optimizer 本身的计算开销可忽略不计——仅需检查 batch 中是否存在 prefill tokens。

涉及论文标题：
- LYNX: Enabling Efficient MoE Inference Through Dynamic Batch-Aware Expert Selection
