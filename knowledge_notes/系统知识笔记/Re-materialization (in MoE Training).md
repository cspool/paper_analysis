## Re-materialization (in MoE Training)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Re-materialization 是 Hecate 中的内存优化技术，用于减少 sparse materialization 引入的额外参数内存。在 FSSDP 的基本模式下，sparse materialization 物化的 expert 参数在整个 forward-backward 过程中保留在 device memory 中。Re-materialization 则在 forward pass 完成后立即释放物化的 expert 参数，待 backward pass 需要时通过额外的 SparseAllGather 重新物化。这形成 "forward 物化 → release → backward 重新物化" 的流水线。

效果：Hecate-RM 将 materialized parameters 的额外内存占用降低 90.2%（因为只需为当前 MoE layer 的 placement 保留内存，而非所有 layer）。代价：backward 需要额外的 spAG 通信（sparse collectives 通信增加 3.6×），但 Hecate-RM 仍优于 baseline 1.4×。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

```
Hecate Re-materialization Memory Lifecycle:

Without Re-materialization (Hecate):
  Layer 1 Fwd: spAG(P,P') → 物化 expert 参数 [KEEP]
  Layer 2 Fwd: spAG(P,P') → 物化 expert 参数 [KEEP]
  ...
  Layer L Fwd: spAG(P,P') → 物化 expert 参数 [KEEP]
  所有 L 层的 materialized parameters 同时在内存中
  Peak memory: L × materialized_params_per_layer + baseline

  Layer L Bwd: 使用已保留的参数
  ...
  Layer 1 Bwd: 使用已保留的参数
  spRS 梯度 reduce 后释放

With Re-materialization (Hecate-RM):
  Layer 1 Fwd: spAG(P,P') → 物化 [用后 RELEASE]
  Layer 2 Fwd: spAG(P,P') → 物化 [用后 RELEASE]
  ...  仅为当前层保留 materialized 参数
  Layer L Fwd: spAG(P,P') → 物化 [用后 RELEASE]

  Layer L Bwd: spAG(P,P') 重新物化 L → 计算 → spRS → RELEASE
  Layer L-1 Bwd: spAG(P,P') 重新物化 L-1 → 计算 → spRS → RELEASE
  ...
  Peak memory: 1 × materialized_params_per_layer + baseline
  参数内存减少: 90.2%
  额外通信: backward 每层多一次 spAG (= 3.6× sparse collectives)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- Hecate-RM 在 Executor 中实现：forward 后显式调用 `release_materialized_params()` 释放 CUDA memory，backward 前调用 `rematerialize()` 触发 spAG。
- 适用于内存受限场景（如 batch size 需要 scale up 但 GPU 内存不足），Hecate-RM 是唯一能在 batch_size=6 时仍保持性能优势的策略（图 14）。
- Re-materialization 本质是用通信换内存：增加 backward 通信开销（3.6× sparse collectives）但大幅降低内存占用（90.2% materialized params），在 batch size 灵活的 training 场景中提供可调的 trade-off。

涉及论文标题：
- Hecate: Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism
