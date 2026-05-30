## Pipeline-Shared Cache (for Data-Centric MoE Training)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Pipeline-Shared Cache 是 HEXA-MoE 在 data-centric 并行配置下提出的 GPU HBM 内存管理机制，用于解决 data-centric MoE 训练中 backward pass 的内存膨胀问题。此前的 data-centric 方法（如 Janus）在 forward pass 中预取每层所需参数，但为 backward pass 保存了所有层的完整 gathered 参数在 GPU HBM 中，导致巨大的内存占用。Pipeline-Shared Cache 在每设备 HBM 上分配一块额外的共享缓存区域，动态缓存当前 pipeline stage 所需的 gathered MoE shards——forward 时写入，backward 时读取，同一 cache 区域在不同层之间复用。配合 all gather 通信与 attention/router 计算的 overlap，实现通信和内存的双重优化。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Pipeline-Shared Cache 的工作流程:
// HBM 布局: [model_params] [activations] [optimizer_states] [pipeline_cache]
// Cache 大小 = max_per_layer_gathered_params
//           = E × D_i × D_mid × 2 × sizeof(dtype)

for layer in 0..L-1:
    // Stream COMPUTE: Attention + Router
    attn_out = attention(layer_norm(x))
    
    // Stream COMM (overlap): All gather MoE shards → cache
    all_gather_into_cache(local_moe_shards, pipeline_cache)
    sync_streams()
    
    // Forward: ESMM 使用 cache 中的完整参数
    y = ESMM(x, pipeline_cache.W1, pipeline_cache.b1, R(x))
    y = ESMM(activation(y), pipeline_cache.W2, pipeline_cache.b2, R(x))

// Backward: 每层重新 all gather 到 cache (与 attention backward 重叠)
for layer in L-1 down to 0:
    all_gather_into_cache(local_moe_shards, pipeline_cache)
    ∂ℓ/∂W2 = ESTMM(y2, ∂ℓ/∂y, R(x))  // 使用 cache 中的完整参数
    ...
```

消融实验（Figure 9a）：data-centric 配置下若不加 pipeline-shared cache，内存占用会超过 Tutel baseline；加入后 data-centric 的内存占用略高于 model-centric 但明显优于 baseline（10%-48% 节省）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GPU HBM 上预分配 contiguous buffer，大小根据所有 MoE 层中最大参数总量计算。All gather 使用 NCCL 原语，结果直接写入 cache buffer。Backward 时从 cache 读取完整参数做 ESTMM，写回梯度仅需 local shard 部分。开源实现：https://github.com/UNITES-Lab/HEXA-MoE。

涉及论文标题：
- HEXA-MoE: Efficient and Heterogeneous-aware MoE Acceleration with ZERO Computation Redundancy
