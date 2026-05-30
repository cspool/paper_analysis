## Tensor-Parallel Expert Loading（张量并行专家加载）

术语是什么？
Tensor-Parallel Expert Loading 是 MoEsaic 为支持大型 MoE 模型（单个 expert 参数 > GPU 显存）的多实例共享而扩展的 tensor parallelism 支持。在 Tensor Parallel (TP) 模式下，每个 expert 被 shard 到多个 GPU 上（如 4-way TP），每个 GPU 仅持有 expert 的一部分 shard。vLLM 原生支持初始模型的 TP 加载，但不支持——向已部署模型中动态添加新 expert 时的 TP 加载。MoEsaic 新增 Ray worker 机制，使新 expert 继承初始模型的 sharding 策略，并在 shard 级别执行去重。

从kernel调度角度拆解术语：
在 MoEsaic 的 4-way TP 配置下（8×A100 40GB），Tensor-Parallel Expert Loading 的 kernel 流程：

```
// Step 1: 初始模型 TP 加载（vLLM 原生）
// Mixtral-8x7B 被 4-way shard 到 GPU 0-3
for gpu in [0, 1, 2, 3]:
    load_model_shard(base_model, gpu_rank=gpu, world_size=4)
    // GPU gpu 持有每个 expert 的 1/4 shard

// Step 2: 新 model instance 的 TP Expert Loading（MoEsaic 扩展）
// 生成 4 个 Ray workers，每个绑定一个 GPU
ray_workers = [RayWorker(gpu=i) for i in range(4)]

// 每个 Ray worker 在对应 GPU 上执行：
for worker in ray_workers:
    // 新 expert 继承初始模型的 sharding 方式
    for new_expert in new_model_instance.experts:
        shard = extract_shard(new_expert, 
                              rank=worker.gpu_rank, 
                              world_size=4)
        // Step 3: Shard 级别的去重
        shard_hash = compute_128bit_hash(shard)  // 对 shard tensor 计算 hash
        
        if shard_hash in hash_dictionary:
            shard.reference(hash_dictionary[shard_hash])  // 共享已有 shard
        else:
            allocate_gpu_memory(shard, gpu=worker.gpu_rank)
            hash_dictionary[shard_hash] = shard

// Step 4: 推理时 fused gate 路由（跨 GPU 的 TP 执行）
// 每个 token 被路由到 merged expert 的 shard
// shard 在各自 GPU 上执行 partial FFN，通过 all-reduce 聚合结果
```

关键区别：
- **完整 Expert 去重 vs Shard 去重**：在单 GPU 模式下，去重是在完整 expert 级别进行（每 expert ~14GB for Mixtral-8x7B）。在 TP 模式下，去重是在 per-GPU shard 级别进行（每 expert shard ~3.5GB for 4-way TP）。
- **继承 Sharding 策略**：新 expert 必须严格继承初始模型的 sharding 方式——若初始模型是 4-way TP 按列切分（column-wise sharding），新 expert 也按同样方式切分。否则去重对象（shard）的语义不一致。
- **Ray Worker 并行加载**：每个 Ray worker 独立在绑定 GPU 上执行 shard 加载和 hash 计算。论文表 3（Table 3）显示 TP 模式下加载速度更快——多个 Ray worker 并行后，Mixtral-4x7B（4 GPUs）加载 4 models 需 135s，比单 GPU 的 Mixtral-4x1B 加载 4 models 需 110s 相对更快（考虑模型尺寸差异）。

术语一般如何实现？如何使用？
- 实现方式：vLLM 的 Ray-based distributed executor + MoEsaic 自定义 `load_weights()` 中的 shard-aware dedup 逻辑。
- 与 vLLM 原生 TP 的关系：vLLM 原生 TP 在初始化时一次性 shard 所有参数；MoEsaic 扩展使其支持增量添加新 expert shard 并在运行时去重。
- 必须性：大型 MoE 模型（Mixtral-8x7B, DeepSeek-V2/V3）因单 expert 参数过大（14GB+），单 GPU 无法容纳完整 expert + runtime state（KV cache），TP 是实际部署的前提。
- 去重粒度权衡：shard 级别 vs expert 级别——shard 级别去重更细粒度（即使完整 expert 不同，其部分 shard 可能相同），但增加了 hash 计算次数和 dictionary 条目数。

涉及论文标题：
- MoEsaic: Shared Mixture of Experts
