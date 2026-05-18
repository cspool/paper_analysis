## Dynamic SM Repartitioning（动态SM重分区）

术语是什么？

Dynamic SM Repartitioning是Bullet提出的一种运行时SM资源管理策略：SLO-aware scheduler根据实时请求队列状态、prefill/decode执行进度和SLO压力，周期性下发repartition command，resource manager立即修改相应CUDA stream的SM mask，使后续kernel在新分配的SM子集上执行。与静态SM分区（如NanoFlow的固定overlap pipeline、MuxServe的固定SM quota）不同，动态重分区允许SM在prefill和decode之间灵活流动，适应动态workload。

从系统架构角度拆解术语：

动态SM重分区在Bullet中的运行时流程：

```
// Scheduler主循环（每个prefill layer group或decode step后触发）

while serving:
    // 1. 读取全局状态
    state = read_shared_metadata_buffer()
    // state包含: prefill_queue_depth, decode_batch_size, current_ttft, p90_tpot,
    //           current_sm_prefill, current_sm_decode, prefill_progress

    // 2. SLO-aware决策：评估是否需重分区
    if state.p90_tpot > TPOT_SLO * 0.9:
        // decode接近SLO边界 → 减少prefill SM，增加decode SM
        new_sm_decode = min(state.sm_decode + 16, total_SMs)
        new_sm_prefill = total_SMs - new_sm_decode

    elif state.prefill_queue_depth > threshold and state.p90_tpot < TPOT_SLO * 0.7:
        // prefill队列堆积且decode余量充足 → 增加prefill SM
        new_sm_prefill = min(state.sm_prefill + 16, total_SMs)
        new_sm_decode = total_SMs - new_sm_prefill

    else:
        continue  // 维持当前分区

    // 3. 下发repartition command
    resource_manager.repartition(new_sm_prefill, new_sm_decode)
    // resource_manager内部:
    //   - 构建新SM mask
    //   - libsmctrl_set_stream_mask(stream_prefill, new_mask_prefill)
    //   - libsmctrl_set_stream_mask(stream_debug, new_mask_decode)
    //   - 耗时: 平均4.1us

    // 4. 搜索并发射下一批prefill layer / decode step
    // 后续kernel自动在新SM子集上运行
```

与静态方案的对比：
- **NanoFlow静态overlap**：固定SM分配和pipeline，无法应对workload波动
- **MuxServe固定SM quota**：每个模型/请求类型有固定SM配额，无法动态流动
- **Bullet动态重分区**：微秒级SM重配，适应prefill burst和decode SLO压力变化

术语一般如何实现？如何使用？

实现依赖底层SM分区机制（libsmctrl或GreenContext）的快速mask更新能力。Bullet的resource manager在约4.1us内完成SM mask修改。调度决策周期对齐prefill layer group或decode step，确保重分区反应足够快。SM配置以16 SM为粒度（A100上6种配置，H100上7种配置）。

涉及论文标题：
- Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration

---
