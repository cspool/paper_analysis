## Closed-Loop Feedback Scheduling for LLM Inference（LLM推理的闭环反馈调度）

术语是什么？
Closed-Loop Feedback Scheduling（闭环反馈调度）是一种调度器与执行引擎持续交互的控制架构——执行引擎在每个关键执行阶段（如每 layer 的 attention 后、router 后）回调调度器报告当前状态，调度器根据反馈（如新的 LS 请求到达、当前 batch 的执行进度）实时决策是否抢占/插入/调整执行流。QLLM 采用此架构替代传统的 open-loop scheduling（调度器仅在 iteration boundary 获得控制权）。

从系统架构角度拆解术语：
Closed-Loop vs Open-Loop Scheduling 对比：

```
# Open-Loop Scheduling (传统 HF TGI / vLLM)
Scheduler:  batch = select_requests()  →  提交给 Engine
Engine:     for layer in 1..N:
                forward(layer, batch)  # 不回调 Scheduler
Engine:     return tokens  →  Scheduler 取回控制权
# 控制权间隔: N layers × per-layer time = 300-400ms

# Closed-Loop Feedback Scheduling (QLLM)
Scheduler:  batch = GetNextBatch()  →  提交给 Engine
Engine:     for layer in 1..N:
                attention(batch)     →  feedback_scheduler()  [Checkpoint 1]
                router(batch)        →  feedback_scheduler()  [Checkpoint 2]
                for expert in experts:
                    expert_ffn(batch)
Engine:     return tokens
# 控制权间隔: 1 layer execution time = ~10ms (30-40× finer)
```

反馈回调的内容（QLLM 的 checkpoint protocol）：
```
on_checkpoint(type, batch_state, queue_state):
    type: "post_attention" | "post_router"
    batch_state:
        current_layer: int
        active_sequences: list[SequenceID]
        execution_progress: {seq_id: (completed_experts, total_experts)}
    queue_state:
        LS_PrefillQueue.size()
        LS_DecodeQueue.size()
        BE_PrefillQueue.size()
        BE_DecodeQueue.size()

    # Scheduler 的决策逻辑
    if !LS_PrefillQueue.isEmpty() or !LS_DecodeQueue.isEmpty():
        return Decision.PREEMPT_BE  # 抢占 BE，优先执行 LS
    else:
        return Decision.CONTINUE    # 继续当前执行
```

术语一般如何实现？如何使用？
- **QLLM 实现**：在 HF TGI 的 model forward 中插入 Python callback hooks（每次 attention/router 后调用 `scheduler.on_layer_checkpoint()`）。Callback 是同步的（Engine 暂停等待 Scheduler 决策），因此 callback 延迟直接影响 per-layer 执行时间——需保持在 μs 级（<100μs）。
- **与中断驱动调度的对比**：中断驱动（如 GPU 上通过 CUDA stream callback）可实现更低延迟（sub-μs），但实现复杂且依赖 GPU 硬件中断支持。QLLM 选用的 polling callback 模型简单、可移植但增加了 per-checkpoint 的 CPU-GPU 同步开销。
- **User-defined policy 扩展**：QLLM 的 closed-loop feedback 设计支持用户通过 <50 行 Python 代码定义自定义抢占策略（如基于 memory pressure 的 offload 触发、基于 token 数量的 QoS 策略等），通过 dependency injection 注入到 checkpoint callback 中。
- 适用场景：需要实时响应高优先级请求的混合工作负载 serving、multi-SLO 推理、动态资源管理的推理系统。

涉及论文标题：
- Priority-Aware Preemptive Scheduling for Mixed-Priority Workloads in MoE Inference
