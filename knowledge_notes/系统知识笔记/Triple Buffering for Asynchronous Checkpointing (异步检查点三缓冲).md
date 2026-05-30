## Triple Buffering for Asynchronous Checkpointing (异步检查点三缓冲)

术语是什么？
Triple Buffering 是 MoC-System 用于管理异步 checkpoint 的缓冲机制，三个 buffer 分别为 snapshot（GPU→CPU 传输目标）、persist（CPU→Storage 序列化+写入源）、recovery（已完成 checkpoint 供故障恢复）。状态机轮转：snapshot buffer 完成 GPU→CPU 后转 persist buffer → CPU→Storage 完成后转 recovery buffer。第三 buffer 确保 snapshot 和 persist 可并行执行（一个 buffer 在 snapshot 时另一个在 persist），且始终有一个 recovery buffer 可用。与标准双缓冲 (snapshot + persist) 相比，triple 消除了 buffer 争用——无需等待 persist 完成即可开始下一次 snapshot。

从系统架构角度拆解术语：
```
# Triple Buffer 状态机
Buffers: {b1, b2, b3}
States:  {SNAPSHOT, PERSIST, RECOVERY}

Timeline:
|-- b1: SNAPSHOT --|-- b1: PERSIST --|-- b1: RECOVERY --|
                    |-- b2: SNAPSHOT --|-- b2: PERSIST --|
                                        |-- b3: SNAPSHOT --|

# 代理线程管理
agent_thread(rank):
    while training:
        if snapshot_trigger and any_buffer.idle:
            b = get_free_snapshot_buffer()
            async_gpu_to_cpu(model_states, b)
            b.state = SNAPSHOT_READY
        
        if any_buffer.state == SNAPSHOT_READY and not persist_in_progress:
            b = get_oldest_snapshot_buffer()
            async_cpu_to_storage(b)
            b.state = PERSISTING
```

术语一般如何实现？如何使用？
- 每个 buffer 分配在 pinned CPU memory（DMA 友好），大小与单次 checkpoint 体积匹配。
- 在 Megatron-DeepSpeed 中通过异步线程实现，buffer 状态通过原子标志维护。三重缓冲在 buffer 数超过 2 时才需管理复杂性，但避免了 snapshot 完成后等待 persist 完成的阻塞。
- 适用场景：长训练任务（数天至数周）需高频 checkpoint（I_ckpt 分钟级），persist 时间可能远超 F&B 时间，必须解耦 snapshot 和 persist 的执行。

涉及论文标题：
- Partial Experts Checkpoint: Efficient Fault Tolerance for Sparse Mixture-of-Experts Model Training
