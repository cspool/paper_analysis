## CUDA Stream

术语是什么？
CUDA Stream 是 CUDA 编程模型中的 FIFO 命令队列，用于管理和排序 GPU operations（kernel launch 和 memory copy）。同一 stream 内的操作严格按 FIFO 顺序执行，不同 stream 间的操作可以并发执行（是否真正并发取决于 GPU scheduler 和硬件资源）。默认情况下，所有 GPU operations 被提交到单一的 default stream（NULL stream）。程序员可创建额外 stream 实现并发操作。

从kernel调度角度拆解术语：
在本文发现的 TX2 GPU scheduler 模型中，每个 CUDA stream 对应一个 stream queue（FIFO）。GPU operation 的生命周期为：
1. CUDA API 调用时入队到对应 stream queue（Rule G1）
2. Kernel 到达 stream queue 头部时入队到 EE queue（Rule G2）；Copy 到达 stream queue 头部时入队到 CE queue（Rule C1）
3. EE queue 头部的 kernel 的 block 被分配到 SM（受资源约束和优先级规则限制）
4. Kernel 所有 block 完成后从 EE queue 出队（Rule G3），随后从 stream queue 出队（Rule G4）

伪代码——单个 stream 的 operation 流转：
```
// CPU 端: CUDA 程序
cudaStream_t s1;
cudaStreamCreate(&s1);

// Operation enqueue (Rule G1)
kernel_A<<<gridA, blockA, 0, s1>>>();  // GPU op → s1 的 stream queue 尾部
kernel_B<<<gridB, blockB, 0, s1>>>();  // GPU op → s1 的 stream queue 尾部 (在 kernel_A 之后)

// GPU 端: Scheduler
while (stream_queues[s1] is not empty):
    head_op = stream_queues[s1].head
    if head_op is kernel:
        enqueue_EE_queue(head_op)  // Rule G2
        while not head_op.fully_dispatched:
            // Only head of EE queue is eligible (Rule X1)
            for each SM:
                if SM.available_threads >= head_op.threads_per_block
                   and SM.available_shmem >= head_op.shmem_per_block:
                    assign_block_to_SM(head_op, SM)  // Rules R1-R3
        dequeue_EE_queue(head_op)  // Rule G3
    if head_op is copy:
        enqueue_CE_queue(head_op)  // Rule C1
        assign_copy_to_CE(head_op)  // Rules C2-C3
        wait_copy_completion()
    dequeue_stream_queue(s1, head_op)  // Rule G4 / C4
```

术语一般如何实现？如何使用？
CUDA Stream 通过 CUDA Runtime API（cudaStreamCreate, cudaStreamDestroy）管理。CUDA 8.0+ 支持 stream priority (cudaStreamCreateWithPriority)。从 Kepler 架构引入 Hyper-Q 后，GPU 硬件层面支持多达 32 个并发工作队列，允许多 stream 真正并发。开发者通常使用多 stream 实现计算与数据传输 overlap（pipelining）或不同 kernel 的并发执行。注意事项：(1) 避免无意中使用 NULL stream 导致串行化；(2) 对于不需要 NULL stream 同步的 stream，使用 cudaStreamNonBlocking 标志创建。

涉及论文标题：
- GPU Scheduling on the NVIDIA TX2: Hidden Details Revealed
- Nimble: Lightweight and Parallel GPU Task Scheduling for Deep Learning
- HuntKTm: Hybrid Scheduling and Automatic Management for Efficient Kernel Execution on Modern GPUs

---
