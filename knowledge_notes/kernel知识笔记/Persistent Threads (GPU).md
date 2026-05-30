## Persistent Threads (GPU)

术语是什么？
Persistent Threads（PT，持久线程）是 GPU 编程中的一种范式，其中单个 kernel 持续驻留在 GPU 上，其线程不断从工作队列中拉取 task 执行，而非传统的"一个 kernel 完成后退出的 fire-and-forget"模式。PT 通过消除反复的 kernel launch 开销（5-20μs/次）和 CPU-GPU 同步来提升性能，特别适合由大量短小、动态生成的 task 组成的 workload。PT 的核心设计是在 kernel 中使用 while 循环持续轮询 task queue，取到 task 后根据 task 描述符执行相应计算。但 PT 的根本限制是**同质性假设**——所有 task 必须在同一个 kernel 内执行，共享相同的寄存器/共享内存配置，如果 task 之间的资源需求差异大（异构 task），则 PT kernel 必须按最大需求配置，导致资源浪费。

从kernel调度角度拆解术语：
Persistent Threads 的伪代码逻辑：
```
// Persistent Thread kernel
__global__ void persistent_kernel(TaskQueue* queue) {
    while (true) {
        // 1. 从全局工作队列取task (atomic操作)
        Task t = queue->dequeue();
        if (t.type == TASK_TERMINATE) return;  // 终止信号
        
        // 2. 根据task类型执行
        switch (t.type):
            case TASK_COLLISION_DETECT:
                // 需要 64 registers, 8KB shared memory
                detect_collisions(t.data, t.params);
            case TASK_CONTACT_FORCE:
                // 需要 128 registers, 16KB shared memory  
                compute_contact_forces(t.data, t.params);
            case TASK_JOINT_CONSTRAINT:
                // 需要 48 registers, 4KB shared memory
                solve_joint_constraints(t.data, t.params);
        
        // 3. 完成task后继续轮询
    }
}

// 问题: PT kernel必须按最"重"的task配置
// launch配置: 128 registers/thread (max across tasks)
//             16KB shared memory/block (max across tasks)
// 结果: 轻量task (48 reg, 4KB shmem) 浪费大量资源
//       SM occupancy被最重task限制 → 并行度低
```

在 ACS 论文的实验中，使用 Juggler 的 PT 框架处理异构 kernel 时比 baseline 慢 1.35×，原因正是轻量 kernel（如碰撞检测）被迫使用为重量 kernel（如接触力计算）配置的 register/shared memory 资源，降低了 SM 层面的线程并行度（occupancy）。

术语一般如何实现？如何使用？
PT 实现方式：(1) 手动 while 循环 + 原子操作管理 task queue（如 Juggler、Whippletree）；(2) CUDA device runtime 的 persistent launch（CUDA 12.x+）。PT 适合的场景：(1) 同构 task（如 raytracing 中遍历 BVH 树，所有 ray 执行相同计算）；(2) task 数量极大且动态生成（每个 task 执行时间短于 launch 开销）。不适合的场景：(1) 异构 task（ACS 论文中指出的问题）；(2) 需要不同 kernel 参数配置的 task（不同 register 需求）。ACS 通过 CUDA stream + 调度窗口方案避免了 PT 的同质性限制——每个 kernel 保持其原生配置，仅通过 stream 并行实现并发，无需合并到单个 kernel。

涉及论文标题：
- ACS Concurrent Kernel Execution on Irregular, Input-Dependent Computational Graphs
