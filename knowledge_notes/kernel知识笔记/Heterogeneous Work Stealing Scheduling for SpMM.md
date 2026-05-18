## Heterogeneous Work Stealing Scheduling for SpMM

术语是什么？

Heterogeneous Work Stealing Scheduling 是 ASM-SpMM 提出的面向异构 ARM CPU（如 Apple M4 的 P-core+E-core）的 SpMM 多核动态负载均衡策略。与传统的 static task partitioning（按行/非零元/core 能力预分配）不同，它先做 hardware-aware initial task mapping，再用 progress monitoring 和 work stealing 在运行时动态再平衡。核心挑战：M4 的 P-core SME unit 和 E-core SME unit 性能不对称（P-core >> E-core），且不同 row window 的非零分布不可预测→静态分配必然导致某些 core 过早完成而等待。

从 kernel 调度角度拆解术语：

Work stealing 调度伪代码：
```
// Phase 1: Hardware-aware initial task mapping
row_windows = partition_by_svl(sparse_matrix_A)    // 按 SVL 切 row window
core_capability = [core.sme_peak_flops for core in cores]  // P-core > E-core
initial_assignments = weighted_round_robin(row_windows, core_capability)

// Phase 2: Parallel execution with work stealing
shared_state = {remaining_windows: List[RowWindow], 
                progress_per_core: int[cores], 
                lock: spinlock}

function execute(core_id):
    local_queue = initial_assignments[core_id]
    while local_queue or shared_state.remaining_windows:
        // Step 2a: Process local work
        while local_queue:
            window = local_queue.pop()
            result = spmm_kernel(window, B, ZA_tile)  // SME or hybrid kernel
            shared_state.progress_per_core[core_id]++
        
        // Step 2b: Try work stealing
        victim = find_slowest_core(shared_state)       // 最大剩余工作量
        if victim and shared_state.remaining_windows[victim]:
            lock(shared_state.lock)
            stolen = steal_half_windows(victim)         // 从 victim 窃取一半剩余 window
            unlock(shared_state.lock)
            local_queue.extend(stolen)
    
    barrier()
    merge_partial_results()                             // 合并各 core 的部分 C
```

关键设计决策：
1. **Hardware-aware initial mapping**：根据 core 的 SME 性能（P-core vs E-core）按比例分配 row window——P-core 获得更多 window
2. **Steal granularity**：以 row window 为最小窃取单位——row window 内部是不可分割的 SME kernel 调用单元
3. **Progress monitoring**：各 core 通过共享 counter 报告已完成 window 数，用于 victim selection
4. **Steal policy**：从完成度最低的 core 窃取其一半剩余 window——避免窃取过多导致新不均衡
5. **无 OS 依赖**：纯 user-space 实现，不依赖 OS 的异构 core 调度支持

术语一般如何实现？如何使用？

实现为 C/C++ pthread 或 std::thread 多线程程序，使用 spinlock 保护共享队列（window 粒度 coarse enough，lock contention 低）。LX2（对称 SME，12 核等效）上 12 thread vs 2 thread 达 8x-11x scaling（static scheduling 有效）。M4（异构 SME）上多线程增益受 SME unit 数量限制（仅 2 个 SME unit 对应 2 个 cluster），但 work stealing 克服了 E-core SME 性能低的问题——E-core 完成后可窃取 P-core 的任务使总体时间接近 P-core 单独完成的分摊时间。

涉及论文标题：
- ASM-SpMM: Unleashing the Potential of Arm SME for Sparse Matrix Multiplication Acceleration

