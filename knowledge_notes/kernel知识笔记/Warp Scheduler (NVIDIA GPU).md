## Warp Scheduler (NVIDIA GPU)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Warp Scheduler 是 NVIDIA GPU 每个 SM 内部的硬件调度单元，负责在每个时钟周期从该 SM 上所有 resident thread block 的 ready warps 中选择若干 warp 发射指令到执行核心。Ampere 架构每 SM 有 4 个 warp scheduler 单元，每个 scheduler 每两周期可发射一条 warp 指令。Warp scheduler 采用的具体调度策略（greedy-then-oldest 或 loose round-robin）NVIDIA 未正式文档化，由 Olmedo et al. (2020) 等通过实证逆向工程推断。关键特性：(i) Warp 间切换 **零成本**（因 SM 有独立 per-warp register file 和 program counter，无需 context switch）；(ii) Scheduler 通过 warp 交错执行隐藏 memory latency（当某 warp 等待 global memory 时发射另一 warp）；(iii) Priority streams 是否影响 warp scheduler 的选择——官方文档未说明，本文推测 warp scheduler 可能不感知 stream priority，导致高优先级 warp 被 "de-prioritized"。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Warp Scheduler 的 issue 决策逻辑（推测模型）：

```
// Per-SM, per-cycle warp scheduling
function warp_scheduler_cycle(sm):
    for each of 4 warp_scheduler_units in sm:
        // 每个 scheduler 管理 subset of warps
        eligible_warps = []
        for warp in scheduler.warp_pool:
            if warp.is_ready():  // 无 data hazard, 无 memory stall
                eligible_warps.append(warp)
        
        if eligible_warps is empty:
            issue NOP  // idle cycle
        
        // 调度策略（逆向工程推测，未文档化）:
        if policy == "greedy-then-oldest":  // 多数文献认为
            // 优先: 上一周期发射过的 warp（保持指令 cache warm）
            if prev_warp in eligible_warps:
                issue prev_warp
            // 次选: 等待最久的 ready warp
            else:
                issue oldest_ready_warp(eligible_warps)
        
        elif policy == "loose round-robin":  // 先前文献也观察到
            next_warp = round_robin_next(scheduler.warp_pool)
            while not next_warp.is_ready():
                next_warp = round_robin_next(scheduler.warp_pool)
            issue next_warp
```

Ampere SM Warp Scheduler 资源参数（RTX 3090）：
- 4 warp schedulers / SM
- 每 scheduler 每 2 周期 1 条 warp 指令
- 峰值: 4 × 0.5 × 1.7GHz (boost) ≈ 3.4 warp instructions / clock / SM

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Warp scheduler 是 GPU 硬件固件的闭源实现，用户不可编程。影响 warp scheduling 的间接方式：(i) 控制 occupancy（SM 上同时 resident 的 warp 数量）——更高 occupancy 给 scheduler 更多 warp 可隐藏延迟；(ii) 避免 warp divergence（thread 在 warp 内走不同分支 → scheduler 串行化路径）；(iii) 控制 block 的 thread/block 数影响 warp 在 scheduler 间的分布。在 concurrent workload 场景中，本文指出若 warp scheduler 不感知 stream priority，两个应用的 warps 在 SM 内会被同等对待——即使一个来自高优先级 stream。

涉及论文标题：
- Characterizing Concurrency Mechanisms for NVIDIA GPUs under Deep Learning Workloads
