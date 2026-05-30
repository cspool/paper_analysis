## Most-Room Policy (Thread Block Placement)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Most-Room Policy 是 NVIDIA GPU thread block scheduler 在并发 kernel workload 下将 thread block 分配到 SM 的实际调度策略，由 Gilman et al. 通过实证测量真实硬件（Pascal/Volta/Turing）发现。该 policy 定义：scheduler 选择**当前能容纳该 kernel 最多 block 数量**的 SM 来放置下一个 block，每次仅分配一个 block 到该 SM，并在有多个 SM 平票时按 device-specific fixed ordering 打破平票。容纳能力计算基于各 SM 当前的资源可用性（剩余 threads、shared memory、registers、blocks/SM 上限、warps/SM 上限），但**不考虑**与已 resident block 之间的资源竞争（如 L1 cache contention、functional unit contention）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Most-Room Policy 的决策逻辑（伪代码）：

```
# 每个 thread block 被调度时的决策:
function schedule_block(kernel_K, next_block_B):
    best_sms = []   # 能容纳最多 K block 的 SM 列表
    max_blocks = 0  # 当前最大容纳数

    for each SM s in all_SMs:
        # 计算 SM s 当前可容纳的 K block 数量
        room_threads = floor((s.max_threads - s.used_threads) / B.threads_per_block)
        room_shmem   = floor((s.max_shmem - s.used_shmem) / B.shmem_per_block)
        room_regs    = floor((s.max_regs - s.used_regs) / B.regs_per_block)
        room_blocks  = s.max_blocks - s.current_blocks
        room_warps   = s.max_warps - s.current_warps

        blocks_fit = min(room_threads, room_shmem, room_regs, room_blocks, room_warps)
        # blocks_fit 中最小的一项即为 Limiting Resource

        if blocks_fit > max_blocks:
            max_blocks = blocks_fit
            best_sms = [s]
        elif blocks_fit == max_blocks:
            best_sms.append(s)

    # Tie-breaking: 按 per-device fixed ordering 选第一个
    chosen_sm = min(best_sms, key=tie_breaking_order)
    assign B to chosen_sm
    update chosen_sm.used_resources
```

具体例子（Figure 2, Pascal GPU, 5 SMs）：
- Kernel X: 5 blocks × 256 threads，已占满 5 SM 各一个 block
- SM0 先完成 → 空（2048 free threads）
- Kernel Y: 3 blocks × 160 threads 此时发射
- SM0: floor(2048/160)=12 blocks of Y, SM1-4: floor(1792/160)=11 blocks → Y0→SM0
- SM0 更新: floor(1888/160)=11 → 与 SM1-4 平票 → tie-breaking (SM0=first) → Y1→SM0
- SM0 再更新: floor(1728/160)=10 < SM1-4 的 11 → Y2→SM1
- 结果: SM0 有 2 个 Y block，SM1 有 1 个 Y block（非 round-robin）

在单 kernel 场景中，由于所有 block 尺寸相同且行为相似，各 SM 资源可用性基本相同，most-room 与 round-robin 无法区分（Section 4.4）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

这是 NVIDIA GPU 硬件实现的闭源调度行为，非用户可编程 API。论文通过 `smid` 寄存器（读取 SM id）+ `globaltimer` 寄存器（控制 block 执行时间）+ `blockIdx`（识别 block）从真实硬件行为中推导得出。GPU 模拟器（GPGPU-Sim、Accel-Sim）可使用此 policy 改进 concurrent kernel workload 的模拟精度。已知的 limiting resource 包括 threads、shared memory、blocks/SM、warps/SM——但论文声明可能存在其他未识别的因素。Tie-breaking ordering: Pascal=ascending (0,1,2,3,4); Turing=even-then-odds (0,2,4,...,66,1,3,...,67)，可能与 TPC/GPC 组织和负载均衡有关。

涉及论文标题：
- Demystifying the Placement Policies of the NVIDIA GPU Thread Block Scheduler for Concurrent Kernels
- Characterizing Concurrency Mechanisms for NVIDIA GPUs under Deep Learning Workloads
