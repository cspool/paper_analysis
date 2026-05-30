## Workgroup Specialization for Multi-GPU Communication

术语是什么？
Workgroup Specialization是将GPU的compute units在单个persistent kernel内划分为不同角色——部分workgroup专门执行计算（GEMM），另一部分专门执行通信（跨GPU数据传输）——通过atomic-based spin-lock传递tile级就绪信号。这是Iris实现最高效compute-communication overlap的kernel调度技术。在8×MI300X（每GPU 304 CU）上，典型配置：256 GEMM workers + 48 COMM workers。

从kernel调度角度拆解术语：
```
@triton.jit()
def wg_specialized_gemm_all_scatter(A, B, C, locks, GEMM_SMS, COMM_SMS, ...):
    pid = tl.program_id(0)
    if pid < GEMM_SMS:
        # === GEMM Worker (前256个workgroup) ===
        for tile_id in range(pid, total_tiles, GEMM_SMS):
            c = gemm_loop(A, B, C)
            tl.store(C + offset, c, mask=mask, cache_modifier=".wt")
            tl.atomic_cas(locks + tile_id, 0, 1, sem="release", scope="gpu")
    else:
        # === COMM Worker (后48个workgroup) ===
        pid = pid - GEMM_SMS
        for tile_id in range(pid, total_tiles, COMM_SMS):
            while tl.atomic_cas(locks + tile_id, 1, 0, sem="acquire", scope="gpu") == 0:
                pass  # spin-lock等待GEMM worker完成
            for remote_rank in range(world_size):
                if remote_rank != cur_rank:
                    iris.put(C + offset, C + offset, cur_rank, remote_rank, heap_bases, mask=mask)
```
GEMM worker使用write-through cache modifier确保COMM worker通过Infinity Fabric coherence看到最新数据。通信tile_n隐藏于GEMM tile_{n+1}的计算期间。

与NVIDIA warp specialization的区别：warp specialization在同一SM/warpgroup内划分角色，workgroup specialization在不同CU上划分角色——后者更适合AMD架构（因AMD静态寄存器分配限制不支持warp specialization直接移植）。

术语一般如何实现？如何使用？
通过triton.jit内pid范围判断实现，开发者需手动实验确定最优GEMM/COMM workgroup数量分配。

涉及论文标题：
- Iris: First-Class Multi-GPU Programming Experience in Triton
