## Time-Shifted Launching (时移启动)

术语是什么？

Time-Shifted Launching（时移启动）是μShare针对blocksize不可修改的CUDA kernel（如cuDNN/cuBLAS wrapper函数和tiling kernel如Conv2d）设计的互补调度技术。由于这些kernel的blocksize隐藏在闭源代码中或修改后产生CUDA internal error（如Conv2d tiling匹配破坏），无法通过half-plus shaping进行scattered co-location。Time-shifted launching通过控制这些kernel的启动时机（relaunch time），延迟其launch直到SM上正在执行的half-plus kernel与待启动kernel的资源需求互补（combined hardware utilization ≤ 100%且shared memory/registers足够），此时才释放kernel进行co-location。如果等待导致kernel launch slack变为negative（接近SLO violation），则将其升级为half-plus shaping。

从kernel调度角度拆解术语：

Time-shifted launching的调度伪代码：

```
// 输入：kernel set O = X ∪ Y, X=half-plus shaped, Y=time-shifted
// 对每个kernel kj ∈ Y：
function time_shifted_launch(kj):
    while kj not launched:
        // 1. 检查资源互补条件
        can_colocate = true
        for each hw_resource in {FP32, FP64, INT32, LDST, SFU, Tensor}:
            if current_SM_active_util[hw_resource] + kj.util[hw_resource] > 100%:
                can_colocate = false
                break
        if kj.smem + current_SM_active_smem > SM_total_smem:
            can_colocate = false
        if kj.reg + current_SM_active_reg > SM_total_reg:
            can_colocate = false
        
        // 2. 满足条件 → 立即launch (用default blocksize，不修改)
        if can_colocate:
            cudaLaunchKernel(kj.func, kj.gridDim, kj.default_blocksize, ...)
            return
        
        // 3. 不满足 → delay β=10μs 后重检
        usleep(β)
        
        // 4. 更新slack = tLaunch - current_time
        kj.slack = kj.tLaunch - now()
        
        // 5. 重新排序整个kernel set O
        O_sorted = sort_ascending_by_slack(O)
        
        // 6. 若kj进入top-x（|X| = min x s.t. Σ^{x}_{i=1} blocks_i.count > num_SMs）
        if rank(kj, O_sorted) <= x:
            // 升级为half-plus shaping
            kj.blocksize = half_plus_sm_threads()
            cudaLaunchKernel(kj.func, kj.gridDim, kj.blocksize, ...)
            return
```

术语一般如何实现？如何使用？

实现要点：
1. **资源互补判断**：依赖offline profiled per-kernel 9-tuple resource profile，profiler使用NVIDIA Nsight Compute记录6种low-level hardware utilization + Nsight Systems记录launch timing
2. **延迟参数β**：论文通过多次实验确定最优β=10μs。β太小导致频繁重检overhead，β太大浪费co-location窗口
3. **Slack动态管理**：随着等待时间增长，kj的slack递减，可能进入sorted set的top-x位置 → 升级为half-plus。这形成一种优雅的退化路径：worst-case（所有kernel都unmodifiable）→ 所有kernel by time-shifted launching拉满 → 等效于resource-coupled co-location（INFless级别性能）
4. **适用性**：μShare的10模型分析中unmodifiable kernel占48.37%（3290/6802次执行），time-shifted launching是必要但非主要的补充机制。实验显示unmodifiable kernel比例从100%降至48.37%时throughput从47.59单调提升至58.81

涉及论文标题：
- μShare: Non-Intrusive Kernel Co-Locating on NVIDIA GPUs

