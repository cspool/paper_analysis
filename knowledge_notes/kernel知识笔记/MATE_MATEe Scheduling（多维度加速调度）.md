## MATE/MATEe Scheduling（多维度加速调度）

术语是什么？
MATE（Multi-dimensional Acceleration for Torus with Error）是针对 N-D torus 上链路故障场景的容错 All-to-All 多维调度方法。核心思想：利用 torus 的多维正交特性——其他维度的健康链路可在不冲突的前提下构建故障环上相邻节点的双向逻辑连接，使故障环也能使用 HalfRing 执行高效数据传输。MATE 将通信拆分为正常 phase（仅 FoldedRing 或跳过）+ 加速 phase（利用逻辑连接执行 HalfRing）。MATEe（增强版）在正常 phase 也传输部分数据（按 HalfRing/FoldedRing 性能比静态分配），减少加速 phase 数据量。

从kernel调度角度拆解术语：
MATE 调度结构（2N 个 phase，N=维度数）：

```
Procedure MATE_Scheduler(S, N, D_fault, Torus, Link, mode):
    Chunk_Num = N
    Chunk_Size = S / N
    for chunk = 0 to Chunk_Num - 1:
        for phase = 0 to 2N - 1:
            if phase % 2 == 0:             // 正常 phase
                p = phase / 2
                dim = (chunk + p) % N
                if dim != D_fault:
                    Schedule[chunk][phase] = HalfRing_Gen(Torus[dim], Link)
                else:
                    if mode == MATE:
                        Schedule[chunk][phase] = None  // 跳过
                    else:  // MATEe
                        fraction = perf_ratio(HalfRing, FoldedRing)
                        Schedule[chunk][phase] = FoldedRing_Gen(Torus[dim], Link)
                        // (仅传输 fraction 比例的数据)
            else:                           // 加速 phase
                planes = GetAvailPlanes(D_fault)
                Schedule[chunk][phase] = HalfRing_Planes(planes, Torus, Link)
                // + FoldedRing_Gen(Torus[D_fault], Link) [未在正常 phase 传输的剩余数据]
```

加速原理（2D torus 例，X-dim 故障）：Y-dim 链路将故障 X-dim 环上的相邻节点（如 (0,1)-(0,2)-(1,2)-(1,1)）连接为一组双向逻辑 X-dim 链路，共可构建 N-1 组（每个 Y-dim 平面一组）。加速 phase 在这些逻辑链路上执行 HalfRing，将故障环通信"卸载"到健康环。

术语一般如何实现？如何使用？
MATE 需要离线性能分析以分配各加速平面的数据量（确保并发传输时间一致）。MATE 可处理更复杂故障场景：(1) 同环多故障——FoldedRing 失败，所有通信通过加速 phase 完成；(2) 多环各一故障——为每个故障分配独立加速 phase，无链路冲突时可并行；(3) 异维各一故障——各自独立加速 phase。在 2D torus 上，MATE 性能为 fault-free baseline 的 1.36×（超过 fault-free baseline 1.0×）。MATEe 在小数据量下性能不如 MATE（因静态分配未考虑启动时间差异），但在大数据量下性能更优（加速 phase 数据量更少）。

涉及论文标题：
- Optimizing All-to-All Collective Communication with FaultTolerance on Torus Networks
