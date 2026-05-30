## XPUTimer

术语解释
Ling 团队开发的轻量级分布式训练性能分析工具（Cui et al. 2025），集成在 DLRover 中。通过 selective tracing + async CUDA event management + data compression 实现 90% 内存节省（~1.5MB/加速器/step）和 O(1) 错误定位。

术语是什么？
传统 profiler（如 NVTX）全量监控产生海量日志难以在生产环境长期使用。XPUTimer 由两大组件构成：(1) Lightweight Selective Tracing——Python 层通过环境变量动态拦截 API，C++/CUDA 层框架无关 kernel 监控（cuBLAS、Flash Attention、NCCL、自定义算子），CUDA event pool 复用+异步后台线程日志+数据压缩仅记录时间戳和 kernel input layout。(2) Diagnostic Engine——multi-layered diagnostic（call stack analysis + in-kernel tracing）将错误定位复杂度从 O(logN) 降至 O(1)，结合宏观 metric (throughput) 和微观 metric (kernel launch latency distribution) 做细粒度异常检测。

从kernel调度角度拆解术语：
```
=== CUDA Event Pool（低开销核心）===
ev_pool = [cudaEventCreate() for _ in range(MAX)]

# 注入 NCCL kernel 后追踪
cudaEventRecord(ev_start, stream)
ncclAllReduce(...)
cudaEventRecord(ev_stop, stream)

# 后台线程异步检查
while training:
    if cudaEventQuery(ev_stop) == cudaSuccess:
        elapsed = cudaEventElapsedTime(ev_start, ev_stop)
        log({kernel: "ncclAllReduce", time: elapsed, layout: dims})

# 压缩: 仅记时间戳+kernel layout，不记完整 tensor
```

术语一般如何实现？如何使用？
- 集成在 DLRover (github.com/intelligent-machine-learning/dlrover)
- Python 层通过 TRACED_PYTHON_API 环境变量动态配置监控目标
- 生产环境长期运行 profiling 无显著性能影响

涉及论文标题：
- Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs
