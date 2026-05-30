## Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是三个运行时代码/调度层面的基础设施：(1) **XPUTimer**：轻量级分布式训练性能分析工具，集成在 DLRover 中。包含 lightweight selective tracing（Python/C++/CUDA 级截获，异步事件管理）和 diagnostic engine（O(1) 快速定位、细粒度诊断）。通过 event pool 管理、异步后台线程数据收集、数据压缩实现减少 90% 内存开销（~1.5MB/加速器/step）。(2) **EDiT (Elastic Distributed Training)**：基于 Local SGD 的高效异步分布式训练方法，包含 layer-wise synchronization（逐层同步+prefetch 重叠通信计算）、pseudo gradient penalty（异常 worker 排除 + 加权平均 + 梯度裁剪）、time-based synchronization（按时间而非固定步数触发同步，解决 straggler 问题）。(3) **PCache** + **Babel**：PCache 是全闪存分布式文件缓存系统，利用 FUSE（用户空间文件系统）+ shared memory 减少用户/内核态切换开销，metadata cache 加速随机读，AI co-design（分散 DP group checkpoint 写入）降低 checkpoint 写延迟 50%+ 峰值内存 60%。Babel 是跨集群数据同步中间件，支持 PB 级数据并行 metadata prefetch（190M 文件从 >6h 降至 ~10min，36× 加速）和 content-sampling CRC 校验（100GB 文件校验 ~3s）。

  实验比较：(a) XPUTimer 内存使用 vs 其他 profiling 方法（减少 ~90%）；(b) EDiT vs 传统同步分布式方法在速度上的对比（最大加速 66.1%）；(c) PCache vs GPFS checkpoint 写延迟（70s vs 160s @ 128 accelerators, 90s vs 240s @ 512 accelerators）；(d) Babel 并行 metadata prefetch vs 串行（>6h vs ~10min）；(e) Babel CRC 校验 vs MD5（~3s vs tens-to-hundreds of seconds for 100GB files）。

- 后端平台是什么，配置是什么。
  异构 AI 加速器集群（Device A~E，见算法 pipeline 条目）。集群规模从 128 到 10,000+ 加速器。PCache 在 1,000 加速器集群上聚合吞吐 1 TB/s，10,000 加速器线性扩至 8 TB/s。EDiT 性能评估在理想环境和异构环境中分别测试。

- 评估性能的软件/脚本是什么。修改了什么。
  - **XPUTimer** 集成在 **DLRover**（[https://github.com/intelligent-machine-learning/dlrover](https://github.com/intelligent-machine-learning/dlrover)）中，独立开源。修改：(a) 在 Python 层通过环境变量 TRACED_PYTHON_API 动态配置需要监控的 API；(b) C++/CUDA 层通过框架无关的 kernel 截获机制监控 cuBLAS、Flash Attention、NCCL 操作及自定义算子；(c) CUDA event 注入 NCCL kernel launch 后并通过后台线程异步监控完成状态。
  - **EDiT** 修改分布式训练同步逻辑：在每个 worker 上逐层独立 forward→backward→sync（而非等所有 worker 完成整个 step 后 All-Reduce），pseudo gradient 使用 EMA 跟踪检测异常 worker 并排除。
  - **PCache** 修改 checkpoint 写入逻辑：将 Megatron 默认的 DP group rank_0 集中写入改为分散写入不同物理节点，通过 FUSE + shm 消除多次用户/内核态切换和数据拷贝。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  DLRover: [https://github.com/intelligent-machine-learning/dlrover](https://github.com/intelligent-machine-learning/dlrover)。XPUTimer 诊断流程：

  ```
  === XPUTimer Selective Tracing 原理 ===

  Step 1: Python 层截获（错误拦截）
    export TRACED_PYTHON_API = "gc.collect,torch.cuda.synchronize,DataLoader.__iter__"
    # 运行时通过 monkey-patch 注入 hook
    @trace_hook
    def traced_synchronize():
        t_start = record_timestamp()         # 同步 API 记录时间戳
        original_synchronize()               # 执行原始操作
        t_end = record_timestamp()

  Step 2: C++/CUDA 层截获（框架无关 kernel 监控）
    # 以 NCCL AllReduce 为例
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start); cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start, compute_stream);
    ncclAllReduce(...);                       # 原始通信 kernel
    cudaEventRecord(ev_stop, compute_stream);

    # 后台线程异步检查
    background_thread:
      while training:
        if cudaEventQuery(ev_stop) == cudaSuccess:
          elapsed = cudaEventElapsedTime(ev_start, ev_stop)
          log({kernel: "ncclAllReduce", time: elapsed, layout: input_dims})

  Step 3: 低开销设计
    # Event Pool: 预分配并复用 CUDA event，避免动态分配
    ev_pool = [cudaEventCreate() for _ in range(MAX_PENDING_EVENTS)]

    # 数据压缩: 仅记录时间戳 + kernel 输入布局
    log_entry = {ts: t_start, kernel: "gemm", m: 4096, n: 1536, k: 5120}
    # 不保存完整 tensor 内容，压缩后 ~1.5MB/accelerator/step

  === EDiT 异步训练流程 ===

  Workers W_0, W_1, W_2, W_3 (4 workers example):

  for step in training:
    # 各 worker 独立 forward + backward
    for layer in model:
      # Forward (并行)
      worker_i: hidden = layer.forward(hidden)
      # Layer-wise sync: 完成一层后立即同步该层
      if layer % sync_interval == 0:
        broadcast_layer_weights()  # 非阻塞 prefetch

      # Backward
      grad = layer.backward(loss)

    # Pseudo Gradient Penalty
    pseudo_grad_i = (current_params - prev_params) / lr
    # 1. 异常检测: EMA 跟踪 pseudo_grad 的 norm
    if |norm(pseudo_grad_i) - EMA(norm)| > threshold:
        exclude_worker(i)  # 异常 worker 被排除

    # 2. 加权平均: 按 pseudo_grad norm 加权
    weights = softmax([1/norm(pg_j) for j in valid_workers])
    fused_grad = sum(w_j * pseudo_grad_j)

    # 3. 梯度裁剪
    fused_grad = clip_by_norm(fused_grad, threshold)

  # Time-based sync（非固定步数）
  if elapsed_time > sync_deadline:   # 达到时间阈值
      synchronize_all_workers()

  === PCache Checkpoint 写入优化 ===

  # Megatron 默认: rank_0 集中写入
  Default: all DP groups write through rank_0 of each group
            → 集中到少数物理节点 → CPU+网络拥塞

  # PCache AI Co-design: 分散写入
  Optimized: round-robin assign write target per DP group
    dp_group_0 → physical_node_0
    dp_group_1 → physical_node_7
    dp_group_2 → physical_node_3
    ...

  # FUSE + shm 加速
  Write path:
    App → FUSE (userspace) → shm (shared memory, zero-copy) → NVMe SSD
    # 避免传统路径: App → kernel VFS → kernel FS → block layer → SSD
  ```
