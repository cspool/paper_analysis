## Lazy Engine (Deferred GPU Execution)

术语是什么？
Lazy Engine 是 HuntKTm 的运行时组件，负责拦截并延迟执行 CUDA 操作，直到 task 的调度目标 GPU 被确定。编译器（Function Wrapper pass）在编译期将所有 CUDA API 调用包裹为拦截版本；运行时 lazy engine 将操作入队而不执行，在资源需求完全确定后汇总并通过共享内存发送给 task dispatcher，获取目标 GPU ID 后才顺序执行所有延迟操作。

从编译框架角度拆解术语：
Lazy engine 的编译-运行时协作流程：

```
编译期（Function Wrapper pass）:
  - 包裹所有 CUDA memory 和 kernel launch API 调用
  - cudaMallocAsync → __huntktm_cudaMallocAsync
  - cudaMemcpyAsync → __huntktm_cudaMemcpyAsync
  - kernel launch → __huntktm_kernel_launch（带 resource info collection）
  - 在资源需求确定点插入 cudaTaskSchedule / cudaTaskScheduleLazy

运行时（Lazy Engine）:
  Task 启动:
    1. 到达 intercepted CUDA 调用 → 入队到 operation queue（不执行）
    2. 重复直到到达 cudaTaskSchedule 或 cudaTaskScheduleLazy
    3. 汇总 queued operations 的资源需求
    4. 通过 shared memory 发送给 Task Dispatcher → 阻塞等待 GPU ID
    5. cudaSetDevice(target_gpu_id) 绑定设备
    6. 顺序执行 queue 中的操作
    7. 后续操作正常执行（不再 defer）
```

关键设计决策：
- cudaTaskSchedule：在资源需求静态可完全确定处插入
- cudaTaskScheduleLazy：在静态无法确定需求时插入，每次 kernel launch 前基于当前请求资源做决策
- Memory pool 预分配：task scheduler 初始化时预分配 memory pool，lazy engine 的 allocation 从 pool 直接返回，避免频繁系统调用

术语一般如何实现？如何使用？
编译期：LLVM function wrapper pass 对目标 CUDA API 做名称替换和参数转换，注入 C++ 拦截逻辑。运行时：lazy engine 维护 operation queue（内存分配/释放、kernel launch、数据传输），通过 cudaDeviceGetDefaultMemPool 使用默认 memory pool，cudaMemPoolSetAttribute 设置 release threshold 为预测的 memory footprint。与 task dispatcher 通过共享内存通信，避免系统调用开销。memory pool 减少了频繁 alloc/free 的运行时 overhead（对比不使用 memory pool 的 HuntKT 版本）。

涉及论文标题：
- HuntKTm: Hybrid Scheduling and Automatic Management for Efficient Kernel Execution on Modern GPUs

---
