## Resource Analyzer (Compile-Time GPU Resource Profiling)

术语是什么？
Resource Analyzer 是 HuntKTm 的编译期 LLVM pass，在编译时提取每个 task 的 GPU 资源需求信息。它分析两个维度：(1) 计算资源——threads per kernel launch、registers per kernel、shared memory per kernel；(2) 内存资源——per-object allocation size。这些信息在编译期嵌入程序，运行时由 lazy engine 查询并发送给 task dispatcher 用于资源感知的调度决策。

从编译框架角度拆解术语：
Resource analyzer 的编译流程：
```
1. 调用 nvcc --ptxas-options=-v 编译每个 kernel
   → 获取 per-kernel register 数和 shared memory 使用量

2. 遍历 stream graph 中的 kernel launch 和 memory allocation:
   - Kernel launch: gridDim × blockDim → thread 需求
   - cudaMalloc(size): allocation size → memory 需求
   - Per-stream 计算需求 ≈ stream 首个 kernel 的需求（保守近似）

3. 插入调度指令:
   - 资源需求完全静态确定 → 插入 cudaTaskSchedule
   - 部分需求依赖运行时输入 → 插入 cudaTaskScheduleLazy
   - 插入位置：资源需求完全确定后的最早可能点
```

编译期分析的局限性：
- 函数封装或复杂控制流可能阻止静态追踪所有 memory allocation
- 运行时依赖的输入大小（如命令行参数决定的矩阵维度）无法静态确定
→ 这些情况交由 lazy engine 在运行时补充

术语一般如何实现？如何使用？
基于 LLVM pass，调用外部工具 nvcc 获取 kernel 属性。遍历 IR 中的函数调用指令，识别 `__cudaPushCallConfiguration` 获取 kernel launch 配置参数，识别 `cudaMalloc`/`cudaMallocAsync` 获取 allocation size。通过 LLVM 的 def-use chain 追踪 size 表达式到常量或输入参数——可追踪到常量则静态确定，否则标记为运行时依赖。与 nvcc 的集成通过 `--ptxas-options=-v` flag 解析输出获取详细的寄存器/共享内存报告。

涉及论文标题：
- HuntKTm: Hybrid Scheduling and Automatic Management for Efficient Kernel Execution on Modern GPUs
