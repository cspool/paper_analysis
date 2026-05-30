## CUDA Graph

术语是什么？
CUDA Graph 是 NVIDIA 自 CUDA 10 起提供的框架，允许开发者将一系列 GPU 操作（kernel launch、memory copy、memset 等）及其依赖关系预先定义为一个有向无环图（DAG），然后通过单次调用在 GPU 上重放整个图。CUDA Graph 的核心优势是消除重复的 kernel launch 和同步开销：传统模式下每次 kernel launch 需要 CPU→GPU 通信（约 5-20μs/kernel），而 graph replay 只需要一次 launch，所有操作及其依赖关系已在 GPU 端处理。CUDA Graph 支持 graph capture（从现有 stream 操作捕捉）和显式 API 构建两种方式。

从系统架构角度拆解术语：
CUDA Graph 的工作流程：
```
CUDA Graph (静态图) 执行流程:
┌─────────────────────────────────────────────────────┐
│ Step 1: Graph Construction (在CPU端)                 │
│                                                      │
│   cudaGraphCreate(&graph);                           │
│                                                      │
│   // 方法A: Stream Capture (自动捕捉)                │
│   cudaStreamBeginCapture(stream);                   │
│   kernelA<<<..., stream>>>(...);  // 节点A          │
│   kernelB<<<..., stream>>>(...);  // 节点B (依赖A)  │
│   cudaStreamEndCapture(stream, &graph);             │
│                                                      │
│   // 方法B: Explicit API (显式构建)                  │
│   cudaGraphAddKernelNode(&nodeA, graph, ..., kernelA)│
│   cudaGraphAddKernelNode(&nodeB, graph, ..., kernelB)│
│   cudaGraphAddDependencies(graph, nodeA, nodeB);    │
│                                                      │
│ Step 2: Graph Instantiation                          │
│   cudaGraphInstantiate(&instance, graph);           │
│                                                      │
│ Step 3: Graph Launch (可重复多次)                    │
│   cudaGraphLaunch(instance, stream);                │
│   // 仅一次CPU→GPU通信，所有kernel在GPU端按DAG执行  │
└─────────────────────────────────────────────────────┘
```

CUDA Graph 对静态计算图（如固定 DNN 推理）极有效——图只需构建一次即可重复使用。但在 ACS 的目标场景（input-dependent 计算图，即每次输入产生不同的 kernel 依赖 DAG）中，CUDA Graph 的致命缺陷是每次输入变化都需要重新构建整张图。ACS 论文实验表明，Brax 仿真中使用 CUDA Graph 的 DAG 构造时间平均占程序总执行时间的 47%，导致 CUDA Graph 在动态图中反而比单 stream 串行更慢。

术语一般如何实现？如何使用？
CUDA Graph API 包括：`cudaGraphCreate`、`cudaGraphAddKernelNode`、`cudaGraphAddDependencies`、`cudaGraphInstantiate`、`cudaGraphLaunch`。要求 GPU compute capability ≥ 3.5（基本功能），某些高级特性需要更高版本。CUDA Graph 还支持 conditional nodes（条件节点）和 device graph launch（CUDA 12.x+）。AMD 等价框架为 ATMI（Asynchronous Task and Memory Interface），使用 barrier packet 机制在命令队列中编码 DAG 依赖。ACS 论文定量比较了其方案与 CUDA Graph：静态 DNN 中二者性能相当（图只需构建一次），动态 workload 中 ACS 显著优于 CUDA Graph（避免每次重建 DAG）。

涉及论文标题：
- ACS Concurrent Kernel Execution on Irregular, Input-Dependent Computational Graphs
