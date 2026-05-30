## Task-Based Runtime System

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Task-Based Runtime System 是一种分布式编程运行时范式。应用程序将计算分解为 task（用户定义函数，操作于分布式数据之上），并以 task stream 形式提交给 runtime。Runtime 负责：(1) 从 task stream 中提取并行性；(2) 计算 task 间的同步和通信（数据依赖分析）；(3) 调度 task 到 processor 执行。主流 task-based runtime 包括：Legion [15]、StarPU [7]、PaRSEC [26]、Ray [41]、Pathways [11]、Dask 等。关键抽象：task = (function, data accesses)，runtime 通过分析 data access pattern 自动推断依赖关系。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Task-Based Runtime 在 Diffuse 的上下文中的运转流程：

```
cuPyNumeric 程序:
  z = x + y          // NumPy 操作
  w = z * 2.0        // 另一个独立操作

       ▼ 库分解

Task Stream (emit to runtime):
  T1: ADD([(x, R), (y, R), (z, W)])   // index task over partitioned data
  T2: MULT([(z, R), (w, W)])

       ▼ Runtime 处理

  1. Dependence Analysis:
     - T2 depends on T1 (reads z which T1 writes)
     - Synchronization: T2 must wait for T1 (at point-task level)
  
  2. Data Coherence:
     - 追踪 aliasing views 的数据一致性
  
  3. Task Scheduling:
     - 将 point tasks 分配到 processor (GPU/CPU)

       ▼ 执行: GPU 0..N-1 execute T1, then T2
```

Diffuse 定位为 task-based runtime 系统的中间优化层：在库分解出 task stream 之后、runtime 执行 task 之前进行 fusion 分析。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Task-based runtime 通常以 C/C++ library（如 Legion, StarPU）或 Python/分布式框架（Ray, Dask）形式提供。终端用户通过 NumPy/SciPy drop-in replacement 库（cuPyNumeric, Legate Sparse）间接使用。Diffuse 证明：通过在 task-based runtime 之上添加 scale-free 的中间分析层，可在不修改应用代码的情况下自动实现跨库边界的优化。

涉及论文标题：
- Composing Distributed Computations Through Task and Kernel Fusion
