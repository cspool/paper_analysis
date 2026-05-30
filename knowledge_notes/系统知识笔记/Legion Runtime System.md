## Legion Runtime System

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Legion 是 Stanford 大学开发的分布式 task-based runtime system，核心设计理念是"表达局部性和独立性"（expressing locality and independence）。Legion 提供两个关键抽象：(1) logical region：分布式数据集合，支持 content-based coherence——同一数据可以多种方式引用（aliasing），runtime 负责维护跨 processor 的数据一致性；(2) task：对 region 进行操作的用户定义函数，runtime 负责发现 task 间的依赖关系、计算并调度所需的通信。Legion 使用 dynamic dependence analysis [14] 在运行时发现 task 间依赖，而非依赖静态分析。Diffuse 构建在 Legion 之上——cuPyNumeric 和 Legate Sparse 原本直接 target Legion，Diffuse 修改它们以生成 Diffuse IR，在优化后转发 task stream 给 Legion。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Legion 在 Diffuse 系统栈中的位置：

```
用户 Python 程序 (cuPyNumeric / Legate Sparse API)
    │
    ▼
Diffuse (task fusion + kernel fusion 的中间层)
    │  优化后的 task stream → 转发给 Legion
    ▼
Legion Runtime
    ├── Dynamic Dependence Analysis [14]
    │   - 计算每个 task 的 precise data dependence
    │   - 仅计算每个 node 实际需要的部分依赖（非全局 materialize）
    ├── Distributed Coherence
    │   - 追踪 aliasing views 的数据一致性
    │   - 在需要时自动插入数据拷贝/通信
    ├── Task Scheduling
    │   - 将 task 调度到 processor (GPU/CPU)
    │   - 管理 task 间 synchronization
    └── Memory Management
        - 管理 distributed region 的物理内存分配
        - 处理 multi-GPU 间的数据移动
    ▼
Low-Level Execution (CUDA, OpenMP, ...)
```

Legion 的 task granularity 最小有效值约 1ms/task（Task Bench [49]），这意味着仅 task fusion（减少 task 数）不能显著提升性能——真正加速来自后续的 kernel fusion。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Legion 开源地址：https://legion.stanford.edu/。以 C++ runtime library 形式提供。任务注册通过宏完成，partition 通过 API 创建（如 `create_partition_by_tiling`）。终端用户不直接使用 Legion API——通过 cuPyNumeric/NumPy 或 Legate Sparse/SciPy Sparse API 间接受益。库开发者需了解 Legion 的 task 和 region 模型来编写 task implementation。Diffuse 作为中间层减少了对库开发者手工优化的需求——许多 fusion 优化由 Diffuse 自动完成。

涉及论文标题：
- Composing Distributed Computations Through Task and Kernel Fusion
