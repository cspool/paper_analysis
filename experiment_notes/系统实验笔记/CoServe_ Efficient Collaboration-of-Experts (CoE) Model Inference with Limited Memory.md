## CoServe: Efficient Collaboration-of-Experts (CoE) Model Inference with Limited Memory

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 **CoServe**——一个面向 CoE（Collaboration-of-Experts）模型推理的 heterogeneous CPU+GPU serving 系统，专为内存受限的边缘设备设计。核心包含三大技术：(1) **Dependency-aware Request Scheduling**：利用专家依赖关系，将依赖相同专家的请求在队列中分组排列，减少专家切换频率；同时动态分配请求到不同 CPU/GPU executor 队列中平衡负载；(2) **Dependency-aware Expert Management**：两阶段专家淘汰策略——优先淘汰无前置依赖的后续专家，不足时按预评估的使用概率淘汰低概率专家；(3) **Offline Profiler**：通过 microbenchmarks 自动确定最优显存分配和 executor 数量，生成专家性能矩阵和路由规则供在线调度使用。

  实验比较：
  - 吞吐量：CoServe Best/CoServe Casual vs Samba-CoE (FCFS+LRU)、Samba-CoE FIFO、Samba-CoE Parallel，在 NUMA (RTX3080Ti) 和 UMA (Apple M2) 设备上，4 个任务 (A1/A2/B1/B2) 对比
  - 专家切换次数：各方法在 4 个任务上的 expert switching 次数
  - 消融实验：CoServe None → CoServe EM (Expert Management) → CoServe EM+RA (Request Arranging) → CoServe (完整)，吞吐量和专家切换次数分解
  - Executor 数量消融：不同 GPU/CPU executor 组合 (G3C1, G4C1 等)
  - 内存分配搜索：sliding decay window 方法从 CDF 中选择最优加载 expert 数，对比吞吐量变化
  - 调度开销分析：request scheduling latency vs inference latency vs pre-scheduled inference，expert management 时间占比

- 硬件平台是什么，配置是什么。
  NUMA 设备：NVIDIA RTX3080Ti (12GB GPU Memory) + Intel Xeon Silver 4214R CPU (16GB Memory) + MICRON MTFDDAK480TDS SSD (530 MB/s)。
  UMA 设备：Apple M2 (24GB 统一内存) + APPLE SSD AP0512Z (~3000 MB/s)。

- 开源Serving框架是什么。修改了什么。
  CoServe 为**自研 PyTorch Serving 系统**，并非基于现有开源 Serving 框架（如 vLLM、SGLang）修改。其 baseline Samba-CoE 未公开代码。CoServe 在 PyTorch 基础上构建了完整的 CoE serving 运行时。

  **CoServe 架构三阶段**：
  
  **Offline Phase 修改**：
  - **Performance Profiler**：运行 microbenchmarks 为每个专家（同架构专家仅 profile 一次）测量最大 batch size、执行延迟 (K, B)、加载延迟、显存占用，生成 performance matrix
  - **Routing Rules & Usage Probabilities**：从用户提供的路由规则或小样本数据集计算每个专家的使用概率，生成 CDF 曲线
  - **Memory Allocation Optimizer**：在 GPU 上通过 sliding decay window 方法搜索最优加载专家数量，在 CPU 上使用最大 batch size 策略

  **System Initialization Phase**：
  - **Executor Creator**：根据 offline profile 结果创建指定数量的 GPU/CPU inference executors
  - **Expert Initializer**：按使用概率降序，round-robin 将专家加载到各 executor 的 model pool 中，直到内存用尽

  **Online Phase 修改**：
  - **Dependency-aware Request Scheduler**：
    - Prediction: 估算添加新请求到各 executor 队列后的额外推理延迟 = 执行延迟 (K × requests_in_batch + B) + 专家切换延迟 (0 或 expert loading time)
    - Assigning: 选择使当前各队列最大总推理时间最小的队列；平局时选择额外延迟最小的队列
    - Arranging: 将新请求排在同专家请求之后，实现请求分组
    - Batch Splitter: 根据当前可用内存和最大 batch size 将同专家请求组拆分为多个 batch
  - **Dependency-aware Expert Manager**：
    - Stage 1: 优先淘汰无前置依赖的后续专家（按显存降序淘汰直至足够）
    - Stage 2: 按使用概率升序淘汰（预评估概率，非 LRU 历史统计）

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  
  **开源情况**: 论文未公开代码仓库，web 搜索未发现 GitHub 链接。CoServe 使用 PyTorch 实现，评估使用自建的电路板缺陷检测 CoE 模型（论文明确指出"no publicly available CoE model exists for testing"）。

  **CoServe Serving 框架输入到硬件执行全过程（以 NUMA RTX3080Ti, 3 GPU executors + 1 CPU executor, Task A1 为例）**：

  1. **Offline Profiling（部署前执行一次）**：
     - 运行 ResNet101/YOLOv5 专家的 microbenchmarks → 获取最大 batch size (如 GPU batch=6)、执行延迟 K/B 参数、加载延迟、显存占用
     - 计算各专家使用概率（从电路板组件分布可知各组件被检测的概率 → 对应专家使用概率）
     - Sliding decay window 搜索 → 确定 GPU 加载 35 个专家、CPU 加载若干专家

  2. **System Initialization**：
     - Executor Creator 创建 3 个 GPU executors + 1 个 CPU executor
     - Expert Initializer 按使用概率降序 round-robin 分配专家：GPU Executor 1 加载专家 1,4,7...；GPU Executor 2 加载专家 2,5,8...；GPU Executor 3 加载专家 3,6,9...
     - 每个 executor 维护独立 model pool（GPU 显存 ∩ CPU 内存）

  3. **请求到达**：电路板组件图像连续输入（每 4ms 一个图像），每个图像携带目标组件类型信息，路由规则确定需要的分类专家（可能还有目标检测专家）。

  4. **Dependency-aware Request Scheduling**：
     - Scheduler 预测每个 executor 队列的额外推理延迟：
       - 若目标 expert 已在 model pool 中 → 切换延迟 = 0
       - 若队列中已有同 expert 请求 → 切换延迟 = 0（专家在前序请求处理期间加载）
       - 否则 → 切换延迟 = expert loading time
       - 执行延迟 = K × (batch requests) + B
     - 选择使最大队列总时间最小化的 executor → 将请求排在同专家请求之后
     - Batch Splitter 按当前可用内存和最大 batch size 拆分请求组

  5. **Expert Switching（如需要）**：
     - 若所需 expert 不在 model pool → Expert Manager 执行两阶段淘汰：
       - Stage 1: 找无前置依赖的后续专家 → 按显存降序逐一淘汰
       - Stage 2: 仍不足 → 按使用概率升序淘汰
     - 从 SSD/CPU memory 加载新 expert 到 GPU model pool

  6. **Inference Execution**：
     - GPU executor: batch 图像 → ResNet101/YOLOv5 expert FFN → 分类/检测结果
     - CPU executor: 并行处理低优先级 batch（使用 CPU 上的 expert）
     - 多 executor 并行执行，专家切换与推理重叠

  7. **输出返回**：缺陷检测结果（组件类型、缺陷类型、对齐点、焊接方向）返回给产线控制系统。

  **关键对比——CoServe vs Samba-CoE 请求处理差异**：

  | 阶段 | Samba-CoE | CoServe |
  |------|-----------|---------|
  | 请求调度 | FCFS，无重排序 | Dependency-aware: 同 expert 请求分组 + 负载均衡分配 |
  | 专家管理 | LRU（仅历史统计） | 两阶段: 依赖感知 + 使用概率 |
  | 内存分配 | 静态/手动 | Offline profiler + sliding decay window 自动搜索 |
  | 并行度 | 单 executor 或 round-robin 多 executor | 多 executor + 请求级动态分配 |

  **性能提升来源**：Dependency-aware scheduling 将同专家请求集中处理 → 一次加载服务多个请求 → 减少 expert switching；Dependency-aware eviction 更准确预测未来使用 → 减少不必要切换。
