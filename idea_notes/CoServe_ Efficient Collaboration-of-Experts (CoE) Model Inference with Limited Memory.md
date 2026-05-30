## CoServe: Efficient Collaboration-of-Experts (CoE) Model Inference with Limited Memory

- baseline方法是什么？
  **Baseline 为 Samba-CoE（MICRO 2024）**：目前唯一探索大规模 CoE 模型部署的系统。Samba-CoE 使用 **FCFS（First-Come, First-Served）请求调度**和 **LRU（Least Recently Used）专家淘汰策略**，在 NUMA 设备上将专家 offload 到 CPU memory/SSD，按需加载到 GPU HBM 进行推理。

  **Baseline 全栈执行例子（以电路板缺陷检测 CoE, 300+ 专家, NUMA RTX3080Ti 12GB, 推理 3 个请求 R1→Expert1, R2→Expert2, R3→Expert1 为例）**：

  - **算法层**：输入组件图像 → Router 确定所需 Expert（如 Expert1 for 组件A, Expert2 for 组件B）→ Expert FFN（ResNet101 或 YOLOv5）执行分类/检测推理。路由规则由用户预定义（人工指定各组件对应专家），CoE 模型不涉及 token-level 动态路由（与 MoE 不同）。

  - **系统框架层**：Samba-CoE FCFS 调度 → R1 到达：Expert1 加载到 GPU → 推理完成 → R2 到达：Expert1 不在队列中可被淘汰 → LRU 淘汰 Expert1 加载 Expert2 → 推理完成 → R3 到达：Expert1 需从 CPU/SSD 重新加载 → 触发 expert switching。在 NUMA 设备上，从 SSD 切换专家占推理延迟 90%+；UMA 设备上占 60%+。

  - **编译框架层**：论文未明确说明（PyTorch 标准执行）。

  - **Kernel/运行时调度层**：标准 PyTorch GPU kernel（ResNet101 conv + YOLOv5 detection head）。Expert 从 CPU→GPU 或 SSD→GPU 通过 PCIe/NVMe 传输，传输时间串行化在推理前。无 kernel 级优化。

  - **硬件架构层**：NUMA 设备（RTX3080Ti 12GB GPU + Xeon CPU 16GB + SSD）或 UMA 设备（Apple M2 24GB 统一内存 + SSD）。GPU 显存不足以容纳全部 300+ 专家（>60GB），需 tiered storage（GPU→CPU→SSD）。

  **Baseline 的核心缺陷**：
  1. **FCFS 调度忽视请求间专家依赖**：依赖同一专家的多个请求在队列中可能被不相关的请求分隔，导致专家被不必要地淘汰后重新加载。例如 R1(Expert1)→R2(Expert2)→R3(Expert1)，R2 可能淘汰 Expert1，R3 再重新加载 Expert1 → 产生可避免的 expert switching。
  2. **LRU 淘汰依赖历史统计而非未来使用概率**：LRU 仅基于过去访问时间预测未来使用，在 CoE 场景中不准确。CoE 的路由规则是预先定义的，可以精确计算每个专家的使用概率，LRU 未利用这一信息。
  3. **静态内存分配未平衡专家加载与 batch 推理**：更大的 batch size 降低平均延迟但消耗更多中间结果内存 → 减少可常驻 GPU 的专家数量 → 增加 expert switching 频率。这一 trade-off 因不同处理器（CPU/GPU）和不同设备架构而异，手工调优困难。
  4. **单 executor 或 naive round-robin 多 executor 未优化负载分配**：不同专家的计算量和访问频率不同，静态均分请求导致部分 executor 过载而其他空闲。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **论文方法**：CoServe = Dependency-aware Request Scheduling + Dependency-aware Expert Management + Offline Profiler。核心洞察是 CoE 系统具有**专家依赖（Expert Dependency）**特性——请求间的依赖（多个请求需要同一专家）和专家间的依赖（后续专家依赖前置专家的输出）——可以用于减少不必要的专家切换。

  **Defect → Design 映射**：

  | Baseline 缺陷 | CoServe 设计选择 | 解决机制 |
  |---|---|---|
  | FCFS 导致可避免的 expert switching | Dependency-aware Request Arranging: 将新请求排在队列中同 expert 请求之后 | 同 expert 请求成组处理，一次加载服务所有同组请求 → expert switching 减少 78.5%-93.87% |
  | FCFS + round-robin 负载不均 | Dependency-aware Request Assigning: 预测额外推理延迟，选择使最大队列时间最小化的 executor | 动态平衡各 executor 负载，最小化总任务时间 |
  | LRU 淘汰低效（纯历史统计） | Dependency-aware Expert Management: 两阶段淘汰——先淘汰无前置依赖的后续专家，再按使用概率升序淘汰 | 使用预评估的专家使用概率（来自路由规则）替代历史统计 → 淘汰决策更准确 |
  | 静态内存分配无法适应不同硬件 | Offline Profiler + Sliding Decay Window: 通过 microbenchmarks + CDF 搜索最优专家数量 | 自动适应不同设备（NUMA/UMA），找到专家加载 vs batch 推理的最佳平衡点 |
  | 静态 executor 数量配置 | Offline Profiler 搜索最优 executor 组合（如 G3C1 vs G4C1） | 根据 workload 特征自动选择最优 executor 配置 |

  **CoServe 全栈执行例子（以同一场景 R1→Expert1, R2→Expert2, R3→Expert1, NUMA RTX3080Ti, 3 GPU executors 为例）**：

  - **算法层**：同 Baseline——Router 确定所需 Expert，Expert FFN 执行分类。CoServe 不修改算法层（不改变专家模型本身）。

  - **系统框架层**（关键差异）：
    1. Request Scheduler 预测 R1 在各 executor 队列的额外延迟 → 分配给 executor 1
    2. R2 到达 → Scheduler 预测：R2 需要 Expert2，各队列均无 Expert2 → 选择总时间最小的 executor（如 executor 2）
    3. R3 到达 → Scheduler 预测：R3 需要 Expert1 → executor 1 队列中有 Expert1 的请求 → 切换延迟=0 → 分配给 executor 1
    4. Request Arranging: R3 排在 R1 之后（同 Expert1）
    5. executor 1 处理：R1 batch + R3 batch 一起用 Expert1 推理，Expert1 仅加载一次
    6. 若 Expert1 不在 model pool：Expert Manager Stage 1 淘汰无前置依赖的闲置专家 → Stage 2 按使用概率淘汰 → 加载 Expert1

  - **编译框架层**：论文未明确说明（PyTorch eager execution）。

  - **Kernel/运行时调度层**：GPU executor 执行 expert FFN（ResNet101/YOLOv5）→ CPU executor 并行执行低优先级 batch → Expert loading (SSD→GPU) 与 GPU 推理可部分重叠（通过并行 executor）。Request scheduling 由 CPU 执行，与 GPU 推理并行。

  - **硬件架构层**：同 Baseline（RTX3080Ti 12GB + Xeon + SSD / Apple M2 24GB + SSD）。Offline profiler 自动确定 GPU 加载 35 个专家（Task A）或 34 个专家（Task B），剩余内存用于 batch 推理。

  **对比 Baseline 的核心改进路径**：
  ```
  Baseline (Samba-CoE):
  R1(Expert1) → R2(Expert2) → R3(Expert1)
  FCFS: Expert1 load → infer R1 → LRU evict Expert1 → 
  Expert2 load → infer R2 → Expert2 in pool → 
  Expert1 reload → infer R3
  专家切换: 3 次加载

  CoServe (Dependency-aware):
  R1(Expert1) → R3(Expert1) → R2(Expert2)
  Scheduling: R1→executor1, R3→executor1 (same expert, 排在R1后), R2→executor2
  Expert1 load → infer R1+R3 (batch) → Expert2 load → infer R2
  专家切换: 2 次加载（减少 33%）

  更复杂场景下（300+ 专家，大量请求），效果放大：
  CoServe 减少 expert switching 78.5%-93.87% → 吞吐量提升 4.5×-12×
  ```

  **关键创新总结**：CoServe 的核心洞察是将 CoE 系统与 MoE 系统的**关键区别**（CoE 的路由规则是预定义的、可离线分析的）转化为系统优化的机会：
  1. CoE 的路由规则可以预计算专家使用概率（MoE 无法做到，因为 MoE router 在推理时动态输出）→ 替代 LRU 的历史统计实现更准确的淘汰决策
  2. CoE 的专家依赖关系（后续专家等待前置专家结果）可以用于优先级排序 → 两阶段淘汰策略（优先淘汰"尚未需要的"而非"最近最少用的"）
  3. CoE 的请求-专家映射可以从路由规则提前获知 → 请求调度可以在知道未来需求的情况下做出安排（而非仅依赖 FCFS）

  这些优化都源于 CoE 区别于 MoE 的根本属性：路由的预定义性和可分析性。
