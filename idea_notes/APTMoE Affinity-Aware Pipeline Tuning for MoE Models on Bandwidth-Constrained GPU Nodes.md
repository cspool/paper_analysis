## APTMoE Affinity-Aware Pipeline Tuning for MoE Models on Bandwidth-Constrained GPU Nodes

- baseline方法是什么？
  **Baseline 为 Mobius（ASPLOS 2023）**，它是带宽受限 GPU 节点上结合流水线并行和 offloading 技术的微调系统。Mobius 将模型分区为多于 GPU 数量的 stage，相邻 stage 映射到不同 GPU，通过跨根复合体映射（cross-mapping）减少带宽竞争。在前向过程中，Mobius 在执行当前 stage 时预取下一 stage 的参数，完成后卸载参数和激活到 host memory；反向过程再从 host memory 预取参数和激活。

  **Baseline（Mobius）全栈执行例子（以 MoE 模型在一个 micro-batch 上的 forward 为例）**：
  - 算法层：标准 MoE top-2 gating（Linear-Softmax-TopK），所有 expert 在 GPU 上计算，无 CPU 参与计算，无 expert 热度利用
  - 系统框架层：流水线并行（GPipe-style stage partition）+ Offloading（host↔GPU）。每个 GPU 持有多个 stage，当前 stage 执行时预取下一 stage 全部参数到 GPU 内存，无选择性加载
  - 编译框架层：论文未明确说明（标准 PyTorch eager 执行）
  - Kernel 调度层：标准 CUDA stream 执行，无专门的数据移动调度优化。所有数据移动按 stage 粒度顺序执行，不同 stage 的数据移动可能互相阻塞
  - 硬件架构层：NVIDIA A800 GPU (40GB) × 4，PCIe Switch 互联，无 NVLink/InfiniBand。Intel Xeon Gold 6348 CPU 仅用于 host memory 存储，不参与计算

  **Baseline 的核心缺陷**：MoE 架构下 data-to-computation ratio 显著增加（expert 数量多而每个 token 仅激活 k 个），导致 Mobius 的 stage 级全量加载方式出现**计算阻塞问题**——数据加载时间超过计算时间，GPU 等待数据而闲置。此外，Mobius 未利用 MoE 的 expert 热度偏斜特性（少数 expert 承担大部分 token），也未利用 CPU 的计算资源。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  APTMoE 提出**亲和感知 offloading（Affinity-Aware Offloading）**，核心思想是基于 expert 热度和计算亲和性将部分低热度 expert 的计算分配到 CPU，减少 GPU 数据搬移量并提升计算效率。具体通过两个策略实现：

  1. **分层加载策略（Hierarchical Loading Strategy）**：将 Mobius 的 stage 级粗粒度加载细分为三层：
     - **Inter-stage loading**：基于历史 expert 热度（少数 expert 跨时间持续高激活），在 stage 切换时预取高热度 expert 到 GPU
     - **Inter-layer loading**：基于预测器（predictor）提前预测目标层 expert 热度，在当前层计算时预加载下一层高需求 expert
     - **Inter-expert loading**：基于实时 gate 输出，在同一层内动态决定哪些 expert 加载到 GPU、哪些留在 CPU 计算
     使用 Equation 1 (R = ΣCPU_time / ΣLoad_time) 作为 GPU/CPU 分配决策阈值，R=1 为停止加载边界。

  2. **需求优先级调度策略（Demand-Priority Scheduling Strategy）**：解决三层加载对同一 PCIe 带宽的竞争，通过 PriorityQueue 按 inter-expert > inter-layer > inter-stage 优先级动态协调加载顺序，采用 CUDA Event 前探机制在 kernel 启动前调度，隐藏 launch overhead。

  **APTMoE 方法全栈执行例子（以 MoE 模型在一个 micro-batch 上的 forward 为例）**：
  - 算法层：Expert 热度预测器（与 gate 同结构，共享权重初始化，提前若干层放置）预测目标层 expert 激活分布 → Equation 1 计算 CPU/GPU 分配阈值 → 高热度 expert 加载到 GPU 用 cuBLAS GEMM 计算 → 低热度 expert 留在 host memory 由 CPU 就地计算（使用 PyTorch CPU tensor），无需加载到 GPU
  - 系统框架层：流水线并行（相邻 stage 映射到不同 GPU）+ 三层加载流水线（inter-stage→inter-layer→inter-expert）→ 每层仅加载部分 expert 参数（而非全量），减少 PCIe 数据搬移量 → 反向过程中所有 expert 热度已知，inter-stage 阶段一次性全局最优分配
  - 编译框架层：论文未明确说明（PyTorch eager 执行，未涉及编译优化）
  - Kernel 调度层：comp_stream 执行计算与 load_stream 执行数据移动并行 → PriorityQueue 动态调度三层加载 → CUDA Event 前探隐藏 kernel launch latency → Inter-stream event 保证 data dependency 正确性
  - 硬件架构层：NVIDIA A800 GPU × 4（PCIe Switch），Intel Xeon Gold 6348 CPU 参与低热度 expert 计算（非仅存储）。三种设备拓扑（C1+G4/G2/G1）验证 CPU 核心数对性能的影响。最大扩展至 16 GPU（3 节点）

  **关键设计对应关系**：
  | Baseline 缺陷 | APTMoE 解决方案 | 具体机制 |
  |---|---|---|
  | MoE 数据量增大导致计算阻塞（data-to-computation ratio 高） | 分层加载策略 | 三层加载（inter-stage/inter-layer/inter-expert）按 expert 热度选择性加载，非全量加载 |
  | Mobius 未利用 expert 热度偏斜 | 亲和感知分配 | Expert 热度预测器 + Equation 1 阈值决策，高热度→GPU，低热度→CPU 就地计算 |
  | CPU 仅作存储，不参与计算 | CPU 参与低热度 expert 计算 | 低 token 数量时 CPU 计算时间与 GPU 可比（受限于 compute-bound→memory-bound 转换），减少数据搬移 |
  | 多加载阶段竞争同一 PCIe 带宽 | 需求优先级调度 | PriorityQueue + CUDA Event 前探，inter-expert > inter-layer > inter-stage 优先级动态协调 |
