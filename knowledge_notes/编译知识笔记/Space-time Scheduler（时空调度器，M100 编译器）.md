## Space-time Scheduler（时空调度器，M100 编译器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- M100 AI 编译器工具链的第一阶段：把神经网络子图映射到 M100 NPU 硬件，同时决定空间分配（算子/子图放到哪些 TPB/cluster）与时间调度（tensor 沿流水线的阶段顺序）。必要时把大 tensor 沿多轴维度分解成 mini-tensor，按时间调度的阶段流经空间分配的 TPB 流水线（Fig.14 例子：含 OP1~OP4 四个算子的子图被空间分配到 4 个 TPB，输入 tensor 经多维分解后按时相阶段流经各 TPB）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 流程：输入 NN 计算图（如 UniAD 某子图或 LLaMA 的 layer 组）→ 空间映射（subgraph/算子 → TPB/cluster，考虑 cluster 内低延迟 vs 跨 cluster Mesh 通信效率）→ 大 tensor 多维分解为 mini-tensor（匹配 HBSM 容量与带宽）→ 时间阶段编排（每阶段在哪批 TPB 执行、tensor 何时产出/消费、同步点设在哪）→ 输出带空间-时间信息的中间表示给 graph compiler 与 backend compiler。作用：决定数据流拓扑与并行度，是"orchestrated dataflow"在软件端的核心——编译器静态规划数据移动以换取硬件简单。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：自研调度器（论文未明确说明具体算法，如贪心/ILP/启发式）；与硬件 co-design（已知 TPB 数、cluster 层级、Mesh/DRB/ICB 带宽与延迟）。使用：对每个新模型自动生成映射方案；与 graph compiler/backend compiler/固件 JIT 组成完整工具链。未开源（论文未提供实现细节或链接）。

涉及论文标题：
- M100: An Orchestrated Dataflow Architecture Powering General AI Computing
