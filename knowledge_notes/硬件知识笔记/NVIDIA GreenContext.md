## NVIDIA GreenContext

术语是什么？

NVIDIA GreenContext是CUDA 12.4引入的轻量级GPU上下文机制，允许在单个CUDA进程内创建多个独立的执行上下文，每个上下文绑定GPU SM集合的一个子集。与传统CUDA Context相比，GreenContext的创建和切换开销极低，支持在同一GPU上实现空间级别的SM分区和并行执行。

从硬件架构角度拆解术语：

GreenContext在GPU硬件层面的工作原理：

1. **SM空间分区**：通过`cuDevSmResourceSplit`/`cuDevSmResourceSplitByCount` API将GPU的物理SM划分为多个不重叠的组。例如，在H200（Arch 9.0）上最小SM分区粒度为8个SM。
2. **资源描述符生成**：`cuDevResourceGenerateDesc`根据SM分区生成资源描述符，描述该分区内可用的计算资源。
3. **GreenContext创建**：`cuGreenCtxCreate`基于资源描述符创建GreenContext，该Context内的所有CUDA操作只能在分配的SM上执行。
4. **Stream绑定**：`cuGreenCtxStreamCreate`创建绑定到特定GreenContext的CUDA Stream，该Stream上的kernel launch仅使用对应SM分区的计算资源。
5. **非完全隔离**：SM分区提供计算隔离，但**不隔离HBM带宽、L2 Cache、时钟频率和温度**。多个GreenContext同时执行时存在HBM带宽竞争。

架构约束：
| GPU架构 | 最小SM/分区 | SM步长 |
|---------|------------|--------|
| 6.X     | 2          | 2      |
| 7.X     | 2          | 2      |
| 8.X     | 4          | 2      |
| 9.0+    | 8          | 8      |

术语一般如何实现？如何使用？

在LLM推理服务中，GreenContext用于实现PD Multiplexing：
- 创建两个GreenContext：一个绑定少量SM用于Decode（延迟敏感），一个绑定剩余SM用于Prefill（吞吐敏感）。
- 两个Context通过各自独立的CUDA Stream异步并行执行。
- MPS (Multi-Process Service) 也可配合使用，但GreenContext的进程内隔离开销更低。

Bullet使用替代方案libsmctrl：通过`libsmctrl_set_stream_mask()`直接修改CUDA stream的GPC配置掩码实现SM分区，mask更新开销仅~4us（vs GreenContext需重新初始化CUDA Graph等资源）。Bullet使用MPS管理两个独立worker进程，libsmctrl在进程内对各自的CUDA stream进行SM mask控制。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing
- Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration

---

