## Big Switch Network Model (大交换机网络模型)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Big Switch Network Model（大交换机网络模型）是一种将 GPU 集群内部互联网络抽象为单个非阻塞（non-blocking）交换机的简化模型。在该模型中，所有 GPU 通过一个逻辑上的大交换机互联，任意 GPU 可以与任意其他 GPU 以全带宽通信，不会因中间交换路径而产生额外的阻塞或带宽衰减。该模型适用于机架级（single rack）GPU 集群——通常包含数个到数十个 GPU，通过高性能网络（如 NVLink、InfiniBand）连接。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 Aurora 的 MoE 推理场景中，Big Switch 模型的关键含义：

1. **GPU 无需指定到具体物理 ID**：在同构集群中，所有 GPU 通过 big switch 互联且交换能力相同，因此 GPU 分配只需考虑 GPU 类型（计算能力、带宽），无需关心具体物理位置（如哪个 PCIe 插槽）。Aurora 的 GPU assignment（Theorem 5.1）正是基于此：按 expert token 负载降序分配给性能降序 GPU 即可达到最优。
2. **带宽建模**：同构场景中所有 GPU 带宽相同（如 100 Gbps），异构场景中不同 GPU 类型有不同带宽（如 100/80/50/40 Gbps），但同一类型的 GPU 对称互换。
3. **通信时间仅取决于端点带宽**：在 big switch 模型下，从 GPU i 到 GPU j 的通信时间由 min(B_i, B_j) 决定。Aurora 的通信调度（Theorem 4.2, 5.2）直接利用此特性：将 traffic matrix 的元素 d_ij/B_i 归一化后，通信时间由 bottleneck GPU（最大列/行和的 GPU）决定。
4. **限制**：该模型忽略交换机内部阻塞、多跳路由延迟、NIC 缓冲竞争等因素，适用于机架内小规模 GPU 集群。论文在 §2.4 中特别说明 MoE 推理"typically requires several to dozens of GPUs, often housed within a single rack"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Big Switch 模型是理论分析和仿真中常用的网络抽象，实际的 non-blocking 网络通过 fat-tree、Clos 等拓扑结构近似实现。
- 在实际部署中，NCCL 等通信库会自动处理 GPU 间的路由和拓扑感知，上层调度器（如 Aurora）只需关注 GPU 类型和带宽分配。
- 当 GPU 规模超出单机架（如跨 pod/跨数据中心）时，big switch 假设不再成立，需考虑层次化带宽（intra-rack vs. inter-rack）、拥塞控制和路由策略。
- Aurora 的未来工作方向之一即为"extending to handle more complex environments, including those with varying network topologies and communication protocols"。

涉及论文标题：
- Optimizing Mixture-of-Experts Inference Time Combining Model Deployment and Communication Scheduling
