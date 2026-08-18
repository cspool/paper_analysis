## R2D2 联合分配与网络重构算法（Joint Allocation and Reconfiguration）

术语解释
R2D2 软件 runtime 的核心调度算法（Alg.1）：在 datacenter 系统控制器中同时优化 VM/任务放置与机器人网络拓扑重构，以最小化重构次数、降低分配延迟并保持高资源利用率。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 与现有 topology-aware 解耦资源调度器（Clio、Hermit、LegoOS——在静态供给拓扑上优化放置）不同，R2D2 的物理拓扑本身可按需变化，因此算法必须联合决策"放哪里"与"要不要重构网络、由哪个机器人执行"。
- 目标：通过分配决策主动"工程化"流量——让新任务优先落在已连接节点，增强空间稀疏与时间稳定，最小化机器人移动（重构成本计入 fitness），同时保证低分配延迟与高利用率。
- 两级分层保证可扩展性：先选 datacenter row（聚合资源足够、best-fit fitness 最高），再选 row 内 compute-memory 节点。上层 hypervisor 经标准 API（兼容 Azure Protean rules）提交任务，系统控制器返回节点分配。
- 异步分发模型：控制器向目标机器人下发重构命令后立即继续处理后续 VM 分配（不阻塞）；系统控制器与机器人控制器解耦，多个机器人独立并行执行 → 系统级重构延迟不随规模串行化。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- Alg.1 流程：①ALLOCATETASK(t, C, M)：FEASIBLEROWS(C,M) 按 fitness 选最优行；②优先排序并尝试 row 内无需重构的 compute 节点（FEASIBLECNODESNORECONF，COMMIT 成功即返回 SUCCESSFUL）——0 重构路径；③若无，枚举 FEASIBLECMPAIRS，对每对查 AVAILABLEROBOTS 实时空闲列表，RECONFIGURENETWORKBYROBOT（成功则 COMMIT，失败 MARKROBOTASDOWN 换下一机器人/下一对）；④全失败 QUEUE FOR RETRY。重构偏好选择空间分散到不同 pod 的不同机器人 → 最大化并行执行。
- 评估（自定义 discrete-event simulator）：联合算法 vs best-fit（同跑 R2D2 硬件）平均与 p99 allocation latency 低 10-20×（best-fit 机器人无关、触发过量重构级联延迟）；37-45% 重构重叠 2+ 机器人（2/4 robot 配置）；vs fat-tree/OCS 分配延迟高 41-51%(avg)/27-30%(p99)（512 节点）但仅占 VM 总运行时间 0.49%；利用率 99% CPU/69% memory。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：系统控制器（datacenter orchestration 层）执行算法，机器人控制器（嵌入式板）把高层命令（断开端口 A、连接端口 B）翻译为 stepper 轨迹/G-code 并闭环控制（编码器、插入力反馈）。fitness 函数综合考虑资源匹配与链路利用率，促进平衡流量与可扩展 fanout。
- 使用场景：VM 分配（Protean trace，含 1000s 请求/秒 burst）、微服务/serverless 负载（额外分配延迟 <2s/4%）、VM 迁移（CPU-only/memory-only/联合迁移，视为带放置约束的分配事件）。论文未提供代码仓库，联网搜索无法确认公开实现。

涉及论文标题：
- R2D2 Robotized Reconfigurable Network for Disaggregated Datacenters
