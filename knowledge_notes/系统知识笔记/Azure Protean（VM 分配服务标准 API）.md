## Azure Protean（VM 分配服务标准 API）

术语解释
微软 Azure 生产级 VM 分配服务（OSDI'20）：管理大规模 VM 放置请求；R2D2 以其规则/流量 trace 作为输入，并让系统控制器暴露兼容其语义的标准 API，使上层 orchestrator 无需改动即可接入。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Protean 解决"VM 分配服务 at scale"：在混合云规模（数十万台机器）把 VM 需求（CPU/内存/存储/时长）高效映射到物理机，权衡利用率、碎片化与分配延迟。R2D2 引用其最新集群 VM trace 数据集（2064 机器、48 核/384GB 每机；含 1000s 请求/秒 burst；每条目含 VM CPU/内存/存储需求与时长）。
- 论文使用：(1) 流量分析输入——用 Protean trace 评估三种放置策略下的空间稀疏/时间稳定；(2) 任务级微基准——burst 峰值下测 R2D2 分配延迟（§VI-D）；(3) API 兼容——R2D2 系统控制器暴露"标准 API（Azure Protean rules）"，hypervisor/调度器可无改动地请求资源分配与网络配置。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 R2D2 栈中：hypervisor 按 Protean 语义提交任务（compute 需求 C + memory 需求 M）→ R2D2 系统控制器调用联合分配与重构算法（见上）→ 返回 compute/memory 节点分配 → hypervisor 按返回值调度 VM 运行。分配延迟含机器人重构时间（无重构路径为 0；有重构路径为机器人运动+插拔，异步并行化）。
- 故障/超时处理：重构失败标记机器人 down、备选机器人/备选放置兜底、最坏 QUEUE FOR RETRY；4%/天机械故障注入下分配延迟仅 +2.7%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：生产环境为 Azure 内部服务（不公开代码）；trace 与规则对外用于研究复现。R2D2 用其 trace 驱动离散事件模拟，评估分配延迟与利用率（99% CPU/69% memory），并作为标准 API 契约保证与现有编排栈兼容。

涉及论文标题：
- R2D2 Robotized Reconfigurable Network for Disaggregated Datacenters
