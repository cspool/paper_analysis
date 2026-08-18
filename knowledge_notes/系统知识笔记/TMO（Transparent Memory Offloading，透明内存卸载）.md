## TMO（Transparent Memory Offloading，透明内存卸载）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TMO（ASPLOS'22，Meta 的 Transparent Memory Offloading）是数据中心 userland 驱动的内存回收策略：把冷匿名页主动"卸载"到远端内存/内存介质（在 Vistara 场景即 CXL 内存），从而在 DRAM 压力出现前就释放本地 DRAM。与内核 reclaim demote（被动、DRAM 已紧张时才动作）互补，TMO 是主动/前瞻式 demote：基于访问模式预测把冷页提前搬到慢 tier，为热页与新分配保留 DRAM 空间。Vistara 生产栈同时部署两种 demotion：内核 reclaim demotion + TMO 主动 demotion。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
TMO 运转流程（Vistara 部署中）：用户态监控/策略组件跟踪各进程内存访问热度 → 识别冷页（长时间未访问）→ 主动把这些页迁移（offload）到 CXL 内存 → 本地 DRAM 始终保持充足 headroom → 新分配与热页 promote 无需等待 reclaim 的激烈操作。与内核 demotion 协同：DRAM 水位未到压力阈值时主要由 TMO 预先腾挪；压力到达时内核 reclaim demotion 兜底。生产效果：CacheA/CacheB 在把缓存对象 demote 到 CXL 的同时保证 hot heap 留本地，无感知时延劣化（CacheB 平均查询时延反而 -29%）；大模型训练/服务场景 OOM 减少 50%（论文 Motivation）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Linux 内核 mm 的 proactive reclaim/offload 机制（TMO 代码已在上游内核，源于 Meta 的 mm 补丁集），配合用户态策略组件判定冷页；Vistara 部署中与 TPP/NUMA demotion 共用 hint fault 与迁移路径。使用方式：运维设定 demotion 阈值/水位与策略，应用无感；监控通过 TPP 遥测（demotion/promotion 速率、hint faults）与 CXL 带宽。适用场景：内存容量紧张、冷页占比高的数据中心负载（缓存、批处理、ML）；论文数据显示简单 LRU 检测即可驱动 TMO/TPP 达到 <0.5% 开销，无需复杂 OS 优化。

涉及论文标题：
- Vistara: Making CXL Real—Full Path from ASIC Design and OS Support to Hyperscale Deployment
