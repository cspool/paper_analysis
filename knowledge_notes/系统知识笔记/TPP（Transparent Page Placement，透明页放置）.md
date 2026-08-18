## TPP（Transparent Page Placement，透明页放置）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TPP（ASPLOS'23，Meta）是 CXL 使能分层内存的应用透明页放置机制：OS 以低开销的 minor page fault（NUMA hint fault）检测页面访问热度，把热页留在/迁移到本地 DRAM、冷页放到 CXL 内存，全程无需应用感知或修改。Vistara 把 TPP 生产化：结合 NUMA demotion（DRAM 压力触发 reclaim demote）与 TMO（主动 demote），并调优内核参数（numa_demotion_enabled=1、numa_balancing=2、zone_reclaim_mode=7）。论文的关键生产发现：① 简单 LRU 式热检测对大多数应用已足够准确（反驳需复杂分类器的观点）；② TPP CPU/系统开销 <0.5%（反驳 TPP 引入显著开销的观点 [44]）；③ 迁移流量极小（Table VIII：promotions 2–7K/min、MRS 94 MB/s CXL 带宽），故 DMA 加速迁移无必要。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
TPP 决策流程（Vistara CacheA 示例）：① 应用分配页面 → 默认先放本地 DRAM（NUMA 0）；② 内核 reclaim 在 DRAM 压力时扫描 LRU，选出冷页 demote 到 CXL NUMA 节点；③ 页面访问时触发 minor NUMA hint fault，内核记录访问并可能把页 promote 回 DRAM；④ 页迁移由实时访问跟踪 + 解耦的分配/reclaim 水位引导，保证 DRAM 始终保留给新分配与热页 promote 的 headroom。效果：CacheA 把缓存 heap demote 到 CXL，hot heap（连接/hashtable/在途请求数据）留本地——缓存容量 680→890GB、QPS +33%、retention age 1min→5–10min。TPP 遥测（Table VIII）：每服务报告 LocalBW/CXLBW/NUMAhintfaults/Promotions 四类指标用于运维。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：在 upstream Linux 内核（TPP 代码已并入内核 memory tiering/demotion 框架：mm/memory-tiers、mm/vmscan NUMA demotion、NUMA balancing）；Vistara 使用的所有内核 CXL 驱动代码要么已在 upstream、要么在上游路上。使用方式：内建默认开启（numa_demotion_enabled=1 等 sysctl 调优），应用无改；运维通过 sysfs/遥测监控 demotion/promotion 速率与 hint faults。适用负载：容量受限、冷页占比大（缓存、数据仓库、ML 参数服务器、DevInfra）；对时延敏感负载可用 cpuset.mems opt-out 或 Fair Share 限制。

涉及论文标题：
- Vistara: Making CXL Real—Full Path from ASIC Design and OS Support to Hyperscale Deployment
