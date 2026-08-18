## 内存分层与 NUMA demotion（Memory Tiering / Tiered Memory，冷热页跨层迁移）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
内存分层（memory tiering / tiered memory）是 OS 内核内存管理机制：把物理内存按性能划分为多个 tier（本地 DRAM 快层 + CXL/PMem 慢层），基于访问模式识别热/冷页，热页放快层、冷页放慢层，并透明地跨层迁移页面。Vistara 部署的 Linux 分层栈 = TPP（Transparent Page Placement）+ TMO（Transparent Memory Offloading）+ NUMA demotion/balancing：CXL 内存 online 为独立 CPU-less NUMA 节点（ZONE_MOVABLE）；DRAM 压力升高时内核 reclaim 机制把合格冷页 demote（降级）到 CXL，TMO 主动 demote，TMO/TPP 也做反向 promote（把变热的页升回 DRAM）。内核参数调优：numa_demotion_enabled=1（允许 demotion）、numa_balancing=2（NUMA balancing 迁移页）、zone_reclaim_mode=7（开启全部 zone reclaim）。论文测得的 tiering CPU/系统开销 <0.5%，反驳"TPP 引入显著开销"的观点。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程（Vistara MemServer 上的一次分层决策）：应用分配页面 → 默认策略优先落在本地 NUMA（DDR5）；本地 DRAM 水位逼近高水位 → reclaim 扫描 + NUMA hint fault 收集访问信息 → 判定冷页（LRU 尾部、长时间未访问）→ demote 到 CXL NUMA 节点（DDR4-2400）释放 DRAM；应用后续访问该页 → NUMA hint fault 触发 → promote 回本地 DRAM。论文数据（Table I）：大部分负载 ≥75% 页面 idle 超过 4 秒（P25 冷度），因此冷页占比大、CXL 带宽利用率低（Table VIII：CXL 0.5–13 GB/s vs 本地 2–300 GB/s），分层几乎不损害端到端性能。knee-of-the-curve 应力测试：hot footprint <75% 时平均时延 <264ns（相对 10% 基线 +1%），>75% 后时延 +4%~+22%——生产负载远在拐点之下。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Linux 内核 mm 子系统（memory-tiers、NUMA demotion 在 mm/vmscan.c，NUMA balancing、hint faults），配合 ACPI HMAT 表提供 tier 相对时延/带宽；控制面经 sysfs（memory tier、demotion 开关）。使用方式：CXL 内存分层在 Meta 直接内建进内核默认行为，应用无感（transparent）；带宽敏感负载可用 weighted interleave 显式调本地:CXL 比例；容器级用 cpuset.mems 限制或 Fair Share 调节；对时延敏感未验证负载可 opt-out CXL。生产收益：缓存/数据仓库/ML 参数服务器/DevInfra 各类负载吞吐提升 4–33%、服务器数减少 15–25%。与更复杂分层方案（DAMON 驱动、多级分类如 Memtis/HybridTier）对比，论文论证简单 LRU 热检测 + 内核 demotion 已足够，DMA 加速迁移（DSA-2LM）因迁移流量极小而无必要。

涉及论文标题：
- Vistara: Making CXL Real—Full Path from ASIC Design and OS Support to Hyperscale Deployment
