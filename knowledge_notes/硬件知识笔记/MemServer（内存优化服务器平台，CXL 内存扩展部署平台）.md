## MemServer（内存优化服务器平台，CXL 内存扩展部署平台）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MemServer 是 Meta 为部署 Vistara CXL 内存扩展而设计的单路"内存优化"服务器平台：把 CPU、本地 DDR5 与 CXL 附加 DDR4 整合为一个 1TB 内存的节点（Table VI）。组成：① 计算——单路 AMD Turin 处理器（Zen5 微架构），158 核/316 线程（2-way SMT）、≈3GHz、TDP 300W；② 本地内存——768GB DDR5-6400，12 通道、614 GB/s 峰值、空闲时延 ≈130ns；③ CXL——2× Vistara ASIC（各 PCIe Gen5 x8）+ 8×32GB DDR4-2400 RDIMM = 256GB，峰值 ≈76 GB/s、空闲时延 ≈250ns、CXL+DIMM 功耗 ≈50W；④ 总内存 1024GB、总服务器功耗 450–560W；⑤ 平台特性——CXL 访问以 256B 粒度在 2 个 Vistara 设备间交错，设备内再跨 DIMM/rank/bank 交错；CXL 卡装在后部专用插槽，机箱用定向气流 + 高容量风扇冷却高密度内存。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
MemServer 解决"内存容量绑定"瓶颈：约 40%（43.7%）服务器因内存容量耗尽而 CPU/存储/网络闲置（stranding）。运转流程（一次冷页访问）：CPU load → 内存控制器命中本地 DDR5（130ns）或 miss 到 CXL → CPU host-bridge 按 256B 粒度把地址映射到 Vistara 设备 0/1（轮换）→ PCIe Gen5 x8 → Vistara CXL 控制器（空闲时延 ≈50ns）→ DDR4-2400（250ns 空闲）返回；OS 侧 CXL 以 CPU-less NUMA 节点暴露（ZONE_MOVABLE），TPP 用 minor page fault 检测热页、NUMA demotion + TMO 主动把冷页 demote 到 CXL、热页 promote 回本地。生产结果（A/B 测试 ≥1 周）：CacheA 缓存 680→890GB、QPS +33%；CacheB 缓存 590→820GB、查询时延 -29%；Spark executor/服务器 +33%；CI 作业/服务器 +33%；devmachine VM/服务器 +33%；ML 参数服务器（MRS）服务器数 -25% 且吞吐 +12%；OOM 相关任务失败 -33%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现与使用：MemServer 是 Meta 内部专有平台（未开源，联网搜索 2026-08 无法确认公开资料），AMD Turin 即 EPYC 9005 系列（Zen5，https://www.amd.com/en/products/processors/server/epyc/9005-series.html）。使用方式：内存容量受限工作负载直接 bin-pack 到 MemServer——数据仓库（Spark/Cosco/FtStoreX 等）、分布式缓存（CacheA/CacheB）、DevInfra（CI/构建/devmachine）、ML 参数服务器（推荐系统 embedding 表）。DDR4 DIMM 来自退役服务器（mixed-vintage，需筛选测试库存 + 备件管理），实现"零成本"扩容与碳足迹降低（DRAM 占数据中心 embodied CO2 69%）。评估方法：同配置 MemServer 分 test/control 组（cpuset.mems 是否含 CXL 节点）、生产负载均衡器均分流量、运行 ≥1 周，比较服务级（QPS、p50/p99 时延、命中率、OOM 率）与系统级（本地/CXL 带宽、CPU 利用率、TPP promotion/demotion、NUMA hint faults）指标。

涉及论文标题：
- Vistara: Making CXL Real—Full Path from ASIC Design and OS Support to Hyperscale Deployment
