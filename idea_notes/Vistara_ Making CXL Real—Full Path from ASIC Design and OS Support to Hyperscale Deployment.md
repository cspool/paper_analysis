## Vistara: Making CXL Real—Full Path from ASIC Design and OS Support to Hyperscale Deployment

- baseline方法是什么？
  - baseline 有两类：(a) 无 CXL 的纯本地 DRAM 服务器——约 40%（43.7%）服务器 memory-capacity bound，工作集放不进 DRAM → 资源 stranded（25–40% CPU 闲置）、ML 推理 fan-out 升高、服务器数量膨胀、OOM 频发；(b) 市售 CXL 方案——捆绑新 DRAM 的 CXL 内存模块（不能复用退役 DDR4、多数不支持 DDR4、功耗与成本高）、FPGA 类 CXL 设备（[19] 报告并发下尾时延高度不稳定，源于 FPGA SRAM/credit buffer 不足）、以及 TPP 等内存分层软件（[44] 担忧运行时开销高、需复杂 OS 优化）。
  - baseline 全栈执行例子（一次容量受限的推荐系统 embedding 查找，baseline = 纯本地 DRAM 服务器）：
    ```
    算法pipeline层：论文未明确说明（本文非推理算法论文；承载负载为 embedding/参数服务器查表）
    系统框架层：论文未明确说明（无开源 serving 框架；baseline 是标准 Linux + 单层本地 DRAM）
    编译框架层：论文未明确说明（无编译框架参与）
    kernel调度层：单 tier 内存管理——内核页面全部落在本地 DRAM，无分层无迁移；内存耗尽触发
               OOM 或把页换到 flash/远端存储（写放大、时延暴涨）
    硬件架构层：仅本地 DDR5-6400（768GB/614GBps/≈130ns），embedding 表超过 DRAM 容量 → shard
               到更多服务器（fan-out 增大、p50/p99 时延上升、服务器数与 CPU 被 memory 拖累）
    ```
  - baseline（市售 CXL）全栈补充：CXL 模块捆绑新 DRAM（不可复用退役 DDR4、功耗/成本高），FPGA CXL 设备并发下尾时延漂移（[19]），软件分层用复杂热检测（DAMON/多级分类等）产生高 CPU 开销（[44]）。
- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法 = 硬件-软件协同设计的端到端 CXL 内存扩展（首次 hyperscale 全路径部署报告）：硬件上自研 Vistara CXL Type-3 扩展器 ASIC——复用退役 DDR4 RDIMM（2×72-bit 通道、2400MT/s）、低功耗（≈9W/ASIC、CXL 子系统总 ≈50W）、空闲时延 ≈50ns、RS(36,32) ECC + x4 chip-kill、3× RISC-V 管理核（secure/control/boot）、先进工艺 + clock/power gating；软件上用生产化 TPP/TMO 内核分层（简单 LRU 热检测即可、CPU 开销 <0.5%）、cgroup cpuset.mems 级 opt-out 弹性、ZONE_MOVABLE 隔离非可迁移内核数据、per-container Fair Share 公平性、weighted interleave 自动调优。对应解决 baseline 缺陷：① 内存容量瓶颈/服务器膨胀 → 1TB MemServer（768GB DDR5 + 256GB CXL DDR4），MRS 服务器数 -25% 且吞吐 +12%、CacheA QPS +33%、Spark executor/服务器 +33%、CI 作业/服务器 +33%、devmachine VM/服务器 +33%；② 市售 CXL 不能复用旧内存/功耗高 → Vistara 专为 DDR4 RDIMM 复用设计（Power/GB 0.7×、Cost/GB 0.13× 相对本地），并支持混合 vintage 兼容；③ FPGA 尾时延不稳 → ASIC 工程化（低时延 controller 配置、completion buffers、bank 级并行、紧耦合 PCIe Gen5 PHY）使尾分布贴近本地 DRAM；④ 复杂分层软件开销 → 简单 LRU 热检测 + 调优内核参数（numa_demotion_enabled=1、numa_balancing=2、zone_reclaim_mode=7）即可，开销 <0.5%，反驳复杂 OS 方案必要性；⑤ 迁移流量极小（Table VIII promotion 仅 2–7K/min）→ DMA 迁移（DSA-2LM [21]）无必要。
  - 论文方法全栈执行例子（一次 CacheA 缓存请求 / MRS embedding 查找，Vistara MemServer）：
    ```
    算法pipeline层：论文未明确说明（非推理算法论文；负载为缓存/embedding 查表，命中率与容量正相关）
    系统框架层：论文未明确说明（无 serving 框架；资源编排器 Twine [32] 按服务 profile 管理
               cpuset.mems，声明式下发 opt-in/opt-out 内存策略，动态切换）
    编译框架层：论文未明确说明（无编译框架）
    kernel调度层：CXL 暴露为 CPU-less NUMA 节点（ZONE_MOVABLE，隔离内核非可迁移分配）→ TPP 用
               minor page fault 检测热页、NUMA demotion + TMO 主动 demote 冷页到 CXL、hot 页 promote
               回本地；cache-pages-prefer-remote-node 把 warmup 期 file cache 直接放 CXL；weighted
               interleave 自动调优服务带宽敏感负载；Fair Share 按容器 local_high 上限 demote，
               多租户 local fraction 90%/70% 失衡 → 74%/71% 平衡（P99 提升 1.6×/1.8×、QPS 跌幅
               65%→12%、消除 OOM）；overhead <0.5%
    硬件架构层：请求命中 hot 对象 → 本地 DDR5-6400（≈130ns）；冷对象在 CXL → CPU host-bridge 256B
               交错 → 2× Vistara（PCIe Gen5 x8）→ DDR4-2400（≈250ns、聚合 ≈76GB/s）；ASIC memory
               controller 流水线精简 + CXL 栈紧耦合 PCIe Gen5 PHY（空闲时延 ≈50ns）；缓存
               680→890GB（CacheA）/590→820GB（CacheB）→ 命中率↑、QPS +33%、查询时延 -29%；
               knee-of-the-curve 应力测试证明 hot footprint <75% 时延 <264ns（+1%），生产负载
               远低于该阈值（Table I：≥75% 页面 idle>4s）
    ```
