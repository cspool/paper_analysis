## Intel RDT (Resource Director Technology) / CAT / MBA

术语是什么？

Intel RDT是Intel Xeon平台提供的硬件资源分区和控制框架，允许软件动态划分和隔离最后一级缓存(LLC)、内存带宽等共享资源。核心组件：(1) **CAT (Cache Allocation Technology)**：按cache way粒度将LLC划分为不同Class of Service (CLOS)，绑定到特定CPU core集合；(2) **MBA (Memory Bandwidth Allocation)**：按百分比限制per-core或per-CLOS的最大内存带宽；(3) **pqos** (Platform QoS) 用户态工具和库。

从硬件架构角度拆解术语：

RDT在AUM AU共享场景下的资源分区流程：
1. **Profile resource affinity**：用CAT将LLC ways从1增到16→观测AU perf变化（prefill TTFT对LLC sensitive ~80%变化；decode TPOT insensitive <5%变化）→得到per-AU-usage的minimal resource demand R_AU = {R_L2C, R_LLC, R_BW}
2. **Runtime partition**：Runtime Controller通过pqos设置CAT CLOS——High-AU cores分配P^t所需min LLC、Low-AU cores分配更少LLC（decode insensitive）、None-AU shared apps用剩余LLC
3. **BW throttling**：通过MBA限制AU core memory BW→释放BW给memory-intensive shared apps (SPECjbb, OLAP)

术语一般如何实现？如何使用？

```bash
# pqos使用示例
pqos -s                                          # 查看当前CAT配置
pqos -e "llc:1=0x000f" -a "core:1=0-11"          # core 0-11 LLC限制4 ways
pqos -e "mba:1=50" -a "core:1=0-11"              # core 0-11 memory BW限制50%
```
AUM Runtime Controller通过Python pqos library实现运行时资源调优，每次调整<1ms。

涉及论文标题：
- AUM: Unleashing the Efficiency Potential of Shared Processors with Accelerator Units for LLM Serving

