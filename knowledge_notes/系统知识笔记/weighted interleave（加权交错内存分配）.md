## weighted interleave（加权交错内存分配）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
weighted interleave 是内核 NUMA 内存分配策略：按 N:M 权重把分配交错（interleave）到多个内存节点，把 DRAM 与 CXL 的比例调到与负载带宽/时延需求匹配。普通 interleave 各节点等权，weighted interleave 允许不等权（如本地:CXL=3:1），适合带宽受限工作负载：让流量按权重分布在快/慢两 tier，以可接受的慢层访问换取更大可用容量，或反之把更多流量留在快层。Vistara/Meta 实现了 Linux 支持并上游化"自动调优 weighted interleave"：比例经 sysfs 动态配置以匹配负载与硬件特性。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程：带宽敏感负载（如数据仓库扫描）启动 → 运维/自动调优按负载带宽-容量需求设 sysfs 权重（如 3:1 DRAM:CXL）→ 内核按权重轮转分配物理页到本地 DDR5 与 CXL DDR4 → 流量按 3:1 分布在两 tier；负载特性变化（如内存占用上升）→ 在线调权重提高 CXL 份额。对比 TPP 的按热度迁移：weighted interleave 是分配期静态策略（页面落在哪由分配权重决定），TPP 是运行期动态迁移（页面随热度跨层移动）；论文把 weighted interleave 作为 TPP 的补充，专供带宽受限且可容忍固定比例慢层访问的负载。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Linux 内核 NUMA mempolicy 的 weighted interleave 扩展（Meta 已上游化自动调优部分），经 sysfs 暴露权重配置；带宽敏感应用可从用户态声明 mempolicy 使用。使用方式：直接用于需要平衡 DRAM 带宽与 CXL 容量的负载；与 TPP/cache-pages-prefer-remote-node 策略组合构成完整内存策略工具箱。论文给出同类策略验证：cache-pages-prefer-remote-node 在 warmup 期把 file cache/tmpfs 冷页直接放 CXL，20GB 本地 + 76GB CXL 达到 96GB 全本地同吞吐。

涉及论文标题：
- Vistara: Making CXL Real—Full Path from ASIC Design and OS Support to Hyperscale Deployment
