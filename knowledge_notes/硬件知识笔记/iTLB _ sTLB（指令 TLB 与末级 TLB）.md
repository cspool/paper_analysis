## iTLB / sTLB（指令 TLB 与末级 TLB）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TLB（Translation Lookaside Buffer）是缓存虚拟地址→物理地址翻译的硬件结构，按层级分：iTLB（L1 指令 TLB，一级）与 sTLB（secondary TLB，二级/末级 TLB，通常同时缓存数据与指令 PTE）。论文 baseline：iTLB 64 项 4-way 1cc、sTLB 1536 项 12-way 8cc。逻辑链：server 负载指令足迹大 → 指令 PTE 数量多 → 压榨 iTLB 与 sTLB → iTLB MPKI 上升 → 更多指令翻译请求到 sTLB → 数据/指令条目争用加剧 → sTLB miss 率上升 → 更多长延迟页表走查 → 前端停顿（sTLB 指令 miss + L1I miss 占 server 负载执行周期 10%+）。IP-CaT 的 tPB 位于 sTLB 旁，把 L1I 跨页预取翻译从 sTLB 分流出去（避免污染）并复用其服务 demand 访问。相关对比策略：CHiRP（Control-flow History Reuse Prediction，TLB 替换策略）与 Morrigan（指令 TLB 预取器）——IP-CaT 均优于二者。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
TLB 层次在指令 fetch 路径中的运转：L1I 预取/demand 取指产生虚拟地址 → 查 iTLB（1cc）→ miss 查 sTLB（8cc）→ miss 查 tPB（IP-CaT）→ miss 触发页表走查（4/5 级 radix tree + MMU PSC，~100+ cycle）→ 翻译写回相应结构（cb=1 预取翻译→iTLB+tPB；其他→iTLB+sTLB）。sTLB MSHR（miss status holding register）跟踪未决翻译请求并新增 cb bit。sTLB MPKI 是论文关键指标：定义为 sTLB 与 tPB 都 miss 的访问数，IP-CaT 使 EPI/FNL+MMA/Barça 的 sTLB MPKI 降 31.6%/18.2%/32.3%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：iTLB/sTLB 为 SRAM 数组，通常 set-associative（论文 4-way/12-way）+ MSHR；sTLB 可集成 tPB 的额外 set。使用方式：在 ChampSim 中配置 TLB 层次与替换策略（LRU/CHiRP）评估；对指令足迹大的 server 负载，TLB 管理（替换、预取、IP-CaT 式专用缓冲）是前端优化关键。相关：iTP（Chasapis 等的 sTLB 替换策略，最大化指令命中）、Victima（L2C 部分作 L3 TLB）、POM-TLB。

涉及论文标题：
- Enhancing Instruction Prefetching via Cache and TLB Management
