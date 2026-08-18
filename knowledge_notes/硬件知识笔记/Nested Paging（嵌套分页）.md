## Nested Paging（嵌套分页）

术语解释
x86-64 虚拟化（AMD-V Nested Paging / Intel EPT）下的二维地址翻译机制：guest 虚拟地址（gVA）先经 guest 页表转 guest 物理地址（gPA），再经 guest 物理地址索引的嵌套页表（nested page table）转 host 物理地址（hPA）；TLB miss 时需二维页表走查（2D PTW），比原生一维走查多一倍内存访问，翻译开销显著更高。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Nested Paging 由硬件（AMD-V NPT、Intel EPT）为虚拟化提供直接地址翻译：hypervisor 维护 guest 到 host 的第二套页表（nested/EPT page table），guest 的每次翻译都变成 gVA→gPA→hPA 两段，TLB miss 时 MMU 要同时走 guest 页表与嵌套页表（每段 4 级，最多 8 次内存访问），翻译开销约是原生的两倍。这也让投机地址翻译的收益在虚拟化下更大。Revelator 把其分层哈希分配与投机扩展进虚拟化：①Diagonal speculation——hypervisor 用 gVA 做哈希输入分配 hPA（Hash(gVA)=hPA），硬件从 gVPN 直接预测最终 hPA，与整个 2D PTW 并行取数（覆盖整条二维走查）；②Horizontal speculation——hypervisor 用 gPA 做哈希输入分配 hPA，guest PTW 识别出某 guest 页表页的 gPA 后，硬件哈希该 gPA 预测其 hPA 所在帧，在嵌套翻译完成前投机取下一级 guest PTE（加速嵌套走查本身）。Full 配置（Diagonal+Horizontal 组合）在虚拟化环境下较 Nested Paging+THP 平均提速 13.6%（较 DMT 高 6%）；Horizontal 单独收益小（嵌套翻译常被嵌套 TLB 命中）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
硬件运转流程（Revelator Full，Fig.8）：guest 访存 gVA → L1/L2 TLB miss → 硬件投机引擎：①Diagonal 路径——用 CityHash(gVPN, host_seed) 直接算候选 hPA 并发投机取数（重叠整个 2D PTW）；②Horizontal 路径——guest PTW 逐级走查（gVA→gPA），拿到某个 guest 页表页的 gPA 后哈希该 gPA 预测该页表页的 hPA 帧，投机取下一级 guest PTE（使嵌套翻译与 guest 走查重叠）；③PTW 完成后正确性校验、错误投机失效。hypervisor 侧：分配 hPA 时用 gVA（Diagonal）或 gPA（Horizontal）作为哈希输入做分层哈希分配，无需 guest 合作（Diagonal 的 hypervisor 实现细节见 extended version）。2D PTW 的额外开销正是 Nested Paging 的固有痛点——投机地址翻译把取数/PTE 取回与整条二维走查重叠，比一维场景收益更大。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：硬件由 Intel EPT/AMD-V NPT 提供嵌套页表走查（MMU 硬件）；Revelator 的虚拟化扩展在 hypervisor 侧修改其物理页分配（分层哈希分配）、硬件侧扩展投机引擎支持 Diagonal/Horizontal 两路径。评估：Virtuoso 模拟虚拟化环境，对比 Nested Paging+THP（NP-THP）、DMT（Direct Memory Translation，ASPLOS'24 直映射末级 PT）、Revelator Horizontal/Diagonal/Full。使用场景：任何虚拟化数据中心/云环境（翻译开销占比更高的场景）；Revelator 论文显示 Full 组合 13.6% 平均加速。信息缺口：Diagonal 投机如何在不改 guest 的情况下由 hypervisor 实现的具体机制在 extended version，正文未展开。

涉及论文标题：
- Revelator: Rapid Data Fetching via OS-Guided Hash-based Speculative Address Translation
