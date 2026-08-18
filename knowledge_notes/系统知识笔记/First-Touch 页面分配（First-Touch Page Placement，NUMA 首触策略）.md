## First-Touch 页面分配（First-Touch Page Placement，NUMA 首触策略）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
First-Touch 是 Linux 在 NUMA 系统上的默认内存分配策略：页缺页异常时，物理页分配在触发缺页的 CPU 所在 NUMA 节点（页粒度 4KB）；一次分配后不迁移——后续其他节点的线程访问即为远端访问。其目标是让"最先使用该页"的线程享受本地访问；缺点是多节点共享数据时无法同时保证本地（对比 interleave 策略：轮转分布保证均衡不保证局部性，性能取决于访问模式）。管理工具：numactl / libnuma、migrate_pages/move_pages；"next-touch"（下次访问时迁移，Solaris MADV_ACCESS_LWP）未合入 Linux 主线（https://stackoverflow.com/questions/12196553）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Dorado 把它用作"虚拟页→物理簇内存"的映射策略：论文设想一个系统调用标记初始化结束、并行区开始，此后第一个访问某页的核把该页分配（或重分配）在其所在簇的内存，页内各行再经 Intel 式哈希函数落到该簇的本地 LLC 分片——从而"行的 Global home = 首触核所在簇"，把共享数据钉在生产者/初始化者附近。论文指出这同时模拟了 NUMA 优化的应用/OS 行为（数据初始化与并行计算同亲和、或 OS 按最强亲和自动迁移页）。关键结论：first-touch 是偏向 baseline 的保守选择——它人为提高了 local-homed 访问比例、压低 baseline 的远端事务，使 Dir2B 表现比随机分配时更好；换用其他分配策略，Dorado 相对 baseline 的增益会更高（更低的 baseline）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Linux 内核缺页路径按 faulting node 本地分配（默认 mempolicy）；用户空间用 mbind/set_mempolicy/numa_alloc_onnode 控制。使用要点：大规模多簇一致性模拟中，页放置决定 home 分布，直接影响目录协议的本地/远端事务比；测量协议增益时应说明页分配策略（first-touch 会低估本地化收益），并可把"初始化与计算同亲和"作为最坏情形处理。论文未明确说明重分配（re-allocation）的触发条件细节。

涉及论文标题：
- Dorado: Clustered Hardware Cache Coherence for 1,000+ Cores
