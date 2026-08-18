## 页表走查（Page Table Walk）与多级 radix-tree 页表、MMU Cache（PSC）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
页表走查（page table walk）是 TLB miss 时由硬件页表走查器（hardware page table walker）逐级遍历页表（radix tree）获取虚拟页→物理页映射（PTE）的过程；多级 radix-tree 页表是 x86 的 4 级（或 5 级 LA57）结构，每级索引一段虚拟地址位，末级条目含物理页号与权限位。MMU Cache（Page Structure Cache，PSC）缓存各级页表目录条目（split PSC：并行查各级），减少走查的访存次数。论文 baseline：5 级 radix tree 页表、x86 硬件页表走查器、4 级 split PSC（L5 1 项/L4 2 项/L3 8 项/L2 32 项，并行搜索 1cc）。逻辑链：sTLB miss → 走查最多 5 级内存 → 长延迟（数百 cycle）→ 预取翻译的走查延迟破坏 L1I 预取及时性；IP-CaT 用 tPB 缓存 L1I 跨页预取走查结果避免重复走查。页表走查本身也访问 L2C（tPB 减少走查→L2C 争用下降→TIPRP 更有效，这是 IP-CaT 组件协同的来源）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
走查流程（4KB 页，x86）：虚拟地址分为 PML4/PDPT/PD/PT 四级索引 → 硬件走查器用 PML4 索引查 PSC L4（命中则省一级访存）→ 逐级：查 CR3 指向的 PML4 表→PDPT→PD→PT→得 PTE（含 4KB 页 PPN）→ 返回 TLB。每级未命中 PSC 则访存（L2C/LLC/DRAM）。IP-CaT 中：L1I 跨页预取的走查结果（cb=1）写入 iTLB+tPB；demand 走查结果写入 iTLB+sTLB；tPB 命中即免走查。论文用 2MB 大页场景验证：2MB 页减少 sTLB miss → tPB 收益随 2MB 占比上升而下降（0%→100% 2MB 时 IP-CaT speedup 7.5%→1.8%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：现代 x86/ARM 处理器内置硬件页表走查器 + 多级 PSC/MMU cache（如 Intel 的 PSC、ARM 的 MMU caches），软件侧由 OS 维护页表。使用方式：模拟器中配置走查器与 PSC 以评估翻译开销（ChampSim 支持）；大页（2MB/1GB）是软件侧减少走查的标准手段，但 server 长 uptime 下 4KB 页仍普遍（内存连续性与碎片问题）。相关：DVMT（应用自定义页表格式）、Elnawawy 等（sTLB 钉住高频数据 PTE）。

涉及论文标题：
- Enhancing Instruction Prefetching via Cache and TLB Management

Revelator 补充视角（ISCA'26）：Revelator 用"哈希分配末级页表帧"让 PTW 的最后一级也可被硬件投机预取——OS 分配末级 PT frame 时用单次哈希尝试 H_1(VPN>>9)（右移 9 位因每个 PT frame 覆盖 512 个连续 VPN），硬件在 L2 TLB miss 时按同一公式重算该帧 PPN、拼上 PTE offset 即可在 PTW 开始时并行投机取末级 PTE，把"末级 PTE 取数与 PTW 前几级走查"重叠（Revelator 的两级并发之一，另一级是数据取数与整个 PTW 重叠）。好处：即使前几级走查已加速（对比 ASAP/DMT/ECH 等加速 PTW 的方案），处理器仍需等至少一次内存访问才能取数据，Revelator 对残留的最后一次访存延迟也能隐藏（low/medium 碎片下较 ASAP/DMT/ECH 高 20%/13%）。配合其数据页哈希投机，Revelator 在 high 碎片下较 THP 提速 25%。
涉及论文标题：
- Revelator: Rapid Data Fetching via OS-Guided Hash-based Speculative Address Translation
