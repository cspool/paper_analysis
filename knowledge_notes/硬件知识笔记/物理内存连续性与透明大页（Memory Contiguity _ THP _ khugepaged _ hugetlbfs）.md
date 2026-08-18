## 物理内存连续性与透明大页（Memory Contiguity / THP / khugepaged / hugetlbfs）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 物理内存连续性（physical memory contiguity）= 应用虚拟地址空间映射到物理内存时，物理页连续（相邻虚拟页落在相邻物理页上）的程度。它是 TLB 覆盖、大页支持、TLB 压缩、页表格式（如 RISC-V Svnapot）等硬件/OS 设计的关键工作负载属性。BULLETTIME 的度量方法：对每段物理连续区域，计算覆盖它所需的最小 power-of-2 页数（页大小可为 4KB 起的任意 2 的幂），最大化大页覆盖的物理内存占比，再按页大小统计覆盖内存的 CDF——这正是"假设的 TLB 能从多大页受益"的近似。
- Transparent Huge Pages（THP）：Linux 对应用透明地用 2MB（PMD 级）大页备份匿名内存；khugepaged 是周期性扫描、把符合条件的一段普通 4KB 页"折叠"为 2MB 大页的内核守护线程（Web 证据：内核 transhuge.rst 与 LWN 报道，折叠只到 PMD 级；压缩/碎片化是主要制约，MAX_ORDER 限制使其无法到 1GB 级）。hugetlbfs 则是显式预留大页的伪文件系统（可 boot 预留或运行时分配，无内核内分配接口），BULLETTIME 用它备份内部 trace buffer。O_DIRECT 是绕过 page cache 直写磁盘的 I/O 标志——page cache 对 trace 落盘的 4KB 页反复分配/释放是 tracing 碎片化物理内存的主要来源（条件 C1 要消除的对象）。
- BULLETTIME 的关键发现：tracing 的 I/O 延迟使应用总运行时间变长，khugepaged 相对应用更早、更频繁地运行，页大小分布从 untraced 的 128–512KB 集中变成"4KB + 大页"双峰，从而误导 TLB 设计（untraced 显示 TLB 应服务 64–512KB 页，traced 却暗示需要宽谱页大小支持）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 与硬件的关系链：物理页连续性 → 大页能否成立（需要 buddy allocator 提供高阶连续块；外部碎片、不可迁移页（MIGRATE_UNMOVABLE）与缺页分配路径共同决定连续性，Web 证据：buddy 按 order 分配、迁移类型隔离）→ 大页决定 TLB 覆盖率与页表 walk 开销（一条 2MB 大页 TLB entry 覆盖 512 个 4KB 页；TLB 压缩/聚簇类设计依赖连续性分布）→ 反过来决定内存管理硬件/OS 设计（Svnapot、Contiguitas、Translation Ranger 等）。
- 具体运转例子（BULLETTIME 评估流）：应用执行分配 → 缺页从 buddy allocator 取 4KB 页 → khugepaged 按周期醒来折叠成 2MB 页 → 周期性地读取应用页表（每 30s 快照）得到虚拟-物理映射 → 结束时对物理内存覆盖做 power-of-2 页分箱统计 CDF → 与 untraced 对比得出 Misplaced Memory（总变差距离）。tracing 失真路径：Pin 落盘经 page cache 分配 4KB 页碎片化物理内存 + I/O 停顿延长运行时间使 khugepaged 超比例运行 → 双峰分布。
- 修复路径（消融顺序）：O_DIRECT 直写（消除 page cache 4KB 碎片，去掉小页过度分配）→ hugetlbfs 大页内部 buffer 并远离应用分配 → 应用线程时间膨胀 → 内核线程睡眠膨胀（+KD 消除 2MB+ 大页过度分配，即"THP 守护线程过快"问题）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用（Web 证据：内核文档）：THP 由 khugepaged 守护线程 + 缺页时同步/异步 hugepage 分配实现，sysfs（transparent_hugepage/enabled、defrag）可调 always/madvise/never；碎片化靠 compaction（kcompactd 后台压缩、khugepaged 折叠，二者分工：defer 模式先唤醒 kcompactd 后由 khugepaged 安装）。hugetlbfs 经 mount -t hugetlbfs 使用、受 vm.nr_hugepages 管理。O_DIRECT 要求对齐的 buffer 与 I/O 大小，用户态直接 DMA 到盘、不经过 page cache。
- 研究使用场景：内存管理子系统设计（TLB 压缩、Svnapot、Contiguitas）、大页策略评测都依赖准确的连续性 trace；BULLETTIME 表明这类研究必须做行为保真（O_DIRECT + 大页 buffer + time dilation），否则 traced 数据会系统性高估低连续区域占比（论文 Fig.2：traced 显示约一半内存 <64KB 页、untraced 不足 20%）。信息缺口：论文未报告除 2MB 外的大页（1GB）策略与 NUMA 影响。

涉及论文标题：
- BULLETTIME: Time Dilation for High-Fidelity Tracing

Revelator 补充视角（ISCA'26，Revelator+THP）：THP 的收益依赖碎片——2MB 连续区域稀缺时（数据中心常态，论文实测可用 2MB 页仅 10–50%）THP 回退 4KB 页、翻译开销回归。Revelator+THP 把大页收益与哈希投机组合成四级分配：①哈希 2MB 分配（2MB 对齐虚拟区哈希选 2MB 对齐物理区）→②传统 Linux THP 分配回退→③Revelator 4KB 哈希分配（严格限制在未标记/未保留给 2MB 的区域，避免破坏未来 THP 机会）→④buddy allocator 兜底。效果：low 碎片下与 SpOT 持平（≈11% vs THP）、medium 碎片 19%、high 碎片 26%，同时保留 THP 的 TLB reach 收益；16 核下较 THP 达 1.40×（medium）/1.50×（high）碎片，能耗较 THP 降 5.5%。对比：L2 TLB-64K（64K 项 L2 TLB）仅在 low 碎片有效（≈4%），high 碎片只 3%——增大 TLB 只把 miss 变 hit，miss 仍付全部 PTW 成本，而 Revelator 隐藏的是残留 PTW 成本。
涉及论文标题：
- Revelator: Rapid Data Fetching via OS-Guided Hash-based Speculative Address Translation
