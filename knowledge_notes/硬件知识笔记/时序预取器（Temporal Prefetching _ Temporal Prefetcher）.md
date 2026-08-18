## 时序预取器（Temporal Prefetching / Temporal Prefetcher）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 时序预取器是面向不规则访存的一类硬件预取器，其核心思想是记录"被访问内存地址 → 其相关后继地址"的时序相关性作为元数据（metadata），当某个地址在执行中再次出现时，用该元数据预取其历史后继地址。逻辑链：(1) 不规则访存（如指针追逐、间接访存）既无固定步长也无固定空间模式，stream/stride/spatial 预取器失效；(2) 但若同一地址在不同时间重复访问，其后续访问序列往往也重复——这就是"时序（temporal）"局部性；(3) 因此把 (地址 A → 后继地址集合 S(A)) 记入元数据表，A 再现时预取 S(A)。代表设计：Domino（HPCA 2018）、PPT、SMS（off-chip metadata）、Triage（MICRO 2019，on-chip 与 LLC 共置）、Triangel（ISCA 2024，state-of-the-art on-chip temporal prefetcher）、Streamline/Tetrahedrangel 等。ICP 论文指出时序预取器的两个根本局限：Factor 1——并非所有不规则访存都有地址级时序重现（间接访存负载如 GAP 的地址几乎不复现，Triangel 在其上性能仅 ~5.9% 提升、元数据复用率极低）；Factor 2——元数据存储巨大（与 LLC 共置可占 LLC 一半容量即 1MB，Triangel 另有 17.6KB 控制状态），即便用非重现过滤（Triangel）或压缩（Streamline 0.5MB）仍是数百 KB 量级。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 时序预取器在缓存层级中的运转流程（Triangel，ISCA 2024，Web 证据 arXiv:2406.10627）：集成在 L2/LLC 级。采样记录阶段——demand 请求的地址序列作为"历史"存入采样表；相关性记录阶段——把历史地址与其后继地址的马尔可夫链式相关性写入元数据表（Triangel 用与 LLC 共置的 metadata 分区 + 采样方法学控制激进性：能处理长期模式时激进及时，不能时避免不准确预取）；预取阶段——某地址再次被访问时查表，按记录的度（prefetch degree）预取其后继地址到缓存。ICP 论文在 gem5 FS 中把 Triangel 集成在 L2（遵循其原设计），与 ICP（L1 集成）对比：Triangel 在 SPEC（地址有重现）上有效，在 GAP（地址不复现）上几乎无效；其元数据复用率（总元数据访问数/插入数）比 ICP 的指令级相关性低约 10^5 倍。Web 证据：Triangel 相对仅带 stride 预取器的 baseline 加速 26.4%（vs Triage 14.2%），内存流量仅增加 10%（vs Triage 28.5%）；后续 Tetrahedrangel 进一步降低能耗。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现方式：早期设计（SMS、PPT）把元数据放 off-chip DRAM（容量够但访问延迟高）；现代设计（Triage、Triangel、Streamline）把元数据与 on-chip LLC 共置，需要 way 分区/元数据插入替换策略，占用可达 LLC 一半容量；Triangel 开源实现基于 gem5（GitHub SamAinsworth/gem5-triangel）。使用场景：作为不规则访存预取的 SOTA baseline 评估新方案（ICP 即以其为最强时序 baseline）；参数敏感性包括预取度（degree，ICP 对 DMP+Triangel 组合扫 1–6，度 4 最优）与元数据表大小（Streamline 0.5MB 压缩元数据在 SPEC 上更好、GAP 上更差，因压缩把多个连续后继绑定一次查找，GAP 地址少有重现导致无用预取更多）。

涉及论文标题：
- ICP: Exploiting Instruction Correlation for Prefetching Irregular Memory Accesses
