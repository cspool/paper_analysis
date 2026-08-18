## glibc Memory Tagging（GLIBC_TUNABLES=glibc.mem.tagging，malloc 内存标记）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
glibc 的 malloc 子系统对 ARM MTE 的软件支持（glibc 2.33+，仅 AArch64+MTE 平台生效，其它平台忽略）：通过环境变量 GLIBC_TUNABLES=glibc.mem.tagging=N（0–255 位掩码）控制是否给分配打随机 tag 以及选择精确或延迟 fault 模式。位掩码语义：bit 0 = malloc 子系统分配 tagged 内存（每分配随机 tag）；bit 1 = 精确 fault 模式（tag 违规立即报，可能变慢）；bit 2 = 精确或延迟模式由系统偏好决定；非零值即自动在内核启用内存标记支持。论文评估的 SYNC 配置为 GLIBC_TUNABLES=glibc.mem.tagging=3（tagged 分配 + 精确模式）。glibc 内部实现：libc_mtag_tag_region() 在 malloc/free 热路径给新返回内存设置 allocation tag、释放时清 tag；开启 tagging 时禁用 brk(2) 改用 mmap(2)——brk 只是推进程序断点、无法提供 MTE 需要的页属性（PROT_MTE）与 tag 存储初始化，mmap 让分配器请求受保护匿名映射（选虚拟地址范围、创建/合并 VMA、更新页表、first-touch 与 tag 初始化）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
论文（AmpereOne，Fedora 40/Linux 6.10.6/glibc 2.39）把 MTE 开销拆为"硬件 always-on（B/A）"与"软件 tag checking（C/B）"两个决策；SPEC CPU 2017 192-copy refrate 上 C/B geomean 0.924（7.6% 降），软件开销来源：(1) 指令数增加——malloc 路径 tag 初始化/清除指令 + mmap 替代 brk 的 VMA 创建/页表更新/first-touch 开销（图 8：502.gcc Mode B 32142 次 brk+94 次 mmap vs Mode C 0 次 brk+28313 次 mmap）；(2) eager tag initialization——大虚拟区间被分配器全量初始化 tag，失去惰性填充（502.gcc/523.xalanc 大量"大区间小触碰"）；(3) 高频小对象瞬时分配——每次 malloc 半固定 tag 初始化成本无法摊销（502.gcc/520.omnetpp/523.xalanc 最严重）；碎片化虚拟地址空间加重内核 VMA 管理、TLB miss 与 realloc 的 de-tag+re-tag 双倍开销。jemalloc（不支持 tag，以 Mode B 为基线）替换后 523.xalanc 提速 1.363×、520.omnetpp 1.093×、geomean 1.059×，证明 MTE 降速源于内存管理而非 tag checking 本身。方向参考：Apple xzone malloc——小块 free 即重赋 tag、大块惰性重赋。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
使用：`GLIBC_TUNABLES=glibc.mem.tagging=3 <binary>` 开启 SYNC tagging（需 glibc 2.33+、Linux 5.10+ CONFIG_ARM64_MTE、MTE 硬件；Fedora 36+ 内核默认开 MTE）。glibc 文档：https://sourceware.org/glibc/manual/2.33/html_node/Memory-Related-Tunables.html。内核侧配套：Linux 7.0 收紧 PSTATE.TCO 处理（Ampere 2025 补丁）减少内核空间多余 tag 检查——此前 memcached 开启 tagging 后 25–50% 开销来自内核入口处全量 tag 检查（251B 内核读 vs 4B 用户读），收紧后消除 97%+ 内核 tag 检查。本术语为近似分层：glibc malloc tagging 属运行时/OS 级内存管理软件，与本层"请求调度"主题不完全吻合，但作为系统软件运行时行为归入本层最接近；论文中它同时是硬件架构层 MTE 术语的软件使能配置。

涉及论文标题：
- Optimized Memory Tagging on AmpereOne® Processors
