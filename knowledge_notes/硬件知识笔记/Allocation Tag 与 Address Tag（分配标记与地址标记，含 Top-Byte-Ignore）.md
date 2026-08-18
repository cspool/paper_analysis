## Allocation Tag 与 Address Tag（分配标记与地址标记，含 Top-Byte-Ignore）

术语解释
MTE 的双 tag 机制：allocation tag 是关联物理内存的"锁"（每 16B granule 一个 4-bit 值，由 malloc 用 tag 指令写入，存储位置由 SoC 实现决定）；address tag 是关联虚拟地址/指针的"钥匙"（4-bit，放在指针虚拟地址最高字节 bits 56:59）。Top-Byte-Ignore（TBI）是 AArch64 指示处理器在地址翻译时忽略虚拟地址最高字节的特性，MTE 利用它放置 address tag。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
容量数学：每 16B granule 4 bit → tag 存储占软件可见内存的 (4 bits / (16 bytes × 8 bits/byte + 4 bits)) = 3.03%。若用 sequestered 方式须为平台全部物理内存预留（无法预知哪些区域会被 tag），CSP 平台的 VM 密度与 TCO 直接受损（Meta 数据：DRAM 占平台成本从第一代 15% 升至第六代 35%）。TBI：AArch64 允许虚拟地址最高字节不参与地址翻译（用于 tag 等），保证带 address tag 的指针仍正常翻译。分配流程例子（论文 Fig 1）：顺序两块堆分配 P（tag=4）与 Q（tag=1）——P 的 granule 设 allocation tag=4、指针 bits 56:59 编程为 4；Q 的 granule 设 allocation tag=1、指针 bits 56:59 编程为 1；各自范围内访问通过；P 越界写入 Q 区域时 address tag 4 ≠ allocation tag 1 → SYNC 模式触发 tag mismatch fault、访问不完成。概率性：4-bit 只有 16 值，tag 复用使 use-after-free 等非顺序攻击被检测的概率 = 1 − 1/16（用满 tag 值后约 94%）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
tag 的存储与传输是 SoC 实现选择（架构只定义语义）：AmpereOne 把 allocation tag 存 DRAM ECC 位（每 16B granule 4 bit）、经 mesh 保留 implementation-specific metadata 位与数据同传、L1/L2 缓存加宽与数据共驻，PE 在 cache-lookup 点做地址 tag vs allocation tag 比较。运转流程：malloc 返回指针 P（bits 56:59=4）→ load [P+offset] 时虚拟地址最高字节被 TBI 忽略不参与翻译、但 bits 56:59 被取出作 address tag → 与目标 granule 的 allocation tag 比较 → 匹配通过。store-to-load forwarding 场景中 allocation tag（address tag）被记录在 store buffer，年轻 load 需 address tag 匹配才可转发（保正确性）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
软件实现：Linux 中 malloc 返回带 address tag 的指针需 PROT_MTE 页 + tag 指令（IRG 取随机 tag、STG 写 allocation tag、ADDG 把 tag 传播到返回指针）；glibc 的 libc_mtag_tag_region() 在 malloc 热路径设置 tag、free 时清除。内核须在页迁移/交换时保存恢复 allocation tag（swap 支持见内核补丁，v6.6+ 的 MTE swap 系列）。KASAN HW_TAGS 复用该双 tag 机制做内核内存错误检测（随机 tag 碰撞即误报率）。参考：Linux 文档 https://docs.kernel.org/arch/arm64/memory-tagging-extension.html、ARM 文档 https://developer.arm.com/documentation/102925/0100。

涉及论文标题：
- Optimized Memory Tagging on AmpereOne® Processors
