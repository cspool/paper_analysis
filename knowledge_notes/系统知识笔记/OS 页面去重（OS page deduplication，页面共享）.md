## OS 页面去重（OS page deduplication，页面共享）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
OS 页面去重是操作系统对只读页的共享机制：Linux 对 DRAM 页采用 copy-on-write（CoW）语义——内容相同的物理页只保留一份，多个进程映射同一物理页；对只读代码页（如 .text、.nv_fatbin 段）保持去重，直到某进程写入才复制。内核级 KSM（Kernel Same-page Merging）进一步主动合并内容相同的匿名页。PRowhammer（ISCA'26）观察 O1：GPU 共享库的 .nv_fatbin 段同样被去重——OS loader 在 kernel launch 时用 mmap(MAP_PRIVATE)+PROT_READ|PROT_EXEC 把 GPU 共享库映射进 hDRAM，任何进程都能以相同 flag 映射这些库；OS 不区分 CPU/GPU 共享库，内存管理与访问控制一视同仁。因此攻击者（无特权用户进程）可映射同一 .nv_fatbin 页、对其做 Rowhammer 位翻转，受害者进程被迫使用同一份被篡改的 GPU kernel 代码——OS 页去重是 PRowhammer 成立的关键前提。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程（Listing 2 系统调用跟踪）：victim 进程启动 GPU kernel → loader openat("/usr/local/lib/libcublasLt.so.12", O_RDONLY) → mmap(<VA>, <size>, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, fd, 0) 把共享库映射进 hDRAM → 只读代码页进入 page cache、被去重（多进程共享同一物理帧）→ 攻击者以相同 flag 映射同一库，物理上看到同一份 .nv_fatbin → Rowhammer 翻转其中 bit → victim 动态链接时从 hDRAM 把（被篡改的）kernel 代码送到 GPU。攻击时序控制：若库在攻击前已被加载、代码页已分配 page-frame 并驻留 page cache，攻击者用 vmtouch 工具刷 page cache，再通过 memory massaging 把可利用 bit 移到可翻转位置。去重被禁用时的替代：PRowhammer 不根本依赖去重，可改用 Frame Feng Shui 类技术操控 page-frame 分配。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Linux 内核的 mm/memory.c 页表管理 + page cache（file-backed 页）+ KSM（匿名页去重）。使用：对 GPU 共享库攻击而言，攻击者只需要普通进程权限即可 mmap 只读共享库（读权限对所有用户开放）；由于只读页去重，攻击者的 Rowhammer 位翻转即作用于受害者也将使用的物理页。安全影响（论文核心）：OS 页去重把"库代码只读、无法常规修改"变成可利用——攻击者不需要 root 修改库文件，只需内存级位翻转；防御上论文建议 GPU 压缩/解压管线加 ECC/CRC、kernel dispatch 前做哈希完整性验证。

涉及论文标题：
- PRowhammer Propagating Bit-flips from CPU to GPU
