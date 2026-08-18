## Memory Massaging（内存按摩）与 Frame Feng Shui（物理页风水）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Memory massaging 是 Rowhammer 类攻击的第二阶段：攻击者操控受害者敏感数据/代码的物理页分配，使其落在 DRAM 中可翻转（且偏移合适）的 page-frame 上。原理：OS 页分配不尊重进程边界——利用 page-frame 分配策略（Linux buddy allocator、per-CPU page frame cache），通过精心构造的分配/释放序列，攻击者能把目标页"按到"自己可控（可 hammer）的相邻物理帧。Frame Feng Shui 是其中一种具体技术（源自 Flip Feng Shui，Razavi et al., USENIX Security'16；Drammer 的 Phys Feng Shui 是其 ARM/Android 变体）：利用 Page Frame cache 与 buddy allocator——分配/释放塑造 page frame cache，buddy allocator 复用最近释放的物理页满足后续分配，攻击者借此把受害者页引导到可预测物理帧，再 hammer 相邻行诱发 bit-flip。Web 证据：FRAMER（arXiv 1905.12974）展示 per-CPU page frame cache 可被用作物理页分配侧信道；Rubicon（IEEE 2025）在 Zoned Buddy Allocator 上实现 page-granular 确定性 massaging（首个近 100% 成功率的确定性 x86 提权 Rowhammer 利用）。PRowhammer（ISCA'26）在 hammering 前用 memory massaging 让 cuBLASLt/GGML 共享库的可利用 bit 落在可翻转的 DRAM 偏移；page cache 已驻留时先用 vmtouch 刷掉再 massaging（微秒级）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程（PRowhammer 版）：(1) memory profiling——分配大块内存，探测 hDRAM 中与可利用 bit 同页内偏移的易翻转 page-frame；(2) 加载目标共享库（cuBLASLt）到自身地址空间，通过分配/释放与刷 page cache（vmtouch）使可利用 bit 落在可翻转位置（memory massaging；去重关闭时用 Frame Feng Shui——塑造 Page Frame cache，让 victim 的分配被 buddy allocator 从缓存页满足，落在攻击者可控相邻帧）；(3) 对 aggressor 行 hammering 完成单 bit-flip；(4) 受害者启动，被迫使用被篡改的库。攻击时间：memory massaging 微秒级，首次 flip 需几分钟，首次可利用 flip 一小时内（DDR3/DDR4）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Linux 物理内存管理（buddy allocator 的 free lists、per-CPU page frame cache、page cache 的驻留/换出）。工具：vmtouch 刷 page cache；攻击者通过 mmap/malloc/free 模式操控分配（Drammer 用 ION DMA buffer 耗尽特定阶 chunk 后释放、再 spray 目标页）。使用场景：Rowhammer 攻击需要"在正确页、正确偏移、正确时间"发生位翻转，massaging 把随机物理分配变成确定性引导。局限：页分配器行为随内核版本/配置变化，攻击者需先逆向本地分配行为（如读 /proc/buddyinfo 观察）；Linux 4.0+ 的 pagemap/PFN 需 CAP_SYS_ADMIN，攻击者依赖分配器侧信道而非直接读物理帧号。

涉及论文标题：
- PRowhammer Propagating Bit-flips from CPU to GPU
