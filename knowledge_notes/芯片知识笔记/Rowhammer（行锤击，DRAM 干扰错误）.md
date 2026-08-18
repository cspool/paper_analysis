## Rowhammer（行锤击，DRAM 干扰错误）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Rowhammer 是 DRAM 的可靠性缺陷被利用形成的攻击：DRAM 单元存在 cell-to-cell 干扰——在单个刷新周期（DDR3/DDR4 通常 64ms）内对某一行（aggressor row，攻击行）重复激活（hammering）成千上万次，通过电容耦合与电磁干扰加速相邻行（victim row，受害行）的电荷泄漏；相邻行单元电荷损失到一定程度后，下次访问被 sense amplifier 误读，产生 bit-flip（位翻转）。攻击者通过在自己地址空间内高频访问某些行，就能让其他进程（受害者）使用的 DRAM 行发生位翻转，无需任何特权。成功攻击需三阶段（Razavi et al.，Flip Feng Shui）：(1) memory profiling——分配大块内存，找出易翻转的 page-frame 及其位翻转偏移；(2) memory massaging——操控受害者敏感数据/代码的物理页分配，使其落在可翻转位置；(3) hammering——反复访问攻击行诱发受害者行翻转。DRAM 厂商的缓解：TRR（Targeted Row Refresh，目标行刷新，DDR3/DDR4 检测可疑激活后刷新受害行，多次被绕过如 TRRespass、Blacksmith）、DDR5 的 PRAC（Per-Row Activation Counter，JEDEC JESD79-5C 加入，DRAM 内每行激活计数并通知内存控制器）、RFM（Refresh Management 背压信号）；Web 证据显示 DDR5 的 TRR/PRAC 仍被新攻击绕过（如 Phoenix，ETH Zürich/Google，IEEE S&P 2026 最佳论文，对 15/15 块 SK Hynix DDR5 触发翻转并在约 109 秒内提权；ECC/on-die ECC 只能减缓多 bit-flip 变体）。PRowhammer（ISCA'26，IIT Bombay）利用 CPU 侧 Rowhammer 在 hDRAM（宿主 DRAM）中对 GPU 共享库的 .nv_fatbin 压缩代码页实施位翻转，把 bit-flip 传播到 GPU 计算（区别于 GPUHammer 直接攻击 GPU 显存 dDRAM/GDDR6）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
Rowhammer 根源于 DRAM 芯片物理组织：行（row）是 DRAM bank 内共享 wordline 的一排单元，行间由浅沟槽隔离（STI）隔离，电容耦合导致邻行干扰。运转流程：CPU 向内存控制器发 activate 命令打开某行（行缓冲读）→ 反复对同一 row 的 ACT/PRE 序列（如 1-1 交替访问同一 bank 的两个 aggressor 行，双面攻击）→ 电荷在 victim row 单元通过邻近电容漏电 → 超过 tRET 容限后数据反转。芯片级缓解在 DRAM die 内实现：TRR（die 内目标行刷新）、PRAC（die 内每行激活计数器）、on-die ECC（ODECC）。PRowhammer 的平台细节：DDR3（Kingston 8GB 1600MT/s，Intel i7-4790 Haswell）与 DDR4（Corsair 8GB 2400MT/s，Intel i7-8700 Coffee Lake），无噪声成功率 DDR3 50%/DDR4 80%，有噪声（GAP PageRank 并发）DDR3 30%/DDR4 73%；首次 flip 几分钟、首次可利用 flip 一小时内，memory massaging 微秒级（vmtouch 刷 page cache）。Web 证据补充：PRAC 被广泛视为未来 DDR5 的必要缓解，但 MOAT（ASPLOS'25）证明其前身 Panopticon 思路可被绕过，Rowhammer 仍未解决。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：攻击者用高带宽内存访问指令（如 CLFLUSH/clflush、非临时 load）绕过 cache 直接访问 DRAM 行，或使用 SledgeHammer 式 bank 级并行放大；DRAM 地址反演（DRAMA）软件逆向出物理地址→DRAM 行映射以定位同一 bank 的相邻行。使用场景（PRowhammer）：对 hDRAM 中共享库页面做单 bit-flip，翻转 SASS 指令为"不同但合法"的指令；攻击前先用 vmtouch 刷 page cache、必要时用 Frame Feng Shui 操控物理页分配。缓解使用：芯片侧 TRR/PRAC/RFM/ECC；系统侧 CATT/Soft-TRR（保护页表）、ZebRAM（guard row，容量损失 50-67% 不实用）、Siloz/Citadel（DRAM 隔离域）等。论文结论：PRowhammer 只有彻底阻止 DRAM bit-flip 或 CPU→GPU 传输链路加完整性校验（压缩/解压管线 ECC/CRC、kernel dispatch 前哈希验证）才能缓解。

涉及论文标题：
- PRowhammer Propagating Bit-flips from CPU to GPU
