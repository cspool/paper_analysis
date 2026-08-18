## Double Map（Apple macOS 的 Meltdown 缓解）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Double Map 是 Apple 为缓解 Meltdown 引入的地址翻译方案（macOS 10.13.2 / iOS 11.2，XNU 4570.31.3，2017-12）。核心思想与 Linux KPTI（完全 unmapping 内核地址空间）不同：**保留内核翻译但使它们在 EL0 执行期间不可访问**，从而避免不必要的 TLB 失效、降低特权转换开销。逻辑链：(1) ARMv8-A 提供两个 Translation Table Base Register——TTBR0_EL1 指向当前进程地址空间翻译表、TTBR1_EL1 指向所有进程共享的内核地址空间翻译表；(2) 两个 TTBR 都携带 ASID 参与 TLB 查找，更新 ASID 只改变翻译上下文而无需全量 TLB 失效；(3) TCR_EL1 定义翻译 regime（虚拟地址大小、粒度、table-walk 行为），XNU 在 TCR_EL1_BOOT（EL1 完整内核映射）与 TCR_EL1_USER（EL0 受限范围）两个配置间切换；(4) 每次 user→kernel 转换，异常入口必须先更新 TTBR0_EL1 与 TCR_EL1 才能以内核 regime 访问内存——这是 TIDE 能被构造的根源。
- 在本文中：Double Map 的异常入口实现（MAP_KERNEL / BRANCH_TO_KVA_VECTOR 宏）把 x18 当 scratch 覆写，macOS 又在返回用户态前显式清零 x18，构成 TIDE 的检测信号（见 TIDE 条目）。Linux 无此行为（KPTI 恢复 x18 为普通寄存器）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件-内核协同流程（一次用户→内核→用户转换）：
```
① EL0 进程执行 svc 或收到中断
② XNU 异常入口 Lel0_irq_vector_64（保存用户寄存器之前）:
     MAP_KERNEL:
       mrs x18, TTBR0_EL1                  # 读当前 TTBR0（含进程 ASID）
       orr x18, x18, #(1<<TTBR_ASID_SHIFT) # 切换到内核 ASID
       msr TTBR0_EL1, x18                  # 更新 TTBR0（TLB 条目在新 ASID 下失配）
       MOV64 x18, TCR_EL1_BOOT             # 载入内核翻译配置
       msr TCR_EL1, x18                    # 恢复完整内核映射 regime
       dsb ish / isb sy                    # 同步翻译上下文
     BRANCH_TO_KVA_VECTOR:
       ldr x18, [TPIDR_EL1]                # 线程指针
       ldr x18, [x18, ACT_CPUDATAP]        # per-CPU 数据
       ldr x18, [x18, CPU_EXC_VECTORS]     # 异常向量表（Double Map 下 handler 动态可达）
       ldr x18, [x18, #(9<<3)]             # 索引 EL1 IRQ handler
       br x18                              # 间接跳转
③ 内核处理中断后返回 EL0 前，macOS 显式清零 x18（防内核信息泄露，4570.61.1 起）
```
- Annotations：②中 TTBR 与 TCR 的更新都需要 GPR 操作数（TCR_EL1 无立即数写），x18 因此成为必用的 scratch；③是 TIDE 信号的直接来源；TTBR1_EL1 不变（内核空间跨进程共享）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：XNU 内核 osfmk/arm64/locore.s（https://github.com/apple-oss-distributions/xnu 的 xnu-4570.31.3 / xnu-4570.61.1 标签，行号见论文引用 [60]/[64]）。使用：作为 Meltdown 缓解被所有 macOS/iOS 用户进程透明使用。
- 从攻击者角度看：Double Map 的 x18 副作用被 TIDE 利用；从防御者角度看，完整缓解需保留用户原始 x18——由于 x18 必须在建立内核映射前被覆写，像其他寄存器一样先保存再恢复"非平凡"，论文认为 OS-only 缓解可能需要基于 KPTI 原则重设计 Double Map，或引入隐藏 scratch 寄存器、虚拟化 x18（需 trap 用户态 x18 读写）等硬件支持；注入随机 jitter 可降低信号强度（93.8%→61.8%）。

涉及论文标题：
- Towards Practical Interrupt Side-Channel Attacks on macOS for Apple Silicon
