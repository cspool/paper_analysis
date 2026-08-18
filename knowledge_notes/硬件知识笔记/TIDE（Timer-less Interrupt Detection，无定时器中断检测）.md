## TIDE（Timer-less Interrupt Detection，无定时器中断检测）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- TIDE 是本文提出的、在 macOS/Apple Silicon（M1–M4）上无需任何架构定时器即可精确检测本核中断的技术。逻辑链：(1) 中断是硬件信号，把接收核从 EL0 拉入 EL1 执行内核 handler，返回时恢复用户态；(2) 每次 user→kernel 转换，XNU 异常入口（Lel0_irq_vector_64）在保存用户寄存器之前，必须先用 MAP_KERNEL 宏恢复内核翻译 regime（mrs/orr/msr TTBR0_EL1 改 ASID、MOV64/msr TCR_EL1 载入 TCR_EL1_BOOT）并用 BRANCH_TO_KVA_VECTOR 宏经 TPIDR_EL1→per-CPU→CPU_EXC_VECTORS 表间接跳转 handler——两个宏都把 AAPCS64 平台寄存器 x18 当 scratch 覆写（TCR_EL1 写源值必须来自 GPR，无法用立即数）；(3) 为防内核信息泄露，macOS 又在 kernel→user 返回前显式清零 x18（XNU 4570.61.1 起）；(4) 因此用户态给 x18 赋非零值后，观察到 x18 变 0 即证明本核刚经历一次内核上下文切换（中断/异常），与任何定时器无关。论文验证：baseline 定时器检出的中断中 x18 100% 被清零；与架构定时器（cntvct_el0，阈值 1μs）比对 >99% 一致，且排除其 0.2%–0.5% 误报（非中断的时间戳跳变不清 x18）；counting-thread 定时器仅检出 45.0%–80.5%。
- 派生两个原语：①中断计时——x18 赋非零后同线程递增计数器直到 x18 清零，计数器值即相邻中断间隔；②去噪——测量代码前后各一次 x18 写/读（至多 1 写 1 读开销），清零则丢弃/重做该测量。基于 TIDE 的成果：网站指纹 closed-world top-1 93.8%、open-world 91.5%、视频 78.1%；SysBumps 去噪后成功率 54%→81%（开销 0.06%）；loop-counting 增强 92.8%/93.6%；跨核隐蔽信道 111.65 bps、误码 4.45%；SIKE（CIRCL v1.1）密钥提取成功。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- TIDE 是架构级（architectural-level）运行时检测原语，处于用户态软件与硬件特权转换的交界，其"传感器"是一个硬件寄存器在特权切换路径上的副作用。运转流程（一次 SPI 到达攻击核）：
```
mov x18, #0x1          # ① 攻击线程给 x18 赋非零值
loop:                  # ② 进入检测循环
  cnt += 1             #    计数器自增（代表流逝时间）
  cmp x18, #0
  b.ne loop            #    x18 仍非零 → 未中断
# ③ 硬件将 SPI 投递到本核（AIC 均匀投递，见 AIC 条目）
# ④ 本核陷入 EL1：MAP_KERNEL 用 x18 作 scratch 覆写其值（mrs x18,TTBR0_EL1; orr; msr; MOV64 x18,TCR_EL1_BOOT; msr TCR_EL1,x18）
#    再 BRANCH_TO_KVA_VECTOR：ldr x18,[TPIDR_EL1] → ldr x18,[x18,ACT_CPUDATAP] → ldr x18,[x18,CPU_EXC_VECTORS] → br x18
# ⑤ 内核处理完毕返回 EL0 前，macOS 显式清零 x18
# ⑥ 用户态循环观察到 x18==0 → 跳出，cnt 即中断计时值
```
- Annotations：①③步骤在攻击线程同一核完成；④是硬件+内核路径，x18 在保存用户寄存器前已被覆写，故用户原值丢失；⑤是 macOS 特定行为（Linux KPTI 不会清零 x18）；⑥的 cnt 反映用户态流逝时间，与 CPU 频率相关（论文消融实验用固定 24 MHz 的 cntvct_el0 替换计数器，准确率基本不变）。
- 去噪原语的硬件流程：测量代码前写 x18=非零 → 执行被测代码（若中途被中断则 x18 被内核清零）→ 测量后读 x18，若为 0 则丢弃该测量，每次测量至多 1 写 1 读；寄存器读写远快于原循环里的内存写，故开销仅 0.06%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Zenodo artifact https://zenodo.org/records/19450550（TIDE-Artifact.zip，CC BY 4.0，含三组实验：x18 行为验证 E1、AIC 反推 E2、网站指纹端到端 E3；其余代码按需提供；早期 GPL v3.0）。硬件依赖：Apple Silicon Mac（M1 Pro–M5）；软件依赖：macOS+Safari、gcc/make/Python3、100 网站指纹另需 GPU 服务器+TensorFlow 2.x。使用场景：网站/视频指纹（LSTM 分类器）、SysBumps 去噪、loop-counting 增强、跨核隐蔽信道、SIKE 密钥提取。
- 限制与可移植性：TIDE 只检测本核中断；Linux 用 KPTI（非 Double Map）且恢复 x18 为普通寄存器，不适用；iOS 与 macOS 同 XNU，已在 iPhone 16 Pro（A18 Pro/iOS 26.3）确认可行；Apple Virtualization.framework 客户机 macOS 内也可用。缓解：保留用户 x18（需重设计 Double Map/新增隐藏 scratch 寄存器/虚拟化 x18）或注入随机 jitter（论文把准确率 93.8%→61.8%，网站加载 +10.8%）。

涉及论文标题：
- Towards Practical Interrupt Side-Channel Attacks on macOS for Apple Silicon
