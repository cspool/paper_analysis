## IPI（核间中断，Inter-Processor Interrupt）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
IPI 是一个 hart 上运行的软件向另一个 hart 发送的异步中断请求：接收 hart 挂起当前执行、转到指定中断处理程序。RISC-V 中 IPI 通过软件中断位实现——发送方无法直接写目标 hart 的中断 CSR，必须用普通 store 写内存映射外设（通常是 CLINT 的 msip 寄存器）置位目标 hart 的 mip.MSIP（机器软件中断挂起位）；mip.MSIP 对 M 模式只读，只能经 MMIO 写。RISC-V 不规定中断从发送到被评估的延迟（无 liveness 保证），因此 IPI 到达时刻是非确定的。Web 来源：RISC-V 特权手册（mip.MSIP 语义）、Linux RISC-V IPI 三种机制（CLINT MMIO / SBI SEND_IPI / IMSIC MSI，见 https://lkml.org/lkml/2022/9/4/342 ）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
IPI 在硬件架构中走 CLINT（发送）→ 目标 hart 的中断仲裁（mip/mie 门控）→ trap 分发（mtvec）→ mret 返回 的完整通路，每个环节都可能出 bug。HARTBREAKER 发现的三个 IPI 相关 bug：(1) T1（Toooba）——mret 返回时应恢复先前使能状态并立即检查 pending 中断，Toooba 却先去无效地址取指，导致写入错误的 trap cause；(2) X1（XiangShan）——MIP 允许乱序读，错过新到达的中断；(3) N2（NaxRiscv）——CLINT 只接受全字 store，字节 store 错误抛异常（违反规格要求的任意尺寸 store 支持）。HARTBREAKER 把 IPI 作为测试程序的主要非确定性来源：每个测试平均 12.1 个 IPI，通过 cf-anchor 安全注入。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：硬件侧 CLINT 提供 msip/mtimecmp/mtime 三组 MMIO（如 SiFive 布局：BASE+0x0 msip、BASE+0x4000 mtimecmp、BASE+0xBFF8 mtime）；软件侧发送方 `sw 1, 4(clint_base)` 触发，接收方处理程序 `csrr x2, MIP` 查询、写 0 清 msip 防重触发。验证侧：HARTBREAKER 用 IPI 覆盖测试（100% 测试含中断）检验中断子系统的时序正确性；对比工具 RISCV-DV/单核 fuzzer（INSTILLER、DifuzzRTL 支持中断但不支持 IPI）。

涉及论文标题：
- HartBreaker: Deterministic Fuzzing of Multi-Hart RISC-V CPUs with Non-Deterministic Programs
