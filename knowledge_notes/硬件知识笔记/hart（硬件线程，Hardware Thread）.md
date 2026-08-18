## hart（硬件线程，Hardware Thread）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
hart（hardware thread）是 RISC-V 对"可独立取指执行指令的执行单元"的称谓——一个多核处理器由多个 hart 组成，每个 hart 拥有独立的程序计数器（PC）、通用寄存器组、CSR 与中断状态，共享全局内存空间；hart 间通过共享内存与中断（IPI）通信。RISC-V 手册用"hart"而非"core/thread"以精确表达硬件执行上下文：一个核可以有多个 hart（SMT），一个芯片可以有多个核。RISC-V 的 hart 计数用 mhartid CSR 标识，指令集以 hart 为粒度定义并发语义。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
HARTBREAKER 的研究对象就是"多 hart 交互"：非确定性源于多个 hart 并发执行时共享内存的读写交错（数据流非确定性）与跨 hart 中断（控制流非确定性）。论文形式化（IV-B）：定义 hart 观察状态 s（通用寄存器 R(s)、PC(s)）与环境状态 e（CSR、内存视图 M(e)），指令级非确定性 = 同一观察状态在不同环境下执行同一指令产生不同 s'。测试程序在三 hart 配置（两个交互 hart + 一个独立访存噪声 hart）上运行；评估覆盖五个多 hart 设计（Rocket、BOOM、Toooba、NaxRiscv、XiangShan），均配置三个 hart。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：硬件上每个 hart 对应一套取指/执行流水资源（或共享流水线的线程槽）；软件上 RISC-V 提供原子指令（AMO、LR/SC）、FENCE 与 acquire/release 注解供多 hart 程序同步。验证侧：HARTBREAKER 针对 hart 间通信通道（共享内存 + IPI）随机生成程序、用同步锚跨 hart 对齐分区，验证 RVWMO 下的执行合法性。

涉及论文标题：
- HartBreaker: Deterministic Fuzzing of Multi-Hart RISC-V CPUs with Non-Deterministic Programs
