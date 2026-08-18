## CLINT（Core Local INTerrupt controller，核本地中断控制器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CLINT 是 RISC-V 的内存映射外设，为每个 hart 提供软件中断（IPI，msip 寄存器）与定时器中断（mtimecmp/mtime），直接发往指定 hart、不经仲裁，区别于处理全局外部中断的 PLIC。典型 SiFive 兼容布局：BASE+0x0 起每 hart 4 字节 msip（bit0 有意义，置 1 触发机器软件中断）、BASE+0x4000 起每 hart 8 字节 mtimecmp、BASE+0xBFF8 全局 mtime。RISC-V 规格要求 CLINT 接受任意尺寸（字节/半字/字）的 store 写入 msip。Web 来源：CVA6/Ariane CLINT README（https://github.com/openhwgroup/cva6/blob/1ca5e47d/corev_apu/clint/README.md ）、RISC-V 特权手册。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 HARTBREAKER 中 CLINT 是"跨 hart 控制流非确定性"的关键硬件入口：发送 hart 用普通 store 写目标 hart 的 msip（图 4 示例：`la x4, CLINT_BASE; li x5,1; sw x5, 4(x4)`），接收 hart 在任意周期陷入中断。论文发现 NaxRiscv 的 CLINT 实现违反规格——只接受全字 store，store-byte 指令错误触发异常（bug N2），暴露"内存映射中断外设的访问宽度契约"这类单核 fuzzer 测不到、litmus 测不到的接口 bug。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：硬件 RTL 中 CLINT 常以内存映射 IP 形式挂在总线（如 TileLink/AXI slave）；软件（M-mode 无 MMU 内核直接访问，S-mode 经 SBI SEND_IPI 由固件代写，AIA 时代经 IMSIC 以 MSI 注入）。验证侧：HARTBREAKER 用"随机宽度的 CLINT store + 目标 hart 轮询 MIP"组合施加压力，N2 由此触发；cf-anchor 依赖 CLINT 语义（写入后延迟到达）构造确定性 landing zone。

涉及论文标题：
- HartBreaker: Deterministic Fuzzing of Multi-Hart RISC-V CPUs with Non-Deterministic Programs
