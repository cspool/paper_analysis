## AAPCS64 x18 平台寄存器（Platform Register）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- AAPCS（Procedure Call Standard for the Arm Architecture）是 Arm ABI 的关键组件，规定函数调用约定、寄存器使用、栈管理与参数/返回值传递。Apple 自 2020 年从 x86 转向 M 系列后采用 AArch64 变体 AAPCS64，提供 31 个硬件通用寄存器 x0–x30，分两类：程序调用类（x0–x8 参数/返回值、x9–x15 caller-saved 临时、x19–x28 callee-saved 局部、x29 帧指针、x30 链接寄存器）与平台用途类（x16/x17 intra-procedure-call 用于动态链接与系统调用；x18 保留给平台特定用途，由执行环境定义）。
- x18 的具体用途因平台而异：Apple 在需要跨特权级 Spectre 缓解的硬件上把它用作上下文切换 scratch；Windows 用 x18 作 TLS 指针；通用应用被建议避免使用 x18（LLVM 用 x15 替代 x18 防崩溃，2025-06 起版本 21.1.0）。
- 在本文中：x18 是 TIDE 的"传感器"——macOS 在每次 kernel→user 返回前显式清零 x18，用户态据此无定时器检测中断（详见 TIDE 条目）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- x18 是硬件实现的 64 位寄存器，EL0 可读写。本论文通过实验确定：任意系统调用（read/fsync/stat/gettimeofday/mmap，用 svc 触发避免 commpage 优化）后，x0/x1/x16/x18 四个寄存器被修改，其中只有 x18 与系统调用执行结果无关、总是被清零。寄存器在特权转换中的运转流程：
```
① 用户态给 x18 写非零值（mov x18, #0x1）
② 中断/异常 → user→kernel：XNU 在保存用户寄存器前用 x18 作 scratch（MAP_KERNEL/BRANCH_TO_KVA_VECTOR），用户原值被覆写
③ kernel→user 返回前：macOS 显式清零 x18（防内核数据经 x18 泄露）
④ 用户态观察到 x18==0 → 判定发生中断；非中断的时间戳跳变不清 x18（0.2%–0.5% 误报被排除）
```
- Annotations：②的覆写发生在保存寄存器之前是"漏洞"关键（本可像其他寄存器一样恢复）；③的显式清零是 2018-03（XNU 4570.61.1）的补充修复，反而固化了可观测信号；④是 TIDE 的检测逻辑。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Apple Silicon 硬件寄存器 + XNU 内核 locore.s 中的 MAP_KERNEL/BRANCH_TO_KVA_VECTOR 宏与返回路径清零代码。使用：攻击者 mov x18,#1 → 循环检查；编译器/运行时避免使用 x18（Apple 平台 LLVM trampoline 改用 x15，Windows 约定把 x18 当 callee-saved）。限制：Linux（KPTI）不具此行为；iOS 同 XNU 可用（已在 iPhone 16 Pro/A18 Pro/iOS 26.3 验证）。

涉及论文标题：
- Towards Practical Interrupt Side-Channel Attacks on macOS for Apple Silicon
