## JIT Checkpointing（Just-in-Time 即时检查点）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- JIT checkpointing 是 EHS 最流行的崩溃一致性方案：电压监视器检测到电容电压低于 V_backup 阈值时，向核心发出即将断电信号，核心暂停程序执行、利用剩余电容能量把程序计数器（PC）与其他寄存器持久化到 NVM（或非易失寄存器），然后休眠；当电压回升到 V_restore 阈值后恢复寄存器、从 PC 继续执行。V_backup 与 V_restore 都需高于额定电压，保证 checkpoint/restore 本身不被断电打断。
- 在 MANATEE 的 SPM-NVM 层级中，JIT checkpointing 只负责寄存器/heap/stack 等程序状态，脏页持久化由 MANATEE 运行时按 WTQ（Write Tracking Queue）以页粒度完成——这是与"整 SPM checkpoint"（NVSRAM、Mapi-Pro）的本质区别：后者把整个 SPM 加密写入 NVM，能量开销巨大，无法在小电容下运行较大负载（如 MNIST/CIFAR）。
从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程例子：①电压监视器检测 V_backup → ②若正处 page manager 原子操作中，允许其先完成（V_backup 阈值被调高以覆盖"2 个 WTQ 条目 flush + page manager 调用 + 程序状态 checkpoint"三件事）→ ③按 WTQ 条目把脏页 AES-XTS 加密持久化到 NVM → ④保存寄存器/heap/stack → ⑤在 NVM 中把完成标志位翻转 0→1 → ⑥休眠。恢复时：唤醒 → 读标志位，若为 1 则复位为 0 并从 checkpoint 继续；若为 0 则判定 checkpoint 不完整、abort 防止不一致恢复或攻击。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：依赖电压监视器（voltage monitor [52,71,88]）与足够大的电容能量缓冲。MANATEE 中 checkpoint 的内容是"寄存器 + 脏页（按 WTQ）"，而非整个 SPM；这是其能量效率的关键（1mF 电容即可运行，而 NVSRAM 需要 10mF 且性能差）。论文未给出公开代码，无法确认是否开源。
涉及论文标题：
- Intermittence-aware Speculative Page Coloring for Secure NVM
