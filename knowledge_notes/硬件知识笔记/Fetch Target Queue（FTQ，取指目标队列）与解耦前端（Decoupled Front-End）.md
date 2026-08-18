## Fetch Target Queue（FTQ，取指目标队列）与解耦前端（Decoupled Front-End）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FTQ 是当代高性能 CPU 前端中位于 BPU（分支预测单元）与 IFU（指令取指单元）之间的队列，实现二者的解耦：每个 FTQ 项代表一个 fetch block（在预测的 taken 分支处结束，或达到最大尺寸如一行指令），BPU 预测下一基本块的起始/结束地址并追加到队尾，取指流水线从 Fetch Head 弹出 cacheline 对齐地址向 cache 层次发需求请求。解耦的好处是预测与取指并行推进（深度推测），前端重定向（re-steer，由 BTB miss 或误预测引起）时从 FTQ 对应位置重新取指。Bumper 论文的 baseline 为 32-entry FTQ（Table III）；FTQ 尺寸是 FDIP 激进程度的调节旋钮——FTQ-Size-OPT（每应用离线最优 FTQ 尺寸）平均为 26，而 Bumper+FTQ-Size-OPT 的平均最优尺寸增大到 51，因为 Bumper 降低了 L2C 污染，允许更深的推测窗口。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程（结合 Bumper Fig.1/Fig.9）：BPU 预测基本块 → 追加 FTQ 项 → Fetch Head 弹地址发需求取指 → FDIP 的 Prefetch Head 用剩余 L1I 带宽沿更远的 FTQ 项发预取；若后续发现预测错误，FTQ 从错误点被冲刷。Bumper 的观察：>50% 插入 L2C 的代码行来自随后被 FTQ 重定向冲刷的错误路径（取指或预取），是 L2C 污染的根源。Bumper 的 Hint Request 复用 FTQ 请求路径访问 L1I tag（Fig.9 ⑤），并在 FTQ 请求未占满 ITLB 带宽时机会式翻译（Fig.9 ④），不抢占关键取指带宽。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现为 FIFO（环形队列），每项含 fetch block 起止地址与元数据（如预测分支信息）；常见于 Intel/ARM 高端核。UDP 论文（ISCA'24）的 UFTQ 机制动态调整 FTQ 尺寸，以平衡 FDIP 的预取覆盖与精确度；Bumper 论文以每应用离线最优 FTQ 尺寸（FTQ-Size-OPT）作为 UFTQ 动态尺寸的上界参照。

涉及论文标题：
- Bumper: Hinting Instruction Usefulness for Robust Unified Caches
