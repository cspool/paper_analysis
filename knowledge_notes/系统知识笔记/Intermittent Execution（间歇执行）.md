## Intermittent Execution（间歇执行）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 间歇执行是 EHS 独有的程序执行模型：程序不是在连续供电下从头跑到尾，而是被频繁断电切成"运行一小段（功率开启周期）→ 断电休眠充电 → 恢复继续"的循环。论文引用 [68] 将其定义为 EHS 程序行为特征。由于每次断电会丢失易失状态（寄存器、SPM），系统必须实现某种崩溃一致性（通常为 JIT checkpoint/restore）才能在唤醒后从断点继续。
- 对 MANATEE 而言，间歇执行的关键性质是：给定电容尺寸，执行距离（execution distance）往往一致 [54]，因此功率开启周期 T_on 可被预测，这是"speculative（投机）"页着色能够成立的前提——编译器按推测的 T_on 设置滑动窗口，仅在窗口内保证页冲突自由。
从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程例子：一次功率周期 = 唤醒（电容满）→ 执行 N 条指令（访问若干 NVM 页，页在 SPM 中驻留/交换）→ 电压降至 V_backup → JIT checkpoint（按 WTQ 持久化脏页 + 保存寄存器）→ 休眠。MANATEE 的投机点：编译器假设"断电会在冲突点前后某处发生"，使被断电隔开的两个页可共享同一 SPM frame；若实际没断电（misspeculation），两页同周期访问同一 frame，由运行时 page manager 加密驱逐/解密载入兜底，保证正确性。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：电压监视器在 V_backup 触发 checkpoint、在 V_restore 唤醒；运行时按完成标志位（NVM 中 0→1）判定 checkpoint 是否完整。评估上，论文用 1mF 电容 + 三条真实功率 trace 复现间歇执行，测得平均功率开启周期 thermal 2701.7ms / RFHome 2662.8ms / solar 2680.0ms，并做 100µF/1mF/10mF 电容灵敏度实验。论文未给出公开代码，无法确认是否开源。
涉及论文标题：
- Intermittence-aware Speculative Page Coloring for Secure NVM
