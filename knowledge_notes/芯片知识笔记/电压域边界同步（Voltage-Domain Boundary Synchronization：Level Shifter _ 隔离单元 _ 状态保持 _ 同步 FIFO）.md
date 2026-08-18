## 电压域边界同步（Voltage-Domain Boundary Synchronization：Level Shifter / 隔离单元 / 状态保持 / 同步 FIFO）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 芯片划分为多个独立电压域后，跨域信号线必须处理两件事：(1) 电平转换（Level Shifter, LS）——把源域电压摆幅的信号转换到目标域电压摆幅；(2) 隔离与状态保持——域关断/降压时用 isolation cell 隔离输出、用 state-retention flop 保存状态；(3) 时钟/相位同步——不同电压域可能不同频率/相位，需要同步 FIFO（如 gray-coded 指针异步 FIFO）在跨域边界缓冲数据。这是多电压域设计的固有"税"：没有它，域间数据在 V/f 切换时损坏。PowerWeave 把它建模为空间 DVFS 面积成本的最大项。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程（PowerWeave 模型）：SM 侧 L1 与芯片级 L2 之间的跨界数据路径——取上界 L1_L2_BITWIDTH=2048 bits（文献报道 L1 带宽 128–256 B/cycle）；实例化 2048-bit 宽 gray-coded 异步 FIFO + 电平转换器 + 隔离单元 + 状态保持 flop；DVFS 域链路按 AXI-like 通道建模，V/f 变化前 de-assert READY 使流量 quiesce，故 FIFO 只需吸收短 pipeline/CDC 延迟与突发尾部，深度 64 条目/通道为安全上界；在 130nm 综合并 place-route 得到具体面积，再按数字/模拟缩放因子缩到 5nm：A_cross,5nm = a_FIFO,130nm/S_dig + a_LS,130nm/S_ana；N 域增量 A_LS(N) = A_cross,5nm × (N−1)。结果：每额外域 0.0359 mm²（占 die 0.00224%），比 DLDO（0.0023 mm²）+ PLL（0.0036 mm²）合起来大约一个数量级——空间 DVFS 的主要硅成本在隔离与同步逻辑而非稳压/时钟。
- 功耗：边界 crossing 逻辑 130nm 综合，148 域保守上界 5.65W。设计权衡：per-SM 148 域时 boundary sync 主导总开销，故论文建议 per-GPC 粒度可捕获大部分节能收益、避免大部分跨界复杂度。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：跨界同步是 SoC 多电压域设计的标准构件（对应 AMBA AXI VALID/READY 双向流控的 quiesce 语义）。论文基于 Mr.Wolf PULP SoC [45]（https://github.com/pulp-platform/mr_wolf）的异步 FIFO RTL 与 130nm PDK 综合，用 mflowgen 做 place-route，ASAP7/IRDS 缩放至 5nm。用于回答"空间 DVFS 硬件是否划算"：即使最细 per-SM 粒度，boundary sync 0.0359 mm²/域、148 域总计 <0.5% die 面积，换取 30–40% 级能量节省。

涉及论文标题：
- PowerWeave: Unlocking Energy-Efficient ML on GPUs with OS-Level Spatial Power Management
