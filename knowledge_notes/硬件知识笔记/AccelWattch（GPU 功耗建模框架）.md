## AccelWattch（GPU 功耗建模框架）

术语解释
AccelWattch（MICRO 2021，Kandiah et al.）是基于 Accel-Sim/GPGPU-Sim 的 GPU 功耗建模框架，用合成得到的每访问能量（per-access energy）与活动统计把逐周期性能模拟扩展到功耗/能量估计，常用于加速器与 Tensor Core 扩展的能耗评估。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
逻辑链：GPU 功耗需要知道各硬件单元（SM 流水线、寄存器堆 RF、cache、互连、Tensor Core）每类访问/活动的能量与动态活动计数；AccelWattch 在 Accel-Sim 逐周期模拟之上跟踪单元活动，乘上校准后的 per-access 能量得到功率/能量；论文用它配合 MXFFP 的 RTL 综合功耗数字评估 MXFFP vs MXFP vs BF16 的 GEMM 能量分解。Web 证据：https://github.com/accel-sim/accelwattch（Kandiah et al., "AccelWattch: A Power Modeling Framework for Modern GPUs", MICRO-54）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
论文用法（GEMM 能量分解，Fig.19）：输入 = Accel-Sim（RTX 5090 派生配置）模拟 MXFP/MXFFP 4/6/8-bit GEMM（256/512/1024 矩阵）产生的活动统计 + RTL 综合（Design Compiler + FreePDK 45nm）得到的 MXFFP 新增逻辑（bit mapper、converter、扩展 dot-product）功耗数字；AccelWattch 输出按单元（RF、core execution、cache、DRAM 等）分解的能量。结果：MXFFP 与 MXFP 能量几乎相同（新增元数据与映射/转换逻辑可忽略），RF 与 core execution 主导能量；4-bit 1024³ GEMM 总能量降至 BF16 的 0.35×（更窄数据通路、更少操作数流量与更快执行）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：作为 Accel-Sim 配套工具，从模拟器收集单元活动计数，与面积/功耗标定数据（可来自 RTL 综合如 FreePDK、CACTI、Cadence Joules）相乘汇总；本论文把综合得到的功耗数字注入能量模型。使用：比较不同低位宽格式/数据通路扩展的能耗与分解、做 energy-delay 权衡；局限：依赖模拟活动与实际功耗标定的一致性，硬件未实现时需综合估计。

涉及论文标题：
- MXFFP Microscaling Flexible Floating Point Format for Large-Scale AI Model Acceleration
