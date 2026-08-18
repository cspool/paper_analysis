## PICS（Per-Instruction Cycle Stacks，逐指令周期栈）与 TEA（Time-Proportional Event Analysis，时间比例事件分析）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PICS（Per-Instruction Cycle Stacks，逐指令周期栈）是细粒度的性能归因数据结构：为每个静态指令报告"该指令对总执行时间的贡献"，并把每条指令的时间按性能事件组合（如 L2 miss、TLB miss、分支误预测、store buffer 满）分解成栈。PICS 统一了性能剖析（profile）与事件分析（event analysis）：既报告每个静态指令占多少执行时间，又按事件组合分解每条指令的时间去向，直接告诉开发者"哪个 PC、哪种 stall 事件、占多少 cycle"值得优化。TEA（Time-Proportional Event Analysis，时间比例事件分析，ISCA 2023，Gottschall/Eeckhout/Jahre）是实现 PICS 的专用硬件方案：为每条 in-flight 指令维护一个 Performance Signature Vector（PSV，约 9 个事件位的 bit-mask），按时间比例（而非事件计数）把 cycle 归因到事件，在 BOOM 核上实现（约 249B 存储、3.2mW、1.1% 性能开销、平均 2.1% 误差），曾演示 lbm/nab 1.28×/2.45× 加速。论文把 TEA 作为"专用固定功能分析硬件"的 baseline：它对该窄任务有效但不可变、增加设计验证负担，且每类新分析都要造一个新硬件块。IPU 的 Capability 2（Software-Defined Performance Attribution）用 IPU_lite 的软件 PSV 复制 TEA/PICS 功能：每 cycle 更新 PSV、每 400,000 cycles（TEA 设计值）归并输出 PICS 条目，结果与 TEA 一致但无需 BOOM RTL 改动、且可任意扩展事件集合。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
TEA/PICS 专用硬件在 CPU 核中的运转流程：TEA 采样器以 ~4kHz 采样 ROB 与流水线事件，为每个 in-flight 指令维护 PSV（事件位图），事件发生时按时间比例把 cycle 归因到对应指令的 PSV；指令退休时若其 PSV 非零，则把（PC，事件组合，cycle 数）写入 PICS 栈表；主机侧按 PC 汇总成逐指令 cycle 栈。IPU 软件版运转流程（论文 Capability 2，gem5 验证）：HIT=CPU 核流水线，17 个信号（长延迟事件控制信号 + 4 处流水线 6 个虚拟地址 PC）=215 bits/cycle 进 IORegs → IPU 顺序核每 cycle 执行 introspection kernel：if-else-if 事件链把发生的事件置位到对应 PC 的 PSV（load-modify-store），flush 时保存 PC 供 commit 后引用，每 400,000 cycles 扫描活动 PSV 列表把（PC+签名）归并成 PICS 条目经 FIFO 发主机 → 主机后处理成表 II 格式（PC 0x7912d0: DTLB miss+DCache Miss=50000000 cycles；0x80dda0: Branch Mispredict=200000 cycles）。近似误差：单 cycle 模拟 vs 每 cycle 模拟的平均相对误差 <3%（3 个应用 10-14%），PC 排序始终正确，丢 PC 覆盖 ≤0.37% cycles。验证：3 个 DARCHR microbenchmark（https://github.com/darchr/microbench）各只期望一个 PC 出现在 PICS 栈，结果吻合。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：TEA 硬件实现位于开源 Chipyard 分支（https://github.com/uv-xiao/chipyard-ntnu，基于 BOOM + FireSim，ISCA 2023 论文与配套工具 bgottschall/pythonTools 等）；IPU 软件版用 RISC-V introspection 程序实现（75 行 _main 代码），直方图/hash/循环指令作 intrinsics。使用方式：开发者在应用源代码顶部加 API 调用（IPU_CONFIG_IMAGE("PICS-generation")、IPU_CONFIG_START(ROI_BEGIN)、IPU_CONFIG_STOP(ROI_END)）限定感兴趣指令区域，运行后得到 PICS 输出文件（表 II）用于定位慢速指令与事件根因（如 arr[lfsr].p1=lfsr 行的 STL2 导致 LSQ Full=127878 kCycles）。论文显示 IPU 版 PICS 与 TEA 专用块同等的 per-instruction 瓶颈分解，但把性能归因从 pre-silicon 固定设计选择变成 post-silicon 可编程软件任务，展示"软件定义性能归因"能力类别。

涉及论文标题：
- Enabling Continuous, In-Field Introspection: The Programmable IPU Architecture
