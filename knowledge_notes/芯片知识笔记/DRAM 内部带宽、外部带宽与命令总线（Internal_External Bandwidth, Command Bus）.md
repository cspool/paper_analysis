## DRAM 内部带宽、外部带宽与命令总线（Internal/External Bandwidth, Command Bus）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
外部带宽 = 跨内存总线的带宽（DRAM 芯片↔内存控制器，移动端 LPDDR5X 典型 <80 GB/s）；内部带宽 = bank 阵列/行缓冲到 I/O 电路或 PIM 单元之间的带宽，利用全 bank 并行可达外部带宽的数倍（[Web] 三星 Hot Chips 33：单颗 LPDDR5 芯片外部 I/O ~51.2 GB/s vs 内部全 bank ~409.6 GB/s）。命令总线 = 控制器向各 bank 发控制命令（ACT/PRE/RD/WR/PIM）的串行通路：多 bank 命令在总线上串行化，是内部带宽利用率的硬上限。COSM Eq.(1) 量化该上限：

$$Util_i = \frac{tBL + (tRP + tRCD)\cdot(1 - R_h)}{\#bank \cdot tBL}$$

Annotations：tBL=8、tRP+tRCD≈30（LPDDR5）、#bank=32（2rank×16banks）时，完全随机访问（R_h=0）Util_i 上限 = 38/256 ≈ 15%；分母 #bank 说明命令串行化使 bank 越多瓶颈越重。CPU 与 PIM 的带宽使用互补（CPU 偏外部、PIM 偏内部），是 COSM 收割空闲内部带宽的物理基础。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
bank 层级（channel→rank→bank group→bank）决定并行度，但行打开开销远大于 burst（tRP+tRCD≈30 周期 vs tBL=8 周期）使"高 bank 数"不自动等于"高内部利用率"。COSM 把命令总线视为与内/外带宽并列的第三种稀缺资源：(1) PIM 命令过短则总线饱和——命令长 ≥64 才能支撑 2rank×16banks 全 bank 并行（考虑行打开开销），≥128 才使命令总线占用 <40%；(2) 命令过长则 CPU 延迟受损——两难由可抢占 PIM 命令解决（见硬件架构库同名条目）。芯片级支撑：每 bank 旁 1kB SRAM buffer + 16-bit PIM-bank 线宽，把内/外带宽的占用在命令层面显式分离（PIM_LdBuf/StBuf 只占内部、PIM_RdBuf/WrBuf 只占外部）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：DRAM 物理组织与 JEDEC 时序（tRCD/tRP/tBL/tCCD）定义，移动端用 LPDDR5-6400（JESD209-5C）；PIM 侧按三星 LPDDR5-PIM 流片芯片参数建模（1GHz、6.4 TFLOPS、6.4 GB/s）。使用：利用率公式用于调度器设计——IWE 估计 bank/bus 空闲窗口、PIM scheduler 用 nPTL 权衡命令总线；设计启示：把"命令总线、内部带宽、外部带宽"作为三个独立资源分别调度（COSM 解耦传输命令即此思想），而非笼统的"带宽"。

涉及论文标题：
- COSM: A Cooperative Scheduling Framework for Concurrent PIM and CPU Execution on Mobile Devices
