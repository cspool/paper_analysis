## BLIMP（Bank-Level In-Memory Processor，bank 级存内处理器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BLIMP（Bank-Level In-Memory Processor）是一类 PIM 架构：在 DDR DRAM 每个 bank 内部署小型通用（通常 RISC-V）处理器，核心通过本地 bank 的 row buffer 读写数据。BLIMP 是工业界近期关注的 PIM 方向（论文引商用 BLIMP 类平台 [3][24]，如 UPMEM、GDDR6-AiM），因处理器紧邻存储电路、访问时延低且对存储单元密度影响小（论文引 prior work 显示 bank 面积开销 <4%）。BLIMP 核限制在本地 bank 内存内工作；目前无广泛使用的 DRAM 规范支持 bank 到 bank 的远程通信而不经 host，因此 BLIMP 核之间的数据传递是 CPU 驱动的 read-then-write 操作。论文定义两种核：BLIMP-S（scalar）——每 bank 一个单线程 200MHz RISC-V RV64GC 处理器（1KB 指令缓冲、1KB scratchpad、5×1KB R/W buffer）；BLIMP-V（vector）——RV64GCV 处理器带本地向量引擎（32×64b vALU、5×1KB vector register），可执行 RISC-V "V" SIMD 指令。同组开源框架 dovedevic/blimp（https://github.com/dovedevic/blimp，"A PIM instrumentation, compilation, execution, simulation, and evaluation repository for BLIMP-style architectures"）提供 BLIMP 的编译、relayout、仿真与评估工具链。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
论文配置：PIM 使能 16GB（8GB×2）2133 DDR4 DIMM，2 channel、2 rank、8 chips/rank、16 banks/chip，共 512 个 BLIMP 核，每核 32MB bank、1KB row buffer（tRP=tRCD=21ns、tRFC=640ns、tREFI=7.8us）。运转流程（一个 offload）：host 把 kernel 可执行体、PIMDT 列分区与辅助数据（如 host 建好的哈希表）经 relayout 载入各 bank → BLIMP 核进入 compute mode 接管本地 bank → 核以 1KB row buffer 粒度 FetchMem 读入数据、执行 RISC-V（-V）指令逐元素计算 → 结果写回指定输出区 → 计算结束核让出控制，host 才能访问该 bank 数据。由于核间无直连，join 的哈希表由 host 构建后复制广播到各 bank；哈希表超过 32MB bank 容量时按分区多轮 build-probe。评估时每次 offload 只模拟单个 bank 的执行，假设数据同质、计算对称、总时延等于最慢 bank；BLIMP-V 相对 BLIMP-S 整体 1.7×（select 场景 2.5×、join 为主场景 1.4×），向量引擎主要加速可向量化的 hash 计算与逐元素谓词。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
BLIMP 的实现/评估依赖 validated cycle-level simulators（prior work [25] ISCA'22 与 [2] riscvovpsim/Imperas RISC-V ISS，https://github.com/riscv-ovpsim/imperas-riscv-tests）+ DRAMSim2（https://github.com/umd-memsys/DRAMSim2）DDR4 时序建模；同组 dovedevic/blimp 仓库的 /simulation、/compilation、/relayout 目录实现 BLIMP 风格仿真流程（instrumentation → compilation → execution → simulation → evaluation）。商用 BLIMP 类平台（UPMEM DDR4-PIM 的 DPU、SK hynix GDDR6-AiM、Samsung HBM-PIM）已在市场出现，但论文选择仿真以支持向量引擎等可配置硬件与细粒度参数。使用场景：OLAP 数据库的 select/join/aggregate 等数据并行、可向量化、无跨数据依赖的算子（论文 SSB SF100 端到端相对 DuckDB 3.1×/5.8×）；而排序、全局依赖聚合、字符串谓词等跨数据依赖/高单元素计算开销的算子不适合 BLIMP，应留在 host。

涉及论文标题：
- Taking Analytic Databases to the Bank
