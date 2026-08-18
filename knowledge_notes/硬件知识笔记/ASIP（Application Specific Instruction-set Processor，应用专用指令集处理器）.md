## ASIP（Application Specific Instruction-set Processor，应用专用指令集处理器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ASIP（应用专用指令集处理器）是为特定应用领域定制 ISA 与微架构的处理器：通过牺牲通用性换取针对目标负载的面积/功耗/能效/时延优化。与通用处理器（GPP，面向任意程序）和固定功能 ASIC（只跑单一算法、不可编程）不同，ASIP 保留可编程性（域内不同程序可重新编译运行），但 ISA/微架构针对领域特征裁剪。动机来自 Dennard scaling 终结后嵌入式领域（植入式、IoT、印刷/柔性电子、低温量子控制）严格的功率/面积约束：如稀释制冷机在 millikelvin 级仅提供微瓦级冷却、脑组织热耗须低于约 40 mW/cm²、印刷电子（1-10 µm 特征尺寸）晶体管预算比硅低 2-3 个数量级。历史上 ASIP 设计走"加性"路线（在基础 ISA 上添加自定义扩展指令提升性能，如 Xtensa [31]、MESCAL 方法学 [33]），但加指令必然使核更大而非更小；æSIP（ISCA 2026，Univ. of Michigan）是首个"系统性硬件-软件协同设计"的 ASIP 自动生成框架，走"减性"路线（重写程序 + 裁剪硬件）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
ASIP 在硬件架构层面的运转流程（以 æSIP 用 Ibex RV32IM 生成 ASIP 为例）：输入 = 目标应用 + baseline 通用处理器 RTL（Ibex Small 配置，2-stage pipeline、无分支预测、RV32IM）+ 内存配置 + 时延约束 → ① 程序重写收窄 ISA 足迹（mul/mulh/div 重写为 add/shift/分支序列，distinct 指令平均降 31.8%、Mul/Div 全消除）→ ② 提取微架构约束（ISA 级 opcode 白名单、数据级 shamt/立即数受限、时序级 cache miss 周期）→ ③ SVA assume（EDC）注入 netlist → ④ abc scorr/dsec 在约束下 k-induction 证明并裁剪门 → 产出裁剪后的 ASIP RTL。裁剪后的 ASIP 不新增指令（保持 RV32IM）、不改 pipeline 深度/发射宽度/分支预测，与原 RISC-V 工具链完全兼容（可编译/仿真/验证）。评估流程：Yosys/OpenSTA 快速综合（面积/频率，扫 27 个 λ 的变体）→ Verilator cycle-accurate RTL 仿真（时延，cache 开启）→ Cadence Innovus PnR（SkyWater SKY130 130nm iso-frequency 25 MHz 与 PPDK 印刷电子标准单元库，输出物理面积/功耗）→ spike+pk 端到端功能验证（22 个 MiBench/EmBench benchmark 全部通过）。结果：与 SOTA 生成器 PDAG 相比面积几何平均降 17.0%（SKY130，无约束）、功耗降 12.3%；1.2× 时延约束下面积降 5.9%；能量最优变体降 4.1%（SKY130）/6.3%（PPDK）；Rocket 5-stage core 上面积降 24.6% 验证跨微架构泛化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式两类：①"减法"——从现成 RISC-V 核（Ibex、Rocket）自动裁剪（æSIP、PDAG [11]、Bespoke [22]、RISSP [58] 路线）；②"加性"——按 MESCAL 方法学从算法→架构→微架构逐级定制。æSIP 作为开源工具（GitHub https://github.com/CrucibleComputingGroup/aesip，GPL-3.0；artifact Zenodo https://zenodo.org/records/19560118；Docker 镜像 polasip/esip:rv32im 含 egglog/Yosys/ABC/OpenSTA/riscv-gnu-toolchain/spike/Verilator；Gurobi 与 Cadence Innovus 需商用许可）自动生成：输入程序 + baseline + 约束，输出 ASIP RTL 与可验证等价的重写程序；artifact_evaluation.ipynb 自包含复现四实验（最小指令集、多目标优化、area-latency Pareto、ecosystem 共享）。使用场景：印刷/柔性电子（PPDK 静态功耗占 99%、RISC-V 核 die 面积达 67.53 cm²，面积削减收益最大）、植入式神经接口（体积/热耗受限）、主动 RFID（能量收集，功率即生命线）、低温量子控制。ecosystem 级使用：多个相关 workload 共享少量 ASIP（num-chip=5 时 17.3% 面积降 + 11.9% 时延，NRE 摊薄）。

涉及论文标题：
- æSIP μArch-aware ASIP-ISA Co-Design via Program Synthesis, Equality Saturation, and External Don't Cares
