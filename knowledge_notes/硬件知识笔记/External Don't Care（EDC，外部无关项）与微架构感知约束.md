## External Don't Care（EDC，外部无关项）与微架构感知约束

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
External Don't Care（EDC，外部无关项）是逻辑综合中 Don't Care 的一种，由用户规格引入：声明某些输入模式永远不会出现（可控性 DC，CDC_in——环境不会产生的输入）或某些输出不被观测（可观测性 DC，ODC_out）。经典 Don't Care 分三类（De Micheli《Boolean Methods》；Savoj & Brayton ICCAD-91 "Observability relations and observability don't cares"）：SDC（Satisfiability DC，由 netlist 结构拓扑导致的不可达输入组合）、ODC（Observability DC，节点变化在输出不可观测的输入模式）、EDC（外部明确给出的约束）。EDC 的关键价值：把优化问题从"完全指定布尔函数的最小化"转变为"布尔关系（Boolean relation）的最小化"——函数在这些输入上可取任意值，从而获得额外门级裁剪（Savoj & Brayton DAC 1990 "The use of observability and external don't cares for the simplification of multi-level networks"；EPFL/De Micheli EXTDC23 形式化为 EXOEC 等价类）。文献同时指出：per-output EDC 是布尔关系的特例且不完备（完整灵活性需布尔关系表示），EDC 与 ODC 组合使用可使节点内每条连接变不可冗余（可测性完备）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
æSIP 首次把 EDC 从逻辑综合系统性地引入处理器微架构裁剪（µArch-aware 约束），分三级（论文 §IV-C1，Fig.7）：
- ISA 级：decoder 只允许重写程序出现的 opcode，如 `assume ((i_rdata[1:0] != 2'b11) || (((i_rdata[31:0] & 32'hfe00707f) == OP_ADD) || (... == OP_MUL)))`——比 PDAG 的 ISA 约束更进一步（直接注入 netlist 作为 assume）。
- 数据级：操作数宽度/立即数取值受限，如 `assume ((opcode == SRAI) |-> (shamt inside {1, 2}))`（桶形移位器只需支持 1/2 移位量）；mul 只处理 16-bit 操作数时高位乘法逻辑冗余。注意：操作数宽度约束需动态 profiling（cycle 级仿真）精确刻画，论文为保证通用性只用静态可推导约束。
- 时序级：访存子系统时序假设，如 `assume (dcache_miss |-> ##[1:5] dcache_response)`（dcache miss 5 cycle 内响应）——简化 memory 接口状态机，删掉超出规格的状态/转移。
运转流程：静态分析重写汇编提取 opcode/立即数使用 → SVA generator 转成 SystemVerilog assume（注入 baseline netlist）→ abc scorr/dsec 在 EDC 约束下 k-induction 证明节点等价/冗余 → 裁剪。消融实验（Fig.9）显示 µArch-aware 约束（ISA+数据+时序）在保留复杂指令的 workload（dijkstra、rijndael）上比仅 ISA 级多降面积（桶形移位器与访存子系统简化是额外收益来源）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：经典综合工具中 EDC 由用户约束文件声明；æSIP 自动化——静态分析自动导出 ISA/立即数约束，SVA 生成器（论文自研）转 SystemVerilog assume，用户只需提供内存配置（时序约束来源）；裁剪引擎为 abc 的 scorr/dsec（k-induction，见独立条目）。使用注意：时序约束与 baseline 微架构/应用负载强相关，用户必须验证约束有效性（无效约束会在验证集外场景产生错误行为）；EDC 裁剪在 and-inverter graph 层操作，下游综合工具可能无法从裁剪后表示恢复更小 netlist（论文观察到个别 variant 面积反而微增）。参考：Savoj & Brayton（DAC 1990/ICCAD 1991）、Marakkalage et al. DATE 2024 "Scalable sequential optimization under observability don't cares"。

涉及论文标题：
- æSIP μArch-aware ASIP-ISA Co-Design via Program Synthesis, Equality Saturation, and External Don't Cares
