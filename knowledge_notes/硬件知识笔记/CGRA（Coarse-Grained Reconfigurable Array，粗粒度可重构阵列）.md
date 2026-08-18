## CGRA（Coarse-Grained Reconfigurable Array，粗粒度可重构阵列）

术语解释
由字级（word-level）处理单元（PE）阵列与可重构互连组成的空间可重构计算结构：配置位流同时决定 PE 功能与互连布线，在能效、灵活性与性能间折中；按时空资源维度分 spatial-only / temporal-spatial，按调度方式分静态调度 / 动态调度。DICE 采用硬件开销最小的"静态调度 + spatial-only"组合。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CGRA 是介于 FPGA 与 ASIC 之间的可重构计算结构：PE 执行字级算术/逻辑运算，互连（crossbar、switch box、wire-switched 直连）路由 PE 间数据流，两者由配置存储器（CM）中的位流统一编程。粒度比 FPGA（位级 LUT）粗，故重配置快、能效高；比 ASIC 灵活。程序映射逻辑链：程序 → 控制数据流图（CDFG）→ 划分基本块/子图 → 每个子图编译为纯数据依赖的 DFG → placement & routing 生成位流 → PE 间数据流直传。两个关键设计维度（DICE 论文 II-B）：① spatial-only（单一配置贯穿执行、每 PE 固定一个操作，如 Amber/SNAFU/RipTide）vs temporal-spatial（每 PE 存多配置字每周期轮换、时间复用资源，如 HyCUBE/CGRA-ME，[56] 测得相对 spatial-only 基线 +160% 面积开销）；② statically scheduled（编译期固定执行顺序，无弹性互连）vs dynamically scheduled（操作数就绪即发射，需 valid-ready/credit 流控（+110% 面积）或 tagged-token 数据流（+480% 面积））。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
DICE CP 内实例：4×5 CGRA（16 个 ALU PE + 4 个 SFU），wire-switched 静态互连（类 AHA），每 PE 带 1-bit 谓词输入；配置经双缓冲 CM0/CM1 装载（一执行一装载，隐藏重配置延迟）。执行流程：位流装载完成 → Dispatcher 按 tid 升序以 II=1 每周期派发活跃线程 → 线程沿 PE 流水逐级推进，中间值 PE 间直连转发（不经 RF）→ 输出直写 RF 或进入 LDST Unit 访存 → e-block 完成后下一位流切换。p 级深度的 p-graph 执行 t 个线程耗时 $T = t + p$ 周期（无 stall 时，多出 p 为流水填充/排空气泡），数百线程摊薄 p/t 开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：PE 按功能分型（INT/FP/SFU）、可配置互连（静态 wire-switched 如 AHA/Amber，或时间复用 crossbar 如 HyCUBE）、CM 组织（单缓冲/双缓冲）、位流装载通路（从片上缓存取入 CM）；映射工具链做子图划分与布局布线（ILP/SAT/启发式）。使用场景：规则数据流、空间并行度高的负载（DSP、线性代数、脉动式流水）；DICE 证明其可承载通用 SIMT 负载——代价是 p-graph 划分约束与谓词/访存边界机制。相关开源生态：OpenCGRA、CGRA-ME、AHA 等。Web sources：Wikipedia CGRA 词条；Liu et al. CGRA Survey（ACM Comput. Surv. 2019）；Van Essen FPL'09《Static versus scheduled interconnect in CGRAs》；AHA（Koul et al., ACM TECS 2023）。

LoRA 的异构 CGRA 实例（ISCA'26）：通用 CGRA 主要支持线性运算，无法高效处理非线性函数（激活函数、三角函数、幂/对数），传统做法要么卸载到 host CPU（性能退化）、要么加专用 IP（硬件开销随函数数增长）、要么用 PICACHU 式 Taylor+MAD 展开（精度越高 MAD/PE 占用越多）。LoRA 在通用 CGRA 上集成可重构非线性功能单元 XCore-PE（见 XCore 条目），使 CGRA 直接执行非线性函数。CGRA 组件：PE（普通计算，partial-prediction 处理分支，需额外细粒度输入输出做 select）、XCore-PE（复杂运算：多项式、x^y）、IOB（输入输出块，与 SPM 交换数据，loop 元信息隐式编码在访存中由 IOB 控制器管理，仿射访问由配置自动生成 SPM 地址、非仿射访问由其他单元算地址经第二输入喂入）、GIB（通用互连块，port-to-port/track-to-port/track-to-track 三类互连，track-to-track 按 Wilton 模式支持长距离通信）。每个 FU 加载一个配置执行所有迭代再切换配置（自带配置内存），FU 内寄存器同步器保证数据同步；CGRA 尺寸/PE 类型/每 PE 支持运算均可参数化。配置流程：LLVM 前端生成 DFG → 模拟退火空间映射 + 内存分区算法 → 生成配置，经 RoCC 指令下发，数据经 L2↔SPM（TileLink+DMA）交换。评估配置：6×6=36 PE + 12 IOB + 12×4KB SPM bank，L2 128KB，其中 2 个 XCore-PE；与 PICACHU 相比 LoRA 平均 2.18× 性能、2.13× 能耗效率，与 STM32H750 MCU 相比 23.33×（见 LoRA 论文 VIII-C）。

涉及论文标题：
- DICE: Enabling Efficient General-Purpose SIMT Execution with Statically Scheduled Coarse-Grained Reconfigurable Arrays
- LoRA: Towards Improved Applicability of Reconfigurable Architecture for Versatile Nonlinear Functions
- Optimizing Spatial Data Structure with Near-Cache Acceleration by Exploiting Physical Locality（RoboCortex）

RoboCortex 视角（ISCA'26，近缓存数据流搜索加速）：RSU（Reconfigurable Searching Unit）即一个 8×8 可编程 CGRA 空间阵列，紧邻 L1 缓存部署，把空间数据结构（Kd-Tree/Octo-Tree/R-Tree）最近邻搜索的 DFS 计算图映射到 PE 上——每个数据流原语对应一个 PE（Read/Write 除外，经 LSU 直连 L1）。与通用 CGRA 的关键差异：①新增显式 Stack 与 Priority Queue 数据流组件支持递归/回溯（Join 端口顺序 + LIFO 表达 DFS 分支优先级），解决"现有数据流加速器不支持显式递归"的问题；②新增 Reg 原语（Enable 为真更新、否则持续输出旧值并冲刷下游队列）维护共享 ResList 的 MaxD，解决搜索函数非纯的问题；③采用 flexible 2D mesh NoC（非固定数据流方向）提升映射灵活性——映射只影响性能不影响功能；④编译期用开源 CCF CGRA 软件栈 + 模拟退火做静态映射，把一次搜索拆成三部分（叶节点距离计算 / 子节点选择 / NeedExpand+递归），并加 pragma 指示划分。面积：RSU CGRA 8×8 fabric 28nm 综合 1.195 mm²（占 RSU 98.19%），配合 16-entry Path Buffer（0.344mm²）总开销约 512B，8nm 缩放到 ~2.3mm²，适合 Jetson Orin 级 SoC。
