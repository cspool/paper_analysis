## 指令阻塞策略（hazard-aware blocking policy，DS-ISA 控制器调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DS-ISA 控制器 Instruction Scheduler 处理指令间依赖的显式规则集。它把 9 条指令按功能分四类——data-related（N/C_LOAD、N/C_STORE）、evolving（N/C_EVOLVE）、config（CFG_CONN）、locking（N/C_LOCK）——并用内部 scoreboard 跟踪在飞指令状态，按三类关系调度：红色 Blocking = 无条件硬停；黄色 Conditional Blocking = 满足特定条件才停（控制/数据冒险）；绿色 Non-Blocking = 独立可并行。规则：(1) data↔data 红阻塞（Data Memory 与 DAC/ADC 阵列的结构冒险，串行访问）；data→evolving 黄阻塞（RAW/WAR）；data↔locking 绿（数据与掩码走分离的并行通路）。(2) evolving 执行期间，访问或重配置演化中组件的后续指令黄阻塞。(3) config 最严格：CFG_CONN 重配置连通性有全局扰动风险，对 data 与 evolving 全部红阻塞，仅对 locking 绿。(4) locking 最宽松：对 data/config/locking 全绿（并行建状态），唯一关键交互是对后续 _EVOLVE 的黄阻塞（RAW，保证 _LOCK 掩码写完才能触发）。与传统流水 stall 不同，其依赖源于物理演化过程与全局配置的时序约束，而不仅是寄存器/内存数据依赖。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
控制器前端调度流程（Fig.8/9）：主机指令流 → Instruction Buffer → Instruction Decoder 解析 → Instruction Scheduler 查 scoreboard 决定发/停。例子（ML 训练一轮的合法重叠）：N_LOAD(新数据) 与 C_LOCK 同时发出（绿）→ N_LOCK 后跟 N_EVOLVE 需等待掩码写完成（黄）→ C_EVOLVE 演化期间 N_STORE 若访问演化中的耦合组则等待（黄）→ 下一轮 C_LOAD 与正在进行的演化黄阻塞；若中途插 CFG_CONN 改拓扑，则其后所有 data/evolving 红阻塞直至配置完成。对照评估：serialized baseline（模拟先前无重叠执行的 DSU 控制器，每条指令后强制阻塞）延迟更高——本策略靠绿放行让独立指令跨阶段重叠，Fig.15 显示四类负载延迟一致降低；Fig.14 显示迭代类负载 stall 占比 <25% 且随问题规模下降（推理的 C_LOAD 结构冒险因 100ns 演化太快摊销不足而偏高，但趋势仍下降）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：scoreboard 表跟踪每条在飞指令的目标组件与类别，按四类关系查表决定 stall；数据与掩码两套物理通路是绿放行的硬件前提。使用方式：程序无需显式插入屏障——调度器自动保证"标签先于触发、演化期间不可改配置/数据"的正确性，同时自动重叠独立指令；面向未来编译器时，阻塞策略即指令级并行度的机器约束模型。局限：论文未给出 scoreboard 面积/功耗分解（含在控制器总 4.6W 内）。

涉及论文标题：
- DS-ISA: Instruction Set Architecture for Dynamical System Units
