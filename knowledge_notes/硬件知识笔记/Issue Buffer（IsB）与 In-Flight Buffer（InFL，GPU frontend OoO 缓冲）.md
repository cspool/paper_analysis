## Issue Buffer（IsB）与 In-Flight Buffer（InFL，GPU frontend OoO 缓冲）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Issue Buffer（IsB）是 frontend OoO 方案（类 GhOST/SIMIL）在 Issue 阶段引入的指令重排缓冲：缓存译码后的指令条目，每个条目经 Dependence Checker 标注依赖位向量，由 Issue Arbiter 从"独立指令"池选择发射，从而让独立指令越过依赖未决的旧指令。In-Flight buffer（InFL）跟踪已越过 IsB、尚未写回的在飞指令的目的寄存器：每项含分配位 + 目的寄存器字段（隐含 warp ID），用于对 IsB 内指令做依赖检查（后续指令与在飞指令的 RAW 依赖）。逻辑链：baseline GPU 的 Issue 阶段按序发射 + scoreboard 阻塞，一条依赖未决的旧指令会把流水线卡住，即使新独立指令可执行；IsB+InFL 把"依赖检查"与"发射选择"解耦，形成前端重排窗口。sCROOGe 相比 GhOST 的简化：省略 IBuffer（Fetch/Decode 每 cycle 一条指令，吞吐足够）与同步控制 flag（同步调度由 Schedule 阶段处理）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
sCROOGe frontend 指令流：新译码指令由 IsB 分配仲裁器放入 IsB（需至少一个空项 + UUID 越界条件满足，否则留在供给寄存器）→ 下一周期 Dependence Checker 处理该指令并更新其依赖位向量（RAW：3 个 rs 与 InFL/IsB 所有 rd 比较；WAW：选中 rd 与其他 rd 比较；WAR：选中 rd 与 IsB 内 rs 比较，因 Issue 之后执行有序、无需查 InFL）→ "独立"与"per-warp 最老"位向量由顺序电路从依赖向量 + UUID 生成（非最老的访存指令被判为不独立）→ Issue Arbiter 在有空 InFL 项（或无需写回）时选独立指令发射，无独立指令时退回选 UUID 最小的最老指令 → 发射后清 IsB 内所有指向该指令 ID 的依赖位、更新 InFL 对应位 → 下一周期 InFL 项分配、IsB 项腾空 → 执行后 eop 到达 Writeback，刷新 InFL 项并清依赖位。设计优化：只把独立指令供给 arbiter 额外带来 2% 增益，并允许移除"per-warp oldest"缓冲、scoreboard 的 SRAM 阵列及伴随逻辑。评估：IsB 4/8/12 项扫描，frontend 平均 6.7% 加速、面积开销峰值 7.5%（功耗 8.2%），小配置甚至出现减速（额外流水级成本超过重排收益）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：SystemVerilog RTL；IsB 每项含指令 + 依赖位向量 + UUID；InFL 每项含分配位 + rd（隐含 warp ID）；配套 UUID 生成单元与两个顺序电路（独立/最老位向量生成）。使用：配置项 IsB 项数（4-12）进入设计空间扫描（ADP/EDP FoM，最优 IsB 数随 {warp,thread} 变化，高 ILP 负载最优 IsB 比低 ILP 高约 0.75）。Web 证据：GhOST（ISCA'24）与 SIMIL（Microprocessors and Microsystems 2024）为 frontend OoO 的仿真版先导工作，sCROOGe 的 RTL 实现差异（省略 IBuffer、调度优化移除 scoreboard）论文已详述。

涉及论文标题：
- sCROOGe Circuit-level Design and Optimization Framework for RISC-V Out-of-Order GPUs
