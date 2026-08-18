## LIT / LIS（Loop Information Table 循环信息表 / Loop Information Segment 循环信息段）

术语解释
LIT 是 CPU 上按 PC 查找的硬件表，保存编译器生成的循环描述符（header PC 与 exit 分支）；LIS 是二进制中承载这些描述符的专用段，由 LLVM pass 生成并经 profile 排序，O/S 调度进程时装入 LIT。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
逻辑链：(1) 编译器视角的循环：每个循环有唯一 header block（单一入口、支配循环内所有块），迭代计数 = header block 在退出前被执行的次数；(2) 层级循环迭代标识符的维护需要知道三条"特殊控制流边"：enter loop（循环外 → header）、continue loop（循环内 → header）、exit loop（循环内 → 循环外，带 pop amount）；(3) enter/continue 边统一由"header PC"表示（到达同一 header PC 时用栈顶 header PC 对比区分 enter 还是 continue），exit 边由 {分支 PC, exit 方向, pop amount} 表示；(4) 循环描述符格式 {header PC, 出口数, exit1, exit2, ...}，全部描述符排进二进制的 LIS 段；(5) CPU 侧 LIT 分 LIT-H（150 项 × 62 bit，按 PC 查 header）与 LIT-E（300 项 × 66 bit = PC:62, dir:1, popcnt:3，按 PC 查 exit 分支）；(6) ISA 规定 LIT 存在与格式，容量由 CPU 定并在配置寄存器中暴露给 O/S；O/S 在进程调度时从 PCB（扩展了 LIT 副本）把 LIS 顶部描述符装入 CPU LIT；SMT 核均分 LIT 分区。LLVM pass 用 profiling（header PC 动态出现次数）给描述符排序（最热循环在前），profiling 依据与"H2P 分支"无关、只看 header 频率。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
LIT 是 SS 的输入源：fetch 每周期对取到的分支/目标做 LIT-H 与 LIT-E 查找，命中信号驱动 SS 状态机（见签名栈条目 Table I）。运转流程例子（bfs BUStep，外层循环 B 内层循环 D）：fetch 观测边 A→B，LIT-H 命中 header B 且 hpc 不同 → hit_header(B) → SS push + sig 更新（enter B）；观测 G→B → hit_header(B) 且与 hpc 相同 → sig 旋转（continue B）；观测 D→F，LIT-E 命中 exit 描述符且预测方向匹配 → hit_exit(1) → SS pop 一次（exit D）。重叠特殊边（同一跳边同时是多个 exit 或 exit+enter/continue）按"先 exit 后 enter/continue、exit 取最大 pop amount"处理。容量敏感性：150/300 的 LIT 相对无界 LIT 仅对少数 benchmark 造成 ≥0.1% 的收益差。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现链路：LLVM 编译器 pass（基于 release/16.x commit 464bda7）分析 CFG 生成循环描述符 → assembler 写入 LIS 段 → O/S loader 按容量截取顶部描述符写入 PCB → 调度时用 load+move 指令对把 PCB 副本装入 CPU 的 LIT-H/LIT-E。本文用 -O3 编译；SPEC profiling 用全部 train SimPoint，GAPBS 用 Kronecker/random 合成图（2^19 顶点）与 road-PA 全跑。开销：LIT-H 1162.5B + LIT-E 2475B（约 3.6KB，占 11KB 总预算的 1/3）。LIS 大小由编译器决定、与 CPU 表容量解耦。通用背景：LLVM 原生有 llvm.loop.* 元数据（并行性/向量化提示），本文把类似思路扩展到"循环结构描述符直供硬件"，属编译器-硬件契约（software-hardware contract）类机制。论文未给出该 pass 的公开代码（模拟器亦不开源）。

涉及论文标题：
- Augmenting the Branch Predictor with a Squashed-Branch Reuse Buffer
