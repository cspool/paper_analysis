## Register Alias Table（RAT）与 Register Renaming Stack（RRS，GPU 后端寄存器重命名）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RAT 是 backend OoO 中做寄存器重命名的查表结构：把架构寄存器名映射到 RRS entry ID，从而在 OC 阶段消除 WAR 与 WAW 冒险（写后写/写后读不再冲突，因为同一逻辑寄存器被不同指令写入时各自映射到不同的 RRS 项），并用 RRS 项追踪 RAW 依赖（读方等写方的结果 broadcast）。RRS 是轻量标识符存储：为每条需写回的指令分配一个 RRS ID 存入 RAT，使 CU 能在 Dispatch 后立即释放（指令数据与结果改由 RRS 项跟踪到写回），避免 CU 长期占用。逻辑链：CPU 乱序处理器用物理寄存器堆 + Rename Map Table（见本库"Rename Map Table（RMT）"条目，CPU Issue Queue 视角）做重命名；GPU backend OoO（LOOG/sCROOGe）用"RAT + RRS 保留站式重命名"替代 scoreboard 依赖跟踪——RAT 每项含 2 个字段：寄存器数据是否需从 RF 取用的位 + RRS ID 字段。sCROOGe 与 PipeIMC 的"显式寄存器重命名"（计算 SRAM 物理 wordline 重命名）不同：后者重命名目标是存内计算的 wordline，前者是 GPU OC 阶段的保留站 ID。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
sCROOGe backend 运转流程：指令进 OC 分配 CU 与（如需要写回）RRS entry，RRS ID 存入 CU → 下一周期 CU 查 RAT：源寄存器重命名为对应 RRS 项，若该 RRS 项尚无数据（producer 未写回）则 CU 等待 broadcast；目的寄存器 rd 的 RAT 项被改写为本 CU 的 RRS ID → 后续依赖此 rd 的指令查 RAT 得到该 RRS ID，在 broadcast 总线上旁听该 ID 的结果 → Commit 后结果写入 RRS 项的专用数据字段，eop 到达时检查 RAT：若 RAT 中该 rd 仍指向此 RRS ID 则把结果写回 RF 并更新 RAT，否则由拥有同一 rd 的其他 RRS 项负责 → 下一周期 RF 写 + 所有依赖该 broadcast 的 CU 数据更新 → RRS 项可释放。以 `r1=r2+r3; r4=r1*r5; r1=r6-r7` 为例：第二条经 RAT 读到 r1 的 RRS ID，等第一条 broadcast 后执行；第三条把 r1 重命名为新 RRS ID，与第二条写 r1 的 WAW 冲突被消除，可与第二条乱序执行。量化：RRS 使无 CU 可用 stall 显著下降（见 Collector Unit 条目）；RAT/RRS 右尺寸化是 sCROOGe 的设计优化点之一。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：RAT 为每逻辑寄存器一项（含 RF-来源位 + RRS ID 字段）的多端口小表；RRS 为循环分配的轻量项池（含 ID、结果数据字段），其面积远小于 CU（0.873µm² vs 21.382µm²，见 Collector Unit 条目）。使用：与 CU、broadcast 总线（含"广播流水线"优化，把互连从 3×N×R×T×datasize 降到 4×N×T×datasize，12+ CU 用双缓冲分接两组）协同；评估配置含 RRS=12/16/20/28 与 CU=4-14 的组合扫描。sCROOGe 论文说明：backend 方案删除 baseline 的 Scoreboard 阶段，依赖跟踪全部迁移到 OC（RAT+CU+RRS）。

涉及论文标题：
- sCROOGe Circuit-level Design and Optimization Framework for RISC-V Out-of-Order GPUs
