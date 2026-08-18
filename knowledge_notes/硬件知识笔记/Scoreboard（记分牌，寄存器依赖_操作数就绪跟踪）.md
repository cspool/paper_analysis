## Scoreboard（记分牌，寄存器依赖/操作数就绪跟踪）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Scoreboard 是跟踪寄存器操作数是否就绪、阻塞依赖冒险指令发射的硬件结构：指令发射时把其目的寄存器标记为"未就绪"，写回时清除标记，后续指令发射前检查源寄存器标记，未就绪则 stall。逻辑链：在 in-order 流水线中，scoreboard 保证指令按发射顺序访问寄存器、避免 RAW/WAW（并配合顺序发射处理 WAR）；在 GPU 中（如 NVIDIA SM 的 RFU 内 scoreboard 单元）用于检测 register file 数据未就绪导致的 stall（vault 笔记 knowledge_notes/硬件知识笔记/GPU Scoreboard and Throttle Stalls：scoreboard stall 是 warp 等 data dependency、指令无法 issue 的两大 stall 类型之一）。Vortex baseline 的 Issue 阶段用 scoreboard 跟踪寄存器依赖 + IBuffer 缓存指令。sCROOGe 对 scoreboard 的处理：frontend 方案通过"只向 arbiter 供给独立指令"的调度优化移除了"per-warp oldest"缓冲与 scoreboard 的 SRAM 阵列；backend 方案删除 Scoreboard 阶段，依赖跟踪整体迁移到 OC 阶段（RAT+CU+RRS 重命名）。NV 专利视角（vault 笔记 human_notes/GPU架构笔记/NV scoreboard机制、NV GRF的并发访问）：scoreboard 记录发射后指令的目标寄存器，后序指令在 I-Buffer 中等待掩码归零后发射，理论上应在指令进入 I-Buffer 前就登记目标寄存器以检测依赖。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
baseline Vortex 的运转流程：Decode 后指令按 warp 进入 IBuffer，Issue 阶段每 warp 每 cycle 一条指令——发射前 scoreboard 检查源寄存器是否就绪（未就绪则指令留在 IBuffer 等待）→ 发射时把 rd 置为未就绪 → Execute 完成后写回 RF 并清除 rd 的就绪位。以 `I1: r1=ld(A); I2: r2=r1+1` 为例：I1 发射后 r1 未就绪，I2 查 scoreboard 发现 r1 未就绪 → stall 直到 I1 写回；若此时同 warp 后序 `I3: r4=r5*r6` 无依赖，baseline 仍被 I2 阻塞，而 sCROOGe frontend 的 Dependence Checker 判定 I3 独立、Issue Arbiter 越过 I2 发射 I3（对应 sCROOGe 移除 scoreboard 的动机）。后端方案则以 RAT 重命名替代 scoreboard：不再需要"寄存器就绪位"这一全局结构，依赖以 RRS ID 追踪。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：寄存器号索引的就绪位数组（每位对应一个物理寄存器），随发射置位/随写回清除；GPU 中位于 RFU 内（NVIDIA）或 Issue 阶段（Vortex）。使用：in-order GPU 用 scoreboard + TLP 切换隐藏 stall；OoO 方案中 scoreboard 被 RAT/重命名或依赖检查器替代。sCROOGe 论文报告 stall 分解中的 dependence（dpnd）stall 类别并测量：frontend/backend 总 stall 相对 baseline -11.8%/-14.8%，sched stall -51%/-61%，backend 的依赖专属 stall 显著减少（重命名 + RF 访问减少）。Web 证据：NVIDIA 的 scoreboard stall 概念见 Nsight Compute stall reasons（vault 笔记 GPU Scoreboard and Throttle Stalls，Infera 用 data hazard 分析估计 IPC）。

涉及论文标题：
- sCROOGe Circuit-level Design and Optimization Framework for RISC-V Out-of-Order GPUs
