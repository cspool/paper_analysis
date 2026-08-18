## 乱序执行调度（Out-of-Order Dispatch：操作表 Operation Table + Dispatcher 算法）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
乱序执行调度是 PipeIMC 解决 in-order 存内计算串行化（控制流/数据依赖导致长 idle）的机制：各 control block 内的 IMC 执行单元采用 in-order issue + out-of-order execution（操作按序取指解码，但 phase 可乱序执行）。核心硬件：**操作表（operation table）**——跟踪每个操作的各 phase 状态（pending/on-fly/completed），向 dispatcher 输出调度信息，从 commit unit 接收完成信息，操作全部 phase 完成即被逐出；**dispatcher**——发现空闲端口时按 Algorithm 1 遍历操作表，检查：(a) 操作是否有在飞 phase（每操作至多一个，Lines 4-5）；(b) phase 是否与在飞 phase 端口冲突（Line 17，重命名后只需查数据依赖）；(c) barrier/fence 必须到操作表顶部才能执行（Lines 15-16，保证前序操作完成）；(d) 特殊 phase 优先调度——memory phase（早执行减少等待数据时间、早进内存层次便于 coalescer 合并省带宽，Line 12）与控制流 phase（分支等，早执行减少 warp stall，Lines 13-14）。实现上每个操作表槽位向优先级 MUX 输出信息，决定派发到空闲端口的 phase。每周期只派发一个 phase 防双端口竞态（两端口同时空闲只调度 memory port）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
执行流程（论文 Algorithm 1 伪代码）：dispatcher 空闲时，对操作表 T 中每个操作 O：若 O 有在飞 phase 则跳过；取 O 的下一个 pending phase P；若 P 是 memory phase 且处于 fencing 或 memory port 无法接受则跳过；若 P 有端口冲突则跳过（不设 special 标志的继续保留为候选）；遇到 barrier 且不在表顶则 break；若 P 是 special phase（memory/控制流）且无冲突则直接派发并 break；否则记 P 为候选，循环完派发候选。commit unit 处理 sequencer 完成的 phase：通知 warp scheduler（分支/同步等影响控制流的操作提交时保持控制流干净——分支操作提交前对应 warp 停取指，保证窗口内操作不会回滚，因此无需回滚机制）与操作表更新。效果：相对 in-order SIMT-EVE，PipeIMC 控制流 hazard stall 显著减少；乱序执行使计算能跨过内存等待时间，内存访问时间显著减少（memory-bound benchmark 上效果明显）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：操作表（16 条目）+ dispatcher 优先级 MUX + commit unit 集成在 IMC controller；每个 warp 一个 IMC 执行单元、同一 control block 的 warp 共享同一前端流水（取指-解码-派遣）。使用：配合显式寄存器重命名（见本库条目）与细粒度发射，形成 PipeIMC 的完整乱序流水执行模型；评估中 Pipe-1（单端口+乱序）相对 EVE 平均 1.73x、Pipe-2r（双端口+乱序+重命名）2.55x。Vault 无专门笔记证据（omnisearch 对"乱序"无命中）。

涉及论文标题：
- PipeIMC a Pipelined In-SRAM Computing Architecture
