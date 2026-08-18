## Collector Unit（CU，操作数收集单元 / GPU 保留站）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Collector Unit 是 GPU 寄存器文件单元（RFU）中为执行单元收集指令操作数的硬件槽位：接收译码/寄存器地址单元（RAU）发来的指令操作数读写地址，从寄存器 Bank、data cache、常量或立即数收集操作数，集齐后由 Dispatch 单元选出发射到执行单元。在 LOOG/sCROOGe 的 backend OoO 方案中，CU 被重新定义为 Operand Collect（OC）阶段的重排结构，等效于 Tomasulo 保留站（Reservation Station）：CU 跟踪指令元数据（PC、warp ID、活跃线程）、分配状态、操作数就绪状态（RF 直读或结果 broadcast），从 Issue 到 Dispatch 期间持有数据与立即数，从而在 OC 阶段实现乱序重排。逻辑链：GPU 传统用 CU 只做操作数收集（NVIDIA 专利 OPERAND COLLECTOR ARCHITECTURE 中每个执行单元配多个 CU，SFU 2 个 CU 并发 4 个读请求、MACU 4 个 CU 并发 12 个读请求，Bank 单端口需 1 cycle/操作数）；OoO 方案把 CU 数量扩大并让其驻留指令直到操作数就绪，形成重排窗口——CU 越多重排深度越大、越能挖掘 ILP，但面积/互连开销非线性增长。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
sCROOGe backend 的运转流程：指令进入 OC 时分配一个空闲 CU（分配条件：有空 CU + UUID 越界检查通过 + 有写回资源（RRS entry 或无需写回））→ 下一周期 CU 查 RAT，把重命名后的源寄存器填入字段（决定从 RF 读还是等 broadcast），若需写回则把 RRS ID 写入 RAT 的 rd → CU 收集源操作数：需 RF 直读的经三个仲裁器之一（最低 CU ID 优先 / Round-Robin / RR 组合，实测最低 CU ID 优先性能与逻辑复杂度最优）每 cycle 顺序取一个操作数 → 操作数全部 valid 后 CU 标记 ready → Dispatch arbiter 每 cycle 从 ready CU 池选一个进 Execute → 写回经 broadcast 路由回 OC 供其他 CU 旁听。vault 笔记（human_notes/GPU架构笔记/NV GRF的并发访问）给出 NVIDIA 专利中 Collector Unit 的经典运转：RAU 送地址 → CU 发读请求给 Bank Request Arbitration Unit（每 Bank 每 cycle 一个请求，两阶段裁决：先按策略选 collector、再为每个 bank 选请求）→ 数据经 crossbar 返回 CU 的 slot → 集齐后 Dispatcher 按线程年龄顺序发射避免 WAR。sCROOGe 实测：CU 从 8 扩到 12 带来 10.3% 面积 / 10.4% 功耗开销（仿真器 LOOG 报 32→48 CU 仅 2.57% 面积 / <1% 功耗），突显 RTL 建模的价值。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：SystemVerilog RTL，可配置 CU 数量（4-14）与线程数；每个 CU 含分配位、操作数就绪标志、RF/broadcast 来源字段、数据与立即数存储。使用：与 RRS（见"Register Alias Table 与 Register Renaming Stack"条目）配合——Dispatch 后 CU 立即释放（CU 面积约 21.382µm²、占总设计 0.9387mm² 的 2.28%，而一个 RRS entry 仅 0.873µm² / 0.09%），解决 LOOG 初版 CU 长期占用导致的无 CU 可用 stall（无 RRS 时 80% 应用的无 CU stall 占比 >42%，RRS=12 时降至 >23%）。Web 证据：NVIDIA"OPERAND COLLECTOR ARCHITECTURE"专利（vault 笔记 human_notes/GPU架构笔记/NV GRF的并发访问、NV GPU专利 引用）。

涉及论文标题：
- sCROOGe Circuit-level Design and Optimization Framework for RISC-V Out-of-Order GPUs
