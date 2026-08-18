## 指令级并行（ILP）与线程级并行（TLP，GPU 延迟隐藏与 OoO 动机）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ILP（Instruction-Level Parallelism）指单条指令流内可并行/乱序执行的独立指令数量；TLP（Thread-Level Parallelism）指并行线程（GPU 中为 warp）提供的并行度。逻辑链：GPU 的性能模型依赖 TLP 上下文切换隐藏延迟——warp 遇长延迟指令（访存、依赖）时切换其他 warp 执行；当负载 TLP 不足时资源闲置，而盲目增加并发 warp 会加剧 cache 争用（相关研究 [38][40][49][78] 已证明）——这正是 OoO 执行挖掘 ILP 的动机。sCROOGe 对 Vortex 负载做了 ILP 分析：动态指令流按 memory fence 与控制指令切成基本块，块内遵守真实数据依赖与访存依赖，平均 ILP = warp 0 总指令数 / 最长依赖链长度，测得各负载平均 ILP 2.02-2.92（Fig.10）；并据此把负载分为低/高 ILP 两类（两端各 8 个）分析最优 OoO 结构差异（高 ILP 类最优 CU/IsB 数比低 ILP 类高 +1/+0.75，EDP 改善 +1.62%/3.06%）。kernel 调度视角（vault 笔记 knowledge_notes/算法知识笔记/ILP_TLP_Arithmetic Intensity Trade-off in DNN Kernels）：ILP/TLP/算术强度是 DNN kernel 的三维 trade-off 空间——ILP↑ 需更多寄存器→TLP↓，本质是同一资源竞争的另一种形态。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
GPU 硬件中的运转：warp scheduler 每 cycle 从多个可调度 warp 中选一个发射（A100 每 SM 4 个 scheduler，每 cycle 每 warp 至多 1 条指令）——TLP 提供"同一周期多条不同 warp 的指令并行执行"，ILP 提供"同一 warp 内多条独立指令并行/乱序"。sCROOGe 的测量：baseline Vortex 的 IPC 对 warp 数不敏感、对线程数近线性扩展（图 12，线程扩展直接对应吞吐），多数负载 4 个并发 warp 即饱和——说明 TLP 扩展收益递减；backend OoO 在 stall 场景把 ready 指令越过阻塞的旧指令重排（重排距离统计：多数指令只越过 1 条，<10% 越过 4 条以上，CU 越大重排越深），把 ILP 转化为真实加速（图 14：backend 最高 27.3% 通用 / 53% ML；sched stall -61%）。{64,32} 这类大配置反而低效（结构成本超过 ILP 收益），16 warps 设计点最优——直接佐证"TLP 已接近饱和、ILP 是下一个增长点"的论文论点（iso-area 对比：OoO SM 以更小面积达到增加 warp 的 in-order SM 的 IPC 并平均 +14.4%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：TLP 靠硬件多线程（warp 调度 + 独立寄存器上下文）实现，ILP 靠超标量/乱序/流水化实现；GPU 传统只做 TLP，OoO 方案（sCROOGe frontend/backend）把 ILP 显式化。使用：ILP 分析以指令 trace 为输入（动态指令流→基本块划分→依赖链长度→平均 ILP），用于：(1) 决定是否值得上 OoO 结构；(2) 指导 OoO 结构尺寸（CU/IsB 数）；(3) 负载分类（低/高 ILP）做结构右尺寸。sCROOGe 论文指出先前仿真工作（LOOG/GhOST/SIMIL）在 64 warps/32 threads 下报告的 IPC 增益（23%/6.9%/31%）被 RTL 实测大幅下调（4.18%/0.86%），部分配置的 ILP 收益落在误差范围内——强调 RTL 验证对 ILP 评估的必要性。Web/vault 证据：FlashAttention-T 的 ILP/TLP 调度范式（knowledge_notes/kernel知识笔记/Tensor-Vector Parallelism Scheduling）、DNN kernel 的 ILP/TLP 权衡（算法知识笔记）。

涉及论文标题：
- sCROOGe Circuit-level Design and Optimization Framework for RISC-V Out-of-Order GPUs
