## 流水化存内计算（Pipelined In-SRAM Computing：memory phase / calculation phase + 细粒度发射）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
流水化存内计算是 PipeIMC 的核心贡献：把每个 in-SRAM 计算操作分割为**两个阶段（phase）**——memory phase（从低层内存层次取数到指定 wordline，或把 wordline 上的结果写回内存，含地址生成→内存访问→写回三个 step）与 calculation phase（位线计算完成运算）。硬件上每个 IMC 执行单元内设 memory stage 与 calculation stage 两个阶段，各自含 sequencer 与一个 SRAM 端口，计算 SRAM 阵列被两个 stage 共享、充当两者间的流水寄存器：当前操作的 calculation phase 在 calculation stage 执行时，memory stage 可以同时为下一条无数据依赖的操作执行 memory phase 取数，从而重叠计算与取数、缩短关键路径。为消除 in-order 串行执行留下的长 idle，论文还提出**细粒度发射（fine-grained issue）机制**：sequencer 在 memory phase 完成地址生成步骤后即把剩余步骤直接交给 memory unit、释放 SRAM 端口；并在每个 1-bit 外围电路的 sense amplifier 中实现一个 **latch** 支持"计算 phase 冻结/恢复"——当 memory unit 需要写回计算 SRAM 或 memory port 需要为新 memory phase 算地址时冻结对应 sequencer，正在进行的计算 phase 数据保存在 SA latch 中，sequencer 恢复后继续执行。这样 memory port 在 memory step 期间保持活跃，最大化流水利用率。IMC 操作抽象为 dest := IMC(src1, src2, op)，源可为内存指针/wordline/立即数，操作数已在 wordline 时无 memory phase，因此每个操作至多三个 phase（load/compute/store）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
执行时间线例子（论文 Fig.3，SIMT 程序 for 循环每轮 O1..O4+Branch 五个操作）：(a) in-order：memory phase（地址生成蓝/内存访问白/写回灰三步）与 calculation phase 严格串行，关键路径覆盖整个操作序列；(b) 加额外端口后：无依赖操作的计算与取数重叠，关键路径缩短；(c) 加重命名 + 重排序后：数据依赖/控制流（分支）造成的 idle 进一步消除；(d) 加细粒度发射后：memory step 期间 memory port 也能执行计算 phase，得到理想时间线。以 matmul 的 O1（load A1/B1 + 乘法）与 O2（无依赖）为例：O1 的 calculation phase 占用 calculation port 执行 105–634 cycles 乘法时，O2 的 memory phase 在 memory port 上并行取数；若 O3 的源是 O2 的结果（数据依赖），则需等 O2 提交（commit）后才能调度，由操作表 + dispatcher 处理。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：dual-port 计算 SRAM（见本库双端口计算 SRAM 条目）+ 每 stage 的 sequencer（从预写微程序 ROM 取对应 SRAM 操作序列）+ memory unit（data transpose unit 重排为 hybrid-8 布局、request buffer 跟踪在途请求、request coalescer 合并访存省带宽）+ SA latch（计算冻结）。使用：在 cycle-approximate 模拟器中重建（计算 phase 周期由 cycle-accurate 计算 SRAM 模拟器给出，add 9、sub 17、mul 105–634、div 145–1174 cycles）；结果 Pipe-2r 相对 SIMT-EVE/Duality Cache 平均提速 155%/113%，计算 SRAM 利用率 2.15x–3.96x / 1.13x–4.77x。Vault 无专门笔记证据。

涉及论文标题：
- PipeIMC a Pipelined In-SRAM Computing Architecture
