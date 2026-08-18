## TISA（Tile-level Instruction Set Architecture，tile 级指令集架构）

术语解释
TISA 是一种 tile 级指令集架构，作为叠加在现有每单元执行 ISA 之上的"调度语义层"（scheduling-semantics layer）：每条 TISA 指令编码算子类型（OpType）、类型化依赖、资源意图（UnitMap）和 tile 级内存范围（TileMem），供硬件调度器在运行时做合法性、就绪性与重叠判断。类比 CPU ISA 中寄存器名之于 scoreboard：TISA 的 OpType/UnitMap/TileMem 字段构成硬件调度器的"架构契约"，但粒度在 tile 级并跨 tensor/vector/DMA 异构单元。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 论文定义：TISA 由奕行智能（EVAS Intelligence）+ 湖南大学团队提出（ISCA 2026 条件接收，论文 "Dynamic Scheduling for AI Accelerators via TISA"）。设计动机：现有 AI 加速器编译流程在 tiling/fusion 后把算子 lower 成不透明指令流，硬件只看到有序指令序列，无法区分真依赖与人为依赖（隐式屏障），静态调度因此保守同步、牺牲利用率。TISA 把调度所需语义（算什么、用哪些资源、数据与哪些 tile 交互）编码进轻量指令格式，让硬件能区分"真资源冲突"与"可恢复停顿"。
- 数据结构：Operand = (TileShape, TileMem, AccessType)，其中 TileShape 为符号/参数化计算边界，TileMem = (base, scope)（base 为符号/常量地址，scope ∈ {Private, Local, Shared} 表示内存层级），AccessType ∈ {R, W, RW}。TISA 指令 TISA_I = (OpType, Operands, Attributes, UnitMap)：OpType ∈ {GEMM, SOFTMAX, ...} 为语义标识符；Operands 数组硬件约束 outs ≤ 3、ins ≤ 7；Attributes 编码重排约束与同步需求；UnitMap = (unit, quantity, affinity) 指定资源。
- 三类语义：计算语义（OpType 标识算子并映射到张量/向量/标量执行单元类）、数据语义（Operands/TileMem 定义数据时空范围，支持细粒度冲突检测）、调度语义（Attributes/UnitMap 约束重排、指定资源亲和性）。依赖 Deps = {(src, type, condition)}，type ∈ {RAW, WAR, WAW}，由 TileMem 区间重叠分析自动导出，condition 支持部分就绪（子区域有效即提前唤醒依赖 tile）。
- 与既有 ISA 的区别：Cambricon/TPU/IPU 等 domain-specific ISA 只定义执行语义（算什么、在哪个单元算），跨 tile 顺序靠 fence/BSP barrier 显式强制；NVIDIA PTX 是细粒度虚拟 ISA（add.f32/ld.global）；TISA 在 tile 粒度（tisa::gemm<me>、tisa::softmax<ve>）保留算子语义、依赖与资源需求。Web 佐证：Epoch 芯片官方营销称 VISA 虚拟指令集（解耦软硬件迭代），TISA 即论文中的 tile 级动态调度架构。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件调度器直接读 TISA 字段（OpType/TileMem/UnitMap）做分发决策，字段不被软件解释——这是 ISA 级抽象的本质属性。违例或省略 TISA 语义只会保守（不正确性不会发生），类似绕过 CPU scoreboard 退化到 in-order issue。
- Epoch 上 TISA 有具体二进制编码（细节因篇幅省略）。每核集成硬件调度器：ME 增加自定义 block 张量/矩阵指令，VE 扩展 tile 友好向量操作，DE 暴露 DMA 风格异步非阻塞描述符，全部遵守 TISA 接口。运转流程例子（FA3 融合注意力）：编译器自动生成 TISA kernel 指令流 tisa::load<de>(s_Q,Q) → tisa::load_transpose<de>(s_K,K) → tisa::gemm<me>(s_P,s_Q,s_K) → tisa::softmax<ve>(s_S,s_P,state) → 循环内 tisa::load<de>(s_V,V) → tisa::gemm<me>(s_R,s_S,s_V) → tisa::rescale<ve>(s_O,s_R,...) → tisa::store<de>(O,s_O)，零显式屏障；每 tile 描述符进入硬件调度器 Reception Buffer 后被按语义路由到各单元等待队列，依赖清除即乱序发射。
- 调度开销：RTL 综合测得每 tile dispatch 7~9 cycles @1GHz（纳秒级）；对比软件运行时（控制处理器上取指/译码/分支/访存做依赖检查）需微秒级（慢 100~1000×），而 tile 执行本身 10³~10⁵ cycles，因此硬件 ISA 接口是必要条件。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：TISA 作为调度语义 ISA 扩展补充现有每单元执行 ISA；Epoch 上有具体二进制编码，编译器（TISA-NPU 后端 LLVM 定制 lowering）把 TISA 元数据嵌入最终二进制，硬件调度器消费。上游编译器把 TISA 作为语义目标 IR（保留算子身份/依赖/资源需求），下游硬件由统一契约抽象（OpType 可对应软件算子或粗粒度硬件指令），同一契约支持通用 GPU 与 domain-specific NPU。
- 使用：编译器经 framework bridge → Graph compiler → Fusion compiler（TISA dialect）→ TISA generator 产出 TISA 指令；运行时硬件调度器按 OpType 路由、TileMem 做区间重叠测试、UnitMap 做逐单元仲裁。可移植性：NVIDIA GPU 上可加语义感知协调器叠在 warp/CTA 调度器之上；domain-specific 加速器可用 TISA 做 native ISA 的 thin wrapper。
- 开源情况：论文未给出 TISA/二进制编码开源链接，联网搜索未发现公开仓库（Epoch 为商用芯片）。同类开源 tile 级编程参考 TileLang（https://github.com/tile-ai/tilelang）为编译期静态调度。

涉及论文标题：
- Dynamic Scheduling for AI Accelerators via TISA
