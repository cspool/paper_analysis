## TISA dialect（MLIR 自定义 tile 级方言）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 论文（Section VI-c）：Fusion compiler（FC）在 MLIR 上定义的自定义 tile 级方言，其操作（tisa.gemm、tisa.softmax 等）通过 OpType 编码算子语义、通过 UnitMap 编码资源意图（到执行单元类的映射）、通过符号化 TileMem 区间与 scope 编码内存访问模式；编译器用该方言把软件调度 tile 图翻译成保留算子身份、数据依赖、资源亲和性的 TISA 指令流，这些属性构成运行时调度器消费的"语义契约"。
- 定位：区别于 MLIR 现有方言（如 Linalg/Affine、StableHLO）——它把 lower 截断在 tile 粒度，不继续 lower 到细粒度 ISA；硬件调度器直接消费 tile 级语义，因此无需循环展开/指令重排等传统优化（ping-pong 缓冲只需交替发射 TISA tile）。OpType 分类与 StableHLO 算子抽象对齐（Framework bridge 阶段保证），数据语义定义在适配 L1/L2 或共享 SRAM 容量的 tile 上。
- 背景（web）：MLIR（https://mlir.llvm.org）是 LLVM 的可扩展多方言编译器基础设施；StableHLO（https://github.com/openxla/stablehlo）是 OpenXLA 的高层 ML 算子集（框架与编译器间可移植层）。本论文 FC 在既有 MLIR 基础设施上新增 TISA 方言，类似 Tenstorrent TT-MLIR（TTIR→TTNN-IR→TTMetal-IR→TTKernel-IR 渐进 lower、32×32 硬件 tile 化）等工业界 dialect 设计范式，但 TISA 方言的独特处是把调度语义（依赖描述符/UnitMap/符号化 TileMem）显式编码给硬件调度器而非只做 kernel 生成。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程：GC 输出的软件调度 tile 图（含融合子图与类型化依赖边）→ FC 匹配融合子图并特化为 TISA dialect 操作（tisa.gemm、tisa.softmax、tisa.load、tisa.store，模板参数 <me>/<ve>/<de> 表示单元映射）→ 经 TISA generator 统一表示 → 后端（NPU: LLVM lowering 嵌入元数据；CPU: 参考执行）。每条 TISA 指令携带 Operand=(TileShape, TileMem(base,scope), AccessType) 与 Attributes/UnitMap。
- 具体例子（FA3，论文 Fig.5 右）：tisa::gemm<me>(s_P, s_Q, s_K) 表示 GEMM tile 的 OpType、操作数 s_Q/s_K/s_P 的 TileMem 区间（Q/K/P 的符号化地址范围与 Local scope）、UnitMap=(ME,1,affinity)——调度器据此路由到 ME 的 WQ 并与 F_me 做区间重叠检查。edge tile（不可整除边界）直接在方言层编码精确 Shape/TileMem，免 padding。
- 与 TISA-CPU 后端配合做端到端验证：两后端保留相同语义描述符，CPU 上重叠 tile 串行执行但语义一致，保证验证与 NPU 行为一致。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：在 MLIR 上以自定义 dialect 形式实现（定义操作、类型与 pass），FC 为 built atop MLIR 的编译器组件；TISA generator 提供虚拟 tile 级指令集统一多后端。论文未开源 FC/TISA dialect 代码（联网搜索未见公开仓库）。
- 使用：编译器把 StableHLO 算子按 OpType 对齐映射到 tisa.* 操作，供上游框架直接发射；运行时调度器只认 TISA 语义字段做合法性检查与跨单元重叠。可移植：同一 dialect 表示的语义契约可用于 GPU 与 NPU（OpType 可对应软件算子或粗粒度硬件指令）。
- 对比（web）：TT-XLA/TT-Forge 用 torchxla→StableHLO→TT-MLIR 方言链做 32×32 tile 化与 fusion；TISA dialect 的差异点在于输出语义是给硬件调度器消费的调度元数据而非仅最终 kernel。

涉及论文标题：
- Dynamic Scheduling for AI Accelerators via TISA
