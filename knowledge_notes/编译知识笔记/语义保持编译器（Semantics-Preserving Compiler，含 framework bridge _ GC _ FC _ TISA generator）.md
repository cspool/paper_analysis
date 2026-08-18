## 语义保持编译器（Semantics-Preserving Compiler，含 framework bridge / GC / FC / TISA generator）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 论文（Section VI）定义：在编译流水线全程保留算子身份、类型化依赖与资源亲和性（传统编译器在 tiling/fusion 后把这些语义丢弃为不透明指令流），最终以 TISA 指令作为目标 IR 输出的编译器栈，是 TISA 动态调度框架三组件之一。它镜像传统 DL 编译的层级分解流程（框架桥→图编译→融合编译→TISA 生成→后端代码生成），但关键差异在于每个阶段都携带"语义三元组"（算子边界、类型化依赖边、资源意图）直到二进制输出。
- 五级构成：① Framework bridge——用 torchxla 前端接入 PyTorch/JAX/TensorFlow，导出框架图到 XLA/StableHLO dialect，TISA 的 OpType 分类与 StableHLO 算子抽象对齐，保证跨框架算子语义一致映射；② Graph compiler（GC）——MLIR 基础上消费 StableHLO IR 做架构感知优化（fusion、tiling、locality 驱动重排），输出软件调度 tile 图（如 FA3 的 M0/S/M1 三 tile 流水），显式保留算子边界与类型化依赖边（自定义 MLIR dialect）；③ Fusion compiler（FC）——把 GC 产出的融合子图特化为 TISA 兼容算子，定义自定义 TISA dialect（tisa.gemm、tisa.softmax），OpType 编码算子语义、UnitMap 编码资源意图、TileMem 编码符号化内存范围/scope，翻译为保留算子身份/依赖/资源亲和性的 TISA 指令流；④ TISA generator 与后端——虚拟 tile 级指令集统一多硬件后端，OpType 静态绑定张量/向量/DMA 单元类；双后端：TISA-NPU（Epoch，LLVM 定制 lowering 把 TISA 元数据嵌入最终二进制，硬件调度器消费）与 TISA-CPU（发射 CPU kernel 做功能验证/参考执行，tile 串行但保留相同语义描述符）；⑤ Runtime interface——执行时每 tile 发射描述符（OpType/UnitMap/TileMem）组成就绪集填充运行时 WQ/IQ。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 关键设计：编译器在 tile 粒度截断 lowering——不需要 lower 到细粒度 ISA 指令，因为硬件调度器直接消费 tile 级语义。这简化优化：ping-pong 缓冲只需分配两个 buffer 并交替发射 TISA tile，无需循环展开或指令重排（重叠由运行时调度器处理）。tile 尺寸最大化算术强度、受 SRAM 容量约束（如 Epoch 256 KB staging buffer 用 64×64）；不可整除张量边界生成 edge tile，TISA 直接编码精确 Shape/TileMem 区间，免 padding 开销、无未对齐惩罚。
- 运转流程例子（FA3 融合注意力）：输入 PyTorch Q/K/V → torchxla 导出 StableHLO → GC 融合（QK^T/softmax/AV → M0/S/M1 三 tile 类型）并 64×64 tiling → 输出软件调度 tile 图（带 RAW 依赖边 M0→S→M1）→ FC 用 TISA dialect 表达 tisa::gemm<me>(s_P,s_Q,s_K)/tisa::softmax<ve>(s_S,s_P,state)/tisa::load<de> → TISA generator 统一表示 → TISA-NPU 后端 LLVM lowering 嵌入元数据成二进制。对比 CUDA 版（论文 Fig.5 左）8 处手工同步（bar.sync/wgmma::wait/warpgroup_commit_batch），TISA 版零屏障。量化效果：TISA 生成 FA3 代码量 -30%、同步频率 -50%、性能在手调基线 5% 内。
- 扩展性：数千并发 tile 时分层调度——每核本地管理 256 tile，编译器做全局协调（静态 tile-to-core 分配）；多核执行用空间划分（attention head/batch 维独立 tile 组静态分配核）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：基于开源编译基础设施构建——MLIR（GC 与 FC 均 built atop MLIR）、StableHLO（IR 消费）、torchxla（https://github.com/pytorch/xla 框架桥前端）、LLVM（TISA-NPU 后端 lowering）。论文未给出该编译器代码开源链接，联网搜索未发现公开仓库（Epoch 商用）。
- 使用：作为语义目标 IR 供上游框架发射 TISA 指令（保留算子身份/依赖/资源需求），下游硬件由统一契约抽象（GPU 与 NPU 均可）。CPU 可移植性验证：Triton-TISA（语义指导编译优化：kernel 融合、循环序、内存局部性）vs Torch-Manual（标准 PyTorch 算子组合）——ResNet50 1.13–1.19×、BERT 1.02–1.20×、LLaMA2 1.14–1.18×（图 7），证明无运行时调度时语义仍指导编译期优化。同类对比：Triton/TileLang/ThunderKittens 保留 tile 语义但缺运行时自适应；本编译器通过 TISA 同时服务编译期与运行时。
- 评估：Epoch 上三配置（Naive/Static/Dynamic）共享同一编译器优化与算子库，仅隔离调度策略，证明语义保持单独贡献 1.2×（消融）。

涉及论文标题：
- Dynamic Scheduling for AI Accelerators via TISA
