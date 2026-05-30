## Streaming Dataflow / Spatial Operator Fusion（流式数据流 / 空间算子融合）

术语是什么？
Streaming Dataflow 是一种不同于传统 GPU kernel fusion 的算子融合范式：将计算图编译为 coarse-grained pipeline，tensor 被 tiled 并通过硬件 pipeline 流式处理。每个 operator 作为 pipeline 的一个 stage，中间结果存储在片上 PMU stage buffer 中（而非写回 HBM），通过 RDN 在 stages 之间流式传输。空间融合（spatial fusion）支持任意 access pattern 的算子链（如 GEMM → Mul → Transpose → GEMM），Transpose 通过 PMU data alignment unit 的 diagonally striped write 实现为 in-place access pattern 变换（无需实际数据搬移）。编译器自动将 20+ PyTorch operators 融合为单个 kernel（GPU 传统 fusion 通常仅 1-5 个 operator 且有 access pattern 限制）。效果：FlashFFTConv 操作强度从 39.5 Ops/Byte 提升至 410.4 Ops/Byte（13× speedup）；decoder layer 融合为单 kernel 后 HBM 带宽利用率~85%（GPU 通常 ~50%）。

从编译框架角度拆解：
编译器工作流程：
```
PyTorch模型代码 (Python)
  ↓ 图捕获
计算图 (FX Graph / TorchScript)
  ↓ 编译器分析
空间融合pass:
  (1) 符号生命周期分析 → 确定每个tensor的live range
  (2) 操作强度分析 → 计算融合后的Ops/Byte (Roofline)
  (3) PCU分配: systolic模式给GEMM, SIMD模式给element-wise
  (4) PMU分配: stage buffer partition (按带宽/容量需求)
  (5) 地址模式优化: 利用PMU ALU生成读写地址, diagonally striped write for transpose
  (6) 静态带宽建模 → 分配RDN vector/scalar fabric带宽
  (7) Place-and-Route → 配置RDN routing table, flow ID, multicast
  (8) 生成单个fused kernel binary (含PCU/PMU/AGCU/RDN配置)
  ↓
硬件执行: 单个Kernel Execute指令 → 片上全部数据流pipeline化执行
```

与GPU传统fusion的差异：(1) GPU fusion受限于shared cache/HBM的中间materialization，streaming dataflow直接在片上PCU-PMU pipeline中传递；(2) GPU SIMT编程不支持跨operator pipeline parallelism，streaming dataflow天然支持coarse-grained pipeline；(3) GPU需手写fused kernel（如FlashAttention），SN40L编译器自动生成。

术语一般如何实现？如何使用？
需要硬件支持：(1) 可组合的片上memory units (PMU)，支持可编程interleaving实现不同capacity/bandwidth分配；(2) 高吞吐灵活地址生成（PMU ALU pipeline）；(3) systolic + SIMD双模compute units (PCU)；(4) 支持multicast/flow control/reordering的片上网络(RDN)；(5) 编译器中的静态带宽模型和Place-and-Route层。使用方式：用户以标准PyTorch编写模型，编译器自动分析并生成空间融合kernel，无需手写kernel代码。

涉及论文标题：
- SambaNova SN40L: Scaling the AI Memory Wall with Dataflow and Composition of Experts
