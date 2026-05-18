## Tile-Based Zero-Tuning Compilation

术语是什么？通过联网搜索让回答具体和精准。
Tile-Based Zero-Tuning Compilation是Infera compiler的核心编译策略：将DNN operator切分为tile/micro operators，为每个tile生成多种ILP/TLP/arithmetic intensity trade-off配置的micro-kernel candidates，编译过程完全基于静态分析（资源约束、instruction scheduling、warp specialization），无需GPU profiling或性能模型评估（zero-tuning）。相比Ansor/MetaSchedule的搜索式编译（2-3个数量级的编译时间差异），zero-tuning通过固定数值化配置直接生成高质量kernel。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
Tile-Based Zero-Tuning Compilation在Infera compiler中的运转流程：
```
ONNX Model
  ↓ TVM Relay frontend
Computation Graph
  ↓ Tile-tailored TVM Compiler (§4.1)
Tile-based TensorIR
  ├── Large operator → micro operators (tiles)
  └── Small operators → shepherd operator (merge)
  ↓ TVM Code Generator
CUDA C++ Program
  ↓ Multi-level Code Optimization (§4.2)
  ├── Instruction Reconstruction (cut-and-patch)
  ├── Warp Specialization (4 mainloop + 4 copy warps)
  └── Source Code Optimizer
CUDA Binary Program
  ↓
Static Library (.a)
```
1. **Tile Size决策**（top-down策略）：
   - Register file level: 32-bit register/thread limit = 64/96/128 → 平衡ILP vs TLP
   - Shared memory level: usage/thread block limit = 48/80/112/144 KiB → spatial tile size = thread tile × thread count, reduction tile auto-constrained
   - Global memory level: spatial tile = block tile × grid size (fixed 64), reduction tile = kernel argument
2. **Kernel生成**（multi-version）：pipeline stage 2/3/4, asynchronous copy + warp specialization for global→shared, padding消除bank conflict for shared↔register, wide types (STG.128) for register→global
3. **Zero-tuning特性**：所有tile size、register/shared memory分配、pipeline配置由静态约束直接计算，无需profiling反馈

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Infera compiler基于TVM 0.16.0实现，修改其tile-based compilation和code generation pipeline。Zero-tuning的关键：Roller证明了仅考虑tile size因素即可生成高性能kernel，Infera进一步发现varying tile size可平衡ILP/TLP/intensity trade-off。编译完全并行化，CPU资源增加时编译时间按比例缩短。相比Ansor/MetaSchedule编译时间低2-3个数量级，相比Roller（仍使用performance model评估）节省66%-86% CPU时间。硬件兼容性通过固定数值配置与现代GPU numerical standards对齐保证。

涉及论文标题：
- Automated End-to-End Model Serving with Cooperative Compilation and Scheduling
