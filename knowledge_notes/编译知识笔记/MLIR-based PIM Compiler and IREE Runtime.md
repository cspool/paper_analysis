## MLIR-based PIM Compiler and IREE Runtime

术语是什么？通过联网搜索让回答具体和精准。

MLIR-based PIM Compiler是利用MLIR（Multi-Level Intermediate Representation）框架为PIM硬件生成优化指令序列的编译器。MLIR提供多层次dialect基础设施，允许编译器逐步将高层模型表示（如StableHLO、Linalg、ONNX）lowering到PIM-specific operations，最终生成PIM可执行的command sequences。PIMphony基于MLIR和IREE（Intermediate Representation Execution Environment）runtime stack实现其编译栈：扩展MLIR dialect表达PIM-specific operations，实现custom pattern-matching和code generation passes，通过IREE HAL（Hardware Abstraction Layer）对接commercial PIM SDK。社区中已有多个类似项目：pim-iree（aiha-lab）将StableHLO lowering到PIM device；Raptor（Politecnico di Milano）将ONNX经由Spatial dialect lowering到PIM JSON；IREE社区推荐通过out-of-tree plugin（参考iree-amd-aie）添加PIM backend支持。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。

PIMphony compilation flow：(1) Frontend: Transformer decoder model graph输入→lowering到MLIR dialect；(2) Pattern-matching pass：遍历MLIR IR→识别attention子图（QK^T matmul + softmax + SV matmul）和FFN子图→标记为PIM-amenable kernels；(3) TCP partitioning pass：根据模型config（num_heads、channel count）计算token-centric partition scheme→生成per-channel token segment range metadata；(4) DCS-aware code generation：分析GBuf/OBuf entry-level data dependency→生成dependency annotations（per-command GBuf/OBuf entry read/write info）供runtime DCS controller使用；(5) DPA code generation：生成Dyn-Loop指令（loop bound = runtime Tcur）和Dyn-Modi指令（stride-based address）→嵌入VA2PA table index reference；(6) IREE HAL deployment：编译产物（PIM instruction sequences + metadata）通过IREE HAL提交到PIM SDK→runtime自适应dispatch。Compilation离线完成（不计入inference latency），每个model编译一次。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方式：(1) 扩展MLIR dialect——定义PIM-specific op（如pim.wr_inp, pim.mac, pim.rd_out）和type（如gbuf_entry, obuf_entry）；(2) 编写custom MLIR passes——使用MLIR的Pattern Rewrite infrastructure实现subgraph识别和code generation；(3) IREE HAL plugin——实现PIM device的buffer management、command submission和synchronization接口。PIMphony的compiler不依赖新hardware primitive，确保与CENT/NeuPIMs simulator兼容。编译产物包含静态指令序列和动态metadata（Dyn-Loop/Dyn-Modi参数、dependency annotations、VA2PA hints），runtime根据当前系统状态（token lengths、batch composition、VA2PA mappings）最终dispatch。

涉及论文标题：
- PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System
