## CENT (PIM-only) and NeuPIMs (xPU+PIM) Multi-Node PIM Systems

术语是什么？通过联网搜索让回答具体和精准。

CENT和NeuPIMs是两种代表性的多节点PIM-based LLM推理系统架构，PIMphony以其为baseline和target platform。CENT[16]（ISCA 2024）：PIM-only系统，所有LLM计算（包括compute-intensive GEMM和memory-bound GEMV）均由PIM/PNM（Processing-Near-Memory）处理，通过CXL-based memory expansion支持大模型的多PIM node互联。每module 16GB，internal bandwidth 16TB/s，PNM提供3 TFLOPS compute，32 PIM channels。7B使用8 modules (128GB)，72B使用32 modules (512GB)。NeuPIMs[21]（ISCA 2024）：xPU+PIM heterogeneous系统，compute-intensive kernel（GEMM/FC）交由xPU/NPU（8 Matrix Units, 256 TFLOPS）处理，memory-bound Attention offload到PIM。每module 32GB，internal bandwidth 32TB/s，32 PIM channels。7B使用4 modules (128GB)，72B使用16 modules (512GB)。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。

```
// CENT (PIM-only) Decode Step:
// 1. host CPU sends command → all PIM modules
// 2. Each PIM module:
//    a. Q/K/V projection: PNM执行GEMM (compute intensive)
//    b. Attention QK^T: PIM channels并行GEMV (memory bound)
//    c. Attention SV: PIM channels并行GEMV
//    d. Output projection: PNM执行GEMM
//    e. FFN: PNM执行GEMM + activation
// 3. All computation in PIM/PNM, no external accelerator

// NeuPIMs (xPU+PIM) Decode Step:
// 1. xPU/NPU handles:
//    a. Q/K/V projection (GEMM, compute intensive)
//    b. Output projection (GEMM)
//    c. FFN (GEMM + activation)
// 2. PIM modules handle:
//    a. Attention QK^T (GEMV, memory bound)
//    b. Attention SV (GEMV, memory bound)
// 3. Data transfer: xPU ↔ PIM via interconnect (CXL/PCIe)
//    - xPU sends Q to PIM for attention
//    - PIM returns attention output to xPU for subsequent layers
```

CENT在长上下文下的问题：PIM-only执行compute-intensive GEMM时MAC效率低（GEMM需要大量MAC但PIM MAC unit为GEMV优化），且pipeline parallelism下attention KV cache的sparse channel activation导致整体utilization崩溃（1M context退至2%）。NeuPIMs通过xPU offload GEMM缓解此问题，但attention侧的channel underutilization和I/O bottleneck仍存在。PIMphony同时提升两种系统的PIM侧效率。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

CENT和NeuPIMs均使用Ramulator-based cycle-accurate simulator建模和评估，结合AiMX PIM specification校准。两者通过MLIR compiler或hand-crafted PIM instruction sequences部署模型。Multi-node scaling通过CXL/PCIe互联实现——CENT使用CXL-based memory expansion连接多个PIM node，NeuPIMs使用standard interconnect连接xPU和PIM。PIMphony的compiler/runtime兼容两者，通过TCP/DCS/DPA提升PIM侧的channel utilization、MAC efficiency和capacity utilization。

涉及论文标题：
- PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System
