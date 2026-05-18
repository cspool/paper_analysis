## Hexagon NPU HVX+HMX Architecture

术语是什么？通过联网搜索让回答具体和精准。

Qualcomm Hexagon NPU是集成在Snapdragon SoC中的移动端AI加速器，采用典型的"vector + matrix"混合架构。其向量单元称为HVX (Hexagon Vector eXtension)，矩阵单元称为HMX (Hexagon Matrix eXtension)。HMX负责低精度GEMM/Conv等核心运算，HVX负责通用向量计算（normalization、activation、dequantization等）。这种分工使NPU在矩阵密集运算上获极高吞吐和能效，但向量单元的通用计算能力相对薄弱。HMX的FP16指令未公开文档化（需reverse engineering）。三代演进：V73 (Snapdragon 8 Gen 2)、V75 (Snapdragon 8 Gen 3)、V79 (Snapdragon 8 Elite)。类似架构包括华为Ascend NPU、AMD XDNA NPU、Intel NPU/Gaudi HPU，均采用"vector + matrix"组合。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Hexagon NPU内部架构层次：
- **Scalar Cores**: 6-8个VLIW硬件线程，每线程4个VLIW slot (S0-S3)，所有vector/matrix指令从VLIW slot发出，control logic开销低
- **HVX Vector Units**: 4-6个HVX单元，每单元含32个1024-bit vector register (V0-V31)。FP16 GEMM吞吐约33 GFLOPS/单thread。支持vgather (gather elements from TCM)、vlut16 (16-entry LUT per byte)等特殊指令。可访问L2 cache或TCM
- **HMX Matrix Units**: 1-2个单元，FP16 GEMM吞吐约12 TFLOPS（~300× HVX单thread）。基本数据单元为32×32 FP16 tile (2KiB)，内存layout特殊：tile级column-major + tile内2-row permutation。HMX只能读取TCM内数据。内部accumulator为FP32高精度，output tile可独立加per-column scale/bias
- **内存子系统**: 1MiB shared L2 cache + 8MiB TCM (Tightly Coupled Memory, software-managed)。DMA引擎DDR→TCM ~60GB/s，l2fetch预取到L2 ~20-30GB/s。HVX scatter/gather/HMX指令仅访问TCM

典型decode GEMM数据流：DMA从DDR搬weight tile + activation tile入TCM → HVX vlut16 dequantization (INT4→FP16) + scale broadcast → HMX load tile pair → 32×32 tile-level inner product (FP32 accum) → output tile → HMX add scale/bias per column → DMA写回DDR。CPU-NPU通信通过FastRPC + rpcmem/dmabuf共享物理内存，仅CPU→NPU单向cache coherence。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

编程方式：(1) QNN (Qualcomm AI Engine Direct)：闭源DNN推理框架，仅支持per-tensor/per-channel量化和固定shape图，不允许低层kernel定制；(2) Hexagon SDK LLVM：提供C/C++编译器和inline assembly，但HMX指令需从binary library reverse engineering；(3) 论文llama.cpp-npu：基于Hexagon SDK LLVM + reverse-engineered HMX FP16指令，约7K行C/C++和inline assembly，无QNN依赖。NPU operator library编译为独立Hexagon DSP shared object，CPU侧llama.cpp backend通过FastRPC启动remote NPU session → rpcmem共享内存通信 → NPU thread pool轮询请求。Hexagon NPU仅有32-bit虚拟地址空间，限制了大模型部署（V73无法运行≥3B模型）。

涉及论文标题：
- Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

