## NVLink / NVSwitch (GPU Interconnect)

术语是什么？
NVLink是NVIDIA的高带宽GPU-to-GPU互联链路（H100: 4th-gen 450 GB/s单向，B200: 5th-gen 900 GB/s单向）。NVSwitch是交换芯片（H100 DGX使用4颗TSMC 4N NVSwitch，每颗含64个port，总双向带宽3.2 TB/s），将所有GPU互联成全互联fabric——任何两GPU间单跳可达。NVSwitch还内置SHARP ALU（400 GFLOPS FP32）支持in-network reduction（参见知识库_芯片设计）。

从硬件架构角度拆解术语：
NVLink port是每个GPU对外通信的物理端点，H100有多个NVLink port连接到所有4颗NVSwitch。数据路径：GPU SM/TMA/CE → NVLink port → NVSwitch crossbar → 目标GPU NVLink port → 目标HBM。通信层次由低到高：(1) PCIe (64 GB/s) 用于CPU↔GPU和跨节点；(2) NVLink (450/900 GB/s) 用于GPU↔GPU；(3) NVSwitch实现all-to-all non-blocking switching；(4) NVSwitch SHARP在fabric内执行reduction（multimem.ld_reduce/multimem.red）。从A100到B200，tensor core TFLOPS提升7.2x，HBM BW提升5.1x，但NVLink BW仅提升3x（intra-node）—通信成为瓶颈的硬件根源。

术语一般如何实现？如何使用？
CUDA层面：IPC (cudaIpcGetMemHandle) 或 VMM (cuMemCreate + export fd) 建立跨进程虚拟地址映射。VMM还支持multicast object (cuMulticastCreate)利用NVSwitch broadcast和reduction。NCCL通过NVLink实现ring/tree/NVLS collective。ParallelKittens直接使用VMM+multicast memory+TMA实现device-initiated通信，绕过NCCL的双向同步和中间缓冲。

涉及论文标题：
- ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels
