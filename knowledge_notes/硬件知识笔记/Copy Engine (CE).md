## Copy Engine (CE)

术语是什么？
Copy Engine (CE) 是 NVIDIA GPU 中专用于执行主机内存与设备内存之间数据拷贝（DMA 传输）的硬件单元。它独立于执行 CUDA kernel 的 Execution Engine (EE) 运行。在现代多GPU系统中，CE也用于GPU间数据拷贝（GPU-to-GPU via NVLink），支持host-initiated的大块连续内存传输。CE通过专用的DMA硬件实现，不占用SM计算资源，是三种主要的inter-GPU传输机制之一（另两种为TMA和register-level指令）。

从硬件架构角度拆解术语：
CE存在于每个GPU上，独立于SM阵列，通过NVLink/NVSwitch fabric直接访问peer GPU HBM。ParallelKittens论文通过微基准测试量化了CE的性能特征：在H100上CE可达368.82 GB/s（82%的理论最大450 GB/s），在B200上可达726.13 GB/s（81%的理论最大900 GB/s）。关键限制在于消息粒度——CE需至少256MB的消息才能饱和带宽，对于细粒度通信（如MoE all-to-all，消息通常为KB级别），带宽利用率显著下降（图2显示16KB消息时利用率仅约20%）。因此CE最适合大块连续数据传输（如FSDP weight gathering），但在需要tile级细粒度overlap的场景下效率低下。

术语一般如何实现？如何使用？
CUDA通过cudaMemcpyAsync在指定stream上异步调用CE，与compute kernel做到stream-level overlap。现代NVIDIA GPU通常有1-2个Copy Engine。在multi-GPU setting中，CE通过cudaMemcpyPeerAsync实现peer-to-peer数据搬运。NCCL默认使用CE进行collective操作的数据传输。然而，ParallelKittens发现对于AG+GEMM等融合kernel，使用CE的baseline方法（Triton Distributed, Flux, CUTLASS）在小矩阵尺寸（N=2048）上比非overlap基线更慢，因为CE需要host发起、仅支持连续传输、且无法利用in-network acceleration。

涉及论文标题：
- GPU Scheduling on the NVIDIA TX2: Hidden Details Revealed
- ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels

---
