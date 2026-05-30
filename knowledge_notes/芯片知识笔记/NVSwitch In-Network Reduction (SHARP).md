## NVSwitch In-Network Reduction (SHARP)

术语是什么？
NVSwitch In-Network Reduction是将归约计算（reduction: sum/min/max等）从GPU SM下沉到NVSwitch交换芯片fabric内部执行的技术。NVIDIA SHARP (Scalable Hierarchical Aggregation and Reduction Protocol) 最初用于InfiniBand交换机，第3代NVSwitch (Hopper H100配套) 将其集成到GPU互联fabric中。NVSwitch芯片内含SHARP ALU（400 GFLOPS FP32吞吐），在数据经过fabric时执行归约——GPU发送partial结果到NVSwitch，Switch内部ALU计算归约结果，然后返回给GPU。这使all-reduce从O(N)步ring算法降为O(1)操作。

从芯片设计角度拆解术语：
NVSwitch chip (TSMC 4N, 25.1B transistors, 294mm²) 的SHARP相关硬件组成：
- **64 NVLink 4.0 ports**: 总3.2 TB/s双向带宽
- **SHARP Controller**: 管理最多128个并发SHARP group
- **Embedded SHARP ALUs**: 400 GFLOPS FP32, 支持FP16/FP32/FP64/BF16和整数op (add/min/max/logical)
- **Embedded SRAM**: SHARP计算中间缓冲
- **Enhanced crossbar bandwidth**: 适应SHARP流量

芯片级数据路径（以multimem.ld_reduce为例）：
```
GPU_i → TMA/ld指令 → NVLink port → NVSwitch crossbar
                                    ↓
                              SHARP ALU (collect from all GPUs, reduce)
                                    ↓
                              结果写回crossbar → NVLink → GPU_i (仅归约结果返回)
```
对比传统方法：无SHARP时，每个GPU需写N次(N个peer)或需N-1步ring传输，SHARP将N次传输变为1次读操作（multimem.ld_reduce从multicast memory读→Switch自动收集+归约→返回）。

PK利用SHARP实现all-reduce：通过VMM创建multicast memory → compute SM将partial结果以local address写入各自GPU → communication SM调用multimem.ld_reduce/multimem.red PTX指令从multicast地址读取→Switch自动执行in-network reduction。相比NCCL ring all-reduce（需2(N-1)步、占用16 SM），SHARP方法只需2步、占用~6 SM。

术语一般如何实现？如何使用？
CUDA API流程：cuMulticastCreate → cuMulticastAddDevice → cuMulticastBindMem → cuMemMap（映射multicast address）→ kernel内使用multimem.ld_reduce/multimem.red PTX指令。NCCL 2.27+通过NVLS (NVLink SHARP)和Symmetric Memory集成SHARP。PK直接通过PGL + all_reduce/reduce原语暴露。硬件要求：H100 SXM + NVSwitch (DGX H100)，PCIe版本不支持。B200继续支持5th-gen NVSwitch SHARP。

涉及论文标题：
- ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels
