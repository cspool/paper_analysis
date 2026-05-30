## GPU Memory Hierarchy for Low-Precision Kernels (低精度Kernel的GPU内存层次)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GPU内存层次（Memory Hierarchy）由三个显式可编程的层级组成：Registers（寄存器）——最快、thread-private，每线程最多255个32-bit寄存器；Shared Memory（共享内存）——block内所有线程共享，比global memory快约100倍，由SRAM实现，每SM典型配置100-228KB；Global Memory（全局内存）——全grid可访问，容量最大（HBM2e/HBM3，数GB至数十GB）但延迟最高。Tilus的thread-block级编程模型显式暴露这三个层次，允许开发者精确控制数据placement和movement，而Triton等高层编译器将此层次抽象隐藏。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。
低精度kernel在GPU硬件上的实际数据流（以NVIDIA L40S为例）：
1. Global Memory (HBM)：权重tensor以紧凑u8格式存储（48 GiB总容量），通过cp.async指令以vectorized访问（v4=16 bytes/cycle）读取
2. L2 Cache：硬件自动缓存最近访问的global memory数据（L40S约98MB L2），软件流水线的预取利用L2减少DRAM延迟
3. Shared Memory (SRAM per SM)：cp.async直接写入SM专属的shared memory buffer（双缓冲，每buffer约BK*BN*bits/8 bytes），共享内存bank conflict通过layout设计避免
4. Registers：ldmatrix或lds指令从shared memory加载到每线程的registers（每线程最多255个32-bit寄存器），View指令在registers内做零开销reinterpretation，Cast指令使用PRMT/LOP3在registers内完成类型转换
5. Tensor Cores：从registers读取operands，执行mma.m16n8k16矩阵乘累加，结果写回registers

整个数据路径：HBM → L2 → Shared Memory (SRAM) → Registers → Tensor Cores → Registers → HBM。Tilus通过CopyAsync流水线重叠了HBM→Shared Memory和Tensor Core计算的时间。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在CUDA编程中，shared memory通过__shared__限定符声明，寄存器由编译器自动分配，global memory通过cudaMalloc分配。Titus在编译时通过Shared Memory Planner计算总shared memory需求并映射各shared tensor，通过Hidet IR精细控制register usage。Titus的CopyAsync利用Hopper/Ada Lovelace架构的TMA（Tensor Memory Accelerator）或Ampere+的cp.async进行异步DMA传输。

涉及论文标题：
- Tilus: A Tile-Level GPGPU Programming Language for Low-Precision Computation

---
