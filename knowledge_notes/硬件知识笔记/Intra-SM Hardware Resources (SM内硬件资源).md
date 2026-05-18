## Intra-SM Hardware Resources (SM内硬件资源)

术语是什么？

Intra-SM Hardware Resources（SM内硬件资源）指NVIDIA GPU每个Streaming Multiprocessor（SM）内部包含的多种专用计算和存储硬件单元。以Ampere架构A40 GPU为例，每个SM包含：64 FP32 cores（单精度浮点）、32 FP64 cores（双精度浮点）、64 INT32 cores（整数运算）、4 Tensor cores（矩阵乘加加速器，支持FP16/BF16/TF32/INT8/INT4）、16 SFU units（Special Function Unit，处理超越函数如sin/cos/exp/sqrt）、32 LD/ST units（load/store，处理global/shared memory访问）、102,400 bytes shared memory（程序员管理的on-chip scratchpad）、65,536 registers（线程寄存器文件，每个thread最多255 registers）。这些资源在SM内为所有resident blocks所共享，是μShare进行scattered co-location时要匹配和互补的基本单位。

从硬件架构角度拆解术语：

A40 SM的资源容量和CUDA scheduling约束：

```
NVIDIA A40 SM (Ampere GA102):
  Thread capacity:            1536 threads (max warps=48)
  Register file:              65,536 × 32-bit registers
  Shared memory:              102,400 bytes (configurable up to 164KB w/ L1 reduction)
  
  Execution units (per SM partition, 4 partitions/SM):
    Partition 0-3 (each):
      FP32/INT32 cores:  16 FP32 + 16 INT32 (or 32 FP32 in uniform mode)
      FP64 cores:        8 (A40: 2 FP64/SM partition, Ampere reduced vs Volta)
      Tensor cores:      1 (Gen3, supports sparse MMA)
      SFU units:         4
      LD/ST units:       8
      Warp scheduler:    1 (handles 16 warps)

  CUDA scheduling constraints:
    - max blocks per SM: 32 (Ampere, via CUDA compute capability 8.6)
    - max threads per block: 1024 (CUDA limit)
    - max registers per block: min(65536, blocksize × registers_per_thread)
    - max shared memory per block: min(102400, requested_smem)

  μShare的关键约束：
    - blocksize ∈ [32, 1024], multiple of 32 (warp alignment)
    - half-plus for A40: b > 768 → b ∈ {800, 832, ..., 1024}
    - 1/3-plus for A800: b > 683 → b ∈ {704, 736, ..., 1024}
```

不同NVIDIA GPU的SM资源配置差异影响μShare的shaping策略：
- A40/A40/RTX 4090/RTX 3080 Ti：1536 threads/SM → half-plus策略 (b_min=800)
- A100/A800/H200：2048 threads/SM → 1/3-plus策略 (b_min=704)
- 不能对2048 threads/SM的GPU使用half-plus：即使max blocksize=1024，两个1024 block (2048 threads) 仍可在1SM内stacked co-locate

术语一般如何实现？如何使用？

NVIDIA GPU的SM硬件资源通过以下方式管理和使用：
1. **NVIDIA Nsight Compute**：测量SM内各hardware unit的active cycle ratio，μShare的profiler使用该工具获取per-kernel的6种resource utilization
2. **CUDA Occupancy Calculator**：基于blocksize、registers/thread、shared memory/block计算SM的理论occupancy（resident blocks数）
3. **Nsight Systems**：测量kernel launch timing和execution timeline，μShare用它记录tkLaunch for slack calculation
4. **CUDA inline PTX**：直接读取SM ID register (`%smid`) 和clock counter (`%clock64`) 确定block的实际placement和timing
5. μShare的贡献不在于修改SM硬件资源本身，而在于通过软件手段使不同kernel的complementary resource demands在同一SM内同时得到满足

涉及论文标题：
- μShare: Non-Intrusive Kernel Co-Locating on NVIDIA GPUs

