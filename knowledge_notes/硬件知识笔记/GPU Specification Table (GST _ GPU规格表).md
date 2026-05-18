## GPU Specification Table (GST / GPU规格表)

术语是什么？通过联网搜索让回答具体和精准。

GPU Specification Table (GST) 是QuCo论文提出的256-byte只读数据block，由GPU vendor在制造时写入（fused into the GPU die），存储该GPU型号的关键architectural parameters。内容包含：memory latencies（L1/L2/DRAM）、clock frequency、LDS size、number of compute units、arithmetic throughput（如FP32 FMA operations per cycle per SIMD）、DRAM bandwidth等QuCo firmware所需的全部硬件参数。GST的数据不暴露给host software或programmer，仅QuCo internal RISC-V firmware可读取——确保GPU proprietary micro-architectural details不外泄。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
GPU Boot Sequence:
  1. GPU powered on → QuCo RISC-V core starts
  2. QuCo firmware 第一条指令: load GST base address
  3. Sequential read GST entries:
     Offset  Description                Value (R9 Nano example)
     0x00    # of Compute Units          64
     0x04    Clock Frequency (MHz)       1000
     0x08    LDS Size per CU (KB)        64
     0x0C    L1 Latency (cycles)         190
     0x10    L2 Latency (cycles)         300
     0x14    DRAM Latency (cycles)       450
     0x18    DRAM Bandwidth (GB/s)       ...
     0x1C    SIMD Muls per Cycle         64
     0x20    Cache Line Size (bytes)     64
     ...     (up to 256 bytes total)
  4. Firmware 将 GST 值存入 data buffer 
     (2KB) → 用于后续所有 kernel launch 
     的 tile/slot 计算
```
GST的architecture-specific nature是实现QuCo portability的关键：同一compiled binary在不同GPU上运行时，QuCo firmware从该GPU的GST读取不同参数→产生不同的tile/slot配置→适配各架构特性（MI-100: 120 CUs 1.5GHz vs R9 Nano: 64 CUs 1.0GHz vs Radeon 530: 6 CUs 1.0GHz）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GST通常通过以下方式实现：
1. **On-die ROM/fuse**：在GPU制造或calibration阶段写入，一次性fuse或mask ROM（类似芯片的device ID或calibration data存储方式）
2. **私密性保证**：GST在QuCo hardware block内部，仅通过QuCo的private address space访问，host software无任何API访问GST内容→保护vendor proprietary micro-architectural details
3. **更新**：通常fixed at manufacturing time；若需更新（如新stepping修正时序参数），通过GPU firmware update更新QuCo firmware中的GST offset mapping，而非修改物理GST本身
4. **容量**：256 bytes足够存储~30-50个关键参数（4-8 bytes each），覆盖tile sizing算法所需的所有architecture-specific常数

类似概念在商业GPU中已有先例：NVIDIA Blackwell的AMP（AI Management Processor）也可能internal access类似的architectural descriptor table来执行context scheduling决策。

涉及论文标题：
- QuCo: Efficient and Flexible Hardware-Driven Automatic Configuration of Tile Transfers in GPUs

术语是什么？

ARM SME (Scalable Matrix Extension) 是 Armv9 ISA 的矩阵加速扩展，在 SVE（Scalable Vector Extension）基础上增加二维矩阵寄存器文件（ZA）和二维矩阵运算指令。SME 的核心目标是加速矩阵乘法——通过 outer product accumulate 指令（MOPA 家族），将两个 vector register 的外积累加到 ZA tile 中。SME 支持的数据类型包括 INT8、INT16、BF16、FP16、FP32、FP64，以及 SME2 新增的 multi-vector 操作和 lookup table 指令。SME 在 Armv9 架构中是可选的（FEAT_SME），目前已知的公开商用实现为 Apple M4 系列处理器。

从硬件架构角度拆解术语：

Apple M4 上 SME 的硬件组织和运转流程（基于 ASM-SpMM 论文和 m4-sme-exploration 公开分析）：
1. **SME compute unit 为 cluster 级共享**：M4 的 P-core cluster（4核）共享一个高性能 SME unit，E-core cluster（6核）共享一个大幅缩减的 SME unit（仅约 P-core 的 1/8-1/16 性能）。这意味着同一 cluster 内同一时刻只有一个线程能使用矩阵加速器。
2. **向量长度**：M4 的 SVL（Streaming Vector Length）为 512-bit，ZA 矩阵寄存器为 512×512 = 4096 bytes，FP64 下可容纳 8×8 矩阵。
3. **ZA tile 划分**：ZA 寄存器可按行/列划分为多个 tile，每个 tile 独立用作 outer product 累加目标。M4 上 FP64 可将 ZA 划分为 8 个 tile。
4. **SME unit 通过 L2 Cache 获取数据**：SME 指令直接从 cluster L2 cache 读取 operand，不经过 per-core L1（与 AMX 继承的架构一致），因此 SME 指令延迟较高，需要多 tile 并发和 prefetch 隐藏延迟。
5. **Streaming SVE 模式**：M4 仅支持 streaming SVE（SME 所需模式），不支持 non-streaming SVE；常规 SIMD 使用 128-bit NEON。
6. **峰值算力**：M4 P-core SME FP64 outer product 峰值约 500 GFLOPS，FP32 约 2000 GFLOPS，FP16→FP32 约 4000 GFLOPS，INT8→INT32 约 16 TOPS。

术语一般如何实现？如何使用？

SME 编程通过 C/C++ intrinsics（`arm_sme.h`）实现，编译器需 Clang 16.0+。ASM-SpMM 论文使用的主要 intrinsics 包括：
- `svld1_f32`：从内存 vectorized load 数据到 Z register
- `svmopa_za32_f32_m`：FP32 outer product accumulate，predicate mask 控制有效元素
- `svst1_hor_za32`：将 ZA tile 按水平方向写回内存
- `_svprfw`：显式软件 prefetch 指令，预取 sparse/dense operand 到 cache

KleidiAI（ARM 官方 AI 库）提供了 SME2 优化的 MatMul kernel（FP32/FP16/INT8），可通过 XNNPACK 在 ExecuTorch 1.0 中使用。

涉及论文标题：
- ASM-SpMM: Unleashing the Potential of Arm SME for Sparse Matrix Multiplication Acceleration

---

