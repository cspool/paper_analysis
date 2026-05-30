## AMD CDNA Architecture (CDNA3 / CDNA4)

术语是什么？
AMD CDNA（Compute DNA）是 AMD 面向数据中心和 HPC 的 GPU 计算架构。CDNA3 用于 MI300X/MI325X（5nm+6nm，146B 晶体管，256GB HBM3），CDNA4 用于 MI355X（3nm+6nm，185B 晶体管，288GB HBM3E）。CDNA4 新增 FP6/FP4/MXFP 数据类型，peak BF16 2.5 PFLOPS，MXFP6 10.1 PFLOPS。

从硬件架构角度拆解术语：
CDNA4 MI355X 硬件组织：256 CU → 8 XCD chiplet（每 XCD 32 CU + 4MB L2 cache）→ LLC（全局 Last Level Cache）→ 288GB HBM3E (8.0 TB/s)。每 CU 含 4 SIMD，每 SIMD 512 个 32-bit 寄存器 + Matrix Core。与 NVIDIA B200 对比：wave 64 vs warp 32，register 512 vs 256/SIMD，MFMA 16x16x32 vs WGMMA 256x256x16，静态寄存器分配（不支持 reallocation），buffer_load（等价 TMA 但绕过寄存器文件），无 wgmma 从 shared memory 直接矩阵乘。在 kernel 执行中，thread block→CU→SIMD→wave→MFMA/ALU/load store 单元流水线执行。

术语一般如何实现？如何使用？
通过 ROCm 软件栈管理（HIP runtime + AMDGPU driver）。编译目标：gfx942 (CDNA3 MI300系列) / gfx950 (CDNA4 MI355X)。ISA 手册公开。MFMA 通过 __builtin_amdgcn_mfma_* 或 inline assembly 使用。

涉及论文标题：
- HipKittens: Fast and Furious AMD Kernels

---
