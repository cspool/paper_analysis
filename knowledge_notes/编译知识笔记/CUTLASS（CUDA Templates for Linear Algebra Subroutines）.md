## CUTLASS（CUDA Templates for Linear Algebra Subroutines）

术语是什么？通过联网搜索让回答具体和精准。

CUTLASS（CUDA Templates for Linear Algebra Subroutines）是NVIDIA开源的高性能GPU线性代数模板库，提供C++模板抽象用于构建自定义GEMM（通用矩阵乘法）kernel。CUTLASS将GEMM分解为thread block tile、warp tile和thread-level fragment的层次化tiling结构，通过模板参数配置数据精度（FP16/BF16/INT8/INT4/FP8等）、tile size、pipeline stages等，自动生成针对特定GPU架构优化的kernel。CUTLASS支持Tensor Core的所有精度模式（包括INT4的mma.sync.aligned.m16n8k32等），并提供software pipeline、epilogue fusion（如dequantization+scaling）等高级特性。

从编译框架角度拆解术语：

CUTLASS的层次化GEMM抽象（以INT4 GEMM为例）：
```
// CUTLASS INT4 GEMM template结构
using Gemm = cutlass::gemm::device::Gemm<
  int4b_t,                          // ElementA
  cutlass::layout::RowMajor,         // LayoutA
  int4b_t,                          // ElementB
  cutlass::layout::ColumnMajor,      // LayoutB
  int32_t,                          // ElementC (accumulator)
  cutlass::layout::RowMajor,         // LayoutC
  int32_t                           // ElementAccumulator
>;

// Tiling配置（编译时参数）:
// - ThreadblockShape: {128, 128, 64} (M, N, K tile size)
// - WarpShape: {64, 64, 64} (per-warp M, N, K)
// - InstructionShape: {16, 8, 32} (MMA instruction m, n, k)
// - Stages: 3 (pipeline stages)
// - 编译器使用这些模板参数展开循环、分配register和shared memory
```

在RoMeo中，CUTLASS被用于实现四种separate cross-precision GEMM kernel。RoMeo利用CUTLASS的模板参数配置能力：INT4-INT4 kernel配置较小tile+更多stages（shared memory小→更多register用于ILP），INT8-INT8 kernel配置较大tile+较少stages（shared memory大→限制occupancy）。CUTLASS的epilogue机制用于融合dequantization（scaling factor multiply）和post-mul overwrite。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 开源地址：https://github.com/NVIDIA/cutlass
- 使用方式：C++ header-only模板库，包含在CUDA项目中通过CMake集成。用户定义Element、Layout、TileShape等模板参数，CUTLASS在编译时生成优化kernel。
- RoMeo使用方式：基于CUTLASS开发custom mixed precision GEMM kernel（W4A4/W4A8/W8A4/W8A8），编译为动态库通过Python ctypes/CUDA Runtime API调用。JIT机制根据模型维度auto-tune tiling size和pipeline stages→编译并缓存binary。
- CUTLASS 3.x（Hopper架构优化）支持TMA（Tensor Memory Accelerator）、WGMMA（warp group MMA）和persistent kernel设计。RoMeo使用CUTLASS 2.x targeting Ada Lovelace (RTX 4090)的cp.async + MMA pipeline。
- 与Triton对比：Triton提供更高层的Python-based kernel编写体验（block-level programming），CUTLASS提供更精细的C++ template-level控制（warp-level和instruction-level）。RoMeo对GEMM kernel使用CUTLASS（精细控制），对fusion操作（outlier detection+quantization+packing）使用Triton（开发效率）。

涉及论文标题：
- RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization
- FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection
