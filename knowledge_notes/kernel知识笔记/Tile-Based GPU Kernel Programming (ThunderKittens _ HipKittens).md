## Tile-Based GPU Kernel Programming (ThunderKittens / HipKittens)

术语是什么？
Tile-based programming 以 tile（二维数据块）为基本数据结构，提供 PyTorch/NumPy 风格的 bulk 操作符注册 tile 上。ThunderKittens (NVIDIA) 和 HipKittens (AMD) 通过 C++ template 元编程实现，内部直接包装 PTX/CDNA assembly。核心抽象：register tile (rt_bf/rt_fl)、shared memory tile (st_bf)、load/store operators 和 compute operators (mma、exp、add 等)。

从kernel调度角度拆解术语：
HipKittens tile 接口示例（BF16 GEMM）：
```
// shared memory tiles (double buffered)
st_bf<128, 64, st_16x32_s> As[2][2], Bs[2][2];
// register tiles
rt_bf<64, 64, row_l, rt_16x32_s> A_tile, B_tile;
// accumulator
rt_fl<64, 128, col_l, rt_16x16_s> C_accum[2][2];

G::load(Bs[t][0], g.b, ...);         // HBM→LDS
load(B_tile, subtile(Bs[t][0], ..)); // LDS→register
mma_ABt(C[0][0], A_tile, B_tile, C[0][0]); // compute
store(g.c, C[0][0], ...);            // register→HBM
```
框架自动处理 AMD 特有复杂性：异构 MFMA layout、phase/bank behavior、buffer_load swizzle 地址计算。tile 抽象已被验证可从 NVIDIA 移植到 AMD，表明统一的 tile-based 编程模型可能成为跨厂商通用 kernel 开发范式。

术语一般如何实现？如何使用？
HipKittens 是 C++ header-only 库（https://github.com/HazyResearch/HipKittens），`#include` 使用。ThunderKittens (https://github.com/HazyResearch/ThunderKittens) 用于 NVIDIA。Python bindings 通过 pybind11 集成到 PyTorch。tile 的 row/col 尺寸必须为 MFMA 形状的整数倍。

涉及论文标题：
- HipKittens: Fast and Furious AMD Kernels
- ThunderKittens: Simple, Fast, and Adorable Kernels

---
