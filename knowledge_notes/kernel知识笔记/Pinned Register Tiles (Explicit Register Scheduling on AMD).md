## Pinned Register Tiles (Explicit Register Scheduling on AMD)

术语是什么？
Pinned register tiles 是 HipKittens 的开发者显式控制 GPU 寄存器分配的机制。绕过 HIPCC 编译器的限制（如不允许 AGPR 作为 MFMA 输入操作数），直接指定每个 tile 映射到哪些物理寄存器（VGPR/AGPR 编号范围），实现 AGPR 直接作为 MFMA 的 A/B operand。在 attention backwards kernel 中，将性能从 855 TFLOPS 提升到 1024 TFLOPS（匹敌 AITER 汇编 1018 TFLOPS）。

从kernel调度角度拆解术语：
开发者通过寄存器范围定义 tile：
```
// 定义寄存器范围: v[24:27], v[28:31], v[32:35], v[36:39]
using Q_ranges = split_many_t<type_list<range<24, 39>>, 4>;
// 绑定到 tile
rt<bf16, 16, 128, row_l, rt_16x32_s, Q_ranges> Q_i;
```
API 与标准 compiler-managed tile 完全一致（load/mma/store），开发者可选择控制粒度。FP6 GEMM 中，explicit register scheduling 完全消除了 54-register scratch spill。

术语一般如何实现？如何使用？
C++ template 元编程实现，ranges 参数映射到具体寄存器编号。仅在编译器限制导致性能损失时使用（attention backward、FP6 等），正常 kernel 仍用 compiler-managed tile。

涉及论文标题：
- HipKittens: Fast and Furious AMD Kernels

---
