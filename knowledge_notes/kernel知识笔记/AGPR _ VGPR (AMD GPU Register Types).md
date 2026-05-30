## AGPR / VGPR (AMD GPU Register Types)

术语是什么？
AGPR（Accumulator GPR）和 VGPR（Vector GPR）是 AMD CDNA SIMD 的两种寄存器。每 SIMD 512 个 32-bit 寄存器，单 wave 时分为 256 VGPR + 256 AGPR。VGPR 用于向量运算和访存，AGPR 用于 MFMA 累加器。HIPCC 不允许 AGPR 作为 MFMA 输入，即使硬件支持，导致编译器插入冗余 v_accvgpr_read 指令。HipKittens 通过 pinned register tiles 绕过此限制。

从kernel调度角度拆解术语：
在 attention backwards 中，MFMA 累加结果在 AGPR，后续需做 softmax vector 运算，HIPCC 必须插入 v_accvgpr_read AGPR→VGPR 搬移。Pinned tiles 直接指定 AGPR 范围作为 MFMA 输入，消除搬移开销。LLVM Dec 2025 patch (PR #170335) 开始自动支持 VGPR→AGPR rewrite。

术语一般如何实现？如何使用：
由硬件和编译器管理，__launch_bounds__ 限制每 wave 寄存器数。HipKittens pinned tiles 在需要精确控制时使用。CDNA4 新增的 scaled MFMA 支持 FP6/FP4 时，AGPR 管理更加关键。

涉及论文标题：
- HipKittens: Fast and Furious AMD Kernels

---
