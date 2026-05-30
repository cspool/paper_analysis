## MFMA (Matrix Fused Multiply Add) — AMD Matrix Core Instruction

术语是什么？
MFMA 是 AMD CDNA GPU 的矩阵核心指令，执行 D=A*B+C。由 wave 的 64 线程协作完成。AMD 使用较小形状（16x16x32、32x32x16、16x16x128）vs NVIDIA 的 256x256x16，且各形状使用完全不同的 thread-to-element mapping（无 NVIDIA 的统一 16x16 core matrix 结构）。CDNA4 新增 FP6/FP4 scaled MFMA。

从kernel调度角度拆解术语：
HipKittens BF16 GEMM 中，每个 warp 使用 16x16x32 MFMA 计算 128x64 输出子 tile：
```
mma_ABt(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
// 底层: v_mfma_f32_16x16x32_bf16 指令, 16 cycle 发射延迟
```
HipKittens 默认用最小 MFMA 形状（16x16x32）以最大化调度灵活性 + deep pipeline，与 NVIDIA 偏好大 MFMA 形状（利用 wgmma 从 shared memory 直接矩阵乘）相反。

术语一般如何实现？如何使用？
通过 LLVM builtin（__builtin_amdgcn_mfma_*）在 HIP kernel 中使用。HipKittens 的 mma_ABt 等自动选择正确指令变体。AMD Matrix Instruction Calculator（https://github.com/ROCm/amd_matrix_instruction_calculator）可查询各形状的寄存器布局。

涉及论文标题：
- HipKittens: Fast and Furious AMD Kernels

---
