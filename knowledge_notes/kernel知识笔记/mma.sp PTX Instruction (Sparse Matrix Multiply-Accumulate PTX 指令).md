## mma.sp PTX Instruction (Sparse Matrix Multiply-Accumulate PTX 指令)

术语是什么？
`mma.sp` 是 NVIDIA PTX ISA 中用于调用 Sparse Tensor Core（SpTC）的 warp-level matrix multiply-accumulate 指令，从 PTX ISA 7.0（CUDA 11.0+，SM80 Ampere）开始可用。它允许程序员在 CUDA inline PTX assembly 中直接触发 SpTC 硬件执行 2:4 结构化稀疏矩阵乘法。与标准 `mma.sync` 指令的关键区别在于：(1) A 操作数必须是压缩稀疏格式（data+metadata pair）；(2) metadata 作为独立操作数传入，编码每 4 元素组中 non-zero 值的位置；(3) B 操作数必须是密集矩阵；(4) 计算吞吐为同 shape 密集指令的 2×。

从 kernel 调度角度拆解术语：
`mma.sp` 在 Samoyeds kernel 中的使用流程伪代码：

```
// Samoyeds kernel 中 mma.sp 的使用
// 场景：C[m][n] += A_sparse[m][k] × B_sparse[k][n]
// A 已编码为 data[m][k/2] + metadata[m][k/2] (2-bit each)
// B 通过 SEL 选择有效列，packed in transposed format

// Step 1: 从 SMEM 加载数据到寄存器（使用 ldmatrix）
// ldmatrix 按 SpTC spec 排列：每个线程持有 A 的特定片段
asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];"
    : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(smem_addr));

// Step 2: 加载 metadata 到寄存器
// 每个 32-bit 寄存器包含 16 个 2-bit metadata（Samoyeds 自定义 packing）
ld_metadata_to_reg(metadata_reg, metadata_smem_addr);

// Step 3: 调用 mma.sp 执行稀疏 MMA
// m16n8k32: M=16 rows of A, N=8 cols of B, K=32 (50% sparse → 16 effective)
asm volatile(
    "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, "     // D = C registers (4×f32 = 16×8×f32/32-threads)
    "{%4,%5,%6,%7}, "     // A = compressed sparse data (4×f16×2)
    "{%8,%9}, "           // B = dense matrix fragment (2×f16×2)
    "{%10,%11,%12,%13}, " // C = accumulator (4×f32)
    "%14;"                // metadata operand for A sparsity
    : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
    : "r"(a0_reg),"r"(a1_reg),"r"(a2_reg),"r"(a3_reg),
      "r"(b0_reg),"r"(b1_reg),
      "f"(c0_init),"f"(c1_init),"f"(c2_init),"f"(c3_init),
      "r"(metadata_reg)
);

// Step 4: Data stationary shuffle（跨越 Sub-Row 边界时）
// 每 V/k_h 次迭代，按 indices 矩阵 shuffle C 寄存器
if (compute_iter % (V / k_h) == 0) {
    // C 寄存器重新映射到正确的行
    shuffle_C_registers(C_regs, indices);
}
```

Samoyeds kernel 中 `mma.sp` 的调度关键点：
1. **Tiling 对齐**：innermost tile `(m_i, k_i, n_i)` 必须满足 mma.sp 指令形状——`m16n8k32`（FP16→FP32）或 `m16n8k16`（更小 tile，延迟更低但吞吐不如 k32）。
2. **Pipeline overlap**：fetch stage（cp.async GMEM→SMEM）与 compute stage（ldmatrix + mma.sp）通过 CUDA pipeline group 机制 overlap。
3. **Metadata packing**：2-bit metadata 需自定义 packing 方案（Samoyeds 将 16×16 metadata 子矩阵映射为 32-bit 对齐的 memory transaction），否则无法配合 ldmatrix 使用。
4. **Data stationary**：C 保持于寄存器跨越多轮 compute iteration；仅在 Sub-Row 边界处 shuffle。

术语一般如何实现？如何使用？
- **支持架构**：SM80+（Ampere A100/A30/RTX 3090 等）、SM89（Ada Lovelace RTX 4070/4090）、SM90（Hopper H100，使用 `wgmma.mma_async.sp` 替代）。AMD CDNA3 有等效指令但语法不同。
- **支持数据类型**：FP16×FP16→FP32、BF16×BF16→FP32、TF32×TF32→FP32、INT8×INT8→INT32。不支持 FP8（Hopper 新增但 mma.sp 尚无 FP8 变体）。
- **约束**：(1) A 操作数需 16-byte alignment（`layout.aligned`）；(2) 每个 warp（32 线程）协作完成一条 mma.sp 指令，各线程持有矩阵的部分 fragment；(3) metadata 格式为每 4 元素 2-bit，每 32-bit 寄存器存 16 个 2-bit 向量；(4) 稀疏仅支持 A 侧（Ampere），Hopper 后扩展但仍是 A-only。
- **集成方式**：可直接在 .cu 文件中用 inline PTX assembly 调用，或通过 CUTLASS/cuSPARSELt 等高层库间接使用。Samoyeds 的 kernel 用 NVCC 编译为 .so，通过 pybind11 注册为 Python module。

涉及论文标题：
- Samoyeds: Accelerating MoE Models with Structured Sparsity Leveraging Sparse Tensor Cores
