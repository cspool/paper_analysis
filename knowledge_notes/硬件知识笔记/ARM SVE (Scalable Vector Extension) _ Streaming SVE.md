## ARM SVE (Scalable Vector Extension) / Streaming SVE

术语是什么？

ARM SVE（Scalable Vector Extension）是 ARMv8.2-A 引入的可变向量长度 SIMD ISA 扩展，允许向量寄存器宽度从 128 到 2048 bits（以 128 为步长），实现二进制兼容的向量化代码。SVE 的核心特性包括：(1) predicate-driven 执行——每条向量指令由 predicate register 控制有效 lane；(2) 向量长度不可知（Vector Length Agnostic, VLA）编程模型；(3) gather/scatter、per-lane predication 等高级数据并行指令。Streaming SVE 是 SVE 的一种执行模式，专为 SME 设计——在 streaming 模式下，CPU 放弃部分 SVE 特性（如 FFR、某些 predicate 指令）以换取 SME ZA 寄存器和 outer product 指令的访问权限。

从硬件架构角度拆解术语：

ASM-SpMM 中 SVE 和 SME 的协同使用：
1. **SME path（主路径）**：streaming SVE 模式 + ZA tile → 使用 FMOPA 做 outer product SpMM
2. **SVE vector path（辅助路径）**：对低密度/碎片化 block，SVE/Neon 执行传统 vectorized dot-product/saxpy。SVE 的 predicate-driven 语义使其可以跳过稀疏 block 中的空行/空列
3. **M4 特殊限制**：只支持 streaming SVE（128-512 bit VL），不支持 non-streaming SVE。常规 SIMD 仍使用 128-bit NEON
4. **SVE vs NEON 在 SME 上下文**：NEON 保持 128-bit 固定宽度，与 SME ZA 无法直接交互；streaming SVE 可以通过 Z register 向 MOPA 提供 operand，也可通过 predicate register 做 mask 转换

术语一般如何实现？如何使用？

SVE 编程通过 C/C++ intrinsics（`arm_sve.h`）或自动向量化（编译器如 Clang/Arm Compiler for Linux）。Apple M4 上 SVE 的 VL 为 512-bit（不可动态改变）。ASM-SpMM 的 hybrid kernel 根据 SME/SVE microbenchmark 延迟估计，优先把最稀疏 block 分配给 SVE/Neon vector path，vector 工作量通过 interleaved scheduling 隐藏在 SME 固定执行窗口内。

涉及论文标题：
- ASM-SpMM: Unleashing the Potential of Arm SME for Sparse Matrix Multiplication Acceleration

---

