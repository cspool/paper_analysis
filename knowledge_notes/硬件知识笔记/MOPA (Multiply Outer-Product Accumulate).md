## MOPA (Multiply Outer-Product Accumulate)

术语是什么？

MOPA（Multiply Outer-Product Accumulate）是 ARM SME 的核心矩阵运算指令家族。它将两个向量寄存器（Zn, Zm）的外积计算结果，与 ZA 矩阵寄存器中的目标 tile（ZAda）累加，结果写回 ZAda。该指令的关键特征是：(1) 计算外积而非内积——两向量直接形成2D矩阵；(2) 支持独立 predicate mask——Zn 和 Zm 可分别掩码，无效 lane 视为零；(3) 是 destructive accumulate——结果覆盖目标 tile 原有内容。SME 指令变体包括 UMOPA/SMOPA（整数）、FMOPA（浮点）、BFMOPA（BF16）等。

从硬件架构角度拆解术语：

MOPA 的硬件执行流（以 FP64 FMOPA 在 SpMM kernel 中为例）：
1. **操作数准备**：Zn = sparse values vector（从 compressed slot 加载的稀疏值），Zm = dense B tile 的一列（或一行，取决于 outer product 方向），ZAda = 目标 ZA tile
2. **Predicate 应用**：Pn predicate mask 指示 Zn 中哪些元素是有效非零值（ColumnPositionMaskBit 转换），Pm predicate mask 控制 Zm 的参与范围
3. **外积计算**：Zn[i] × Zm[j] → 2D 结果矩阵，累加到 ZAda[i][j]。无效 predicate 位置的乘积为零，不参与累加
4. **关键洞察（ASM-SpMM）**：MOPA 的外积语义天然适合稀疏计算——sparse vector 直接作为外积的一个操作数，predicate 消除空位而非 zero padding 浪费算力。这与 GPU Tensor Core 的 inner-product 语义（需要 left-aligned dense tile）本质不同

术语一般如何实现？如何使用？

MOPA 通过 SME intrinsics 调用，如 `svmopa_za64_f64_m(ZAda, Pn, Pm, Zn, Zm)`。使用要点：(1) ZAda 必须在 streaming SVE 模式下；(2) 需先通过 svld1 类指令将数据从内存加载到 Z register；(3) Pn/Pm 需预先通过 whilelt 或 compare 类指令设置；(4) 多个 MOPA 可流水线化到不同 ZA tile 以提高吞吐；(5) MOPA 延迟较高（Apple M4 上约 10-20 cycles），需要通过多 tile 并发和 compute/prefetch overlap 隐藏。

涉及论文标题：
- ASM-SpMM: Unleashing the Potential of Arm SME for Sparse Matrix Multiplication Acceleration

---

