## ZA Matrix Register / ZA Tile

术语是什么？

ZA 寄存器是 ARM SME 的核心二维矩阵累加器，大小为 SVL×SVL bits（Apple M4 上为 512×512 = 32KB）。ZA 寄存器可被逻辑划分为多个 tile，每个 tile 作为 outer product 指令的独立累加目标。Tile 的划分粒度由 outer product 指令的操作数类型和 SVL 决定：FP64 下每行 8 个元素（512/64=8），ZA 被划分为 8×8 的 tile 阵列。

从硬件架构角度拆解术语：

ASM-SpMM 中 ZA tile 的使用流程（FP64，SVL=512）：
1. **ZA 初始化**：执行 SpMM row window 前，清空对应 ZA tile（svzero_za 或直接覆盖）
2. **Tile 分配**：FP64 下 8×8 的 ZA 可容纳 1 个 8×8 tile（全矩阵）或多个更小 tile（多 tile 并发模式）。每个独立的 outer product 可映射到不同 ZA tile/slice
3. **累加过程**：对每个 compressed slot，svmopa_za64_f64_m 将 sparse vector（Z register）× dense tile fragment（Z register）的外积累加到指定 ZA tile，predicate mask 控制有效元素
4. **写回**：所有 slots 处理完成后，svst1_hor_za64 将 ZA tile 水平逐行写回输出矩阵 C
5. **多 tile 并发**：kernel 可将多个 independent outer products 流水线化到不同 ZA tile，提高 ZA/Z register 占用率，同时剩余 Z register 做 operand streaming（预加载下一 slot 数据）

术语一般如何实现？如何使用？

ZA 寄存器通过 SME intrinsics 操作。关键要点：(1) ZA 在 streaming SVE 模式下才可用；(2) ZA 的访问粒度由指令决定——horizontally（按行，svst1_hor_za）或 vertically（按列，svst1_ver_za）；(3) 多 tile 并发需手动管理 tile 分配和依赖，无硬件自动调度；(4) ZA tile 间无数据通路——tile 间累加需通过 Z register 中转。

涉及论文标题：
- ASM-SpMM: Unleashing the Potential of Arm SME for Sparse Matrix Multiplication Acceleration

---

