## Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search

- baseline方法是什么？
  - **Baseline 方法**：原始 RaBitQ [27] 仅支持高压缩率（32x，每个向量用 D 位表示），以及其简单扩展 RaBitQ(pad)——通过将 D 维向量零填充至 B·D 维再应用 RaBitQ。同时包括传统量化方法：SQ（全局均匀标量量化）、LVQ（逐向量标量量化）和 PQ/OPQ（乘积量化，k=8）。
  - **全栈执行例子（Baseline: RaBitQ(pad) + IVF @ B=4, 8x 压缩）**：
    - **算法pipeline**：将 D 维向量零填充到 B·D 维空间，在扩展空间中构造超立方体顶点码本（±1/√(B·D)），随机旋转后量化。距离估计基于 Lemma 2.1 的无偏估计器，误差界 O(1/√(B·D))——误差随 B 线性根号衰减而非指数衰减，导致即使增加 bit 数也不能有效提高精度。理论要求 Θ(B·D) bits 才能达到误差界 ε，而理论上最优仅需 Θ(D log B) bits。
    - **系统框架**：C++ 实现，IVF 索引（4,096 聚类），Raw vectors → IVF 分区 → RaBitQ(pad) 量化每条向量为 B·D 位 → 存于 RAM。查询时：q → 找最近聚类质心 → 对候选向量批量 FastScan SIMD 估计距离 → 返回最小估计距离的向量。需要访问所有候选的全部 B·D 位才能估算距离，无剪枝。
    - **编译框架**：C++ 由 GCC 11.4.0 编译，-Ofast -march=native，使用 AVX512 SIMD 指令集。
    - **kernel调度**：FastScan [4] 通过 SIMD 批量计算 `<二进制码, query>`，但对 B>1 需额外将 B-bit 整数拆分为多位二进制码分步计算。论文未修改 kernel。
    - **硬件架构**：Intel Xeon Gold 6418H CPU（Sapphire Rapids, 48 cores），1TB RAM。无 GPU/加速器。
  - **Baseline 缺陷**：(i) RaBitQ(pad) 的填充策略将向量升维后再量化，误差界仅 O(1/√(B·D))，B 增加时误差衰减慢，与理论上最优的 O(log B / √D) 存在巨大差距；(ii) 原始 RaBitQ 和 RaBitQ(pad) 需要高压缩率（32x）配合 re-ranking 才能产生合理 recall，不重新排序时 recall 低（<90%），但 re-ranking 需要存储原始向量，违背了压缩省内存的初衷；(iii) SQ/LVQ 在 B<4 时误差比 RaBitQ 大数个数量级，PQ/OPQ 在 B≥4 时精度不如 SQ，且依赖 RAM 查表导致效率低。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **论文方法**：Extended RaBitQ——通过在 D 维空间（而非 B·D 维）中构造包含 2^{B·D} 个向量的码本，将 B-bit 无符号整数网格向量归一化后随机旋转。结合高效量化编码算法（O(2^B·D log D)）和两阶段距离比较（先用 MSB 快速剪枝）。
  - **全栈执行例子（Extended RaBitQ + IVF @ B=5, 6.4x 压缩, >95% recall 无 re-ranking）**：
    - **算法pipeline**：
      - **码本构造**：G = {-(2^B-1)/2 + u | u=0,...,2^B-1}^D（D 维空间中的均匀网格），然后 G_r = {P·y/||y|| | y∈G}。码本向量是随机旋转的单位向量，继承了 RaBitQ 的无偏估计器。误差界理论：∀ε, 仅需 B = Θ(log(ε^{-2}·log(1/δ)/D)) bits，达到渐近最优（对比 RaBitQ(pad) 需要 Θ(ε^{-2}·log(1/δ)) bits）。
      - **量化编码 (Algorithm 1)**：利用 Lemma 3.1 的几何性质（∃ t>0 使得 argmax_y ⟨y/||y||, o'⟩ = argmin_y ||t·o' - y||），仅需枚举至多 D·2^{B-1} 个 critical values（每个维度 i 的临界值 = (x+0.5)/o'[i]），使用最小堆维护 O(log D) 插入/弹出，总复杂度 O(2^B·D log D)。B=7 时百万级 3072 维数据集仅需 ~98 秒。
      - **两阶段距离比较**：ȳ_u 的 MSB ȳ₀ 恰好等于原始 RaBitQ 的二进制码 x̄_b。第一阶段用 FastScan 批量计算基于 ȳ₀ 的粗略距离，误差界已知可判定大部分候选；第二阶段仅对未剪枝候选访问 ȳ_last 增量计算 ⟨ȳ_u,q'⟩ = 2^{B-1}·⟨ȳ₀,q'⟩ + ⟨ȳ_last,q'⟩。距离估计公式：⟨ō,q⟩ = (1/||ȳ||)·(⟨ȳ_u,q'⟩ - (2^B-1)/2 · Σq'[i])。
      - **误差控制**：经验公式 ε < 5.75·2^{-B}/√D（>99.9% 置信度），误差随 B 指数衰减（对比 RaBitQ(pad) 仅 O(1/√(B·D))）。
    - **系统框架**：C++ 实现，IVF 索引。Index: raw vectors → 中心化（每聚类局部质心）→ Algorithm 1 量化 → 分离存储 MSB 和剩余位 → RAM。Query: q → 变换 q' → 找 nprobe 个最近质心 → 第一阶段 FastScan(MSB) → 剪枝 → 第二阶段 剩余位增量计算 → 返回最小估计距离向量。B=5 时 >95% recall（6.4x 压缩），B=7 时 >99% recall（4.5x 压缩），均无需 re-ranking。
    - **编译框架**：GCC 11.4.0，-Ofast -march=native，Ubuntu 22.04 LTS。AVX512 SIMD 批量计算。
    - **kernel调度**：FastScan SIMD 批量计算 ⟨ȳ₀,q'⟩；B=4 或 8 时直接复用现有系统的 4-bit/8-bit 整数与浮点内积实现 [1,17]；其他 B 通过拆分（如 B=9 → 1-bit + 8-bit）实现。论文未修改底层 kernel。
    - **硬件架构**：Intel Xeon Gold 6418H CPU（Sapphire Rapids），1TB RAM。无 GPU/加速器。搜索单线程，索引多线程（96 threads）。
  - **对应关系的核心逻辑**：
    - Baseline 因"升维填充导致误差界 O(1/√(B·D)) 次优" → 论文在原始 D 维空间中构造 2^{B·D} 规模码本，误差界随 B 指数衰减，理论达到 Θ(D log B) 渐近最优。
    - Baseline 因"高压缩率必须 re-ranking 但违反省内存目标" → 论文支持 B=5~7 的中等压缩率（4.5x-6.4x），可独立产生 >95%~>99% recall，无需存储原始向量。
    - Baseline 因"全 bit 参与所有候选评估导致效率低" → 论文利用 MSB=RaBitQ 二进制码的特性，第一阶段用 SIMD 快速剪枝，仅对少数候选做完整距离计算。
    - Baseline 因"SQ/LVQ 精度不足" → 论文在同样 bit 数下 error 比 LVQ 小 1.3x-3.1x（B>6），B<6 时差距更大（可差数量级）。
