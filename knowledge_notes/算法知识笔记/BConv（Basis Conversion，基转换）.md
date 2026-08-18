## BConv（Basis Conversion，基转换）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- BConv（Basis Conversion，基转换）是 RNS-CKKS 中把一个 RNS 基（l1 个模数）下的多项式转换到另一 RNS 基（l2 个目标模数）下的算子：先在原始基下与常数相乘，再在目标基下与常数相乘并做模归约，复杂度 O(l1·l2·N)。BConv 是 ModUp/ModDown 的内部核心（模提升/降回本质就是基转换），计算密集型（算术强度 1.60 ops/byte）。
- 在 keyswitch 的 ModUp/ModDown 流水（INTT→BConv→NTT）中，BConv 同时处理来自多个多项式的系数（对分解组内所有 limb 各收一个系数做流水树归约），与 NTT 的高多项式内并行形成互补的并行模式——这一差异是 HE² 设计 tree-based BConvU 并做 NTTU/BConvU 吞吐匹配的核心动机（BConvU 每分解组通常需要不到 15 个 limb）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 一次基转换（原始基 B1 的 l1 个模数 → 目标基 B2 的 l2 个模数）的计算过程：
```
# 输入：a 在基 B1 下的表示 a = (a mod q_1, ..., a mod q_{l1})
# 第一步：原始基下常数乘（解基数重建系数的缩放）
for i in 1..l1:
    s_i = a_i * c_i mod q_i          # 常数乘，得到"缩放后的剩余"
# 第二步：目标基下常数乘 + 归约（逐目标模数重建）
for j in 1..l2:
    a'_j = ( Σ_i s_i * b_{j,i} ) mod p_j   # 常数乘 + 模归约
return a' = (a'_1, ..., a'_{l2})          # 基 B2 下的表示
```
- Annotations：两步常数乘 + 归约正是复杂度 O(l1·l2·N) 的来源；ModUp = BConv（Q→PQ·dnum 域）、ModDown = BConv + 模归约（回 Q 域）；hoisting 交换模域后 BConv 的目标模数变多、计算量上升（见"Hoisting"条目）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：软件库中作为 keyswitch/relinearization 的内部步骤（SEAL/OpenFHE 的 BaseConv）；硬件上 HE² 用 tree-based BConvU——每单元每周期从分解组所有 limb 各收 1 个系数、做流水化树归约，吞吐 672 w/ns（对比 SHARP 16384 w/ns 但配合 NTTU 吞吐匹配仍追平 IRF 关键路径性能）。使用：任何需要跨 RNS 基变换的场合（ModUp/ModDown、乘后降模）；HE² 中 BConv 全部由 xPU 执行（ComOps），与 xMU 的 MemOps 通过 1 TB/s HBM 交换中间结果。
- HyperDrive 补充视角（ISCA'26，GPU 上 BConv 的两阶段分解与 NTT 融合）：采用 fast BConv [6]，把基转换拆成两阶段——BConv1（EWMult，与前置 INTT2 融合为 INTT2-BConv1 kernel）与 BConv2（矩阵乘法，与后置 NTT1 融合为 BConv2-NTT1 kernel）；BConv2 与 BConv1 不直接融合（BConv2 的矩阵乘法结构会使不同 block 基于同一共享输入重复计算）。融合前提是 Row-Major NTT 消除了 NTT 的多 pad 约束：BConv2-NTT1 kernel 中每 thread block 处理单个 limb i、BConv 约减沿 α' 维（GMEM→Reg→SMEM），SMEM 中间系数直接喂 NTT Stage-1（SMEM→Reg→GMEM），避免把 L+α-α' 维中间数据物化到 GMEM（Alg. 2）。BConv 的 stall long scoreboard 占比 60.6%（off-chip 访存）是融合的主要收益点（图 5）。

涉及论文标题：
- HE^2: A Communication-Light Heterogeneous Architecture for Efficient Fully Homomorphic Encryption
- HyperDrive: Hierarchical Exploitation of Memory Efficiency for GPU-Based FHE Acceleration
