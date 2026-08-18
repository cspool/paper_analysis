## Mask-Based In-NAND Computing（mask 门控的 OU 映射与专家放置调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mask-based in-NAND computing 是 DIAMoND 针对"固定 NAND 阵列尺寸 vs MoE 多样化矩阵尺寸"失配提出的灵活计算机制，由三部分组成：① Operation Unit（OU）——把单个 NAND 平面层划分为规则矩形计算单元，尺寸 H=min{ρ_in,d_min}、W=min{ρ_out,d_min·QB}（ρ_in/ρ_out 为硬件可用输入/输出维、d_min 为模型最小矩阵维、QB 为量化位数）；Mixtral-8x7B@512 并行度 → 每 OU 512 TSG × 4096×8 BL、每平面 4×4 OU 阵列。② Mask 设计——利用 NAND string 串接结构：两层 WL 同选（一层权重、一层 mask），只有两层 cell 都导通才形成电流通路，等效权重与 mask 的 AND 门控；对角线/反对角线循环 mask 使 4×4 OU 阵列只需 4 个 read cycle 并行跑完 16 个 OU（无 mask 需 16 个）；Alg.1 mask 生成：方阵用对角/反对角循环模式，C<R 补列成方阵后忽略补列，C>R 拆成多个方阵再合并；mask 开销 4~64 层、1.7%~27.6% 存储（比 sparse mapping 省 2~4× 且专家组合更灵活）。③ 映射与部署——Alg.2 Round-Robin + Mask-Guided：子矩阵轮转分配到多平面 OU 阵列，OU 内按输入竞争分组、依 mask 模式放置；再以 List Scheduling 调度 Transformer 数据流图（顶点=权重矩阵、边=依赖）确定同 cycle 联合部署，同 plane 专家组成 Expert Group。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Alg.2 简化伪代码（子矩阵集合 S、OU 阵列 {OU_1..OU_N}、WL 层集 {wl_1..wl_J}、mask 集 {M_1..M_K}）：
```
Group S by input contention            # 共享输入的矩阵分组
i <- 1; T <- {}
for s in S:
    (j, k) <- (1, 1)
    while s not assigned:
        if OU_i has free OU at wl_j under mask M_k:
            assign s to it; T.update; break
        else:
            k <- (k+1) mod K            # 换下一个 mask
            if k wraps to 1: j <- j+1   # mask 耗尽换下一 WL 层
    i <- (i+1) mod N                    # 轮转下一个 OU 阵列
```
计算过程例子（Mixtral 单个专家的 Down-Projection 矩阵，2 die × 4 plane）：@512 并行度 → 矩阵切 28 个子矩阵 → Round-Robin 分配到各 plane 的 4×4 OU 阵列 → mask 引导把同输入子矩阵对齐到对角 mask 位置 → 一次 read cycle 内 16 个 OU 并行输出、28 子矩阵 2 cycle 完成；@1024/2048 时 OU 阵列布局与切分自适应变化（4×2、2×2 OU）。List Scheduling 部署：把去掉非 in-NAND 算子（W_Q 等）的 Transformer 数据流图按拓扑序贪心调度——依赖满足的矩阵进 ready list、只要 OU 资源够就同 cycle 调度，同 cycle 矩阵子矩阵拼接后整体映射。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：映射与调度离线完成（部署即权重编程布局），运行期只需按 AES 结果选 mask 组合执行 read cycle；mask 状态由硬件 Dynamic Mask Selector（Priority Queue/Conflict FIFO/Mask Pattern RAM/Pattern State Handler）管理。使用方式：把任意形状权重矩阵对齐到固定 NAND 阵列——Mask 设计使解码加速至多 1.73×（vs 无 mask 的 Base 异构架构）；与 AES 联合后 FFN 层固定 3 cycles（Up/Gate/Down 各 1 read cycle）、合计加速 1.95×；专家冲突率从 Mask-only 的 10.2%~93.5% 降至接近 0。评估：SSDsim 基座 cycle-accurate 模拟器。局限：mask 层占用 NAND 存储（1.7%~27.6%），矩形 OU 阵列的 mask 归约（补列/拆方阵）有映射复杂度。

涉及论文标题：
- DIAMoND Dynamic Inference for Adaptive Edge MoE with Heterogeneous In-NAND and Near-DRAM Compute Architecture
