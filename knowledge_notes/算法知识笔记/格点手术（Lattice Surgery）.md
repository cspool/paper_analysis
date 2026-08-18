## 格点手术（Lattice Surgery）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
格点手术（lattice surgery，Horsman et al. 2012）是 surface code 架构上实现多逻辑量子比特操作（尤其是联合逻辑 Pauli 测量）的主流容错方案：在相邻代码 patch 的边界临时修改稳定子集合——merge 阶段沿边界加入联合稳定子把两个 patch 耦合（其乘积等于联合逻辑可观测量，如 ∏S_k^(XX)=X_L^(L)·X_L^(R)），split 阶段恢复原 patch；奇偶测量结果作为经典边信息用于更新 Pauli frame，无需物理纠错。由联合测量可构造 CNOT/S/H 等 Clifford 门；非 Clifford 门（T）则经 gate teleportation 消费 magic state 实现。时间成本与码距 d 线性相关（约 d 轮 code cycle 或 "code beats"）。Web：arXiv:1808.02892（Litinski, Quantum 2019）把表面码操作抽象为 tile-based game——patch 占 tile，虚线边=X 算子、实线边=Z 算子，操作含单 patch 测量（0 代价）、multi-patch 测量（1 代价）、patch deformation（enlarge 1/shrink 0），时间步=surface-code cycle（d 轮测量）。Triage 论文用 multi-patch measurement / patch rotation / idle 作为指令集（[38] 编译器）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
格点手术是 Triage 空间并行性的来源：lattice surgery 临时合并相邻 patch 并在边界测量联合稳定子，使错误在合并区域内空间相关——单解码器若把合并后的大 volume 整体解码会承受超线性复杂度惩罚且无法并行。Triage 的做法是把合并体积按 d×d patch × d 轮切成 slice 图，边界相邻的 slice 之间建立互斥边（空间邻居，同层最多 4 个）：
```
# 一个多 patch parity measurement 的 slice 化（Triage 视角）
for t in 1..T_rounds:                    # 每个 syndrome 测量周期
    for p in merged_patches:             # 合并区内的每个逻辑 patch
        slice S(t,p) → 顶点 V
        对同 patch 的 S(t-1,p) / S(t+1,p) 加时间互斥边（时间邻居）
        对相邻 patch p' 的 S(t,p') 加空间互斥边（lattice surgery 边界）
图 G=(V,E) 二染色 → 偶/奇两个独立集 → 各自并行解码
```
每个 slice 的窗口缓冲（要保留的边界 syndrome 量）由未解析邻居数（degree）决定，直接决定其解码延迟 t_dec=A·volume^α（α=1.17）。这样多量子比特操作从"一个巨大的不可分割解码任务"变成"一组可并行的小任务"，突破时间并行无法拆分多量子比特操作的瓶颈。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：量子硬件上沿 patch 边界测量联合稳定子（merge/split），编译器（Litinski 风格）把逻辑电路编译成 lattice surgery 操作序列（LLI：multi-patch measurement、patch rotation、idle）并按 tile 布局放置 patch；调度器（如 Triage）决定哪些 slice 同时解码。Web：Watkins et al. 的高性能 surface-code 编译器（Quantum 2024）、LeBlond et al.（ACM TQC 2023）、Hirano & Fujii 的 locality-aware PBC（arXiv:2504.12091）都基于此抽象。Triage 用它做 benchmark 空间维度的并行性来源，并指出空间并行解码 [27] 是它的直接前身之一。

涉及论文标题：
- Triage An Adaptive Parallel Window Decoding Scheduler for Real-time Fault-Tolerant Quantum Computation
