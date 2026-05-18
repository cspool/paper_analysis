## Gather Tasks Paradigm

术语是什么？

"Gather Tasks"（任务聚合/拼接）是 Uni-STC 提出的稀疏张量核调度范式，核心思想是将稀疏计算的控制流从"gather data"（将稀疏数据 gather 成固定形状任务后送入 MAC array）转为"gather tasks"（生成轻量控制 task，将多个低负载任务灵活拼接后统一执行）。传统稀疏 TC（如 DS-STC、RM-STC）的 gather data 模式遇到长行、窄行或双边稀疏时，固定形状任务导致大量 MAC 利用率不足的周期。Gather tasks 则通过 BBC 格式的两级 bitmap + TMS/DPG 的任务拆分 + SDPU 的 segmented dot-product 拼接，使每周期 MAC 利用率更高且中间数据传输量更小。

从kernel调度角度拆解术语：

Gather data vs Gather tasks 的调度流程对比：

**Gather data 模式（RM-STC / DS-STC）：**
```
for each row in A (warp-level):
    // Step 1: 从 DRAM gather 该行和对应 B 列的数值到 SMEM/register
    load A_row_vals = gather(A[row_indices], DRAM)
    load B_col_vals = gather(B[col_indices], DRAM)
    
    // Step 2: 将数值组织成固定形状的 T2/T3 任务
    task = pack_to_fixed_shape(A_row_vals, B_col_vals, task_shape)
    // 问题：如果 A_row 非零少（短行），task_shape 填不满 → MAC 空转
    //      如果 A_row 非零多（长行），一个 T2/T3 装不下 → 拆成多个独立任务
    //      多个独立任务无法跨 K 维拼接 → 增加无效访存和 network traffic
    
    // Step 3: 送入 MAC array 执行
    result = MAC_array.execute(task)
```

**Gather tasks 模式（Uni-STC）：**
```
// 软件侧：BBC 格式已在预处理时完成 tile 组织和 bitmap 编码
stc.load.meta...   // 仅加载轻量 bitmap metadata 和 values 到硬件 buffer

// 硬件侧自动执行 task generation + concatenation:
// TMS: 从 top-level bitmap → 动态判断 A/B tile 沿 K 维的重叠情况 → 
//      选择拼接策略（outer-product vs row-major ordering）
//      → 生成灵活形状的 T3 (4×4×4) tasks → 入 Tile queue
// DPG: 弹 T3 → 解析 bottom-level bitmap → 生成 T4 (1×1×4) task code
//      → 以 Z-shaped order 入 Dot-product queue → 
//      低负载 task 自然拼接
// SDPU: 弹 T4 → 执行 segmented dot-product → 
//       同 C_target 自动预合并 → 减少写回

// 关键：queue 中存的是 task code（8-bit）而非中间乘积 → 低带宽
```

量化效果：RM-STC 和 DS-STC 分别有 62.78% 和 61.68% 的周期 MAC utilization < 50%，Uni-STC 的低利用率周期比例显著降低。在 SuiteSparse 全量矩阵上，Uni-STC 对 DS-STC/RM-STC 的几何平均 speedup 为 3.35x/2.21x，energy reduction 为 1.97x/1.27x。

术语一般如何实现？如何使用？

Gather tasks 范式要求软硬件协同：(1) 软件侧用 BBC 格式存储稀疏矩阵，bitmap 结构直接供硬件解析；(2) 硬件侧实现 TMS（task merge & split）、DPG（dot-product generation）和 SDPU（segmented dot-product）三级流水线；(3) 指令接口（UWMMA）支持异步 task generation。关键 trade-off 是新增硬件面积和 BBC 预处理成本 vs. gain（MAC utilization 提升 + 中间产品网络带宽节省）。Uni-STC 论文评估额外面积约 2.12%，预处理时间可被迭代应用摊销。

涉及论文标题：
- Uni-STC: Unified Sparse Tensor Core

