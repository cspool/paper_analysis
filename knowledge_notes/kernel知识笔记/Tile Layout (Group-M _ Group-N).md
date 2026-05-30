## Tile Layout (Group-M / Group-N)

Tile Layout 是 GPU GEMM 中 output tile 的空间排列顺序，决定 SM 执行 tile 的顺序以及相邻 SM 访问数据在 L2 cache 中的复用程度。四种基本 layout：Column-Major（按列排 tile）、Row-Major（按行排）、Group-M（以 group size s 将 M 维 tile 编组后排成列）、Group-N（以 group size s 将 N 维 tile 编组后排成行）。Column-Major = (GM, ceil(M/bM))，Row-Major = (GN, ceil(N/bN))。

从kernel调度角度拆解：Tile layout 影响同一 wave 内相邻 SM 的 L2 cache 数据复用——相邻 tile 沿 M 维需要相同 A 行，沿 N 维需要相同 B 列，良好 layout 使这些数据在 L2 中被相邻 SM 共享。HyTiS 分析模型：
- 第一 wave 的 DRAM→L2 流量 V_1 最关键（L2 初始为空）
- Group-M 最优 s：s_opt_GM = min(ceil(sqrt(N_SM · bN/bM)), ceil(M/bM))
- Group-N 类似对称推导
- 最终选择 V_tol = ΣV_i 较小的 layout

计算公式：
```
s_opt_gm = min(ceil(sqrt(N_SM * bN / bM)), ceil(M / bM))
s_opt_gn = min(ceil(sqrt(N_SM * bM / bN)), ceil(N / bN))
// Pick (GM, s_opt_gm) if V_tol_gm <= V_tol_gn else (GN, s_opt_gn)
```

实验：H100 上不同 layout 的 DRAM read 量差异最高 64%，最小 V 的 layout 与最佳性能强相关。HyTiS vs fixed group-M (s=8)：low DRAM read 区 46%→20%，high 区 15%→28%。

术语一般实现：CUTLASS 支持 column-major/row-major 可配置，Triton 用 group-M with fixed s=8，PyTorch Inductor 固定 layout。HyTiS 在 Level-1 自适应选择；Level-2 单 wave 时固定 column-major（单 wave 下 layout 无影响）。layout selection 仅涉及简单数学运算，runtime overhead 可忽略。

涉及论文标题：
- HyTiS: Hybrid Tile Scheduling for GPU GEMM with Enhanced Wave Utilization and Cache Locality
