## Intel AMX / TMUL（Advanced Matrix Extensions / Tile Matrix Multiply）

术语解释
Intel x86 的矩阵计算 ISA 扩展：8 个 1KB 二维 tile 寄存器 + TMUL 矩阵乘指令，Sapphire Rapids 起出货；ATX 论文将其作为 ICA 的代表实例，并在解压用例中让核用 AMX 与 NCA 交错执行。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
AMX 分两块：AMX-TILE（tmm0–tmm7 八个 16 行×64B 的二维寄存器，共 8KB 架构状态）与 TMUL 指令（INT8 的 `TDPBSSD` 族、BF16 的 `TDPBF16PS` 族等，执行 16×16×32 tile 乘加 C+=A×B，Sapphire Rapids 吞吐约 2048 INT8 或 1024 BF16 op/cycle）。状态管理经 XSAVE（XTILECFG 64B + XTILEDATA 8KB），OS 懒保存；编程接口是 intrinsic（`_tile_loadd`、`_tile_dpbssd`、`_tile_stored`）。AMX 的寄存器↔寄存器接口 + 无状态特性使 TMUL 指令可以推测/乱序执行——这是 ICA 抽象的教科书实例。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
论文中的两个角色：(1) 作为 ICA baseline 的模板——论文的"perfect ICA"与 SpMM/SDDMM/GeMM ICA 配置都采用"寄存器 tile 输入 + 核内矩阵单元"的结构，取数依赖核 LSQ；(2) 作为解压用例的核侧计算单元——DECA-like NCA 把压缩 tile 解出来写回 tile 寄存器，核立即用 AMX TMUL 对解出的 tile 执行 GeMM，形成"NCA 解压 → 核 AMX 计算"的细粒度交错流水，任务输入仅 512B–2KB，此时 ATX NCA 相对 LLC OCA 快 18×（小任务下 OCA 调用开销支配）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
使用前提：CPU 支持 AMX（CPUID 检测 AMX-TILE/INT8/BF16 位）、OS 启用（Linux `arch_prctl(ARCH_REQ_XCOMP_PERM)` + XSAVE 状态）、上下文切换必须保存 8KB tile 状态。典型用法：把大矩阵切成 16×64 子块装入 tile 寄存器，循环发 TMUL。局限（论文讨论）：tile 容量 1KB×8 迫使软件切块，取数仍走核通用访存路径，MLP 受限——这正是 NCA 用 UTE 流引擎接管取数的动机。AMX 的 8KB 架构状态也是论文对比 UTE 状态开销（仅 4.5KB）的参照物。
- PowerGrad 补充视角（ISCA'26，AMX 作为 ML 加速平台与性能计数器）：PowerGrad 把 Emerald Rapids（Xeon Gold 5512U）的 AMX 作为"Accelerated"平台（ML 加速支持）的核心硬件，与无 AMX 的 Legacy（Haswell E5-2660 v3）平台对比评估框架可移植性。AMX 相关的两个硬件观测点：(1) 性能建模——功率/性能模型的回归系数训练集用 TorchBench（能触发 AMX 指令）而非 PARSEC 3.0；在线梯度估计额外读两个计数器 exe.amx_busy（AMX 单元忙碌周期）与 fp_arith_inst_retired.vector（向量指令数）来刻画 AMX/向量活动；(2) 度量公平性——1 个 AMX busy 周期计为 16 条指令（等价 16 条向量指令的操作量 [19]），避免 BIPS 低估 AMX 加速的吞吐。结果：Accelerated 平台功率模型 AAE 2.5%（低于 Legacy 的 4.1%，因计数器更细粒度），但单 CPU/节点无本地控制器可用，PowerGrad 收益（9.0%/9.9%）低于双 CPU Legacy（22.9%/23.0%）。

涉及论文标题：
- ATX: Accelerator Task Extensions
- PowerGrad: Hierarchical Power Management for Power-Limited ML Inference Clusters
