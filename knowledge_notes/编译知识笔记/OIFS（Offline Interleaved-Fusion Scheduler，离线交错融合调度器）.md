## OIFS（Offline Interleaved-Fusion Scheduler，离线交错融合调度器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- OIFS 是 CASCADE 的编译器级离线调度器（论文 V 节）：输入 TFHE 应用，构建 BSP 级计算图（同层 BSP 可并行、跨层不可并行），用 Interleaved-Fusion Cost Model（IFCM）评估候选映射的执行时间，用动态规划（DP）求全局最优的二维时空映射矩阵 f(t,c)——f(t,c) 表示映射到"时间层 t × chiplet c"的 HMUX 融合组。目标：最小化 BSP 任务总执行时间，权衡两类惩罚：empty-slot penalty（融合配置不优时映射矩阵出现空槽、浪费计算周期）与 bubble penalty（融合粒度过粗导致流水启动/排空气泡）。与固定融合（所有组同尺寸）不同，OIFS 允许组尺寸不同，容忍轻微负载不均衡以消除空槽。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- OIFS 的运转流程（论文 Fig.11）：
```
1. parse(应用) → BSP 计算图（每 BSP 节点含 HMUX 链 + 轻量非 BSP 操作）
2. B = 可并行 BSP 数; bs = C × intra-HC batch size
3. 对候选 k（融合组数），DP 求最优划分：
   DP[j][r] = 前 j 个 HMUX 用恰好 r 个融合组的最小 T_run
   T_exe(t,c) = max(T_comp × |f(t,c)|, T_comm)   # IFCM：本地计算 vs D2D 时延
   T_task = T_run + T_bubble = ⌈B/bs⌉·Σ_t Σ_c T_exe(t,c) + T_bubble
   剪枝：S_max（最大融合粒度，防大融合气泡）、k_min=C（最小组数，保证 chiplet 级并行）
4. O(n) 扫描 k，取 T_task 最小的 f(t,c) → 输出调度 + BSK 放置
```
- Annotations：步骤 1 是"应用→任务图"（编译前端）；步骤 3 是成本驱动的离线优化（编译优化）；步骤 4 输出指导硬件映射与 BSK 放置（代码生成/部署）。问题被形式化为受约束的 2D 整数划分问题（Completeness：Σ|f(t,c)|=n）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：IFCM 是解析成本模型（T_comp 为单 HMUX 计算时间、T_comm 为 ICT 的 D2D 通信时延、bubble 随融合粒度增长），DP 表 DP[n][k] 预计算后 O(n) 线性扫描选最优 k——整体可扩展（S_max/k_min 剪枝保证可计算）。使用：集成在 CASCADE 的 cycle-accurate 模拟器中（模拟器先调用 OIFS 求映射、再逐 cycle 执行），离线完成、无运行时开销；是"架构-调度协同设计"中的调度侧，与 Interleaved-Fusion 映射策略（见 kernel调度 库条目）配套。评估对比：OIFS vs Segmented HMUX Mapping（SHM，n 均分 C 段，bubble 大、D2D 带宽利用率仅 7.7%）vs Fixed-Fusion Mapping（FFM，固定组尺寸，empty-slot 惩罚）——DeepCNN-50 参数集 I：OIFS 总执行 4.08 ms、HC 利用率 95.9%、D2D 利用率 76.8%（FFM 90.7%/75.7%、SHM 76.8%/7.7%）；DeepCNN-50 上 OIFS 相对"无 OIFS 的 naive 交错"额外 4.1×（共 53.5× vs 单片串行）。

涉及论文标题：
- Unlocking Pipeline Parallelism for Bootstrapping: A Pipelined Multi-Chiplet TFHE Accelerator
