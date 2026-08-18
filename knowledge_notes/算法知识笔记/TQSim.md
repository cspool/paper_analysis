## TQSim

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- TQSim（Wang, Tannu & Nair，ISCA 2025，"Accelerating Simulation of Quantum Circuits under Noise via Computational Reuse"）是加速 noisy 量子线路模拟的近期工作，TUSQ 的主要同类 baseline。核心方法：BFS + memoization + sampling——把 n qubit、深度 d 的电路 C(n,d) 切成 k 个不相交子电路 {sc_i(n,d_i)}，树根为电路起点、深度 i 节点对应子电路 i 结束时的态矢量集合（所有 ER 的 ensemble）；在树的不同深度采样代表性节点并 memoize 其态矢量直到内存饱和，复用保存的态避免从头重算。
- 与 TUSQ 的关键差异：TQSim 的速度依赖可用内存（能缓存多少中间态），内存饱和时必然损失 fidelity（用统计方法在保真度与速度间取舍）；TUSQ 用深度优先遍历（DFTT）实现纯算法加速、不需 memoization（除非非幺正通道），并用 ER 中间表示区分"可无损消除的冗余"（Tallying/Commutation/DFTT）与"可小损消除的不显著计算"（Pruning，α/β 可调）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- TQSim 流程对比（同一 noisy 电路、S shots、40GB GPU）：
  ```
  # TQSim（BFS + memoization）
  chop C(n,d) 为子电路 {sc_0..sc_{k-1}}，Σd_i = d
  tree = BFS 逐层生成所有 ER 路径
  在每层采样代表节点 → memoize 其态矢量，直到内存饱和（如 30-qubit 态 8GB，40GB 只能存 5 个）
  后续计算复用缓存态；缓存不足时该部分丢保真度
  # TUSQ（DFTT）
  ECM 先消除冗余（Tallying/Commutation/Pruning）→ 唯一电路建成树
  DFS 正向 compute、反向 uncompute 复用共享前缀，零额外内存 → 纯算法加速
  ```
- 实测：时间与内存密集区（p=1%、1M shots、40GB）同 fidelity 误差下 TUSQ 平均/最大 39.32×/3134.31× 快于 TQSim（6 个 benchmark：adder/bitcode/bv/phasecode/qaoa/qft，5-25 qubit）；低计算非内存密集区（32k shots）TQSim 反而快（BV 3.26×、QFT 2.25×），因为 TUSQ 的 CPU 预处理收益不足以覆盖开销——确认 TUSQ 面向 time+memory critical 场景。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：TQSim 论文与代码由作者提供（TUSQ 论文致谢 Meng Wang 分享 TQSim 的 GPU 兼容代码用于基准测试）；TQSim 在 CPU（192GB）与 GPU（40GB）上模拟到 20 qubit（8MB 态矢量，可缓存 24576/5120 个中间态）。使用：作为 noisy 模拟加速 baseline，其最优区间是"计算密集但内存富余"；TUSQ 论文用它证明"缓存方案在 time+memory critical 区失效、需算法级冗余消除"的动机。

涉及论文标题：
- TUSQ Tracking, Uncomputation, and Sampling for Noisy Quantum Simulation
