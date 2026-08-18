## Triage An Adaptive Parallel Window Decoding Scheduler for Real-time Fault-Tolerant Quantum Computation

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现是 Triage，FTQC（容错量子计算）经典控制栈中的解码器调度中间件，把"如何把有限 M 个物理解码器动态分派到 N 个逻辑量子比特的 syndrome 解码任务"建模为带约束的动态实时调度问题，属于把有限资源池调度给持续到达任务流、满足实时截止时间（SLO 式）并最大化吞吐的系统级调度器（非 LLM Serving 框架，按最相似层次归入 Serving 调度）。核心调度单元是 slice S(t,p)：一个 d×d 逻辑 patch 在 t 时刻一个 d-round 同步测量周期内产生的 syndrome 数据块。离线阶段：Litinski 风格编译器（A game of surface codes）把高层程序编译为 LLI（Low-Level Instruction）流 → 静态分析器单趟构建带注释的 Timeline（每个 unit 存 layer index t、空间坐标 (r,c)、操作标签、6-bit 邻居 mask(t−1,t+1,↑,↓,←,→)、到最近关键同步点的 deadline（无则为∞）、可能的 causal cone 引用）。在线阶段：离散事件模拟器在 syndrome 到达和任务完成两个事件上触发调度器，把 PENDING slice 以无冲突 independent set 方式分派到 M 个物理解码器；同步失败时插入 idle syndrome layer（所有 t≥ℓ 的层右移一层），目标是最小化 idle layer 数 → 最小化逻辑错误率 LER。
  - Triage 是双模调度架构：①稳态模式（steady mode）——轻量启发式，统一优先级函数 P(V)=w_u·Urgency(V)+w_c·Cost-Efficiency(V)（Urgency=1/Deadline，Cost-Efficiency=1/(Degree+1)，w_u=w_c=0.5），并探索 FIFO/EDF/MDF 三种策略；②紧急模式（emergency mode）——Triage Trigger 在任一 PENDING slice 的 deadline 达到 τ_emergency（默认 4）时触发，用"预测性因果锥着色"（Algorithm 1：把 causal cone 的 PENDING slice 放入优先级队列，离散事件模拟推进时钟，贪心选独立集，MDF 排序，O(n log n)）预先计算关键操作 causal cone 的解码计划并按时间表派发；配套 ScopeCap（<100，巨型因果锥回退稳态模式）、增量重规划（新因果锥超出现有 scope 30% 且距上次重规划 ≥2 层才触发）、机会回填（backfilling：由紧急计划算 M_peak，用 M−M_peak−B_bf(t) 的预算派发非关键、因果无关任务）。
  - 实验比较什么：①与 baseline 对比（串行 sliding window [41]、time-parallel window [24]、SWIPER [26] 投机调度 SOTA），在 Bell4（Litinski compact layout，39 逻辑层、5 个 π/8 关键门）上扫解码器数量与相对解码速度 (τdec/τgen)，对比插入 idle layer 数与热力图/最优调度器地图（Triage 在资源受限下界前沿最优，SWIPER 在资源充裕区最优，左下角黑色区为 backlog 终止失败区）；②跨 15 个 FTQC benchmark（QASMBench 派生：T_injection、rotation_C+T、MSD15to1、bell4、mult15_CL/SL、adder28_CL/SL、adder64_SL、adder118_SL、secal1_SL、variational4_SL、qft4_SL、trotter4_SL、ising26_SL）在两种资源场景（并行富余 count=2×#LQs/speed=0.9；延迟富余 count=#LQs/speed=1.8）比 idle layers 与 LER；③随机解码延迟鲁棒性（log-normal 抖动，σ_base=0.3447/α_d=0.0041/α_p=15.03/p_ref=10^-3，按 pymatching 每 shot 延迟校准，LOO 验证 MAE 0.064）；④调度计算开销（Triage 每层 median 亚毫秒，紧急模式 O(n log n) 拟合 y=a·n log n, a=0.01513, R²=0.8056，Delay Ratio 0-0.20 灵敏度模拟，ScopeCap 在 0.06 处避免 backlog 失败）；⑤窗口缓冲大小（资源受限选小缓冲、资源充裕趋近 d/2）、超参数 (w_u, τ_emergency) 敏感性（性能对参数鲁棒）。主要结果：相比标准时间并行 baseline，平均 LER 降低 52.6%；在慢解码器区（τdec>τgen）仍可工作；总执行时间 T_total=N_total_layers×d×T_meas（d=21 时超导约 21μs/层，离子阱/中性原子 2.1-21ms/层）直接受益于 idle 层减少。
- 硬件平台是什么，配置是什么。
  - 无真实量子硬件或 FPGA 加速器，全部为软件仿真。仿真设备：Intel i9-14900K 处理器 + 188 GB RAM，Python 3.9。模拟的是经典控制流水线（编译器→静态分析→离散事件模拟）。解码器延迟建模：profiling pymatching 解码器在不同解码 volume 下的实测延迟，幂律拟合 t_decode=A·(volume)^α（α=1.17），单个 slice 的延迟由其窗口缓冲大小（约束图中未解析邻居数，即 degree）决定。Monte Carlo LER：用 Stim 模拟 d=9 旋转 surface code、circuit-level depolarizing noise p=3×10^-3，外推到 d=21，每点 ≥10^5 runs。平台时间假设：每个逻辑层需 d 轮 syndrome 测量，T_layer=d×T_meas。
- 开源Serving框架是什么。修改了什么。
  - 论文未基于任何开源 Serving/调度框架修改，作者自建调度仿真框架（Python 3.9 离散事件模拟器）。复用的开源组件：pymatching / Sparse Blossom [16]（MWPM 解码器，用于延迟 profiling 与 LER Monte Carlo）、Stim [42]（快速 stabilizer 电路模拟器）、QASMBench [43]（benchmark 来源）、Litinski 风格编译器 [38]（生成 LLI/tile 布局，指令集含 multi-patch measurement、patch rotation、idle）。Baseline 复现：sliding window [41]（一次处理一个 lattice surgery 操作涉及的 slice 块，不提前处理更晚时间的 slice）、time-parallel window [24]（跨时间维并行逻辑 patch，但不拆分多量子比特操作）、SWIPER [26]（复现其 successor-based 投机策略，10% misprediction rate + 10% speculation time，投机解码模块不计入解码器占用）。
  - 修改/新增：①新增 Triage 调度器本体——slice 约束图（最多 6 邻居：2 时间 + 4 空间）、slice 状态机（UNGENERATED→PENDING→OCCUPIED→ASSIGNED→COMPLETED）、Timeline 构造、稳态加权启发式、紧急模式 Algorithm 1 预测因果锥着色、Triage Trigger（τ_emergency 阈值 + ScopeCap<100 + 30% 扩幅 + 最小重规划间隔 2）、机会回填预算 M_usable(t)=max(0,min(M−M_peak−B_bf(t), F(t)−E(t)))；②新增离散事件模拟框架——syndrome 生成、idle layer 插入（把 t≥ℓ 的层右移并新建 idle 层）、backlog 终止条件（插入 idle 层数超过 benchmark 原始层数 10 倍则强制终止）、随机延迟 log-normal 抖动注入、调度延迟灵敏度模拟（Delay Ratio 0.00-0.20，紧急模式用 O(n log n) 拟合函数按实时 scope 大小计算延迟）。
- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：Triage 论文代码截至检索（2026-08）未公开，arXiv:2605.04459（ISCA 2026 录用），未发现官方 GitHub 仓库，无法确认开源链接；作者属 HKUST(GZ) Wang Research Group / QuAIR Lab。可复用开源组件：pymatching（https://github.com/oscarhiggott/PyMatching）、Stim（https://github.com/quantumlib/Stim）、QASMBench（https://github.com/pnnl/QASMBench）。
  - 使用例子（一个含 T 门（非 Clifford）的 lattice surgery 计算在解码器池调度下的全过程，框架输入到性能输出）：
    ```
    # 输入：高层量子程序（如 mult15_SL，15 逻辑比特、586 层、252 个 T 门）+ 解码器池配置（M 个解码器，各 r_dec 倍 syndrome 生成速度）
    # 1) 离线（编译 + 静态分析）：Litinski 风格编译器生成 LLI 流（multi-patch measurement / patch rotation / idle）
    #    → 静态分析器构建 Timeline：每个 unit 记录 (layer t, 坐标 (r,c), 操作标签, 6-bit 邻居 mask, 到最近 T 门同步点的 deadline, causal cone 引用)
    # 2) 在线（量子硬件 + 调度器）：每个 t 产生一层 syndrome（每个逻辑 patch 一个 slice）→ 触发 Triage 调度器
    #    - 稳态：对所有 PENDING slice 按 P(V)=0.5·1/Deadline+0.5·1/(Degree+1) 排序，选冲突无关（independent set）且 ≤M_available 的 slice 分派给空闲解码器
    #    - T 门临近：某 slice deadline ≤ τ_emergency=4 → Triage Trigger 触发紧急模式
    #      * 用 BFS 从关键 slice 的时间/空间前驱反向展开 causal cone（只扩同层空间邻居与 t−1 时间前驱，COMPLETED 剪枝，LRU 缓存）
    #      * Algorithm 1 预测着色：把 cone 内 PENDING slice 按 t_start 入优先级队列，模拟时钟推事件，每步按 degree 升序贪心选独立集，
    #        记录 (t_sim, s) 到计划 P，更新邻居 t_start 与 degree，直到队列空 → 得到紧急计划（≤ScopeCap=100 切片）
    #      * 机会回填：M_peak 之后的空闲解码器按稳态启发式跑非关键任务
    #    - 若因果锥未在关键操作执行前解码完 → 插入 idle syndrome layer（所有 t≥ℓ 层右移），暂停推进；idle 超 10 倍原层数则判定 backlog 失败终止
    # 3) 解码器执行：pymatching/Sparse Blossom 在每 slice 的窗口缓冲（含邻居边界人工 syndrome）上跑 MWPM，延迟按幂律 t_decode=A·volume^1.17 计入
    # 4) 性能输出：idle layer 数（同步失败度量）→ 总层数 N_total_layers → 墙钟时间 T_total=N_total_layers×d×T_meas；
    #    LER：Stim Monte Carlo（d=9, p=3×10^-3, ≥10^5 runs）模拟 windowed lattice surgery 后按层聚合，外推到 d=21
    ```
    作用：在 M<N 共享资源模型下把有限的解码器动态分配给 N 个逻辑比特的持续 syndrome 流——稳态启发式保证平均吞吐与 backlog 可控，紧急模式在非 Clifford 同步点前用最大并行度预解码因果锥、避免 Pauli frame 同步失败导致的 idle stall 与 LER 上升，机会回填回收空闲解码器吞吐，从而在慢/稀缺解码器资源下维持低 idle 层数与低 LER（平均比时间并行 baseline 低 52.6% LER）。
