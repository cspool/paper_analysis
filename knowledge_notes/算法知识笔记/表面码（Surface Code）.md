## 表面码（Surface Code）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
表面码是当前最主流（尤其超导 qubit 平台）的量子纠错码：把数据 qubit 排列在二维格点上，与两类 ancilla qubit（X 型、Z 型）交错纠缠。每个 ancilla 与相邻数据 qubit 做稳定子（stabilizer）奇偶校验，测量结果组成 syndrome——X/Z 错误改变其邻域 ancilla 的测量值，因此 ancilla 检测的是"错误的边界"而非错误本身：单个 X 或 Z 数据错误触发 2 个相邻 ancilla 产生非零 syndrome，Y 错误触发 4 个；若一个 ancilla 耦合偶数个错误数据 qubit 则奇偶相消报告 0，形成错误链（error chain）。码距 d 是格点每条边的数据 qubit 数（rotated 记法 [[d²,1,d]]，即 d² 物理 qubit 编码 1 逻辑 qubit），距离 d 的码可纠正 ⌊(d−1)/2⌋ 个错误；阈值约 1%（code-capacity 级）。web：低于阈值时每增加 1 码距，逻辑错误率指数级下降（Google Willow 距离 7 码每级约 16× 错误抑制，arXiv:2408.13687）；表面码只需近邻连接，适配平面工艺布线。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
表面码是 syndrome 压缩 pipeline 的输入结构（本论文）。每测量轮对全部 ancilla 做稳定子测量得到一张 syndrome 位图（每位 = 1 个 ancilla 的奇偶结果），该位图按行主序编号即为压缩器的输入位流：
```
# 单轮 syndrome 采样（Stim 内建 surface_code:rotated_memory_z，d=21）
circuit = stim.Circuit.generated("surface_code:rotated_memory_z",
                                distance=d, rounds=n_rounds,
                                after_clifford_depolarization=p)
syndrome = sampler.sample(circuit)   # shape: [shots, n_detectors]
# 每个检测器 detector = 一个 ancilla 的稳定子奇偶结果，0/1
```
本论文的压缩对象是这张位图中的非零位置（index）及其时空模式：空间上 X/Z 错误产生水平/垂直对、Y 产生 cross，时间上测量错误在相邻轮同位置成对出现。错误链使 ancilla 只在链两端报告非零（d=21 时每轮非零 syndrome 稀疏，p=10^-3 时约 0.1% 数量级非零）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实际实现是量子处理器上的稳定子测量电路：每轮把每个 ancilla 与邻域数据 qubit 用 CNOT 纠缠后测量（syndrome extraction），重复多轮构建三维解码图（2 空间 + 1 时间轴，测量错误成为时间轴上的边）。本论文用法：1000 逻辑 qubit、d=11–31、p=10^-4–10^-2，Stim 运行时生成 syndrome 数据集驱动 IcePack 压缩评估；解码仍在 300 K 全精度执行，压缩保证无损。局限：需近邻连通（qLDPC 类非平面码不适用，但 BB 码等周期结构可适配）。

补充（Coset Ensemble Decoder 论文）：该文用 rotated surface code 但取 periodic boundary conditions（周期边界，与 QUEKUF 相同设定）跑全部算法精度评估（码距 d∈{3,5,...,19}）；Micro-Blossom/Helios 的硬件资源数字取自 rotated 变体原始论文。解码图构建：T=d 轮 syndrome 提取，相邻轮 syndrome 输出 XOR 得到 detector（把测量错误与数据错误隔离），顶点=detector 事件、边=潜在错误，构成 3D 解码图 G(V,E)——它是 MWPM/UF/陪集集成解码的共同输入结构。

补充（Triage 论文）：Triage 用 rotated surface code 做蒙特卡洛 LER 评估——d=9、circuit-level depolarizing noise p=3×10⁻³、Stim 每点 ≥10⁵ runs，外推到 d=21；Triage 的 slice 抽象正是以 d×d surface code patch 为空间单位：slice S(t,p)=一个 d×d 逻辑 patch 在 t 时刻（一个 d 轮 syndrome 测量周期）内产生的 syndrome 数据块，解码按 window-based lattice surgery 分 slice 并行后逐层聚合 LER。解码延迟模型直接来自 pymatching 对 surface code 解码 volume 的实测拟合（t_dec=A·volume^α，α=1.17），单 slice 延迟由其窗口缓冲大小（约束图中未解析邻居数，即 degree）决定。

涉及论文标题：
- A Streaming Architecture for Quantum Error Syndrome Compression at 4 Kelvin
- Coset Ensemble Decoder for Quantum Error Correction with Algorithm-Hardware Co-Design
- O3LS: Optimizing Lattice Surgery via Automatic Layout Searching and Loose Scheduling
- Triage An Adaptive Parallel Window Decoding Scheduler for Real-time Fault-Tolerant Quantum Computation

补充（O3LS 论文）：O3LS 把表面码抽象为 patch/tile 模型用于晶格手术编译——每个 patch 是 distance-d 表面码（d² 物理数据 qubit 编码 1 逻辑 qubit），四边为 X/Z 型边界（虚线=Z 算子、实线=X 算子，抽象为 patch 上的 X-/Z-边缘）；编译目标是"用最少 tile 与时间步实现算法"（space-time volume）。代表性布局：compact（[34]，顺序放置、按需加列）、sparse（[25]，每数据 patch 与邻居间隔至少一个空 tile、X/Z 边缘均邻接路由空间）、standard（[25]，类 sparse 但放置不同）、以及 O3LS 自动生成的 squeezed 布局（评分函数 S(B)=C(B)×(N_x+N_z−α_e·N_e) 迭代搜索）。评估设定：d=9 表面码、p=10⁻³ 电路级去极化噪声、STIM 表征原子操作、PyMatching 2 解码、分层 LER 模型（p_total≈Σ_t p_layer，PPM/PR/idle 三类错误复合）。
