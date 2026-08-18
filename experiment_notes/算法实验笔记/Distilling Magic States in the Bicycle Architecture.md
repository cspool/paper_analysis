## Distilling Magic States in the Bicycle Architecture

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：在 Bivariate Bicycle (BB) QLDPC 码上运行魔法态蒸馏（MSD）工厂的算法设计。(i) 用 triorthogonal matrix G∈{0,1}^{m×n} 刻画协议（前 k 个奇权重行对应 k 个输出逻辑 qubit，m−k 个偶权重行对应 ancilla/parity check；每列 c 定义作用于 S_c={r: G_rc=1} 的 m 体 Z 型 π/8 旋转 exp(iπ/8·Z^⊗S_c)）。执行形式为 Pauli-measurement-based Clifford 电路：m 个逻辑 |+⟩ 初始化 → 逐列消费 1 个噪声 |T⟩ 实现旋转 → X 基测量 m−k 个 parity check 并 postselect 全 |+⟩；相比 Bravyi-Haah 原构造省去 n-qubit 稳定子态制备与 T 门 unencoding，仅用 m 个逻辑 qubit。(ii) 协议级压缩（qubit recycling）：对 G 做列置换、块内行置换、F_2 行加法，最小化峰值并发活动 qubit 数 C(G)=max_j|W(j)|（行 i 在列 j "working" 当 j≥f_i 且（偶行 j≤ℓ_i 或 奇行永不释放）），49-to-1 从 13→7、51-to-3CS 从 18→9、64-to-2CCZ 从 17→10 qubit，使大协议塞进单 BB 块。
  - 实验比较：与 surface-code lattice-surgery 工厂（Litinski [46]，(15-to-1)SC(17,7,7) 与两级 (15-to-1)SC(11,5,5)+(15-to-1)SC(25,11,11) 等）及 Gidney 魔法态培养 MSC（Cultivation_SC→d, d=3/5）比较：物理 qubit 数、逻辑时间步 τ_i（含 discard 率）、时空体积（qubits×timesteps）、输出错误率 p_out（union bound 解析 + 密度矩阵仿真双口径）。
- 硬件平台是什么，配置是什么。
  - 纯经典仿真：MacBook Pro（10 核 CPU、32GB RAM），无量子硬件/GPU。仿真为逻辑层：每个逻辑 qubit 抽象为单 qubit 并施加给定逻辑错误率的 depolarizing 信道；物理级错误率直接借用自行车架构论文 [27] Table I 的既有仿真结果（gross/two-gross 的 automorphism、in-module、inter-module 测量错误率）。
- 模型是什么。数据集和bench分别是什么。
  - "模型" = 量子纠错码与蒸馏协议：gross [[144,12,12]] 与 two-gross [[288,12,18]] BB 码（LPU 分别 90/158 qubit）；协议 15-to-1、20-to-4、8-to-CCZ、49-to-1（输出 |T⟩）、51-to-3CS（|CS⟩）、64-to-2CCZ（|CCZ⟩）。bench = 噪声模型：物理错误率 p_phys∈{10⁻³,10⁻⁴}，输入魔法态误差 p_in（depolarizing 信道）、automorphism 误差 p_auto、inter-module 测量误差 p_inter（双 qubit depolarizing）、in-module 误差 p_intra（测量翻转 p_meas + 逻辑 depolarizing，λ=p_meas/p_intra∈{0.5,0.9}）。无传统数据集。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：论文 arXiv:2602.20546（ISCA 2026）；作者代码 https://github.com/kunliu7/bb-code-magic-state-distillation（描述 "Compilation of magic state distillation on BB codes"）：scripts/conj_path/conj_path_searcher_cnt.py（in-module 测量计数）、prj_msd/10mapping/（逻辑 qubit 映射）、prj_msd/20compressed/min_meas_cnt_comp_v3.py（协议压缩）、prj_msd/30min_aut/min_aut.py（最小化 native automorphism）。
  - 算法 pipeline 伪代码（triorthogonal 蒸馏，15-to-1 在 two-gross 块上，m=5）：
    ① init：全部物理 qubit 置 |+⟩，跑 1 轮 syndrome extraction → m 个逻辑 |+⟩；
    ② for c in 1..n（n=15 列）：取 1 个噪声 |T⟩（误差 p_in），经 injection gadget 在逻辑 qubit 上实现 exp(iπ/8·Z^⊗S_c)；测量结果为 1 时补 exp(iπ/4·Z^⊗S_c) 条件 Clifford 校正；
    ③ measure m−k=4 个偶行 qubit 于 X 基，postselect 全 |+⟩ 结果；
    ④ 成功 → 前 k=1 行输出蒸馏 |T⟩；p_out ≈ 35·p_in³ + O(p_L)。
    压缩伪代码：每行 i 求首/末 1 列 (f_i, ℓ_i)，工作集 W(j)={i: j≥f_i 且（偶行 j≤ℓ_i 或 奇行）}，目标 min C(G)=max_j|W(j)|；允许变换 = 列置换 + 块内行置换 + F_2 行加法（全局最优 NP-hard，化简到 cutwidth）；贪心聚类行起止 + 定向行加法求启发式解。
  - 效果（Table III，pivot 注入）：(15-to-1)Gross 378 qubit / τ=6122 / 体积 2.3×10⁶ / p_out≈1.3×10⁻⁶（union bound）vs (15-to-1)SC(17,7,7) 4620 qubit / τ=256 / 1.2×10⁶；(15-to-1)Two-gross 734 qubit / 11249 / 8.3×10⁶ / ≈1.0×10⁻⁸；(49-to-1)Two-gross 734 qubit / 70748 / 5.1×10⁷ / 2.0×10⁻¹¹（p_phys=10⁻³）与 ≤10⁻¹⁷（10⁻⁴）；两级 Cultivation_SC+(15-to-1)Two-gross 454+734 qubit 达 4.1×10⁻¹²（10⁻³）与 ≤10⁻¹⁷（10⁻⁴）。
