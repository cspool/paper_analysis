## Photonic Quantum Computing on Spin Memory Architecture with Tree-Encoded Fusion

- 属于算法pipeline的实现是什么？实验比较什么？
  - Tree-encoded fusion（树编码融合）是论文提出的新型逻辑量子比特编码/融合算法方案（属于"新算法模型"层）：把参与 Type-II fusion 的逻辑量子比特编码为树结构——根量子比特 q_root 连接 b 个分支，每分支是 3 个量子比特的线性图 {q_i^a, q_i^b, q_i^c}，叶子 q_i^c 用于融合测量，q_i^a/q_i^b 是用于间接测量容错的辅助量子比特。按融合结果执行不同测量模式：(1) 融合成功——对 q_i^a、q_i^b 做一对 X 测量，把成功的融合纠缠直接连到 q_root；(2) 融合失败——q_i^c 被测量掉，q_i^a/q_i^b 留在树中，对 q_i^b 做 Z 测量移除它，留 q_i^a 作备份；(3) 融合擦除（光子丢失）——对 q_i^b 做 X 测量、q_i^a 做 Z 测量，基于 stabilizer X_i∏_{j∈E(i)}Z_j 实现对 q_i^c 的间接 Z 测量，无损消除被擦除 qubit 对图态的影响；(4) 全部分支失败/擦除的极端情况——用 (2) 留下的备份 q_i^a 再试一次融合。相比 redundantly-encoded fusion（m 次冗余融合尝试，但每个逻辑 qubit 的 m 个物理 qubit 独立暴露于擦除，逻辑擦除率 P_eras=1-(1-p_eras)^(2m) 随 m 指数恶化）和 repeat-until-success fusion（P_eras=Σ_{i=0}^{m-1}p_fail^i·2p_eras），树编码用间接 Z 测量把每个分支的擦除影响限制在单分支内，逻辑成功率 S_tree=1-(1-(1-p_eras)^2+p_fail)^b（b=分支数）。
  - 实验比较：同一 MemTree 编译框架下对比三种融合方案（固定融合失败率 p_fail=0.25，擦除率 p_eras 0%~10%，程序 2~20 qubit，执行时间上限 6×10^5 ns）：执行时间平均减少 1.9×10^-3×（vs redundantly-encoded）和 1.7×10^-2×（vs RUS）；光子源消耗多 2.55×（vs redundantly）和 1.63×（vs RUS）——用空间换时间，且光子源劣势随 #qubit 增大而缩小；Fig.4(f) 以 10^3 次融合试验统计成功率，显示 p_eras 升高时树编码优势显著。参数研究（Fig.5）：30-qubit caterpillar 限制下 b_prep>6 时光子源急剧增长，执行时间随 b 增大指数下降至 b=4 后收敛 → 选 b=4、b_prep=6（真实硬件 83.3% 单 timestep 制备成功率、97.1% 两 timestep 内）。Ablation study：MemTree+RUS 在 p_eras=0.5%、36-qubit 下除 Grover 外全面劣于 OneAdapt-ET，证明性能提升主要来自 tree-encoded fusion 而非架构差异。
- 硬件平台是什么，配置是什么。
  - 仿真平台：论文自研 realistic error-aware PQC 模拟器（spin memory 架构），配置来自实验工作 [25][29][30][43][52]：InGaAs 半导体量子点（QD）发射器（LA 纵向声学激发脉冲发射光子 + OSRP 光学自旋旋转脉冲定义 caterpillar 结构），caterpillar 初始化 12 ns + 每 qubit 发射 0.6 ns 时间周期，最大 caterpillar 30 qubit；融合成功概率 1-p_fail=0.75（无擦除时，需额外干涉测量装置），HOM 可见度 V_HOM=99.5% → 融合保真度 σ_fus=99.75%，OSRP fidelity 99%，退相干 T2=2.34 μs（由 4-qubit GHZ 95% fidelity 反推），t_cycle=30 ns；模拟 2×10^4 个 caterpillar 发射周期统计成功 shots。
  - 真实硬件：Quandela 云端 PQC 平台（24-photon modes），光学电路用 Perceval PQC 工具包构建（双轨 dual-rail 编码，融合电路 = 光子模式置换 + 相移 + 两个分束器；SNSPD 探测器延迟 <50 ps；经典 feed-forward 总延迟 <5 ns，用 Perceval FFCircuitProvider 实现）；实测硬件特征：HOM 不可区分度 92.0%、透射率 5.16%、g^(2)=2.0%。对比对象：IBM Torino 超导量子计算机（Qiskit 转译）。
- 模型是什么。数据集和bench分别是什么。
  - Benchmark：6 类典型量子算法——Bernstein-Vazirani (BV)、Quantum Approximate Optimization Algorithm (QAOA)、Grover's Algorithm、Quantum Fourier Transform (QFT)、quantum Hamiltonian simulation (QSIM)、Ripple Carry Adder (RCA)、Variational Quantum Eigensolver (VQE)。程序规模：融合方案对比 2~20 qubit（baseline 融合方案执行时间过长不可及）；编译器对比 36/64/100 qubit；真实硬件实验 6~12 qubit QAOA（EfficientSU2 SU2 默认 ansatz + RealAmplitudes RA 变体），指标 PST（Probability of Successful Trial）与 IST（Inference Strength）。模拟器基于 Perceval（[24]）。
- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：论文未提供代码仓库链接（arXiv:2604.21475），联网搜索未找到公开仓库，无法确认开源；对比融合方案（redundantly-encoded [25]、RUS [40]）按其论文最新协议实现（[12]），代码尺寸 m_Redun=5、m_RUS=6；Perceval（https://github.com/Quandela/Perceval，开源）用于真实硬件实验电路构建与 feed-forward 控制。
  - 使用例子（tree-encoded fusion 测量模式伪代码，b=4 分支）：逻辑量子比特 A、B 融合，对每个分支 i，Type-II 融合作用于叶子 q_i^c(A) 与 q_i^c(B)（HWP+PBS 线性光学干涉，双光子到异侧探测器=成功、同侧=失败、缺失=擦除）：
    ```
    for branch i in 1..b:
        outcome = type2_fusion(q_i^c(A), q_i^c(B))   # 成功 / 失败 / 擦除
        if outcome == SUCCESS:
            X_measure(q_i^a); X_measure(q_i^b)       # 融合纠缠直接连到 q_root
        elif outcome == FAILURE:
            Z_measure(q_i^b)                          # 移除 q_i^b，留 q_i^a 作备份
        elif outcome == ERASURE:
            X_measure(q_i^b); Z_measure(q_i^a)        # 间接 Z 测量 q_i^c（stabilizer X_i∏Z_j）
    if all branches failed/erased and backup exists:
        fusion_retry_with(q_i^a)                      # 备份尝试
    ```
    树结构从 caterpillar 态组装（Fig.4(e)）：主路径上的 q_root + b 个叶 qubit 由 caterpillar 提供；b 个 4-qubit 线性图从长线性图经 Z 测量分离，再融合到叶子上；制备参数 b_prep=6（>b），同 timestep 并行尝试 b_prep 次分支制备（失败自动测量掉、擦除用间接测量恢复），成功分支 <b 则下一 timestep 重试。
