## TUSQ Tracking, Uncomputation, and Sampling for Noisy Quantum Simulation

- 属于算法pipeline的实现是什么？实验比较什么？
  - TUSQ 是为 noisy quantum circuit simulation (QCS) 提出的新算法框架，针对"时间密集+内存密集"（time-critical and memory-critical）的多状态向量模拟（SVS ensemble）场景（DMS 内存 O(2^2n) 不可扩展；naive SVS 每个 shot 都要重做完整矩阵向量乘）。实现含两大模块：(1) Error Characterization Module (ECM)——三步消除冗余/不显著电路实例：ER Tallying（对噪声通道预采样得到 Error Realization (ER)，统计唯一 ER 及频率，同一 ER 电路只算一次 statevector、按其频率重复采样输出）、ER Commutation（用 per-qubit 栈 + 6 条 Pauli 门穿通规则把噪声门尽量右移，识别输出等价的不同 ER 并合并 shot 计数）、Pruning（按频率 p_max 与阈值 α=0.01 划分 significant/insignificant 电路，对 insignificant 集合随机采样 β=100 个代表电路并加权采样以保持集体贡献）；(2) Depth First Tree Traversal (DFTT)——把待模拟电路集合按共享前缀组织成树，用 compute（正向乘 U）与 uncompute（反向乘 U†）做深度优先遍历复用重叠计算，操作数从 O(|E|log_b|E|) 降到 O(|E|)；对 non-invertible channel（mid-circuit measurement、erasure 等）用 DFTT+Caching（LIFO 缓存 pre-MCM 状态，容量 K 受内存约束，K=3 即可恢复 60%-100% 性能）。实验比较：TUSQ vs Qiskit 2.1.0、CUDA-Q 0.11.0、TQSim（同为 noisy SVS 加速器，BFS+memoization 方案），及 TNS 场景 vs CUDA-Q tensornet-mps；指标为相对加速比 γ 与 relative fidelity difference δ。198 个 benchmark × 1M shots：平均/最大加速 59.06×/7878.03×（vs Qiskit）、13.38×/439.38×（vs CUDA-Q）；同 fidelity 误差下 vs TQSim 平均/最大 39.32×/3134.31×；TNS+TUSQ vs 未优化 TNS 平均 248.39×；CPU 预处理平均/最大 3.97/18.52 秒。
- 硬件平台是什么，配置是什么。
  - NERSC Perlmutter 超算节点：AMD EPYC 7763 CPU（64 核/128 线程）+ NVIDIA A100 (40 GB) GPU；默认单 GPU（CUDA_VISIBLE_DEVICES=0），多 GPU 实验到 33 qubit，TNS 实验到 40 qubit。DFTT+Caching 性能恢复分析用 Stim 内置 rotated surface code memory 电路（26/64/118 物理比特，d=3/5/7，p=10^-2/10^-3/10^-4，d 轮测量，1M shots/电路，仅统计操作数不实际遍历）。MSC 验证用 18-qubit d=3 电路。代表性结果：30-qubit Adder × 10^6 shots 在单 A100 上约 820 秒（CUDA-Q/Qiskit 同硬件 >10 小时）。
- 模型是什么。数据集和bench分别是什么。
  - 无模型训练；benchmarks 用 Supermarq 套件：QAOA（13-25 qubit，depth 82-770，130-1250 门）、Adder（4-28）、Bitcode（5-25）、Phasecode（5-25）、GHZ（14-28）、QFT（14-24）、BV（4-24），覆盖线性（GHZ/Bitcode/Phasecode）/并行（QAOA）结构、单峰（Adder/Bitcode/Phasecode）/双峰（GHZ）/尖峰（QAOA）/均匀（QFT）输出分布。shots 取 32k/100k/1M/10M（Wang et al. TQSim 用 32k，Patti et al. 用 10^9）。噪声模型：depolarizing、measurement error、amplitude/phase damping（Pauli twirled 化），p=1% 默认（部分实验 p=0.1%）。VQE 正确性验证：10/15-qubit Ising 与 Heisenberg Hamiltonian。TNS 实验：40-qubit QFT/Adder/QAOA(p=2)，bond dimension=16，100k shots，α=0.01、β=100。
- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：论文正文声明 "An open-source implementation of TUSQ can be found here"，指向 GitHub 仓库 https://github.com/tinaoberoi/TUSQ（描述为 "C++ state vector and tensor network simulator for quantum circuits"，MIT license）。截至 2026-08 检索，该仓库仅含占位 README（无源码/文档/运行示例），实际代码未公开，无法按开源文档给出运行命令，复现需按论文方法自实现。baseline 与依赖开源：Qiskit v2.1.0（github.com/Qiskit/qiskit-aer）、CUDA-Q v0.11.0（github.com/NVIDIA/cuda-quantum）、NVIDIA cuStateVec v1.12.0 / cuTensorNet v2.9.1（cuQuantum SDK，github.com/NVIDIA/cuQuantum）、Supermarq（github.com/PrincetonQuantum/Supermarq）、Stim（github.com/quantumlib/Stim，DFTT+Caching surface code 电路来源）、TQSim（论文[41]，作者分享 GPU 兼容代码）。
  - 背景张量计算：noiseless SVS 只需一次矩阵向量乘 |ψ'⟩ = U|ψ⟩（U 为 2^k×2^k 幺正门矩阵，|ψ⟩ 为 2^n 维复向量，内存 O(2^n)）；noisy QCS 把 depolarizing channel 展开为加权经典混合 ρ'=(1-p)ρ+(p/3)XρX+(p/3)YρY+(p/3)ZρZ，等效于对 S 个固定噪声门的电路实例各做一次 SVS 再平均，故需 S 次完整矩阵向量乘（S-fold 开销）。
  - ER Tallying 伪代码：
    ```
    # 输入：带噪声通道的电路 C，总 shots S
    tally = {}                                        # ER n 元组 -> 频次
    for shot in 1..S:
        er = tuple(sample(ch) for ch in C.channels)   # DEP 采 I(1-p)/X/Y/Z(p/3)，measurement 采 I/X
        tally[er] += 1
    for (er_i, s_i) in tally.items():
        c_i = 把 er_i 的固定噪声门并入无噪声电路
        |ψ_i⟩ = SVS(c_i)                              # 只算一次 statevector
        输出分布 += 从 |ψ_i⟩ 采样 s_i 次                # 采样比矩阵向量乘便宜得多
    ```
  - ER Commutation（per-qubit 栈，保持"噪声门尽量靠右"不变量；复杂度从 O(g1·g2) 降到近最优）：
    ```
    # 规则：1) 相邻噪声 Pauli 相乘合并；2) Pauli 穿过无噪声 Pauli；3) X/Y/Z 穿过 R_X/R_Y/R_Z；
    #       4) CNOT：X(control)→X(control)X(target)，X(target)→X(target)
    #       5) CNOT：Z(target)→Z(control)Z(target)，Z(control)→Z(control)
    #       6) CNOT：Y(control)→Y(control)X(target)，Y(target)→Z(control)Y(target)
    stacks = [栈() for q in qubits]
    for gate in transpiled_circuit:                   # 基为单比特门 + CNOT
        if gate 无噪声:
            检查 gate 作用 qubit 的栈顶：若栈顶噪声门按规则可穿通 → pop 并按规则 push 新噪声门；否则直接 push gate
        else:  # 噪声 Pauli 门
            与栈顶已有噪声门按规则1合并，否则入栈
    结束后：ER 相同（含穿通后等价）的 (c_i, s_i) 合并 shot 计数
    ```
  - Pruning：p_max = 最高频电路频率；电路 c_i 显著 iff p_i ≥ α·p_max（α=0.01，用户可调）；insignificant 集合 C_I 中随机采 β=100 个代表 {t_1..t_β}，每个代表按 (p_insig/Σp_t)·p_t 次采样；S_final = Σ_i 1[p_i ≥ α·p_max] + min(β, |C_I|)。示例：10-qubit QAOA 1M shots 中 significant 占 58%、insignificant 合计 42%，不能直接丢弃。
  - DFTT：树节点 = 中间 statevector，边 = 门；DFS 正向边乘 U、反向边乘 U†（uncompute 回滚），共享前缀边只算一次（图5：算完 S1 后回滚到公共节点 d 再走 S6 分支）；T_dftt = 2|E|，T_naive = N_l·h = O(|E|log_b|E|)（b=4 DEP 或 b=2 measurement，h=log_b((b-1)|E|+b)-1，N_l=b^h）；非幺正边（MCM/erasure）用 DFTT+Caching：pre-MCM 节点入 LIFO 缓存，回滚跨非幺正边时取缓存状态而非求逆，K 不足时对"离叶子最近的 K 个 pre-MCM 节点"的子树分别 DFTT+Caching。单/双比特门矩阵向量乘分别计 1/4 个操作。
