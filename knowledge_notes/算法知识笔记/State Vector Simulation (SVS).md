## State Vector Simulation (SVS)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- State Vector Simulation（态矢量模拟）是量子线路模拟（Quantum Circuit Simulation, QCS）最主流的形式：把 n 个 qubit 的量子态表示为一个 2^n 维复向量 |ψ⟩（内存 O(2^n)），把每个量子门表示为一个幺正矩阵 U，门的应用就是一次矩阵向量乘 |ψ'⟩ = U|ψ⟩。从 |0...0⟩ 初态出发按线路顺序乘完所有门得到最终态矢量，再按 Born rule 采样得到经典输出分布。noiseless QCS 只需做一遍完整矩阵向量乘。
- 本论文中 SVS 是 noisy 模拟的基本单元：由于噪声通道的随机性，每个 shot 可能产生不同的"固定噪声门电路"，S 个 shot 就需要 S 次独立 SVS（S-fold compute overhead）；TUSQ 的 ECM+DFTT 正是围绕"减少这 S 次 SVS 的冗余"设计。TUSQ 用 NVIDIA cuStateVec v1.12.0 作为 SVS 的 GPU kernel 后端。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- SVS 计算过程伪代码（一个 noiseless 电路）：
  ```
  # 输入：n 比特线路（门序列 g_1..g_m），初态 |0>⊗n
  ψ = zero_state(n)                    # 2^n 维复向量，仅 ψ[0]=1
  for g in gates:
      ψ = apply_unitary(g.matrix(), ψ) # 门矩阵（2^k×2^k，k=1/2）作用于对应 qubit 的振幅
  counts = sample(ψ, shots)            # 按 |ψ[i]|^2 概率采样 shots 次
  ```
- 张量计算：|ψ'⟩ = U|ψ⟩，U 为 2^k×2^k 幺正矩阵，|ψ⟩ 为 2^n 维复向量；单比特门一次乘 2^n 个元素、双比特门一次乘 4·2^n 个元素（本论文操作计数按 1/4 计）。
- noisy 场景：对每个 shot 把噪声通道采样成固定 Pauli 门（如 DEP 的 I/X/Y/Z），得到 S 个不同电路实例，各自从头做 SVS 再平均输出——即本论文要消除的 S-fold 开销来源。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：主流量子 SDK 都内置 SVS 后端——Qiskit-Aer StatevectorSimulator（qiskit-aer，GPU 版）、NVIDIA CUDA-Q（cudaq 的 statevector 后端，底层 cuStateVec）、cuStateVec 库本身提供 cuStateVecApplyMatrix 等 kernel。使用：Qiskit 中 `StatevectorSimulator().run(circuit, shots=S)`；CUDA-Q 中 `cudaq.sample(kernel, shots_count=S)`。TUSQ 论文把 Qiskit 2.1.0 与 CUDA-Q 0.11.0 的 SVS 作为 baseline，在 NERSC Perlmutter 单 A100 上与 TUSQ 对比，报告平均 59.06×/13.38× 加速。

涉及论文标题：
- TUSQ Tracking, Uncomputation, and Sampling for Noisy Quantum Simulation
