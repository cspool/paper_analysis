## Density Matrix Simulation (DMS)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Density Matrix Simulation（密度矩阵模拟）是把量子态表示为 2^n×2^n 正半定 Hermitian 矩阵 ρ（迹为 1，内存 O(2^{2n})）的模拟范式：无噪声操作是 ρ' = UρU†，一般噪声通道是 ρ' = Σ_i K_i ρ K_i†（Kraus 算子形式），例如去极化通道 ρ'=(1-p)ρ+(p/3)XρX+(p/3)YρY+(p/3)ZρZ。DMS 一次电路执行就能完整刻画噪声统计，不需要对多个电路实例平均。
- 本论文把 DMS 作为 noisy QCS 的"理想但不可扩展"参照：内存 O(2^{2n}) 比 SVS 的 O(2^n) 平方级更大，导致 El Capitan 级超算也只能模拟约 25 qubit，而 30 qubit SVS 在 16GB 笔记本即可运行（论文引 [41] 的估计）。因此 DMS 的内存开销使其在大规模下不可行，SVS 多实例平均成为唯一现实策略。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- DMS 计算过程（一个含噪声的电路）：
  ```
  ρ = |0><0|⊗n                       # 2^n×2^n 密度矩阵，O(2^{2n}) 内存
  for g in gates:
      if g 无噪声: ρ = U ρ U†
      else: ρ = Σ_i K_i ρ K_i†        # Kraus 表示，如 DEP: (√(1-p)I, √(p/3)X, √(p/3)Y, √(p/3)Z)
  p_dist = diag(ρ)                    # 测量：取对角元素
  ```
- 与 SVS 的关系：DEP 通道的展开式把噪声态解释为"加权经典混合"，即 ρ 可看作多个 SVS 电路（固定 I/X/Y/Z 噪声门）输出的加权平均——这正是"noisy 模拟 = S 个 SVS 平均"的理论依据，也是 TUSQ 重要性采样（Pruning）保持集体贡献的数学基础。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Qiskit-Aer 的 DensityMatrixSimulator、CUDA-Q 支持密度矩阵后端（cudaq.DensityMatrixSimulator）、cuStateVec 也有密度矩阵 kernel（cuStateVec 支持 state matrix 运算）。使用：Qiskit 中 `DensityMatrixSimulator().run(circuit, shots=S)` 一次运行得到含噪统计。论文观点：DMS 精度最优但内存平方增长，仅在少 qubit 场景可用；大规模场景必须回到 SVS 多实例 + 冗余消除（TUSQ）。

涉及论文标题：
- TUSQ Tracking, Uncomputation, and Sampling for Noisy Quantum Simulation
