## Pruning（noisy 量子模拟中的重要性采样，α/β 参数）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Pruning（剪枝/重要性采样）是 TUSQ Error Characterization Module 的第三阶段：利用"ER 频率随 Hamming weight 指数衰减"这一分布偏斜特性，把 ER 电路分成显著电路 C_S（频率 p_i ≥ α·p_max）与不显著电路 C_I（p_i < α·p_max，α 为阈值，论文取 0.01）。显著电路正常模拟；不显著电路的个体贡献小但集体占比可很大（10-qubit QAOA 1M shots 中 insignificant 合计占 42%），因此不直接丢弃，而是从 C_I 随机采 β 个代表电路（β=100），每个代表按 (p_insig/Σp_t)·p_t 加权采样，保持整体输出分布贡献。
- 这是 TUSQ 唯一引入 fidelity 损失的步骤：relative fidelity difference δ = |f_A-f_B|/(f_A+f_B) 平均 1.66%、最大 7.15%（α=0.01、β=100），对 VQE/Adder/BV 等算法正确性影响可忽略（Adder 320 例中 289 例、BV 380 例中 368 例推断输出比特串不变）。α/β 是用户可调旋钮：要更低 fidelity 偏差就降低 α、提高 β，代价是更多模拟时间。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- Pruning 伪代码：
  ```
  p_max = max(freq of ER circuits)
  C_S = {c_i | p_i >= α·p_max}                 # 显著电路，α=0.01
  C_I = {c_i | p_i <  α·p_max}                 # 不显著电路
  p_insig = Σ_{c_i∈C_I} p_i                    # 集体占比
  K = 从 C_I 均匀随机采样 min(β, |C_I|) 个代表   # β=100
  for c_t in K:                                # 每个代表
      计算 |ψ_t⟩ = SVS(c_t)
      采样次数 = (p_insig / Σ_{c_t∈K} p_t) · p_t   # 按频率加权保持集体贡献
  S_final = |C_S| + min(β, |C_I|)              # 待模拟电路总数 ≪ 原始 S
  ```
- 与 baseline 对比：TQSim 的 fidelity 损失来自内存饱和时缓存不全（统计方法取舍），TUSQ 的 Pruning 把"丢多少"显式化为 α/β 参数；naive SVS（Qiskit/CUDA-Q）完全不剪枝、无损失但计算量随 S 线性增长。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：TUSQ CPU 预处理阶段按上式执行，Pruning 与 ER Tallying/ER Commutation 串接；在 QAOA p=2→6 深度增加时 ER 分布趋于均匀、剪枝有效性下降（深度越大越难区分信号与噪声，p=10 时几乎失效）。使用：用户按误差容忍度设置 α、β（论文默认 α=0.01、β=100）；希望无损时可用 α=0（保留全部电路）。评估指标：speedup γ 与 relative fidelity difference δ 联合权衡。

涉及论文标题：
- TUSQ Tracking, Uncomputation, and Sampling for Noisy Quantum Simulation
