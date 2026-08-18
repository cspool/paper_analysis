## Error Realization (ER)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Error Realization（错误实现，TUSQ 提出的轻量中间表示 IR）是"从噪声通道采样得到的一组固定噪声门"：对每个 shot，把电路中每个噪声通道（去极化 DEP、测量噪声等）采样成一个确定的 Pauli 门（DEP 采 I/X/Y/Z，测量噪声采 I/X），整条电路就变成一个"固定噪声门电路"；一个 SVS 实例对应一个 ER。含 m 个噪声通道的电路的 ER 就是这 m 个采样的 n 元组，例如 (I₀, X₁, Y₂, ..., I_{m-1})。
- ER 的价值：它是"电路是否产生相同输出"的轻量判据——ER 相同的电路最终态矢量相同，无需实际计算即可合并；不同但等价的 ER（经 Pauli 门穿通后相同）也产生相同输出。TUSQ 用它做冗余检测（ER Tallying、ER Commutation）与重要性加权（Pruning）。低 Hamming weight（更多 I 门）的 ER 出现频率指数级更高（p=1% 时 Hamming weight >2 的概率为零），这是 Pruning 有效性的依据。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- ER 生成与使用流程：
  ```
  # 预采样阶段（CPU，一次完成）
  for shot in 1..S:
      er = tuple(sample(channel) for channel in circuit.channels)  # 每个通道采 I/X/Y/Z 或 I/X
      tally[er] += 1                          # 记录唯一 ER 及频次
  # 使用阶段
  for (er_i, s_i) in tally.items():           # ER Tallying
      c_i = 把 er_i 的固定 Pauli 门并入无噪声电路
      |ψ_i⟩ = SVS(c_i)                        # 同一 ER 只算一次
      输出分布 += 从 |ψ_i⟩ 采样 s_i 次
  ```
- ER Commutation 例子（图4B）：两个 shot 的 ER 分别为 (X, II) 与 (I, XX)，不同但等价——X 门穿过 CNOT 的控制比特会在目标比特上额外产生 X 门，因此两者产生相同输出，可合并 shot 计数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：TUSQ 在 CPU 预处理阶段预采样全部噪声通道、统计唯一 ER 频次（论文报告 CPU 预处理平均 3.97s、最大 18.52s），随后用 ER 驱动 ECM 三步（Tallying/Commutation/Pruning）确定待模拟电路集合。使用前提：噪声通道可采样成 Pauli 门（测量与去极化噪声天然满足；amplitude/phase damping 经 Pauli twirling 近似）。开源：论文声明开源实现位于 https://github.com/tinaoberoi/TUSQ，但截至 2026-08 仓库仅占位 README、无源码。

涉及论文标题：
- TUSQ Tracking, Uncomputation, and Sampling for Noisy Quantum Simulation
