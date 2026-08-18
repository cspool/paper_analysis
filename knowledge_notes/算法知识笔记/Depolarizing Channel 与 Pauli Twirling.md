## Depolarizing Channel 与 Pauli Twirling

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Depolarizing Channel（去极化通道）是量子噪声建模标准模型之一：以概率 p 把量子态"极化"成完全混合态，等价形式 ρ' = (1-p)ρ + (p/3)XρX + (p/3)YρY + (p/3)ZρZ——以 1-p 概率保持原态、各以 p/3 概率被 X/Y/Z 门作用。本论文把它作为主要噪声模型（默认 p=1%，部分实验 p=0.1%），并指出它是 TUSQ 的 ER 采样与 DFTT 的前提：DEP 采样的 ER 是固定 Pauli 门（I/X/Y/Z，b=4），是幺正的、有逆的。
- Pauli Twirling（Pauli 绕化）是把一般噪声通道（如 amplitude/phase damping、thermal relaxation）近似为 Pauli 通道的技术：用随机共轭 Pauli 门包裹噪声并平均，把退相干通道 ρ→(1-p_X-p_Y-p_Z)ρ + p_X XρX + p_Y YρY + p_Z ZρZ 化到 Pauli 形式，其中 p_X=p_Y=(1-e^{-t/T1})/4、p_Z=(1-e^{-t/T2})/2-(1-e^{-t/T1})/4（T1/T2 为弛豫时间）。Pauli twirling 保证噪声门是幺正 Pauli 门，这是 DFTT 树遍历（要求可逆/uncompute）的硬件前提。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- DEP 通道采样（ER 生成的一部分）：
  ```
  def sample_dep(p):
      r = random()
      if r < 1-p: return I
      elif r < 1-p+p/3: return X
      elif r < 1-p+2p/3: return Y
      else: return Z
  # 电路含 m 个 DEP 通道时，ER = (sample_dep(p) for _ in range(m))，b=4
  ```
- Pauli twirling 公式（论文式6）：ρ → (1-p_X-p_Y-p_Z)ρ + p_X XρX + p_Y YρY + p_Z ZρZ；p_X=p_Y=(1-e^{-t/T1})/4，p_Z=(1-e^{-t/T2})/2-(1-e^{-t/T1})/4。X/Y 项只含 T1（amplitude damping）误差，Z 项含 T1+T2。
- 兼容性结论：测量噪声与 DEP 天然满足 DFTT 的幺正性要求；decoherence 经 Pauli twirling 也可纳入；一般非幺正通道（如 erasure）则需 DFTT+Caching（缓存非幺正边前状态）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：Qiskit 与 CUDA-Q 的内置 noise model 均支持 depolarizing error（如 Qiskit `depolarizing_error(p, num_qubits)`）与 amplitude/phase damping；CUDA-Q noisy simulation 例子见 https://nvidia.github.io/cuda-quantum/latest/examples/python/noisy_simulations.html。TUSQ 场景：噪声模型的选择决定 ER 的 b 值与幺正性，进而决定用 DFTT 还是 DFTT+Caching；论文 surface code 实验用 Stim 内建电路 + DEP（after_clifford_depolarization）。

涉及论文标题：
- TUSQ Tracking, Uncomputation, and Sampling for Noisy Quantum Simulation
