## Stim（稳定子电路模拟器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Stim 是 Google quantumlib 开源的稳定子电路（stabilizer circuit）快速模拟器（GitHub: quantumlib/Stim，Apache-2.0，Craig Gidney，arXiv:2103.02202 / Quantum 2021）：对 Clifford 电路做 Tableau 仿真，采样 syndrome 极快（mega-sampling 用 256-bit AVX SIMD 批量并行采样；2 万 qubit、8 百万门的 d=100 电路约 15 s 分析、~1 kHz 采样）。限制：无非 Clifford 门、仅 Pauli 噪声。带 detector/observable 标注的电路可生成 detector error model（解码图/Tanner 图），是 QEC 研究的标配工具。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Stim 是本论文 syndrome 数据生成器与评估前端：
```
# 生成 d=11..31、p∈{1e-4,1e-3,1e-2} 的表面码 syndrome（artifact icepack.py）
circuit = stim.Circuit.generated("surface_code:rotated_memory_z",
                                 distance=d, rounds=R,
                                 after_clifford_depolarization=p)   # 现象学噪声
# circuit-level: after_reset_flip_probability / before_measure_flip_probability
sampler = circuit.compile_detector_sampler()
shots = sampler.sample(n_shots)     # 每 shot = R 轮 detector 位图
# -> 送入 IcePack 压缩 emulator（空间聚类→时间聚类→RGE）
# -> 输出 reduction_rge（与 reference/ CSV 对比，误差 <0.1）
```
评估配置（本论文）：每个 d–p 对 20000 次独立运行、跨多轮；1000 逻辑 qubit；artifact 以 Docker 跑 icepack.py + artifact.ipynb，产出论文图 5/7/8/15 的 CSV 与 PNG。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Python 接口（pip install stim）：stim.Circuit.generated 内建表面码电路、compile_detector_sampler 批量采样、detector_error_model 生成解码图。本论文用法：现象学与电路级两种噪声模型、非 IID qubit（Willow 检测概率分布生成 10 组 × 10 万 ancilla）、错误率漂移、burst errors 场景，全部以 Stim 采样为数据源；还以 Stim 生成的 syndrome index 分布驱动 10 万周期队列仿真求 99 分位延迟。采样随机性导致每次运行数值不同，但 reduction 比例与 reference 相差 <0.1（artifact 自检标准）。

补充（Coset Ensemble Decoder 论文）：该文用 Stim 生成 circuit-level depolarizing noise 表面码电路——depolarizing 以 p 施加于 Clifford 门之后的数据 qubit 与相邻轮之间，测量错误建模为同概率 p 的经典比特翻转，reset 理想，q=p、T=d 轮；biased/unbiased phenomenological noise（p_X/p_Z，bias η=p_Z/p_X∈{0.5,1,10}）亦经 Stim 生成。生成的 syndrome 数据驱动 Python 硬件模拟器评估 LER 与 cycle 计数，并与 RTL 交叉验证。

补充（TUSQ 论文）：该文用 Stim 内建 rotated surface code memory 电路做 DFTT+Caching 性能恢复分析——`stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=R, after_clifford_depolarization=p)` 生成 26/64/118 物理比特（d=3/5/7）、p∈{10^-2,10^-3,10^-4} 的电路，d 轮测量即 d 个 non-invertible 通道；每电路采样 1M 次，统计树遍历操作数（单比特矩阵向量乘=1、双比特=4、非幺正边前向=1、反向=0）求性能恢复 α(K)。结论：容量 3 的 LIFO 缓存恢复 60%-100% 的 DFTT 性能。TUSQ 还用它说明 FTQC 逻辑级模拟的配套角色：物理层 Clifford 电路用 Stim（多项式可扩展）、逻辑层非 Clifford 深电路用 TUSQ（DFTT+Caching 支持 MCM）。

补充（Triage 论文）：Triage 用 Stim 做窗口化 lattice surgery 的 LER 蒙特卡洛——d=9 rotated surface code、circuit-level depolarizing noise p=3×10⁻³、每点 ≥10⁵ runs（Memory Experiment 式逐层 syndrome 生成，同步失败时插入 idle 层再模拟该层 syndrome），先得到 d=9 的逐层 LER 再外推到 d=21 聚合总 LER；Stim 也是解码器延迟校准的数据源之一（pymatching 在 Stim 生成的 rotated surface-code 电路上按 shot 测延迟，15K shots/设置，拟合 log-normal 抖动参数）。

涉及论文标题：
- A Streaming Architecture for Quantum Error Syndrome Compression at 4 Kelvin
- Coset Ensemble Decoder for Quantum Error Correction with Algorithm-Hardware Co-Design
- TUSQ Tracking, Uncomputation, and Sampling for Noisy Quantum Simulation
- Triage An Adaptive Parallel Window Decoding Scheduler for Real-time Fault-Tolerant Quantum Computation
