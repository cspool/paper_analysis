## 最小权完美匹配解码器（MWPM / Blossom / PyMatching）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MWPM（Minimum-Weight Perfect Matching）是表面码最经典、精度最高的解码器：把 syndrome 解码化为解码图（detector 图）上的最小权完美匹配问题——每个非平凡 syndrome（奇数奇偶顶点）需与另一顶点配对，边的权重取错误概率的负对数（权重随空间/时间距离增长），用 Blossom 算法（Edmonds 花算法，工业实现 Kolmogorov Blossom V）求全局最优匹配，配对路径即推断的错误链。它是"物理 ML"解码：解 argmax p(E|s)，在 LP 框架下迭代维护 blossoms。代价：算法与实现复杂度高、串行性强、延迟大；但精度是 baseline 金标准。加速变体：Fusion Blossom（Wu & Zhong, arXiv:2305.08307，Flower 图融合，GPU/FPGA）、Sparse Blossom（Higgott & Gidney，Quantum 9:1600，稀疏技巧 1M errors/core/s）、PyMatching（Higgott 的 Python/C++ 软件实现，detector error model 输入）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 解码图 G(V,E)，V=detector 事件，E=潜在错误(权 w_e=-log p_e)
O = {v ∈ V : v 为奇数奇偶(非平凡 syndrome)}    # 需配对的顶点
M = MinWeightPerfectMatching(G, O)           # Blossom 算法，保证最优
Ê = 由 M 的边组成的最可能错误链
# 缺陷(本文)：只优化物理单链 argmax p(E|s)，忽略简并性/陪集
# 本文用法：软件精度用 PyMatching 实现作为 MWPM 基准
```
本论文中 MWPM 是精度上限参照：p=0.002 circuit-level、d∈{3..19} 时本文 K=24 陪集集成解码 LER 与 MWPM 之比从 d=3 的 1.0× 渐增至 d=19 的 ~2.1×；硬件对照 Micro-Blossom（ASPLOS 2025，d=15 867k LUT @43 MHz，延迟随 d 陡增，d≥5 超单轮提取时限致系统 infidelity 受罚）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
软件：Blossom V（C++ 库）、PyMatching（pip 包，输入 stim.DetectorErrorModel）、Fusion Blossom（带 Python 绑定）。硬件：Micro-Blossom（FPGA/加速）、Astrea（ISCA 2023）、Promatch（ASPLOS 2024，预解码扩展）。使用场景：追求最低 LER 的离线/内存实验解码，或作为其他解码器的精度上限基线；实时场景因延迟常需近似替代（UF、本文方法）。

补充（Triage 论文）：Triage 把 pymatching（Sparse Blossom，v2 的 C++ 核心）当作解码器延迟与抖动分布的事实来源而非评估对象——①延迟建模：profiling pymatching 在不同解码 volume 下的单次解码延迟，幂律拟合 t_dec=A·volume^α（α=1.17），用作调度仿真里每个 slice 的延迟（volume 由窗口缓冲大小/约束图 degree 决定）；②抖动校准：在 Stim 生成的 rotated surface-code 电路上逐 shot 测 pymatching 延迟（每设置 15K shots，warmup 后），拟合平均保持的 log-normal 抖动模型 t_actual=t_est·exp(−σ²/2+σz)，σ(d,p)=clamp(σ_base+α_d·log₂(d/5)+α_p·(p−p_ref), σ_min, σ_max)，得到 σ_base=0.3447、α_d=0.0041、α_p=15.03、p_ref=10⁻³、σ∈[0.30,0.70]，LOO 验证 MAE 0.064、尾部分位数 ~15% 相对误差。Triage 假设"延迟随 volume 单调增长对任何实用解码器成立"，因此相对性能趋势可推广到其他解码器。

涉及论文标题：
- Coset Ensemble Decoder for Quantum Error Correction with Algorithm-Hardware Co-Design
- Triage An Adaptive Parallel Window Decoding Scheduler for Real-time Fault-Tolerant Quantum Computation
