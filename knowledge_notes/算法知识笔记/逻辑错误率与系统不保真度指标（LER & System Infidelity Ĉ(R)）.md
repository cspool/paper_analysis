## 逻辑错误率与系统不保真度指标（LER & System Infidelity Ĉ(R)）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LER（Logical Error Rate）：一个完整 QEC 周期（d 轮 syndrome 提取）内逻辑错误未被纠正的概率，是解码精度的主指标，阈值定理下随码距指数下降。系统不保真度 Ĉ(R) 是本论文自定义的系统级指标：量化 feedback decoding 场景中逻辑 patch B 的解码延迟对逻辑 patch A 保真度的损耗——从经验式 E(n)=½(1−(1−2ε)^n) 出发，用 d 轮 LER E(d) 重参数化（ε 不可直接测量，FTQC 的基本单位是整个 QEC 周期）：有效错误率 Ê(m)=½(1−(1−2E(d))^m)、保真度 F̂(m)=(1−2E(d))^m；B 延迟 R（以提取轮计）使 A 的保真度乘以 (1−2E(d))^{R/d}，反转为不保真度 Ĉ(R)=1−(1−2E(d))^{max(1,R)/d}∈[0,1)。max(1,R) 掩码含义：解码在一轮内完成即无 backlog、不影响 LER。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
E(n) = ½(1-(1-2ε)^n)                      # 文献[26] 经验式
(1-2ε)^d = 1-2E(d)                        # 重参数化：基本量=E(d)
F̂(m+R/d) = (1-2E(d))^{R/d} · F̂(m)         # B 的延迟 R 折算进 A 的保真度
Ĉ(R) = 1-(1-2E(d))^{max(1,R)/d}           # R=L/l, l=单轮提取时长
```
使用效果（本文）：Micro-Blossom 的 LER 最低，但 d≥5 时延迟超过一轮提取时限使 Ĉ(R)>0 受罚，最终系统 infidelity 反超 UF 类；本文在 d=11 时较 Micro-Blossom 降低 74.3%、较 Helios 降低 51.7%——把"精度×延迟"折算为单一可比量，证明低延迟对 feedback decoding（非 Clifford 门条件操作）的真实价值。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Pauli-frame 解码（内存实验）只用 LER + 反应时间即可；feedback 解码必须用 Ĉ(R) 类联合指标。实现仅需各解码器的 LER(E(d)) 与延迟(L) 两个实测量；物理怠机误差、Dynamic Decoupling、Pauli Twirling 的影响全部封装于 E(d)。后续工作可直接复用该公式对比不同解码器在条件逻辑操作负载下的系统影响。

补充（Triage 论文）：Triage 用"插入 idle layer 数 + 总执行层数→LER"的间接度量——同步失败时插入 idle layer，idle 期间 qubit 经历额外纠错轮直接抬高 LER，因此 LER 是总执行层数（含 idle）的单调函数：先模拟 window-based lattice surgery（d=9、p=3×10⁻³、Stim ≥10⁵ runs/点）得到逐层 LER，再按每应用总层数聚合出整体 LER（外推 d=21）；每层时间 T_layer=d×T_meas（d=21 时超导约 21μs、离子阱/中性原子 2.1-21ms），T_total=N_total_layers×T_layer 把 idle 层减少直接折算为墙钟时间节省。结果：Triage 相比标准时间并行 baseline 平均 LER 降低 52.6%；慢解码器区（τ_dec>τ_gen）仍维持低 LER。

涉及论文标题：
- Coset Ensemble Decoder for Quantum Error Correction with Algorithm-Hardware Co-Design
- Triage An Adaptive Parallel Window Decoding Scheduler for Real-time Fault-Tolerant Quantum Computation
