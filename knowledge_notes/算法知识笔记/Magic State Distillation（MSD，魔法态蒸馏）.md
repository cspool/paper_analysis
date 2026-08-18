## Magic State Distillation（MSD，魔法态蒸馏）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- MSD 是一种容错协议：消耗 n 份低保真（误差 $p_{\rm in}$）的噪声魔法态，输出 k 份更高保真的魔法态，用"数量换质量"。逻辑链：①非 Clifford 门（如 $T=\mathrm{diag}(1,e^{i\pi/4})$）受 Eastin-Knill 定理限制，在大多数码上无法 transversal 实现，通常经 gate teleportation 消费魔法态实现；②直接注入的魔法态误差 $p_{\rm inj}$ 远高于算法需要的逻辑误差 $p_L$；③蒸馏码（如 [[15,1,3]] Reed-Muller 码，Bravyi-Kitaev）带 transversal T 门，把 n 份输入映射到 k 份输出，单轮输出误差 $p_{\rm out}\approx c\,p_{\rm inj}^t+O(p_L)$（t=O(d) 为抑制阶数，c 为协议常数；15-to-1 协议 c=35、t=3）；④多轮串联 $p_{\rm out}^{(r)}\sim \tilde c\,p_{\rm inj}^{t^r}$，两轮即够大多数应用。Bravyi-Haah triorthogonal 构造使开销随精度多对数增长 $O(\log^\gamma(1/\epsilon))$、$\gamma=\log_2 3\approx 1.6$。在 surface-code 架构中蒸馏工厂占物理 qubit 的 90% 以上（"T-gate 瓶颈"）。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 本论文的 Pauli-measurement-based 形式（triorthogonal 矩阵 $G\in\{0,1\}^{m\times n}$，前 k 个奇权重行=输出、m−k 个偶权重行=parity check，列 c 对应旋转 $e^{i\pi/8}Z^{\otimes S_c}$，$S_c=\{r:G_{rc}=1\}$）：
  ```
  init: 全部物理 qubit 置 |+⟩ + 1 轮 syndrome extraction → m 个逻辑 |+⟩
  for c in 1..n:                # n=15 个 π/8 旋转
      取 1 个噪声 |T⟩（p_in），经注入 gadget 实现 exp(iπ/8·Z^⊗S_c)
      （测量结果=1 时补 exp(iπ/4·Z^⊗S_c) 条件校正）
  measure 偶行 qubit 于 X 基；postselect 全 |+⟩
  成功 → 前 k 行输出蒸馏 |T⟩^⊗k；失败 → 丢弃本轮
  ```
  相比 Bravyi-Haah 原构造（n-qubit 稳定子态制备 + 逐 qubit T 门 + Clifford unencoding），只用 m 个逻辑 qubit、更少 Clifford 门。本论文把 15-to-1、20-to-4、8-to-CCZ、49-to-1、51-to-3CS、64-to-2CCZ 全部放到单个 BB 码块内执行。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现方式分代：①surface-code lattice surgery 工厂（Litinski）——多 patch + surgery 路由，15-to-1 需 4620 qubit/256 时间步；②QLDPC 单块工厂（本论文）——378（gross）/734（two-gross）qubit，15-to-1 two-gross 输出 1.0×10⁻⁸、49-to-1 达 2.0×10⁻¹¹（$p_{\rm phys}=10^{-3}$）；③作为 MSC 的二级协议突破其 10⁻⁹ 天花板（$35\cdot(10^{-9})^3=3.5\times10^{-26}$）。工业实现：Microsoft QDK 的 RoundBasedFactory（"15-to-1 RM prep"+"15-to-1 space efficient" 级联）。Web 参考：Bravyi-Kitaev quant-ph/0403025（PRA 71, 022316）、Bravyi-Haah arXiv:1209.2426（PRA 86, 052329）、Litinski 1708.07197 系列。
- 补充（O3LS 论文）：O3LS 的评估用 15-to-1 蒸馏协议的 magic state factory（Bravyi-Kitaev），工厂置于自动生成的 squeezed 数据布局之外、与数据区之间保证至少一条路由路径连接；π/4 与 π/8 Pauli-product measurement 用标准 gate teleportation 协议（Litinski [34] Fig.7/11(b)）。MSD 是 O3LS 优化目标的外在约束：布局与调度只优化数据区（架构因素），magic state 供给成本不在其调度范围内，但工厂放置（布局外+连通性）纳入布局设计评分（C(B) 要求 ancilla 与工厂路由连通）。
- 补充（TACO 论文）：TACO 用 15-to-1 蒸馏作为默认协议并给出资源模型：11 个逻辑 qubit tile、每 11 cycles 产 1 个 magic state、错误抑制 35p³（p=10⁻³ 时 3.5×10⁻⁸）。TACO 把 MSD 当作架构一级的"可调资源"而非固定开销：magic state 吞吐（块数）由空间-时间体积最小化决定——18 比特 QFT 从 1 到 10 magic states/cycle 扫描，最优 4 个/cycle（体积比单 block 降 57%）；20 比特 QFT 下 3 个 distillation block × 11 units = 363 tiles 支撑 760,901 cycles（对比 PBC Fast 11 blocks/121 tiles/2,382,355 cycles）。随 T 门成本下降（MSC），TACO 的 magic-state 开销占比几乎恒定而数据体积显著下降——这是与 PBC 的本质区别（PBC 中 T 优化收益递减，TACO 中收益更大）。
- 补充（Triage 论文）：Triage 把 T 门（非 Clifford）当作整个解码调度问题的"绝对同步点"来用——T 门经 gate teleportation 实现，末端的 classically-controlled S-gate 校正不能穿过 T 门吸收进 Pauli frame，执行前必须物理纠正 E_acc（恢复 |ψ⟩），因此解码器必须先解码完相关因果锥完成 Pauli frame 同步，否则逻辑操作 stall、插入 idle 层、LER 上升。这与"Clifford 门可异步"形成 dichotomy：正是魔术态/非 Clifford 操作把解码从吞吐问题变成带优先级的实时调度问题（Triage 的 deadline/causal cone 属性、紧急模式全部围绕同步点设计）。Triage 评估用 15-to-1 MSD 协议与 T-gate 密度不同的 benchmark（T-Den. 7.69%-49.61%，如 rotation_C+T 11.80%、MSD15to1 45.83%），验证调度器在 T 门密集应用上的收益。
- 涉及论文标题：
- Distilling Magic States in the Bicycle Architecture
- O3LS: Optimizing Lattice Surgery via Automatic Layout Searching and Loose Scheduling
- Transpiler-Architecture Co-Design to Curb Clifford Costs in Fault-Tolerant Quantum Computing
- Triage An Adaptive Parallel Window Decoding Scheduler for Real-time Fault-Tolerant Quantum Computation
