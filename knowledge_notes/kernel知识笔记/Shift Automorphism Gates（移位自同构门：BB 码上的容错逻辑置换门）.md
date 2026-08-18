## Shift Automorphism Gates（移位自同构门：BB 码上的容错逻辑置换门）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- automorphism 门是"对物理 qubit 做置换、在逻辑 qubit 上诱导出 CNOT 电路"的容错逻辑门；shift automorphism 是其中保持 X/Z 类型（不混基）的 36 元素子群，源于 BB 码的循环平移对称（qubit 与单项式 $x^i y^j$ 对应，平移即移位）。实现方式：沿度 6 连通图在数据 qubit 与 check qubit 之间做连续 swap 操作（先绿后红两段，图 3）——与 syndrome extraction 共用同一稀疏连接，因此 12 个生成元可容错实现，且任意 shift automorphism 至多用 2 个生成元合成。成本（[27] Table I）：τ=14 个时间步；错误率 10⁻⁶·⁴（gross）/10⁻¹⁴·⁵（two-gross）@ p_phys=10⁻³、10⁻¹²·²/10⁻³⁷ @10⁻⁴——比 LPU 测量快一个数量级且错误低得多，故是调度中的"便宜"原语。
- 从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 本论文中 automorphism 作为"测量重定向 kernel"调度：
  ```
  测量逻辑 Pauli P 的 kernel 流程:
  1. 查表选 automorphism 序列 g_1..g_r（r≤2 生成元）使 g(P) 落入 LPU 原生集
  2. for g in g_1..g_r: 沿连接图执行 g 的 swap 序列（绿段→红段），
     逻辑算子随之置换（X_L0/X_L1/X_L2 的支撑被搬动，改变与 pivot Z_L0 的重叠）
  3. LPU 测量 g(P)（单次原生测量，占 τ=120/216）
  4. 逆置换恢复，byproduct Pauli 记录到跟踪表
  ```
  调度优化：①TSP 排序使相邻旋转间重定向所需 automorphism 轮数最少；②双轨模式（dual-track）——ZX-duality 使 automorphism 群在两个 6-qubit 块上同步作用，当两轨旋转标签一致且为纯 X/Z 型时，一条 automorphism 序列 + 一次同时 LPU 测量完成两轨旋转，吞吐近 2×（pivot Y 测量占双模块时串行）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 代数基础：BB/generalized bicycle 码的 automorphism 结构（arXiv:2412.04181 的 shift automorphism 与 ZX-duality；arXiv:2606.05044 Davenport-Blue-Chuang 给出 cyclic submodule 框架下 block-separable automorphism 的充要条件与 fold-transversal CX 判据）。注意 automorphism 保持码距（弱意义），但电路级 fault distance 可能因单 CZ 对齐逻辑算子而 -1。工业用法：自行车架构指令集中的标准指令之一，编译策略是"多 automorphism、少测量"（本论文因 automorphism 便宜且快而把 LPU 测量次数作为首要最小化目标）。
- 涉及论文标题：
- Distilling Magic States in the Bicycle Architecture
