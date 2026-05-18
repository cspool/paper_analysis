## Deadlock-Free Bridge Module for Inter-Chiplet Communication in Open Chiplet Ecosystem

- baseline方法是什么？
  Baseline class为三类state-of-the-art inter-chiplet deadlock avoidance方法：MTR (boundary router turn-restriction based)、DeFT (virtual channel isolation for upward/downward traffic) 和RC (in-chip permission network for injection control)。它们共同将deadlock处理与chiplet内部NoC实现绑定，不能满足开放chiplet生态中"plug-and-play chiplet"的目标。

  全栈执行例子（以PARSEC workload + MTR baseline + 4-chiplet homogeneous mesh + gem5/Garnet为例）：
  - 算法层：论文未明确说明（PARSEC application不涉及ML算法）。
  - 系统框架/Serving层：论文未明确说明（gem5 full-system simulation，非serving框架）。
  - 编译框架层：论文未明确说明（无编译框架修改）。
  - kernel调度层：论文未明确说明（无GPU kernel调度）。
  - 硬件架构层：MTR在chiplet边界router实施turn restriction，禁止特定方向的包转向以打破跨chiplet CDG环。例如，限制从interposer vertical channel进入chiplet的包只能转向特定方向，避免形成cyclic dependency。执行流：In-Req从interposer到达chiplet boundary router → turn restriction检查 → 若方向违反限制则被阻塞或用其他vertical channel → 进入chiplet内部NoC → 触发cache controller查找 → 可能产生Out-Rsp → 从chiplet出口router发回interposer。MTR成本低，但会限制vertical channel选择造成load imbalance，且依赖TSV/interposer wiring layout，portable性差。
  - 芯片设计层：4个homogeneous chiplet通过shared interposer互连，chiplet和interposer均采用4×4 mesh。MTR的边界turn restriction与具体TSV/interposer wiring绑定，不同floorplan需要重新设计限制规则。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出DFBM (Deadlock-Free Bridge Module)，放在chiplet NoC与interposer NoC边界作为独立bridge module，包含Credit Management (CM) 和Cross-VN Deadlock Buffer (CVN-DB) 两大核心模块，通过transaction-aware packet injection control实现deadlock avoidance，且不要求了解chiplet内部NoC拓扑细节。

  **缺陷1：MTR的turn restriction限制vertical channel选择，导致non-uniform vertical channel distribution下load imbalance**
  → DFBM不依赖转向限制，而是通过CM的Expected Credit Table预测事务后续响应数并在入口预留credit。包可以自由选择任何可用的vertical channel，对非均匀vertical channel分布更稳健。在non-uniform条件下（12条vertical channel分布不均），DFBM相对MTR延迟降低1%-4%。

  **缺陷2：DeFT要求每个VN至少2个VC用于upward/downward traffic隔离，增加路由器资源和面积（约+48%）**
  → DFBM不要求片内NoC做额外VC隔离，而是在bridge module中通过CM管理credit、CVN-DB管理共享buffer。DeFT需要per-VN的VC dedicating，DFBM用CVN-DB共享deadlock buffer将面积开销从~5% (dedicated per-VN buffer) 降到~2.5% (shared buffer)，且开销位于成本更低的interposer侧。

  **缺陷3：RC的permission network需要片内NoC加专用控制网络，增加验证复杂度并产生vendor lock-in**
  → DFBM继承RC的injection control思路但把控制逻辑外置到bridge module，不要求供应商修改片内NoC。CM的two-stage admission arbitration (Stage1 coherence extraction + Stage2 credit/CVN-DB check) 替代RC的片内permission network。同时DFBM不是持续节流，而是根据expected credits、CVN-DB occupancy和congestion状态动态调节，低负载下延迟更接近原生路径。

  **缺陷4：Recovery类方法 (UPP/Steered Bubble) 允许deadlock先发生再恢复，需检测逻辑和escape channel，引入架构修改和验证负担**
  → DFBM是avoidance机制，不依赖死锁检测后再打断环。通过在最坏情况下预留足够credit保证chiplet-to-interposer出口方向的吸收能力，从根源上防止跨chiplet CDG形成环。无需deadlock检测逻辑、escape channel或directional bubble routing。

  **缺陷5：所有baseline都将deadlock处理与chiplet内部NoC实现绑定，破坏开放生态的模块化目标**
  → DFBM作为独立bridge module实现了"plug-and-play"目标：不同供应商的chiplet无需了解彼此内部NoC实现细节即可直接互连。DFBM只需要供应商提供少量协议参数（coherence state machine依赖、cache controller最大outstanding request数、NoC VC数），而不需要修改片内NoC路由算法、VC分配或拓扑。

  论文方法全栈执行例子（以PARSEC workload + DFBM + 4-chiplet mesh + gem5/Garnet为例）：
  - 算法层：论文未明确说明（PARSEC application，非ML场景）。
  - 系统框架/Serving层：论文未明确说明。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明。
  - 硬件架构层：DFBM two-stage admission pipeline执行流：外部chiplet发来In-Req → DFBM CM Stage1提取coherence type → 查询Expected Credit Table得到最大Out-Rsp/Out-Fwd-Req数K → CM Stage2检查current reserved credit + CVN-DB occupancy → 若credit充足则admit In-Req进入chiplet（消耗reserved credit），若不足则block → chiplet内部cache controller处理请求产生Out-Rsp/Out-Fwd-Req → 响应包返回DFBM → 若interposer方向空闲则直接发出，若拥塞则进入CVN-DB → CVN-DB按Response > Forward-Request > Request优先级drain → 包成功发出后CM更新credit。DFBM在请求进入chiplet前就为其后续响应预留边界吸收能力，保证chiplet-to-interposer vertical channel不成为不可释放的依赖点。
  - 芯片设计层：DFBM位于chiplet NoC与interposer NoC边界（interposer侧），连接两侧已有VC接口不引入专用控制线。CM和CVN-DB均在interposer side实现，面积开销（~2.5% with CVN-DB）由成本更低的interposer承担。对非确定性coherence transition，dummy packet只在chiplet-DFBM接口内局部传输，对全局延迟和带宽影响有限。
