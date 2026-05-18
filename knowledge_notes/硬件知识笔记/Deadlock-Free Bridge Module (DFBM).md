## Deadlock-Free Bridge Module (DFBM)

术语是什么？

DFBM (Deadlock-Free Bridge Module) 是本文提出的独立桥接硬件模块，置于 chiplet NoC 与 interposer NoC 的边界，用于在开放 chiplet 生态中实现跨 chiplet 通信的 deadlock-free 保证。DFBM 不要求了解 chiplet 内部 NoC 的拓扑、路由算法或 VC 分配细节，仅需供应商提供少量协议参数（coherence state machine 依赖、最大 outstanding request 数量、VC 数量），从而实现 "plug-and-play" 的模块化 chiplet 集成。DFBM 包含两大核心子模块：Credit Management (CM) 和 Cross-VN Deadlock Buffer (CVN-DB)。

从硬件架构角度拆解术语：

DFBM 连接 chiplet NoC 和 interposer NoC 两侧已有的 VC 接口，不额外引入专用控制线。其执行流水线分为两个阶段和两条路径：(1) 对外部进入 chiplet 的 In-Req/In-Fwd-Req，CM Stage1 提取 coherence type → 查询 Expected Credit Table 获得该事务可能触发的最大 Out-Rsp/Out-Fwd-Req 数量 K → CM Stage2 结合 pre-allocated credit、reserved credit 和 CVN-DB occupancy 判断是否 admit/block → 若 admit 则消耗 reserved credit 并允许包进入 chiplet；(2) 对 chiplet 主动发出的 Out-Req，CM 使用 pre-allocated credit 管理（数量由 cache controller 最大 request 数预先协商）→ 消耗 credit 后允许发向 interposer → 当对应 In-Rsp 返回并进入 chiplet 后释放 credit；(3) 对返回的 Out-Rsp/Out-Fwd-Req，若 interposer 方向空闲则直接发出，若输出端拥塞且超过阈值则进入 CVN-DB → CVN-DB 按 VN 优先级（Response > Forward-Request > Request）drain → 包成功离开后 CM 更新 credit。DFBM 的核心效果是：在请求进入 chiplet 之前就为其后续响应预留边界吸收能力，保证 chiplet-to-interposer vertical channel 不成为不可释放的依赖点，从根源上防止跨 chiplet CDG 环形成。

术语一般如何实现？如何使用？

DFBM 以 Verilog RTL 实现 CM 和 CVN-DB 模块，集成到 chiplet 边界 router 与 interposer NoC 接口之间。Expected Credit Table 通过分析 MESI coherence protocol state machine 离线生成：对每种 In-Req coherence type，穷举所有可能的 state transition → 统计可能产生的 Out-Rsp/Out-Fwd-Req 最大数量 → 填入 table。对于非确定性 transition（同一输入事务可能产生 0 到 K 个响应，Data 与 ACK 组合不固定），引入 dummy packet 将响应数规整到协议最大 K，使 CM 使用保守的 credit 规则保证 worst-case absorption。CVN-DB 以 shared buffer 替代 dedicated per-VN buffer，通过优先级仲裁（Response > Forward-Request > Request）保证高优先级响应先被 drain。面积评估通过 OpenSMART 生成 NoC RTL + EDA 综合：DFBM 使用 dedicated per-VN buffer 约 5% 面积开销，采用 CVN-DB 后约 2.5%，且开销位于 interposer 侧（成本更低）。性能评估在 gem5/Garnet 多 chiplet 模拟器上进行，对比 MTR、DeFT、RC 三类 baseline。

涉及论文标题：
- Deadlock-Free Bridge Module for Inter-Chiplet Communication in Open Chiplet Ecosystem

