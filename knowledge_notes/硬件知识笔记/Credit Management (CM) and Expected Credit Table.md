## Credit Management (CM) and Expected Credit Table

术语是什么？

Credit Management (CM) 是 DFBM 的核心子模块之一，负责基于 coherence transaction type 的 two-stage flow control，控制包在 chiplet 与 interposer 之间的准入（admission）。CM 维护两类 credit：(1) Pre-Allocated Credits —— 管理 chiplet 主动发起的 Out-Req，数量由 cache controller 最大 outstanding request 数预先协商，保证主动请求数量在可吸收范围内；(2) Reserved Credits —— 管理由外部请求触发的被动响应（Out-Rsp、Out-Fwd-Req），数量与 NoC VC 数量和 expected response 需求相关，保证 chiplet-to-interposer 出口方向在最坏情况下有足够吸收能力。Expected Credit Table 是一个静态查找表，将每种 coherence transaction type 映射到该事务可能触发的 passive response 最大数量。

从硬件架构角度拆解术语：

CM 的 two-stage admission pipeline 执行流程：Stage 1 (Coherence Information Extraction)——从到达的 In-Req 包中提取 coherence type、VN、地址等信息 → 查询 Expected Credit Table → 获得该事务可能触发的最大 Out-Rsp/Out-Fwd-Req 数量 K → 转发到 Stage 2。Stage 2 (Admission Arbitration)——检查 current reserved credit 是否 ≥ K → 查询 CVN-DB current occupancy → 若 credit 充足且 CVN-DB 有剩余空间则 admit（允许包进入 chiplet，消耗 reserved credit K）；若不足则 block（暂缓进入，避免包进入后生成无法被 DFBM 吸收的响应导致 backpressure）。CM 还负责 credit 回收：当 Out-Rsp/Out-Fwd-Req 成功离开 DFBM 后，对应的 reserved credit 被释放。Pre-Allocated credits 的回收时机是：Out-Req 发送到 interposer 后消耗，对应 In-Rsp 返回 chiplet 后释放。对非确定性 coherence transition 引入的 dummy packet 也在 CM 的 credit 管理下——dummy packet 消耗 credit 但仅局部传输（chiplet-DFBM interface 内），返回后释放。

术语一般如何实现？如何使用？

Expected Credit Table 离线生成：分析目标 coherence protocol（如 MESI Two Level）的 state machine → 对每种 In-Req/In-Fwd-Req type，穷举所有 cache state → 判断每个 state 下可能产生的响应类型和数量 → 取最大值为 upper bound K → 填入 hardware lookup table（small SRAM/CAM）。CM 的 credit tracking 以 register-based counter 实现，pre-allocated credit counter 由系统集成方根据 chiplet cache controller 规格配置，reserved credit counter 由 CM 硬件根据 Expected Credit Table lookup 动态管理。CM 在硬件实现上与 CVN-DB 紧密耦合：Stage 2 的 admission decision 同时参考 reserved credit 和 CVN-DB occupancy，在低负载下当 credit 充足时几乎不引入额外延迟（包直接通过），在高负载下根据 credit 和 buffer 状态动态节流。

涉及论文标题：
- Deadlock-Free Bridge Module for Inter-Chiplet Communication in Open Chiplet Ecosystem

