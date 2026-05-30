## System-on-Wafer (SoW) Technology (晶圆级系统集成技术)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
System-on-Wafer (SoW) 是 TSMC 的晶圆级封装技术，将多个 compute die 和 HBM die 直接集成在单个硅晶圆上，通过 Local Silicon Interconnect (LSI) 和 XSR SerDes 实现 die-to-die 互联。与传统的 multi-chip module (MCM) 或 2.5D interposer 方案（如 CoWoS）相比，SoW 的显著差异在于 scale——可容纳 up to 24 compute dies + 96 HBM dies，总面积 >200,000 mm²，远超单 die photomask 限制（800-1,000 mm²）。论文使用的 TSMC SoW 配置为 8×3 2D mesh（24 compute dies），每个 die 垂直连接到 local HBM dies 通过 LSI（terabit-level bandwidth），水平方向相邻 compute dies 通过 XSR SerDes 互联。

从芯片设计角度拆解术语：
SoW 的芯片级互联组织和数据访问延迟模型：

```
TSMC SoW 8×3 Topology (俯视图):
┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│ Die0  │ Die1  │ Die2  │ Die3  │ Die4  │ Die5  │ Die6  │ Die7  │
│ +HBM  │ +HBM  │ +HBM  │ +HBM  │ +HBM  │ +HBM  │ +HBM  │ +HBM  │
├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
│ Die8  │ Die9  │ Die10 │ Die11 │ Die12 │ Die13 │ Die14 │ Die15 │
│ +HBM  │ +HBM  │ +HBM  │ +HBM  │ +HBM  │ +HBM  │ +HBM  │ +HBM  │
├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
│ Die16 │ Die17 │ Die18 │ Die19 │ Die20 │ Die21 │ Die22 │ Die23 │
│ +HBM  │ +HBM  │ +HBM  │ +HBM  │ +HBM  │ +HBM  │ +HBM  │ +HBM  │
└───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘

互连方式:
- 垂直方向 (同列): LSI (Local Silicon Interconnect) — terabit-level BW
- 水平方向 (同行): XSR SerDes links — terabit-level BW
- 每对相邻 die: D2D BW = 1.7 TB/s, latency = 200 ns/hop

访问延迟模型 (以 Die0 访问 Die7 的 expert 数据为例):
- Die0 → Die1 → Die2 → ... → Die7: 7 hops × 200 ns = 1400 ns
- Remote HBM access: 300 ns
- 返回 Die0: 7 hops × 200 ns = 1400 ns
- Total: 3100 ns (vs local HBM access: 300 ns)
- 差距: ~10× (随距离增大可到 ~15×)
```

SoW 与 Tesla Dojo 5×5 mesh 的对比：SoW 的矩形布局（8×3）使 die 之间的最大 Manhattan 距离更大（10 hops vs Dojo 的 8 hops），在无优化的 baseline 下产生更多 inter-unit communication。论文结果确认了这一点——SoW baseline 比 Dojo baseline 的 hop count 更高，因此论文策略在 SoW 上的相对加速比也更高（7.5× vs 6.0×）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- SoW 的关键技术：(1) LSI——die 与下方硅 interposer 之间的 local 互联，提供极高的垂直带宽；(2) XSR SerDes——超短距 SerDes（几毫米级别），用于水平方向的 die-to-die 互联，比传统海思 SerDes 功耗低得多；(3) 供电和散热——wafer 级集成对 power delivery network 和 thermal management 提出极高要求。
- SoW 当前状态：TSMC 已在 2025 ECTC 上展示了 SoW-X，但目前仍处于 roadmap/announcement 阶段，尚未在商用产品中实现。Tesla Dojo（5×5 mesh）是已部署的 wafer-scale 系统的代表。
- 编程模型是开放问题：论文采用 single-GPU-like 模型（与 Blackwell, Rubin 一致），但 WSC-LLM 和 MoEntwine 采用 multi-GPU-like 模型。最终哪种成为标准取决于行业演进方向。
- 对 MoE serving 的意义：SoW 可将完整 MoE 模型（200B-1000B）容纳在单个芯片上，消除跨机架/跨节点的网络通信瓶颈，将瓶颈从 inter-node network 转移到 intra-wafer D2D communication。

涉及论文标题：
- Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting
