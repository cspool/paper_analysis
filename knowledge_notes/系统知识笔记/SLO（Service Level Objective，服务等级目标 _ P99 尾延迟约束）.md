## SLO（Service Level Objective，服务等级目标 / P99 尾延迟约束）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- SLO（服务等级目标）是服务提供商对用户体验的量化性能承诺，通常以尾延迟百分位定义，如"P99 延迟 < X ms"。在 AI 推理 serving 中，SLO 是资源/功率供给的硬约束：系统在满足 SLO 的前提下最大化吞吐或最小化功率，SLO 达成率（SLO attainment = 满足 SLO 的请求比例）是与 Goodput 互补的核心评估指标（见知识库 SLO Attainment 条目）。论文（Power Sloshing）把 SLO 定义为"服务器以最大频率运行时在最高可持续负载下观测到的 P99 延迟"（作为 Baseline 基准），并用两个指标评估：Performance/Watt 中"性能 = 不违反 SLO 的最大 QPS"；功率评估中"SLO 违反区间比例 = P99 超过 SLO 阈值的采样区间占比"。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- SLO 在电源管理闭环中的约束作用：控制器以 u_G 目标区间隐含编码 SLO headroom——目标区间越高（接近饱和）功率省得越多但突发吸收能力越差（SLO 违反风险高）；越低则更保守。论文据此实现两个变体：SLO-Optimized（P75 目标区间，C1 上 40-50% 利用率）优先保 SLO，Power-Optimized（P90，60-70%）优先省电。评估流程：加载动态负载 trace（含 idle→峰值→突变 stress）→ 每采样区间算 P99 延迟 → 与 SLO 阈值比较统计违反区间比例 → 结果：C1 上 Baseline 4%、SLO-Optimized 5%（几乎无恶化）、Power-Optimized 14%（激进省电以 SLO 稳健性为代价）；D1 因延迟低全部 0%。非对称频率策略（u_G>u_max 立即拉满）正是为吸收突发、压低违反概率。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：serving 编排层定义并监控 SLO（如 P99 latency、TTFT/TPOT），host 侧的功率控制器不直接读 SLO，而是通过利用率目标区间间接保持 headroom。评估用 load-testing 基础设施：逐步增加 QPS 直到 P99 违反 SLO，记录饱和点。相关概念对照：AdaServe（multi-SLO serving，按请求 SLO 定制投机解码）、MuxWise SLO-aware Dispatcher（按 ITL SLO 分配 SM）、SLO Attainment（达成率指标）。论文未开源；SLO 定义与测量方法在论文 §VI 实验设置中给出，可在任何推理服务上按相同协议复现（QPS 扫描→P99 曲线→取最大可持续负载点）。

Rearchitecting-the-Datacenter-Lifecycle 补充视角（ISCA'26，SLO 作为生命周期 goodput 与硬件供给的硬约束）：论文把 SLO 定义为 LLM 推理的延迟约束——TTFT ≤ 400ms、TBT ≤ 100ms（取自 DynamoLLM [105]），并以此计算 goodput：roofline 模型预测给定 (硬件, 模型, 负载) 的 TTFT/TBT，不断增加请求负载直到任一延迟越过 SLO，取最大可持续 RPS 为 goodput，再据此求最小 GPU 供给量与对应利用率。SLO 在这里既是 operation 层调度的约束（heterogeneity-aware 调度把 prefill 放新 GPU、decode 放旧 GPU，仍须满足 SLO），也是 IT provisioning 层刷新决策的依据（SLO goodput 而非裸 FLOPS 决定 GPU 代际的真实效率）。该用法把 SLO 从"在线调度目标"提升为"数据中心 15 年容量规划与 TCO 优化的统一性能标尺"。
  - SHyLA 补充：SHyLA 的 SLO 形式为每用户吞吐约束 T_min^user = 25 tokens/s/user（[57] 的用户体验参考值）。两阶段 DSE 的 Stage 2 在给定该 SLO 下用进化算法（Top-K 选择，Algorithm 1）在部署设计空间中最大化系统吞吐：满足约束时偏向提系统吞吐（更大微批 b 或 PD aggregation），不满足时转向提每用户吞吐（更小 b 或更大张量并行 pt）；16/64 chiplet 下均以 PD aggregation 优于 PD disaggregation（Weight 复制 + KVCache 迁移的 NVM 写开销）。
涉及论文标题：
- Power Sloshing in Compound Servers for Large-Scale AI Inference Workloads
- Rearchitecting the Datacenter Lifecycle for AI
- SHyLA 3D-Stacked NVM-DRAM Hybrid LLM-Inference Architecture Exploiting Data and Memory Heterogeneity
