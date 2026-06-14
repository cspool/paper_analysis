# 影响并发的架构/机制

**核心价值问题**：**什么架构因素限制或扩展了并发效率**？该限制是否可绕过？

## 引导提问

| # | 引导提问 | 这个问题在探测什么（review 模式来源） |
|---|---------|----------------------------------|
| 4.1 | 该方法的**memory hierarchy**（HBM → L2 → SMEM → Register）中，哪一级是并发瓶颈？多 kernel 并发时这一级的带宽/容量是否被**共享和竞争**？瓶颈是**容量不足**（放不下 working set）还是**带宽不足**（多 kernel 共享饱和）还是**延迟过高**（remote chiplet 访问 10-15×）？ | 探测**瓶颈的根因类型**——review 中「HBM bandwidth 是共享资源——所有并发 kernel 的全局访存经同一组 HBM stack」「Expert Weight Load 42% 是核心瓶颈」「L2 命中 vs HBM miss 性能差异 2×」「SMEM capacity 是纯软件方案的物理硬上限」都在区分瓶颈类型 |
| 4.2 | 该方法的**数据访问模式**（coalesced / scatter / gather / strided）是什么？多 kernel 并发时不同 kernel 的访问模式是否会**互相破坏 coalescing**？SMEM 的 bank conflict 是否被分析过？ | 探测**并发对访存效率的破坏**——review 中「bank conflict 可使有效 SMEM BW 退化 88%」「不同访问模式混合可能使 DRAM 效率从 100% 退化到 3%」「多微算子并发时不同算子的 bank 访问模式不同需 layout 转换」 |
| 4.3 | 该方法是否涉及**跨 chiplet / 跨 die / 跨 NUMA node** 的数据传输？remote vs local 延迟/带宽比是多少？NUMA 效应如何影响并发 kernel 的 placement？ | 探测**分布式硬件下的并发约束**——review 中「remote/local 延迟比可达 10-15× 需 NUMA-aware 调度」「L1.5 cache 专门缓存 remote data」 |
| 4.4 | 该方法的**片上通信**使用什么 NoC 拓扑和路由？通信模式（one-to-one / multicast / all-to-all）是否与 NoC 的拥塞控制机制兼容？链路带宽是否支持该通信模式的高并发？ | 探测**NoC 对并发通信的支撑能力**——review 中「多 VN 隔离不同 coherence 消息类不互相阻塞」「All-to-All 占 MoE 训练 41.5%-95.7% 时间，HalfRing 平均 2.28× 加速」「leaky-bucket 平滑突发流量」「优先级队列按时间戳调度通信 chunk」 |
| 4.5 | 该架构限制是**物理不可绕过**（SRAM 容量、HBM PHY 数量）还是**可通过软件策略缓解**（warp 调度顺序、tile size 选择、block order）？硬件提供的隔离机制（MPS / MIG / VN）能否消除共享资源冲突？ | 探测**硬限制 vs 软限制**——review 中「SMEM 容量是物理硬上限」「硬件限制不可绕过只能减少数据搬运量」「MPS SM Partition 硬约束需 CUDA 初始化前设定」区分了两类限制 |

## 评估标准

| 回答特征 | 价值信号 | 判定 |
|---------|---------|------|
| 瓶颈根因被精确区分（容量/带宽/延迟），且各因素的贡献比例被量化（如「Expert Weight Load 占 decode time 42%，Attention 19%，Other 12%」） | 瓶颈可精确诊断 | **高** |
| 并发对访存效率的影响被量化（如「2 kernel 并发时 DRAM efficiency 从 90% 降至 45%」），且缓解策略被讨论 | 并发代价可量化 | **高** |
| 硬限制和软限制被明确区分，软件可缓解的限制给出了具体策略 | 优化空间可评估 | **高** |
| 仅泛泛提及「memory wall」等概念，无定量分析 | 信息增量有限 | **中** |
| 无并发架构分析 | 无并发相关性 | **低** |
