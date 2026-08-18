## 分离式 vs 单体 Serving（Disaggregated vs Monolithic Serving：kernel 覆盖缺失驱动的部署范式选择）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 分离式 Serving（Disaggregated Serving）指把模型推理流水线按算子类型拆到不同计算 tier 部署：数据预处理（特征转换）在独立 CPU tier、神经网络计算在加速器（GPU/AMD/MTIA）。单体 Serving（Monolithic Serving）指预处理与神经网络计算同机（同加速器或 client 侧）执行。论文（KernelEvolve）核心论点：当关键算子（尤其 200+ 数据预处理算子）在加速器上缺原生 kernel 时，模型被迫走分离式部署，缺失 kernel 成为"二进制部署约束"而非增量性能损失。量化代价（Table 2，生产 MTIA 模型）：Paradigm 1（client→MTIA 单体）P50/P75/P90/P99 = 39/44/46/61ms；Paradigm 2（client→CPU tier→MTIA tier）α=58/65/73/97ms，β=42/48/51/57ms，γ（预处理执行）=4/7/10/16ms，纯网络开销 δ=α-β-γ≈10-20ms——P99 从 61ms 恶化到 97ms（+25%），而这 10-20ms 是零计算收益的架构税。进一步代价：跨节点序列化/通信、级联故障、同步部署与版本兼容的运维复杂度、冗余 CPU 基础设施增加 TCO。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 系统架构决策流程（MTIA 生产模型）：①模型图含预处理算子（bucketize、cryptographic hash、top-k truncate、type downcast 等）→ ②检查 MTIA 原生算子覆盖 → ③缺算子 → 判定无法单体部署（host CPU 不可行：加速器服务器 host 资源不足、与 I/O/系统管理争用、加 CPU 违背加速器整合的经济性/功耗收益）→ ④切分离式：请求 client→CPU tier 做预处理→MTIA tier 做 NN → ⑤每个请求额外 2 跳网络（δ≈10-20ms）与序列化成本。对比：KernelEvolve 生成缺失 kernel 后恢复单体部署（Paradigm 1，P99 61ms），预处理与 NN 同加速器执行、零跨节点开销。该范式选择的两个影响面：性能面（微秒级 kernel 效率直接决定亚秒级/亚 100ms 延迟与 TCO）、架构面（kernel 覆盖决定部署拓扑可行性，缺失则阻块模型 launch，约束算法创新）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：分离式是标准微服务式横向扩展（各 tier 独立扩容），但为 latency-critical 推荐服务付出网络税；单体依赖"完整算子覆盖 + 同机资源充足"。KernelEvolve 的解法不是改部署框架而是填 kernel 缺口：对 MapId/MBDT 等生成 fused Triton kernel（MTIA v2i 上消除 CPU 回退，MapId 最高 4.07×、MBDT 最高 9.25×），使模型满足单体部署条件。这给出可复用的部署决策准则：kernel 覆盖 → 部署范式选择 → 端到端延迟/TCO；新加速器上线时"硬件可用与软件生态成熟度之间的 gap"是自动化 kernel 生成的最大需求窗口。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta
