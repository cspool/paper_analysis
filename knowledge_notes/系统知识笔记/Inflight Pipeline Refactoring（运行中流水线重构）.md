## Inflight Pipeline Refactoring（运行中流水线重构）

术语是什么？通过联网搜索让回答具体和精准。

Inflight Pipeline Refactoring是FlexPipe提出的核心机制：在LLM推理服务不中断的前提下，运行时动态改变pipeline的拓扑结构（stage数量、stage边界、并行实例数），使pipeline粒度实时适应请求分布的变化。不同于传统系统中pipeline配置在部署前由离线优化决定且运行中保持固定，Inflight Pipeline Refactoring将pipeline stage粒度作为运行时连续控制变量——通过监控请求到达的coefficient of variation (CV)、队列长度、吞吐和延迟profile，持续在候选granularity set中选择最优配置，并在高CV时自动拆细pipeline（Pipeline Expansion）、低CV时自动合并pipeline（Stage Consolidation）。决策延迟<5ms。

从系统架构角度拆解术语：

FlexPipe描述的Inflight Pipeline Refactoring运转流程：
1. **持续监控**：Resource Monitor采集请求到达interval的CV（ν_t = σ_t/μ_t）、各stage queue length q̂_j、per-stage latency和throughput。
2. **Granularity评估**：Performance Controller对候选granularity set G={g_1,...,g_K}（每个g_k=(η_k, b_k)包含stage数η_k和batch size b_k）计算优化分数 S_k = α·T_k/T_max + (1-α)·L_min/L_k · exp(-|ν_t-ν_k|/σ)。第一项最大化吞吐，第二项最小化延迟，指数项确保所选granularity与当前CV匹配。
3. **Pipeline Expansion（CV升高时）**：选择更细粒度的pipeline配置→通过HRG查找可用GPU（优先选择host memory中保留该模型参数副本的服务器）→新GPU instance加载split stage→在旧pipeline继续服务的同时异步迁移KV cache→完成迁移后更新gateway routing metadata→后续请求流入细粒度pipeline。
4. **Stage Consolidation（CV回落时）**：选择更粗粒度的pipeline配置→layer-wise merge相邻fine-grained stage的state→重建pipeline→更新load balancing路由→释放多余GPU instance（参数可保留在host memory中）。
5. **一致性维护**：KV cache迁移使用token-level validity mask C(t)=∪_{i∈GPUs} KV_i(t)⊗M_valid，仅同步有效token的KV cache，避免全局状态复制。

Refactoring决策由Algorithm 1驱动：每optimization interval监控请求强度λ_t和特征velocity ν_t=∂λ_t/∂t，更新队列长度q̂_j，遍历所有granularity计算scores，选最优g*，若g*≠g_current则触发parameter migration with consistency → update routing metadata → activate new pipeline。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：(1) 细粒度stage切分是refactoring的前提——模型必须先通过operator-level profiling被分解为可合并的fine-grained stage（如32-stage），保留attention block等层级结构边界以支持未来合并；(2) KV cache迁移优先使用RDMA（低延迟高带宽），不支持RDMA时fallback到sendfile系统调用做kernel-space zero-copy传输，避免NCCL的数秒级connection establishment overhead；(3) 多granularity data parallelism框架（公式5）决定每种granularity的并行实例数M(g_k)=⌊μ_total/μ_k⌋，其中μ_k=T_k/(β_1+β_2·η_k)考虑pipeline coordination overhead；(4) FlexPipe原型约7K LoC实现，其中3.2K LoC用于动态operator-level model partitioning/merging；(5) 在82-GPU集群上CV=4时stall recovery仅9ms（比MuxServe/ServerlessLLM快约82%）。

涉及论文标题：
- FlexPipe: Adapting Dynamic LLM Serving Through Inflight Pipeline Refactoring in Fragmented Serverless Clusters
