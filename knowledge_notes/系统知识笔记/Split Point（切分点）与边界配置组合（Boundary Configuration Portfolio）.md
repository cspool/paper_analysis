## Split Point（切分点）与边界配置组合（Boundary Configuration Portfolio）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Split point（SP）是流水线并行中把模型切成"边缘段 + 云端段"的层边界位置：LLaMA2-7B 有 32 层、候选 SP=0..32，SP=0 即 cloud-only、SP=32 即 edge-only，中间 SP 通过 pipeline 并行在低负载时接近最优延迟、高负载时接近最优吞吐。边界配置组合是 DynoPipe（§4.1）用动态规划离线预计算的 3-5 个 split point 集合，每个针对一种资源 regime：bandwidth-constrained（attention 层后、激活张量最小）、compute-constrained（早切、云端利用率最高）、memory-constrained（边缘 footprint 最小）、balanced（比例分配）。DP 递推（Eq.2）：F(d,l;R)=min_{i≤l} max{F(d-1,i-1;R), T_exec(i,l)+T_boundary(i)}，约束集 R_constraint(l) 保证各配置尊重边缘显存上限/云端 GPU 可用/带宽阈值。组合规模上界 |K|≤min(资源 regime 数, 层数)，均匀 transformer 架构实测 |K|=4 覆盖全部最优（{4,8,12,16}）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
SP 决定总延迟的四维构成（Eq.4）：T_total=T_queue(λ,μ)+T_edge(SP)+T_net(SP,RTT)+T_cloud(SP)，其中有效流水服务率 μ(SP)=1/max(T_edge(SP),T_cloud(SP)) 决定系统容量。规律：低负载（λ≪μ）最优 SP 偏云端（最小化单请求延迟）；负载趋近容量（λ→μ）最优 SP 偏均衡（T_edge≈T_cloud 最大化吞吐）；网络争用下最优 SP 移向传输量小的位置。论文实测 9 种工况（QPS=3/4/5 × 网络 Free/Moderate/Contention）仅出现 4 个不同最优 SP，组合大小 4 即可零残差覆盖；静态单 SP 在条件漂移时劣化 36%（SP=12 网络争用 QPS=4）到 82%（SP=4 于 QPS=5）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：DP 离线生成 + 实时查表选择。Portfolio 生成约 3 min/设备对、查找表 <30KB/模型（128 token、batch=1/4/8 的 per-layer 执行时间+激活大小 profile）。使用场景：一切边云流水划分决策；对比 FlexNN/EdgeShard 的静态单 SP，组合化选择使系统可覆盖多个资源 regime，避免"离线最优边界在条件漂移后失效"。局限：假设单调资源-性能关系（均匀 transformer 成立，MoE/混合模态模型可能需要更大组合）。

涉及论文标题：
- DynoPipe: Heterogeneous Edge-Cloud LLM Serving with Dynamically Orchestrated Pipeline Boundaries
