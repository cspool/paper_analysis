## Fair Share（容器级内存分层公平性 / per-container memory-tier accounting）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fair Share 是 Vistara 为多租户（多容器同 host）引入的内存分层公平机制：为每个容器设置本地 DRAM 用量上下界（local_low, local_high），容器本地 DRAM 使用超过 local_high 即把超额页 demote 到 CXL，从而把本地快层资源按约束公平分配，防止一个容器垄断 DRAM 把邻居挤到 CXL。论文动机：基线 TPP 无 per-container 记账时，两个同负载 CacheA 容器的 local fraction 失衡为 90% vs 70%，P99 时延偏差 20%；noisy-neighbor（邻居流量尖峰）时容器 A QPS 暴跌 65% 且被 OOM-kill。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程（两 CacheA 容器实验）：每容器注册 local_low/local_high（本地 DRAM 预算）→ 内核分层记账每容器的本地/CXL 用量 → 容器本地用量 > local_high 时触发其页面 demote 到 CXL（而不是挤占邻居）→ 容器本地用量 < local_low 时允许 promote。效果（Table IX）：两容器 local fraction 从 90%/70% 失衡变为 74%/71% 平衡，P99 时延从 283/327μs 降到 176/185μs（1.6×/1.8× 提升）；noisy-neighbor 下容器 A 最坏 QPS 跌幅从 65% 缩到 12% 且数秒内恢复，完全消除 OOM 风险。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：扩展内核内存分层（memory tiers/TPP demotion 路径）加入 per-cgroup 记账与水位触发（论文未给出公开补丁细节，联网搜索 2026-08 未找到公开实现）；控制面与 cgroup 集成，可按容器/服务配置 local_low/local_high。使用方式：多租户共置（多容器同 host 最大化利用率）时开启；单租户独占服务器是 Meta 主流模式（论文前文），Fair Share 针对共置场景。适用负载：缓存等分层内存敏感的共置容器；配合 cpuset.mems opt-out 与 weighted interleave 构成完整的多租户内存管理策略。

涉及论文标题：
- Vistara: Making CXL Real—Full Path from ASIC Design and OS Support to Hyperscale Deployment
