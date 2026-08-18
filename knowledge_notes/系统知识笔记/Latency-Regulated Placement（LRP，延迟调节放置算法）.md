## Latency-Regulated Placement（LRP，延迟调节放置算法）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LRP 是 DynoPipe 的运行时边界编排算法（Algorithm 1）：每个决策周期从预计算配置组合中选最小化预测端到端延迟的 split point，而非在线重解优化。核心组件：(1) ComputeCost(S_i,D_j,λ)=α·T_comp(S_i,D_j)+β·T_mem(S_i,D_j)+λ·T_comm(S_i)——组合计算/内存/通信代价；(2) SelectBoundary(system_state)——按触发阈值选配置：带宽<τ_bw→activation-minimal、edge_load>τ_compute→early-cloud、memory_pressure>90%→memory-aware、否则 balanced（多约束同时激活时选最坏 stage 延迟最小者）；(3) λ 为瓶颈感知动态权重——带宽受限时增大（偏向通信最小边界）、计算过载时减小（偏向负载均衡）、内存受限时显存上限优先；(4) hysteresis δ=15-20% + cooldown 防振荡切换，λ 用指数平滑防抖动。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
LRP 把"在线重解 O(n) 超 sub-ms 预算"的问题转成 O(|K|)（|K|≤5）表查找。运转循环（Algorithm 1 伪代码）：Pre-compute 各资源场景的边界配置 → while system active：monitor 网络/计算/内存条件（每 500ms）→ target_boundary=SelectBoundary(state) → 若边界变化且改进>δ：λ=adapt 到主导瓶颈 → foreach stage：Find D* 最小化 ComputeCost（带 hysteresis 阈值 δ）→ Assign S_i→D* → Execute boundary migration。设计取舍：有限组合启发式而非穷举在线重优化（O(n)/次太贵）或学习式控制器（需训练数据、边云条件漂移下难稳定）；假设单一边云边界/请求（跨域延迟比域内高 10-50×）、单调资源-性能关系、代表性离线 profile。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：轻量遥测（带宽、GPU 利用率、内存压力每 500ms 采样）+ 预计算查找表（<30KB/模型，离线 profiling 128 token、batch=1/4/8）+ 触发阈值判定的启发式。使用场景：在线边云 serving 的边界选择；效果：决策延迟 sub-ms、切换频率 <2 次/min（QPS=5）、4 并发源下 <1.1× 退化（对比 EdgeShard 45×、FlexNN 1.8-3.1×）。局限：不保证全局最优——非均匀 per-layer 成本或强非平稳环境需更大组合或学习式控制。

涉及论文标题：
- DynoPipe: Heterogeneous Edge-Cloud LLM Serving with Dynamically Orchestrated Pipeline Boundaries
