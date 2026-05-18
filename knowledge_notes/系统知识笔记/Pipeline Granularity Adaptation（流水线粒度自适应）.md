## Pipeline Granularity Adaptation（流水线粒度自适应）

术语是什么？通过联网搜索让回答具体和精准。

Pipeline Granularity Adaptation是FlexPipe提出的根据实时请求分布动态选择最优pipeline stage数量的调度策略。Pipeline granularity由stage数η_k和batch size b_k共同定义：细粒度pipeline（如32-stage）per-stage compute time低（9.67ms vs 69.94ms）、参数加载快（5.43s vs 47.14s）、max batch大（1024 vs 128），但inter-stage communication overhead高（65.1ms vs 6.3ms per iteration）；粗粒度pipeline则相反。核心insight是：最优granularity是workload-dependent和time-dependent的——bursty workload（高CV）需要细粒度pipeline利用distributed buffering吸收burst，stable workload（低CV）需要粗粒度pipeline最小化communication overhead。

从系统架构角度拆解术语：

FlexPipe的granularity adaptation基于扩展的G/G/S排队模型：
- 系统delay T_total = [ρ^S/(S!(1-ρ))] · [(CV_a²+CV_s²)/2] + Σ_{i=1}^{S} λ_i/(μ_i-λ_i)
  - 第一项：Queue Latency（由系统利用率ρ=λ/μ、stage数S、到达/服务CV决定）
  - 第二项：Stage Congestion Delay（各stage的独立排队延迟累加）
- 当CV_a > 3时，增加stage数S使得per-stage service time τ减小，effect 1（congestion alleviation通过细粒度任务分割）dominates，设置S ∝ √CV_a 实现最优延迟。这解释了16-stage pipeline在CV=4时较4-stage实现约3× latency improvement的原因。

Granularity选择由多目标优化驱动：g* = argmax [α·T_k/T_max + (1-α)·L_min/L_k · exp(-|ν_t-ν_k|/σ)]，并通过SLO验证：(T_j − S_j) · Σμ_{jk} / Q_j ≥ r_j，确保所选granularity能在deadline内处理r_j个请求。

Scale granularity decision function（公式11）：m_j = G_max / (1 + β·e^{−γ(cv_j·q̂_j)})，乘积cv_j·q̂_j越大（高CV+长队列），m_j越接近G_max（最细粒度），通过Sigmoid平滑过渡避免决策振荡。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：(1) 候选granularity set G在离线阶段通过operator-level profiling生成，覆盖从粗（如4-stage）到细（如32-stage）的pipeline配置，每个g_k profiling T_k、L_k和optimal CV threshold ν_k；(2) 在线adaptation sensitivity由σ参数控制——σ越大切换越平滑但不及时，σ越小切换越敏感但可能振荡；(3) 多granularity data parallelism允许不同granularity的pipeline同时存在，系统根据各granularity的效率动态分配GPU资源；(4) 论文在OPT-66B上实测：4-stage pipeline在CV=1时latency~0.5s，CV=4时pipeline stall cycle ratio增22×；16-stage在CV=4时latency仅为4-stage的1/3。FlexPipe在CV=1→4完整spectrum上保持稳定性能。

涉及论文标题：
- FlexPipe: Adapting Dynamic LLM Serving Through Inflight Pipeline Refactoring in Fragmented Serverless Clusters
