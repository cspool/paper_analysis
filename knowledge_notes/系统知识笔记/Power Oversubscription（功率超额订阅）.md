## Power Oversubscription（功率超额订阅）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Power oversubscription 是数据中心的功率管理策略：数据中心按预先定义的总功率预算建设，但利用"节点通常不满载"的事实，在相同功率预算下部署更多节点（超额订阅），已知工作负载下通过功率上限（power capping）实现而不过多损失性能。已被生产环境广泛采用（Google、Meta、Azure 等），典型手段是功率上限限制峰值功率以抑制功率波动。
- Lit Silicon 论文（ISCA'26）把它作为解决方案空间的基础：AI 推理的超订机会充足，LLM 训练几乎占满供电功率、机会较少但存在大的功率摆动，功率上限可有效削减峰值。论文的三个用例全部源自超订场景：GPU-Red（无节点上限时省电）、GPU-Realloc（节点上限下重分配）、CPU-Slosh（节点上限 + CPU 功率转移）。估算成本节省：Google PUE 1.09（行业平均 1.56）、GPU 功率约占供电 50%、训练/推理平均用 TDP 的 75%、电价 $0.14/kWh（2025-08）、4% 功率节省 ≈ 单客户 6GW 部署年省约 $70M。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程：数据中心设定节点级功率上限 P_n（< Σ 各设备 TDP）→ 节点功率管理在设备间分配功率（Lit Silicon Algorithm 3 用 gpu_delta = ceil((node_power - P_n)/G) 均匀回退、再按 TDP 修正）→ 设备 DVFS 在功率上限内调速 → 超订出的额外节点共享供电容量。例子：8×MI300X 节点 TDP 总和 5600W（8×700W），若 P_n=5480W（低 120W），则每 GPU 基准低 15W；straggler +15W 后须全部 GPU 均匀 -15W/8 才能回到 P_n——超订的功率余量（headroom）就是 GPU-Realloc 可重分配的空间。
- 关键收益：不增加数据中心总功耗的情况下提升性能（GPU-Realloc/CPU-Slosh），或在同等性能下降低功耗（GPU-Red）；CPU-Slosh 进一步利用"训练时 CPU 闲置"的功率余量（约 86.5% 核心功率）转移给 GPU。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：数据中心调度/功率管理栈（如 Google 的 power provisioning [Fan et al. ISCA'07]、Aztec 类预测性超订 [Kumbhare et al. ATC'21]）在机架/PDU 层面限制功率并处理峰值；节点内由固件/驱动执行功率上限（amd-smi/nvidia-smi）。使用：运维按工作负载与超订率设定节点功率上限；Lit Silicon 论文指出节点级功率上限由数据中心决定（取决于超订程度与是否支持 CPU 核心 power-gating），其算法以 P_n 为唯一用例间差异参数。注意：训练几乎满负载时超订空间有限，功率波动（power swing）是主要风险点。
- PowerGrad 补充视角（ISCA'26，推理场景的功率受限环境）：PowerGrad 面向集群级功率受限/严重功率受限环境——"每个节点需求功率都超过其最大分配"（demand > supply，如可再生能源波动或 Demand-Response 动作导致的临时限电）。在此环境下功率上限分配的难点升级：全体节点都功率饥饿时，按"功率消耗模式"（如 DPS）或"是否用满分配"（如 SLURM）都无法区分节点优先级，分配退化为均分（≈Fair），总预算压到 Σ节点上限后没有 slack 可回收。PowerGrad 的解法是改用性能梯度 ∂BIPS/∂P 区分功率敏感度而非功率观测本身，在超订功率包络内把功率从低梯度节点转给高梯度节点（评估对集群功率预算 sweep：Legacy 每节点 55–75W、总 880W 起；Accelerated 每节点 115W 起、总 1840W）。严重受限场景收益最大：55W/节点时平均/P95 延迟降 23.6%/27.4%。与 GPU-Red/GPU-Realloc/CPU-Slosh（训练、节点内设备间）不同，PowerGrad 是跨节点+节点内的分层分配，执行器为 Intel RAPL（功率上限而非直接锁频）。

涉及论文标题：
- Lit Silicon: A Case Where Thermal Imbalance Couples Concurrent Execution in Multiple GPUs
- PowerGrad: Hierarchical Power Management for Power-Limited ML Inference Clusters
