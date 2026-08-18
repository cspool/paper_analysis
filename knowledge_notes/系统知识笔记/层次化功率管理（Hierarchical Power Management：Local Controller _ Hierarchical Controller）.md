## 层次化功率管理（Hierarchical Power Management：Local Controller / Hierarchical Controller）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 层次化功率管理是把集群功率分配组织成多层控制器树的分层控制结构：叶子层的 Local Controller 在节点内处理器之间重分配功率预算，上层 Hierarchical Controller 在节点/子集群之间重分配，层间异步执行、各层用同一梯度驱动算法（Algorithm 1），可递归扩展任意层数。它是 PowerGrad（ISCA'26）的框架组织方式，针对"集中式功率控制在大集群、快变 ML 工作负载下太慢"的痛点：本地层用 100ms 细粒度快速调节（RAPL 测量 50ms 以下不可靠），高层因网络 socket 通信开销运行更慢（子集群 1s、集群 4s，由最坏往返网络延迟 100ms×2×层级规模推导）。各层共享同一 ALLOCATE_POWER 算法（对子节点输入 G、f、P、PL），故层次可递归添加（一组 Hierarchical Controller 可向父级上报，父级异步重分配其预算）。
- 与 Power Sloshing（服务器内 CPU-GPU 模块级）的区别：PowerGrad 是集群级（跨节点+节点内）的分层功率预算分配，用性能梯度而非利用率阈值做分配依据；两者都属"功率预算再分配"家族，Power Sloshing 条目已把 PowerGrad 列为相关系统。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 三种拓扑（论文 Fig.6 与 Table I）：(a) 默认两级 PowerGrad——16 节点各配 Local Controller（每 100ms）+ 1 个集群级 Controller（每 4s），仅 Legacy（双 CPU/节点）可用；(b) PG-central——单集群控制器直管所有处理器（Legacy 32 个 / Accelerated 16 个），无本地层；(c) PG-multi——四子集群×四节点三层（子集群 1s、集群 4s），Legacy 下三层、Accelerated 下两层（节点无本地层）。
- 运转流程（Algorithm 1，控制器视角）：
```
function ALLOCATE_POWER(G, f, P, PL, lr, α)
    communicate(parent, G, f, P)                    # 向父级上报状态
    PL_node ← parent 异步下发的本节点功率上限
    for i ∈ children:                               # 梯度驱动重分配
        PL'[i] ← PL[i] + lr×G[i] − α(PL[i]−P[i])    # 高梯度加、低梯度减；回收未用预算
    for i ∈ children: PL'[i] ← PL'[i] − (ΣPL' − PL_node)/N   # 均分校正到节点上限
    for i ∈ children: if PL'[i] < PL[i]+incmin and f[i] < fmin:
        PL'[i] ← PL[i] + incmin; 再调整其他子节点      # fmin 防饿死（cpuinfo 最低频，incmin=1W）
    return PL'
```
- 例子：Legacy 16 节点集群，集群总预算 880W。集群控制器每 4s 按各节点聚合梯度重分配节点预算（把功率从跑低梯度应用的节点转到跑高梯度应用的节点）；节点 Local Controller 每 100ms 再在双处理器间分配（把功率从内存密集的 Llama-low 转到计算密集 prefill 的 Llama-high）。层级收益：Accelerated 单 CPU 节点无本地层可用，故只能靠 4s 粒度 → 收益（9.0%/9.9%）低于 Legacy（22.9%/23.0%）；PG-multi 三层结构在两类平台都是最优变体。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：全用户级软件——节点内 Gradient Estimator 与 Local Controller 为 Java 线程（共享内存同步，100ms 周期），Hierarchical Controller 为 Python 进程（网络 socket 与子控制器通信，1–4s 周期）；功率上限经 Intel RAPL 硬件接口强制到处理器（控制 V-f 状态）。超参数 lr=2.0、α=0.3（按目标系统 benchmark 调）。使用场景：功率受限（power-limited）或严重功率受限（severely power-limited，每节点需求>分配）的 ML 推理集群，特别是不可预先 profile 的动态混合模型负载；也可用于 Demand-Response 动作或可再生能源波动导致的临时限电。层次设计价值：低层快（100ms）处理快变负载、高层慢（秒级）跨节点协调，通信开销只在高层次付出；缺点是单 CPU 节点无法利用本地细粒度层。论文未开源，实现细节以论文 §III 与 Algorithm 1 为准。

涉及论文标题：
- PowerGrad: Hierarchical Power Management for Power-Limited ML Inference Clusters
