## BALD 通信调度算法（Balanced Allocation with Load and Distance awareness）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BALD（Balanced Allocation with Load and Distance awareness）是 BusyBarn 提出的通信调度算法，为 wafer-scale 层次化 2D mesh 上的 point-to-point 与多组 multicast（区域受限 collective 与广播）做链路分配（link allocation），同时优化吞吐与延迟并内建故障容错。算法三步：(1) Path Profiling——对拓扑 T=(N,E) 用 Dijkstra 全源最短路径（Algorithm 1），输出最短距离映射 S 与唯一路径映射 U（记录等长多路径并把 pair 标记非唯一），天然处理非对称节点度与异构边权重；(2) Path Scheduling——对每个通信任务 C=P(s,D)（源 s、目标集 D），按优先级 score = α×branch_cost + β×link_load + γ×neighbor_distance 迭代选择分支与邻居分配链路（Algorithm 2）：branch_cost 为当前分支最早可调度时间、link_load 为链路当前占用（mesh 上 collective 的主要瓶颈）、neighbor_distance 为到最近目的地的距离（预计算）；(3) Heuristic Backtracking——维护 tabu forbidden/candidate 列表，找出最大负载链路 l*、对其上任务按概率 ρ 重分配（成功回退进 tabu candidates、失败进 tabu forbiddens），最多 I 次迭代（Algorithm 3），消除随机选择造成的链路争用。算法基于 BFS/DFS 拓扑无关，只要存活图连通即可容忍任意节点/链路故障。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Path Scheduling 的优先级计算（Algorithm 2 核心）：
```
while P not empty:
    for each task p in P:                       # 每个 multicast 任务
        for each branch in path[p]:             # 当前分支（源/已分配节点）
            for each neighbor in available_neighbors:
                priority = alpha * branch_cost + beta * link_load \
                         + gamma * neighbor_distance   # 选最小优先级
        path[p] += (best_branch, best_candidate)   # 分配链路
        update branch_cost, link_load              # 更新网络状态
        if candidate in destinations: remove from p
```
Heuristic Backtracking（Algorithm 3）伪代码：
```
for iter in 1..I:
    l* = link with maximum load; overloadedTasks = tasks using l*
    u ~ Uniform(0,1)
    if u < rho: pick t from overloadedTasks not in tabuForbiddens
    else:       pick t from tabuCandidates
    re-run Path Scheduling on backtrackedTasks
    if total load of overloaded links decreased:
        tabuCandidates ∪= overloadedTasks      # 成功回退→继续优化
    else:
        tabuForbiddens ∪= overloadedTasks      # 失败→禁止重访
```
执行例子（Fig.7 4×4 mesh 两个并发 multicast：任务1 8→{7,14}、任务2 9→{3,11}）：XY 路由共享 (9,10)(10,11)(11,7) 造成争用热点；BALD 先 profiling 得 8→7 与 8→14 等最短路径与备选等长路径，再按优先级给两任务分配不同链路（如任务2 改走 9→5→6→7→3 之类非冲突路径），β 大时优先均衡链路负载，最后回退重排过载链路任务。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：在事件驱动模拟器中实现三步算法作为通信调度器（替代 XY/XY-YX-FT），输出路径作为路由器 LUT 下发（主机每推理任务加载、执行中更新，类似 TPUv4）。使用与超参：集体通信实验首轮 α=100、β=1、γ=100（偏好最短路径），随后 α=1、β=100、γ=1（均衡链路负载）；ρ 为 tabu 概率、I 为最大迭代。评估结果：All-Gather 峰值有效带宽 533.3 GB/s（与 TACOS 持平，超 MultiTree 1.25×、XY 1.5×、Hierarchical Ring 近 2×）；All-to-All 峰值 213.3 GB/s（XY 的 2.4×，链路故障下 1.84–2.25×）；6×6 mesh 多故障下 All-Gather 1–1.94×、All-to-All 1.56–2.55×（vs XY）。开源：https://github.com/redbird-arch/isca2026-busybarn-artifact.git。

涉及论文标题：
- Mapping and Communication Optimizations with Fault Tolerance for Wafer-Scale LLM Inference
