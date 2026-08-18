## Bin-Aware Workload Mapping（bin 感知负载映射，MLCC/LCC）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
把负载依赖图 WG=(W,D)（节点 = 计算任务如 GEMM/attention 层，边 d_{i→j} = 带权重 p_{i→j} 的数据依赖/通信任务）单射映射到修复后拓扑 TG=(V,E)（Γ(w_i)≠Γ(w_j)，Eqn.5）。目标不是最小化通信距离（hop），而是最小化期望最大链路 contention：MLCC_exp = max_e LCC_exp(e)，LCC_exp(e) = Σ_{d∈D, e∈R(v_i,v_j)} p_{i→j}（Eqn.6，沿 turn-prohibition 路由路径累加传输频率权重）。与 Si-Kintsugi（hop 数代价函数）的本质区别：故障下 contention 是主要延迟源（hop 延迟 5–6 跳后饱和、contention 近线性增长）。优化器：SEGA（Strengthen Elitist GA，Geatpy，种群 100、≤100 代、10 代停滞早停），适应度围绕 bin 目标 MLCC_exp^target（来自 pre-binning），达标后自适应升级到高一 bin（≤2 次，30 代内未达标则采纳上一有效目标最优解）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
伪代码（单个体评估）：
```
LCC = zeros(|E|)                       # 每条修复后拓扑链路的 contention 计数
for d_ij in D:                         # 负载图每条通信边
    path = turn_prohibition_route(TG, Gamma(w_i), Gamma(w_j))
    for e in path: LCC[e] += p_ij      # 沿路径累加传输频率权重
MLCC_exp = max(LCC)                    # 期望最大链路 contention
fitness = -|MLCC_exp - MLCC_exp_target|  # 逼近 bin 目标而非全局最小
# SEGA 进化：选择/交叉/变异 + 精英保留；达标 → target 升级到高一 bin
```
复杂度 O(|D|·h)（h 为平均路径长度），128×136 下 ~18.36 min；空间 O(|E|)。效果：CB*+Ours-SW* 方差降幅 ≥9.16%，Ours-WM（ConBin 硬件 + 仅映射）>45.83%，speedup 至 1.85×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
以 bin 目标（而非全局最优）为适应度中心——芯片只需"够到"本 bin 目标，达标后允许向上一档升级：弱芯片不被强求（避免无效优化）、强芯片不浪费潜力（可晋升 premium bin），这是性能收敛的关键机制。与 CUPOKer（核数贪心放置）和 Si-Kintsugi（hop 代价 + 通信距离建模）对比，bin 感知映射把优化方向从"最大化单芯片加速"改为"收敛到 bin 目标"。Web 证据：NoC 应用映射的 GA 路线成熟（能量/contention 多目标，Hu&Marculescu 分支定界、eMesh GA、GAMR），ConBin 的增量是 contention 代价 + bin 目标自适应的组合。

涉及论文标题：
- ConBin: A Performance-Convergence Framework for Wafer-Scale Chip Binning
