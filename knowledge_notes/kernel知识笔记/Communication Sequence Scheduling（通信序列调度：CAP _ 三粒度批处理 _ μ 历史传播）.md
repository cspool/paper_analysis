## Communication Sequence Scheduling（通信序列调度：CAP / 三粒度批处理 / μ 历史传播）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
负载映射固定后，进一步静态决定各 core 的通信序列（传输执行顺序）以缓解时变 contention。三个机制：(a) CAP（Contention Analysis Phases）——把通信时间线切成 contention 分布准稳定的区间，每 CAP 分析并发任务（含上一 CAP 的在途任务）；(b) 三粒度批处理——早期 batch 用细粒度（≤b1）、后期用中/粗粒度（≤b2/≤b3）控制复杂度（batch = 以第 i 个目的地为目标的全部传输）；(c) 历史感知传播因子 μ∈(0,1)——未解决的高 contention 任务跨 CAP 保留，平衡质量与开销。目标：min Φ = min{φ^0, φ^1, ..., φ^{K-1}}（Eqn.7，φ^k = 该 CAP 期望最大链路/目的端 contention），以 bin 目标 Φ^target 自适应升级。优化器：多染色体 NSGA-III（Geatpy，种群 120、≤100 代、10 代停滞早停）。负载图与依赖固定 → 通信序列可静态预排，无运行时开销。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
伪代码：
```
batches = group_transfers_by_destination(CSS_0)   # 第 i 批 = 所有"第 i 目的地"传输
carry = {}                                        # μ 历史传播：未解决高 contention 任务
for k in 0..K-1:                                  # K 个 CAP（三粒度：b1/b2/b3 截断）
    tasks_k = batches[k] ∪ carry
    phi_k = max_link_dest_contention(concurrent(tasks_k))
    NSGA-III 调序 tasks_k 使 phi_k 逼近 Phi_target
    carry = top_contended(tasks_k, mu)            # μ∈(0,1) 保留进下一 CAP
```
时间 O(|D|·h)，128×136 下 ~28.19 min；空间 O(n_link + n_dest)。效果：Ours-WM（仅映射）方差降幅 >45.83%，Ours-ALL*（+调度）达 79.21%——细粒度通信调序能更精确地把芯片推向目标 contention 水平；Ours-ALL* 端到端 speedup 至 2.39×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
映射与调序解耦的两级优化（先空间后时间），每级都以 bin 目标为中心自适应升级；CAP 把"全局时序优化"降为"分区间局部调序"，三粒度批处理 + μ 传播使开销对 wafer 规模可控。Web 证据：ICAPS 2020 的 contention-aware mapping & scheduling 同样"顺序化可能争用的数据传输"；Kalray MPPA 用离线全局通信调度（软件流水 + 传输抢占）。区别：这些方法最小化 makespan/能量，ConBin 最小化"与 bin 目标的距离"。

涉及论文标题：
- ConBin: A Performance-Convergence Framework for Wafer-Scale Chip Binning
