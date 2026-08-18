## 资源受限调度问题与 GA 调度策略搜索（RCPSP + Genetic Algorithm）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- CPE 虚拟化把逻辑 PE 映射到 K 个物理 CPE，"何时执行哪个 PE"构成资源受限调度问题（resource-constrained scheduling，RCPSP 变体）：给定 DAG G=(V,E)（节点=PE、边=数据依赖）与 K（物理 CPE 数），求 V 的有序时间步划分 S=(S0,...,S_Tmax)，目标最小化总步数（等价最大化利用率 |V|/(K·(Tmax+1))），约束：依赖约束（(u,v)∈E 且 u∈Si, v∈Sj ⇒ j>i）与资源约束（|St|≤K）。该问题 NP-hard，AutoFHE 用遗传算法求近优解并离线固化。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- GA 设计：染色体 = |V| 个基因的优先级序列（基因=节点调度优先级整数，合法染色体须满足数据依赖）；初始种群 = critical-path-first 调度起步 + 随机置换；演化 = order crossover（OX，父序列片段拼接）+ swap mutation（随机交换两基因）+ 依赖校验；选择 = tournament selection；收敛输出最大利用率调度。
- 伪代码（论文 Algorithm 1）：
```
SCHEDULINGSEARCH(G, K):
    P = InitPopulation(G)              # critical-path-first + 随机置换
    for gen in 1..MaxGenerations:
        NewP = []
        for i in 1..|P|/2:
            p1, p2 = Selection(P)      # tournament
            c1, c2 = Crossover(p1,p2)  # order crossover
            Mutate(c1); Mutate(c2); Validate(c1); Validate(c2)
            NewP += {c1, c2}
        P = NewP
    return FindBest(P, G, K)
```
- 效果：vs round-robin 基线 +12.9%–31.6%（表 III Perf.-R 列）；图 9 展示 round-robin 会饿死就绪 PE 使 CPE 空闲，GA 调度利用率更高。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 与文献 RCPSP 的 GA 解法同构（activity-list/优先级编码 + serial/parallel SGS 解码；PSPLIB J30–J120 基准验证）。本论文特点：搜索离线完成、结果固化进硬件调度器（运行时零搜索开销）；与硬件参数 K 耦合（K 变则最优调度变），故嵌入 DSE 内层循环逐候选重搜。

涉及论文标题：
- AutoFHE: An Automatic Hardware Generation Framework for Domain-Specific FHE Accelerators
