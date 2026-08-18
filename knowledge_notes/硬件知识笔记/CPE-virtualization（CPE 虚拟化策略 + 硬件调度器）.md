## CPE-virtualization（CPE 虚拟化策略 + 硬件调度器）

术语解释
- 用 K 个物理 CPE 池时分共享全部逻辑 PE，避免 1:1 实例化 CPE 的面积爆炸；调度策略离线搜索后固化为硬件调度器。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- CPE-virtualization 是 AutoFHE 解决"每个加密 PE 实例化一个 CPE 导致面积不可行"（Challenge 2）的架构策略。生成架构含两部件：(1) CPE 池——多类型物理 CPE；arithmetic+bootstrapping+key-switching 紧耦合序列视为统一 CPE lane，K 为各类型实例数（可为向量）；(2) 硬件调度器——把离线搜出的调度策略固化为控制器，在线确定性编排（PE 完成即释放 CPE 给下一个数据就绪 PE）。逻辑 PE 的时空映射（哪些 PE 何时上哪个物理 CPE）由 GA 离线求优后硬连线，运行时无搜索。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程：FIRRTL 图抽象为 DAG（节点=识别的 PE）→ GA 离线求调度（依赖约束 j>i、资源约束 |St|≤K、最大化利用率 |V|/(K·(Tmax+1))）→ 生成硬件调度器（固化 Sched）→ 运行时：调度器按时间步选出就绪 PE 映射到空闲 CPE 执行（算术→自举→密钥切换 lane），PE 完成释放 CPE，重复直至 DAG 全部执行完。
- 效果：可扩展性实验——Strix 的 1:1 PE-CPE 绑定在 48 PE 时面积超 400 mm² 预算出界；AutoFHE 同预算仍生成合法加速器且性能更高（DSE 自动搜出最优物理 CPE 数）。GA 调度 vs round-robin +12.9%–31.6%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 与时分复用/多租户共享资源的一般思想同构，但作用域是 PE 级时间片共享而非进程级；实现为生成时硬连线（无运行时调度开销、无 OS 参与）。使用前提：CPE 资源比普通 PE 高数量级（bootstrapping/key-switching 昂贵）时共享才有收益；K 与最优调度耦合，须在 DSE 中联合优化（内层逐候选重搜调度）。

涉及论文标题：
- AutoFHE: An Automatic Hardware Generation Framework for Domain-Specific FHE Accelerators
