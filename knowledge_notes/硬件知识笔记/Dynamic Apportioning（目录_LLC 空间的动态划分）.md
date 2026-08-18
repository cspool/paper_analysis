## Dynamic Apportioning（目录/LLC 空间的动态划分）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dynamic Apportioning 是 Dorado 的存储管理策略：目录分片不再为每种条目/指针类型（LLptr/LRptr/RLptr × ED/TD，甚至本地/远端数据 LLC 分区）各设独立结构，而是 TD 与 ED 都用单一目录项结构，所有类型的数据行与共享者指针动态竞争同一存储空间，随负载共享模式自适应。实现只需 1 个类型位区分指针存的是核 ID 还是簇 ID（32 簇×32 核下 5b ID + 1b 类型 = 6b/指针）。动机（论文 Fig.4）：不同负载本地/远端 home 行比例差异大（Redis 以远端为主、FaaSFunc 以本地为主），固定分区必失配。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转例子（Fig.6）：本地核访问本地行→本地分片按目录分配算法在 TD 或 ED 建 LLptr 条目（TD 条目连带数据进 LLC）；该行被远端核访问→无需新结构，直接在原目录项空闲指针位放 LRptr；核访问远端行→本地分片建 Temporary home 条目用 RLptr（可落 TD 或 ED）。效果：以总条目数/总指针数定尺寸即可（按本地线程工作集估算），本地共享多的负载自然填满 LLptr、远端多的填满 LRptr。实验：TLH-Dir4B（固定 60% 远端条目 4RLptr + 40% 本地条目 2LLptr+2LRptr）speedup 1.17×，加 Dynamic Apportioning 的 TLH-Dir3B-Dynamicity 达 1.24×。代价：每指针 6b（多 1 类型位）→同面积条目从 4 指针降为 3 指针。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现（论文）：SST 中 TD/ED 统一条目 + 指针类型位。使用要点：适用于"多种条目/指针类型共生"的目录（如含 Global/Temporary home 的协议）；静态划分需按平均负载调 60/40 之类参数且仍失配，动态划分免调参。类似思想可对比：Fang et al. 的静态分割（有限指针条目/全位向量条目按 way 迁移，仍属固定结构）、Pool directory 的动态全局池（但需全局仲裁与连续分配）。

涉及论文标题：
- Dorado: Clustered Hardware Cache Coherence for 1,000+ Cores
