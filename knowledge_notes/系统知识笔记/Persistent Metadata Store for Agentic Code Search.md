## Persistent Metadata Store for Agentic Code Search

术语是什么？
Persistent Metadata Store是KernelEvolve中用于持久化搜索图状态的存储层，由metadata store（关系数据库）和object store（kernel文件存储）组成的两层架构。Metadata store记录每个搜索节点的：unique identifier (id)、parent reference (pid，编码tree structure)、fitness score (score)、correctness flag (is_buggy)和path reference (path_ref，指向object store中的kernel文件)。这种separation of metadata and content设计支持efficient metadata queries（无需加载大型源码文件或profiling traces）和distributed concurrent exploration。

从系统架构角度拆解术语：
关系型metadata store提供四个关键能力：
1. **Distributed Concurrent Exploration**: 多个agents同时扩展不同节点，数据库提供transaction isolation和consistency guarantees。当KernelEvolve scale到数十甚至数百个concurrent agents探索数千个优化步骤时，in-memory graph representation变得不可行。
2. **Complex Contextual Queries**: 通过SQL recursive CTE实现graph traversal——分析sibling node outcomes、检索high-performing ancestor strategies、识别global best solutions。
3. **Cross-Session Knowledge Reuse**: 持久化使新搜索可以从历史优化的类似operators开始（匹配operator type、input shapes、hardware platform）。例如AMD MI350上的新GEMM变体：metadata查询识别15个历史GEMM kernel，3个达到>1.5× speedup的kernel被检索作为搜索起点，避免重新发现基础优化模式。
4. **Fault Tolerance and Checkpointing**: Crash或中断后从last successful iteration恢复——每个node insertion原子性地持久化exploration state，使multi-hour optimization runs从brittle转变为resilient。

术语一般如何实现？如何使用？
SQL database + filesystem object store。生产部署维护跨月跨数百operator types和多平台的搜索历史，创建持续增长的kernel expertise corpora。Metadata queries在milliseconds内执行（即使有数百万nodes），object store access selective on demand。当high-quality historical solutions存在时，KernelEvolve从这些实现初始化搜索（而非from scratch），减少time-to-solution、inference cost和environmental impact。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

---
