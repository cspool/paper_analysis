## Graph-Based Search for Kernel Optimization

术语是什么？
Graph-Based Search for Kernel Optimization将kernel优化问题建模为搜索图G_t=(V_t, E_t)，节点v∈V_t为kernel implementation artifact，有向边(v_i, v_j)为从kernel v_i到v_j的transformation。根节点v_0为baseline implementation或initial specification。搜索通过select→apply→score三操作迭代推进，支持greedy/MCTS/evolutionary多种策略。与静态编译优化不同，每个节点incorporate祖先的execution feedback（profiling data、compilation status、correctness results）驱动progressive refinement。

从编译框架角度拆解术语：
搜索trajectory展示两阶段行为：
1. **Draft Phase** (steps 0-10): independent sampling without feedback——快速生成初始候选
2. **Tree Expansion Phase** (steps 10+): 每个新节点incorporate executor feedback——performance systematically improves

论文conv1d 300-step搜索可视化（Figure 12）：fitness从~2000收敛到6889。Green nodes表示successful generation，red nodes表示compilation/correctness failures。关键insight：搜索自动发现non-obvious optimization combinations——在conv1d case中包括kernel fusion (5 kernel→2 kernel)、expanded autotuning (20+ configs)、3D grid launch、double-buffered execution和differentiated cache modifiers——这些组合即使对expert kernel developer也需要数周试验。

术语一般如何实现？如何使用？
KernelEvolve中实现为state machine with tree search engine。Selection policy可切换：greedy用于rapid initial solutions，MCTS用于balanced exploration-exploitation，evolutionary用于population-based optimization。Persistent storage (metadata store + object store)提供fault tolerance——每个node atomic insertion确保crash recovery，避免multi-hour runs因硬件故障丢弃所有进展。Cross-session knowledge reuse使新搜索从historically optimized similar operators开始，大幅减少exploration cost。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

---
