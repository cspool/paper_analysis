## Context Remap

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Context Remap（上下文重映射）是一种离线预处理技术，通过改变token到context parallel rank的映射函数$\phi: T \to CR$来改变attention workload在设备间的分布，增强data locality并降低通信开销。其关键特性：仅影响attention模块性能——因为LLM中token间交互仅发生在attention（MLP/LN中每token计算独立），改变映射不影响非attention模块。UltraAttn中用于增强node/device-level tiling前的locality。例如strided attention默认sequential mapping $\phi(t_i)=\lfloor i \cdot CP/S\rfloor$在node-level locality差；改用$\phi(t_i)=\lfloor i \cdot 16/S \rfloor \mod 4$后相邻strides被分配至不同node，locality显著改善。Zigzag ring attention的interleaved mapping (Eq.7)也是context remap的一种——用于causal attention load balance。形式化定义：$T=\{t_0,...,t_{S-1}\}$（token set），$CR=\{0,...,CP-1\}$（rank set），$\phi: T \to CR$（任意feasible映射）。当前手动离线设计，自动化面临remapping与hierarchical tiling的联合搜索空间爆炸挑战。

涉及论文标题：
- UltraAttn: Efficiently Parallelizing Attention through Hierarchical Context-Tiling
