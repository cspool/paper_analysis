## Profile-Guided Model Placement for Dynamic NNs (基于运行时Profile的动态网络模型放置优化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Profile-Guided Model Placement 是 Brainstorm 提出的多 GPU 场景下的动态优化，通过分析 Router JIT profile 中跨层 Cell 的 co-activation 统计模式，重新排列 MoE expert 在不同 GPU 上的物理放置，以最小化 inter-GPU 通信量。其核心洞察来源于脑科学的"功能区组织"类比——大脑皮层中同一功能的神经元聚集在相邻区域。在动态 MoE 网络中，跨层的 expert 之间存在强统计相关性（如 TaskMoE 中同 task 的 expert 在相邻层中大概率同时被激活，图 2c 显示相关性高达 87%），将相关 expert co-locate 在同一 GPU 上可大幅减少 all-to-all 通信。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
全流程（以 TaskMoE, 16 experts/layer, 8 GPUs 为例）：
1. **静态分析约束**：Ahead-of-time 符号执行发现 Self-Attention 层是 Cross-Cell mixing 类型——每个 token 的输出依赖同一 sentence 的所有 token。这产生硬约束：同一 sentence 的所有 token 必须在 Self-Attention 前聚合到同一 GPU。
2. **动态 Profile 收集**：Router 的 JIT Profiler 记录每层每个 expert 被哪些 token 激活，构建跨层 expert co-activation 矩阵 M[i][j] = Pr(Expert-i at Layer-L 和 Expert-j at Layer-L+1 被相同 token 激活)
3. **Placement 求解**：以 M 为输入，在静态约束下（Self-Attention 聚合约束）求解最小化 Σ M[i][j] × distance(gpu_of_expert_i, gpu_of_expert_j) 的分配方案。使用启发式贪心算法（论文未详细说明具体算法）。
4. **部署执行**：Router 感知 placement 映射，将 router_fn 输出的原始 branch ID 翻译为目标 GPU ID。Cell 通过 sparse point-to-point 通信传输，仅传实际跨 GPU 的 Cell，无 all-to-all padding。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Brainstorm 离线执行 placement 优化（非在线调整），在部署前分析 profile 并生成 placement 方案。Paper 显示可减少 42~87% inter-GPU 通信。TaskMoE 在 8 GPU 上加速 1.34×（on top of sparse communication）。SwinV2-MoE 的收益较小（marginal improvement），因为其 communication 时间占比仅 35%。但单层实验显示 best vs worst placement 差距达 1.26×，预示在大规模 MoE 模型中潜力更大。

涉及论文标题：
- Optimizing Dynamic Neural Networks with Brainstorm
