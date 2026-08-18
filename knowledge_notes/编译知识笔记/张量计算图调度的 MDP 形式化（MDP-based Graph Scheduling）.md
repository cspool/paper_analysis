## 张量计算图调度的 MDP 形式化（MDP-based Graph Scheduling）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
把"张量计算图的调度"建模为马尔可夫决策过程（MDP）P=(S,A,T,M,R)：状态 S(G,n) = 计算图 G + 调度节点 n（n 标识图中可施加重写动作的具体位置，按逆拓扑序规范化处理）；动作 A(r,p) = 调度规则 r + 参数配置 p；转移 T 按"状态-动作 pattern 与动作条件"把图改写成新状态；终止态由性能成本模型 M 估奖励 R（真机执行时间倒数）。目标是最优策略 π* = argmax R(S_π,x)，即最大化测得性能。QiMeng-Tensify（ISCA'26）用该形式化把"调度规则应用策略"从传统编译器的单条手写规则序列（hard-coded policy）变成可搜索的 policy space π，从而支持任意顺序组合的图变换（unconstrained graph transformation），把变换空间量级扩大几个数量级（如 GatedMLP 下 >1e10 空间可被系统化覆盖）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
在编译框架中的运转流程（GatedMLP 例子）：初始 S=(G, n=逆拓扑序第一个节点)；循环直到终止：在状态 S 上选择动作 A（如 A2 AutoInline）→ 检查条件（n 是否 inlinable）→ 满足则 S(G,n) → S(G',n')（inline 后跳到下一节点），不满足则 S(G,n) → S(G,n')（换节点不动图）；A1/A3/A4/A5/A6 满足条件时 S(G,n) → S(G',n)（变换当前节点不换位置）。当图到达无法再施加动作的终止态，查询成本模型估奖励 R，真机测得的性能作为 ground truth 反馈用于选最优图重写序列。该形式化把"每个图一个固定规则序列"变成"为每个状态-动作对学习选择策略"，是 LLM-guided MCTS 的求解目标。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：动作表（Table II）7 条规则及其参数——A1 MultiLevelTiling(tiling factors)、A2 AutoInline、A3 ParallelizeVectorizeUnroll(loop, unroll length)、A4 CrossThreadReduction(split factors)、A5 ComputeAtLocation(compute locations)、A6 AutoBind、A7 InlineConstantScalar；转移表（Table III）按 pattern/condition 决定 S→S'。使用方式：搜索引擎每迭代从根出发，沿 MCTS 树选择动作序列 π，施加变换生成 program sketch，细粒度参数规格填参数后真机测量得 R，R 回传更新 Q(s,a)；500 次迭代（或 early stopping K=200）后返回最优程序 p*。与传统 autoscheduler（Ansor/TensorIR 的进化搜索、Meta-Scheduler 的概率程序）相比，QiMeng-Tensify 首次把 LLM 先验注入该 MDP 求解。

涉及论文标题：
- QiMeng-Tensify Scaling up Tensor Computation Optimization via Architecture-Aware LLM-Guided MCTS
