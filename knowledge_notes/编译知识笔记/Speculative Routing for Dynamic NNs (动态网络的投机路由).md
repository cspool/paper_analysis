## Speculative Routing for Dynamic NNs (动态网络的投机路由)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Speculative Routing 是 Brainstorm 提出的动态优化，借鉴 CPU 分支预测的思想：基于 Router profile 的统计分布，预测最可能被激活的 branch 并提前启动其算子执行，跳过 router_fn 的 CPU 计算和 CPU-GPU 同步开销；同时保留一个并行验证路径，当预测错误时 unroll 并重新执行正确 branch。适用于 Router 开销占据推理延迟显著比例的动态网络（如 MSDNet 路由占 65%，DynamicRouting 的 186 个 router 占 44%）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
Brainstorm 编译器的 torch.fx Pass 实现该变换：
1. **Profile 分析**：从 Router 的 JIT profile 计算每个 Router 的 branch 选择概率。若某 branch 的激活概率 > 阈值，标记为"可预测"
2. **图变换**：对于可预测的 Router，生成两路执行路径：
   - 预测路径：预测的 branch 算子前置（在 Router 之前 speculative launch），路由逻辑（router_fn）移到并行验证线程
   - 回退路径：check 算子检查 router_fn 实际输出是否与预测一致；若不一致，unroll（丢弃预测 branch 的 spec 输出，重新启动正确 branch）
3. **运行时**：GPU 同时执行 speculative branch compute 和 router_fn（CPU 侧），若命中则 router 延迟被 hiding；若 miss 则微基准显示 overhead 可忽略（与默认执行相当）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Brainstorm 自动检测并应用。在 DynamicRouting 的 186 个 Router 中，分析发现选择概率 > 90% 的 Router 占比很大，简单策略（选历史最频繁 branch）即达 90%~95% 准确率。预测错误时的 unroll overhead 因仅涉及 kernel 丢弃（已计算但未使用的 spec 输出）而 negligible（per micro-benchmark）。

涉及论文标题：
- Optimizing Dynamic Neural Networks with Brainstorm
