## Cell-level Dataflow Analysis (Cell 级别数据流分析)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Cell-level Dataflow Analysis 是 Brainstorm 编译框架的核心分析技术，分为静态分析和动态分析两部分。静态分析在 AOT 阶段通过符号执行推导 Cell 在静态算子中的跨层传播关系；动态分析在 JIT 阶段收集 Router 的路由决策统计分布。两者结合使得编译器能理解 sub-tensor 级别的完整数据流——从每个 Cell 的创建、经各算子的变换与混合、到 Router 的分发决策，最终形成 Cell 在完整模型执行中的生命周期轨迹。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
**静态分析（AOT）流程**：
```
1. 扫描标注：遍历模型代码，找到所有 brt.annotate_cell 调用
2. 符号初始化：为每个被标注 tensor 的每个 Cell 分配唯符号（如 X0, X1, ...）
3. 算子遍历：沿数据流图逐算子进行符号执行
   - MatMul(cell_tensor, weight): 输出每个 Cell 保持对应输入符号 → 类型 1
   - SelfAttention(q_cells, k_cells): Q·K^T 的矩阵乘法使输出每个 Cell 的符号为输入所有符号的集合 → 类型 3
4. 约束提取：Cross-Cell mixing 点标记为"聚合约束"——所有参与 Cell 必须在同一 GPU
```
**动态分析（JIT）流程**：
```
1. Router 拦截：每次 Router.forward() 被调用，Routes tensor 写入环形 buffer
2. 异步落盘：独立线程定期将 buffer 内容追加到 profile 文件
3. 统计聚合：offline 分析 profile 文件，计算每 Router 的 branch 激活频率、Cell 负载分布、跨层 co-activation 矩阵
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Brainstorm 用 ~3,000 LOC Python 实现静态符号执行引擎，复用 torch.fx 的图追踪能力。符号执行基于算子的 tensor expression 推导（类似 TVM 的 tensor expression），但以 Cell 符号而非数值为操作数。动态 profiling 用 C++ 实现环形 buffer + 异步 I/O 线程，开销 <1.0%。Profile 文件仅在优化触发时被读取（通常在部署前完成），不影响在线推理性能。

涉及论文标题：
- Optimizing Dynamic Neural Networks with Brainstorm
