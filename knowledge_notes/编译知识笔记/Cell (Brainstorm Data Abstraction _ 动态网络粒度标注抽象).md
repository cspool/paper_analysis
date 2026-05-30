## Cell (Brainstorm Data Abstraction / 动态网络粒度标注抽象)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Cell 是 Brainstorm 框架提出的核心数据抽象，让模型开发者标注 tensor 中 dynamism 发生的数据粒度。传统 tensor-centric 编程模型以整个 tensor 为最小调度单位，但动态网络中 dynamism（如 token/patch/pixel 级别路由）发生在 sub-tensor 粒度上，编译器无法追踪。Cell 允许开发者通过 `brt.annotate_cell(tensor, dims, shape)` 指定 tensor 中哪些维度（dims）和什么形状（shape）构成一个 Cell——即路由的基本单元。例如，NLP 任务的 token 定义为 dims=(0), shape=(1,768)，表示第一维的每个 768 维向量是一个 Cell；CV 任务的 32×32 patch 定义为 dims=(0,1), shape=(32,32)，表示前两维的每个 32×32 子矩阵是一个 Cell。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
Cell 是 Brainstorm 编译器进行 Cell-level 数据流分析的基础。在 AOT 编译阶段，编译器为每个标注了 Cell 的 tensor 创建符号版本（每个 Cell 分配不同符号），通过对算子张量表达式进行符号执行，推导 Cell 在静态算子中的传播关系：
- **类型 1（保持顺序）**：如 MatMul(tensor_of_cells, weight)，每个 Cell 与同一 weight 矩阵运算，输出保持 Cell 边界和顺序
- **类型 2（重排序）**：如 permute 或 rearrange 操作，Cell 的相对顺序改变但内容独立
- **类型 3（Cross-Cell mixing）**：如 Self-Attention，两个 Cell-annotated tensor 做 MatMul，输出的每个 Cell 混合了输入所有 Cell 的信息
这种分析使得编译器能理解跨层 Cell 依赖（如 Self-Attention 要求所有 token 在同一 GPU 聚合），从而指导 Profile-Guided Placement 的合法性约束。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
模型开发者只需在现有 PyTorch 代码中添加一行标注：
```python
x = brt.annotate_cell(input_tensor, dims=(0), shape=(1, 768))
```
随后所有基于该 tensor 的 Router 操作都能自动感知 Cell 粒度。编译器在 AOT 阶段扫描标注并进行符号执行，JIT 阶段收集每个 Cell 的实际路由决策统计分析。Brainstorm 论文展示了 6 个动态网络（SwitchTransformer, TaskMoE, SwinV2-MoE, LiveSR, DynamicRouting, MSDNet）仅需 6~24 行代码修改即可完成移植。

涉及论文标题：
- Optimizing Dynamic Neural Networks with Brainstorm
