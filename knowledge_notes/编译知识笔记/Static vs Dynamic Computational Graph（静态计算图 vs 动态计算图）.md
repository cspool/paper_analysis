## Static vs Dynamic Computational Graph（静态计算图 vs 动态计算图）

术语是什么？
静态计算图（Static Graph）和动态计算图（Dynamic Graph）是深度学习框架的两种计算范式。静态图在训练/推理前预先编译整张计算图，编译后所有 tensor shape、operator 执行顺序固定，执行效率高但无法在运行时调整——代表框架：MindSpore (GRAPH_MODE)、TensorFlow 1.x。动态图在运行时逐算子构建和执行计算图，支持 shape 随输入动态变化，灵活但有一定框架开销——代表框架：PyTorch (eager mode)。

从编译框架角度拆解术语：
两种图在 MoE 训练中的差异：

**静态图模式（MindSpore）**：
```
# 编译前
capacity = max_tokens_per_expert  # 必须在编译前确定，训练中不可变
graph = compile(MoE_Model, capacity=capacity)  # shape 固定

# 训练
for iteration i:
    tokens, labels = next_batch()
    # 各 expert 的输入 buffer 已按 capacity 预分配
    # hot expert: 超出 capacity 的 token 被丢弃（token dropping）
    # cold expert: buffer 不满 → zero-padding 浪费计算
    loss = graph(tokens, labels)
    loss.backward()
```

**动态图模式（PyTorch）**：
```
for iteration i:
    tokens, labels = next_batch()
    # 每次 iteration 可动态调整 expert 的输入 shape
    # hot expert: buffer 自动扩容到所需的 token 数
    # cold expert: buffer 自动缩容到实际 token 数
    output = moe_layer(tokens)  # 无 token dropping, 无 padding
    loss = criterion(output, labels)
    loss.backward()
```

EfficientMoE 的核心贡献之一是在**静态图框架内**实现动态性：通过 load prediction cycle 的周期性评估 + 编译前 capacity 注入 + replica placement 修改，在保持静态图计算效率的同时获得接近动态图的效果。key insight：不必每次 iteration 都重新编译——只需在 load prediction cycle (m 次 iteration) 边界重新评估并重新编译即可。

术语一般如何实现？如何使用？
主流框架的图模式选择：PyTorch 2.x 引入 `torch.compile` (Dynamo + Inductor) 支持静态图优化（JIT trace）；TensorFlow 2.x 使用 `@tf.function` 将 eager 代码转为静态图；MindSpore 使用 `ms.GRAPH_MODE` vs `ms.PYNATIVE_MODE` 切换。静态图适用于生产部署和高效训练（减少框架开销），动态图适用于开发调试和需要动态 shape 的模型结构（如可变长度序列、MoE 的 per-expert 变容）。许多框架现在支持混合模式。

涉及论文标题：
- EfficientMoE: Optimizing Mixture-of-Experts Model Training With Adaptive Load Balance
