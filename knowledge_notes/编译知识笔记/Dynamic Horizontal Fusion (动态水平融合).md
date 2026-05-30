## Dynamic Horizontal Fusion (动态水平融合)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dynamic Horizontal Fusion 是 Brainstorm 提出的编译器优化，将动态网络中多个条件激活（conditionally activated）的并行分支水平融合为一个 fused GPU kernel，并根据 Router profile 中的 Cell 负载分布为每种常见负载编译多个不同 shape 的 tuned kernel。与传统的水平融合（假设所有 branch 同时激活、输入相同大小）不同，动态水平融合需要处理：(1) 运行时才知道哪些 branch 被激活；(2) 每个 branch 接收的 Cell 数量不均匀；(3) branch 负载分布随时间可能漂移。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
全流程：
1. **Profile 分析**：从 Router 的 JIT profile 提取每个 branch 的 Cell 负载的多 percentile 分布（如 P50=4 patches, P90=9 patches, P100=27 patches）
2. **Kernel Tuning**：对每类 branch 算子（如 Conv2D），为每种 shape 通过 TVM auto-tune 一个最优 kernel。合并所有 branch 的 tuned kernel 为一个 fused kernel，内含多个 variant。
3. **图变换（torch.fx Pass）**：替换数据流图中多个串行 branch 算子为 fused kernel 节点；修改 Router 节点使其在分发 Cell 时同时 pad 到 nearest tuned shape
4. **运行时调度**：Router 得知实际 Cell 数后，选择 nearest tuned kernel（如 8 patches → 27-patch kernel，padding 19），计算 global input pointer offsets，一次 GPU launch 并发执行所有激活的分支

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Brainstorm 通过 torch.fx 自动应用该优化，无需开发者手动干预。规则性策略决定是否启用：当 branch 数多且单个 branch 无法 saturated GPU CU 时启用（如 LiveSR 的 lightweight branch、SwitchTransformer 的 256 experts）。限制：kernel candidates 数量受控——如 SwitchTransformer 的 256 个 expert 使用相同 FFN 结构，仅需 ~6 个 candidate kernel（因所有 expert 共享同一算子结构，仅权重不同）。

涉及论文标题：
- Optimizing Dynamic Neural Networks with Brainstorm
