## Whole Graph Computation-Communication Overlapping

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Whole Graph Computation-Communication Overlapping 是 Lancet (MLSys 2024) 提出的 MoE 训练优化范式，将 computation-communication overlap 的 focus region 从仅 all-to-all 通信 + expert 计算扩展到整个训练计算图。传统方法（Tutel, FasterMoE）只重叠 MoE 层内的 all-to-all 和 expert 计算，但由于 all-to-all 时间通常远超 expert 计算（可达 3.36x），expert 计算被完全隐藏后 all-to-all 仍是瓶颈。Lancet 扩展 focus region 后识别出两类新的可重叠算子：(1) 前向传播中的 non-MoE 计算（self-attention、前后 Transformer 层的 FFN），通过沿 batch 维度分区并 pipelining 与 all-to-all 重叠；(2) 反向传播中的 weight gradient computation (dW)，与 all-to-all 无数据依赖，可直接调度重叠。在 RAF compiler 上以两个 IR Pass 实现：Weight Gradient Computation Schedule Pass（反向）和 Operator Partition Pass（前向）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

在 Lancet 的 RAF compiler 中通过两个 IR 级别 Pass 实现全图重叠：

```
IR Input: 完整训练图（前向+反向的指令序列 I=[I_1,...,I_N]）

Pass 1: Weight Gradient Computation Schedule Pass (反向传播)
  ├── Dependency Analysis: 构建 G=(I,E), BFS标记与每个all-to-all无路径依赖的dW指令
  ├── Best-Fit Greedy Scheduler: 
  │     for each all-to-all I_a (按backward顺序):
  │         从 W^{I_a} 中选择未使用dW, 使 Σt^W ≈ t^a
  │         直到 all-to-all 被完全重叠 或 W^{I_a} 耗尽
  └── ReorderInstructions: 将选中的dW指令移到对应all-to-all之后

Pass 2: Operator Partition Pass (前向传播)
  ├── DP: T(n) = min_{1<i<n-1}{T(i) + min_{1<k<K} P(i,n,k)}
  ├── PartitionAxisInferencer: OR-Tools CSP求解 所有tensor的分区轴
  ├── PipelineScheduler: 模拟分阶段时间线, 报告P(i,n,k)
  └── 选择最优(i,n,k) → IR变换, 生成不规则all-to-all kernel

IR Output: dW重排 + 前向算子分区 + 不规则all-to-all kernel
```

关键设计决策：前向用 partition+pipeline（几乎所有计算依赖 all-to-all），反向用 scheduling（充足 dW 可调度）。Partition 沿 batch 维度（非 capacity 维度），通过特殊 gating operator 在 partition 间传递容量信息保证数学等价。DP 自动搜索最优 partition range，平衡 overlap gain vs partition overhead（GPU kernel launch + SM underutilization）。Gating 方法约束 partition 范围：Switch/Random gate 可分区前后，Batch-Prioritized gate 只能分区之后。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Lancet 实现约 13K LoC C++（RAF compiler 扩展），用户只需在 RAF pass manager 启用两个 pass，无需修改 Python 训练代码。三个超参通过环境变量设置：ρ（最大分区数）、γ（指令分组大小）、ι（最大 partition range）。优化过程单 GPU 运行 <20 分钟。评测在 GPT2-S-MoE / GPT2-L-MoE + A100/V100 集群上（Wikitext dataset），实现 non-overlapped communication 减少最高 77%，end-to-end 加速最高 1.3x vs Tutel/DeepSpeed。GitHub: https://github.com/hikettei/Lancet 和 https://github.com/awslabs/Lancet-Accelerating-MoE-Training-via-Whole-Graph-Computation-Communication-Overlapping (Apache-2.0)。

涉及论文标题：
- Lancet: Accelerating Mixture-of-Experts Training via Whole Graph Computation-Communication Overlapping
