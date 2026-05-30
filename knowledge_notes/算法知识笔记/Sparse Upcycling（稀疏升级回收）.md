## Sparse Upcycling（稀疏升级回收）

术语是什么？
Sparse Upcycling 是一种将预训练 dense checkpoint 转换为 MoE 模型的方法，由 Komatsuzaki et al. (ICLR 2022) 提出。其核心操作是将 dense 模型的 MLP 层复制多份作为 MoE 的 expert 初始权重（即每个 expert = 原 MLP 的完整复制），然后添加 router 并继续训练。这种方式保证了初始时 MoE 模型输出与 dense 模型等价（所有 expert 相同，router 任意加权结果相同），然后通过后续训练逐步分化 expert。

从算法pipeline角度拆解术语：
Sparse Upcycling 流程：
```
# 输入：预训练 Dense 模型（N 层）
# 输出：MoE 模型（N 层, E 个 expert per MoE layer）

for layer l in dense_model:
    if is_moe_layer(l):                          # 选择部分层转为 MoE
        for i in range(E):                       # 每个 expert
            expert_i[l] = copy(dense_mlp[l])      # 完整复制原 MLP
        router[l] = random_init()                 # 随机初始化 router
    else:
        keep_dense(l)                             # 保留非 MoE 层

# 继续训练：expert 从相同的初始点开始逐步分化
```

术语一般如何实现？如何使用？
在 HuggingFace Transformers 框架中实现：加载 T5 等预训练模型，复制 FFN 层权重为多个 expert，添加 Top-K router，继续预训练。与 Checkpoint Recycling 的关键区别：(1) Sparse Upcycling 完整复制 MLP，expert 大小固定与原模型一致；(2) Checkpoint Recycling 可选择性采样权重，构造不同大小的 expert，更灵活。

**MoE-Pruner 的补充**：MoE-Pruner (Xie et al., 2024) 通过实验验证了 Sparse Upcycling 对 MoE 剪枝策略的影响：(a) Upcycling 初始化的 MoE 模型（如 Mixtral-8x7B、Qwen1.5-MoE-A2.7B、MiniCPM-MoE-8x2B）具有更高的 expert 相似性和更均衡的 expert 激活频率，因此 expert-level pruning 会带来严重性能下降，weight-level pruning 是更好的选择；(b) 从零训练（train from scratch）的 MoE 模型（如 DeepSeek-V2、OLMoE）具有更低的 expert 相似性和更不均衡的激活频率，cold expert 可以被安全剪掉。MoE-Pruner 通过 Load Balancing Score（激活频率的变异系数）量化了这一差异：upcycling 模型的 score 通常更低（更均衡），train-from-scratch 模型的 score 更高（更不均衡）。此发现为"不同初始化的 MoE 需要不同压缩策略"提供了定量依据。

**Nexus 的补充**：Nexus (Gritsch et al., 2024) 进一步推广了 Sparse Upcycling 概念，从**多个独立训练的域特化 dense expert** 而非单一 dense checkpoint 进行 upcycling。其流程：(1) 在 SlimPajama 的各子域（ArXiv, Books, C4, StackExchange, Wikipedia）上分别训练 dense expert；(2) 合并时，seed model 的原始 FFN 作为 shared expert（始终激活），各 dense expert 的 FFN 沿新维度拼接为 routed experts；(3) 非 FFN 参数（attention, norms）通过简单权重平均 merge。Nexus 的关键创新是用基于域嵌入的 adaptive router（见 Adaptive Domain-Embedding Router 术语）替代从零训练的线性 router。这种 upcycling 方式使得 expert 保留域专业化（如 ArXiv expert 对 ArXiv token 的路由概率达 63%），且支持后续高效扩展新 expert。

涉及论文标题：
- MoE Jetpack: From Dense Checkpoints to Adaptive Mixture of Experts for Vision Tasks
- MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router
- Nexus: Specialization meets Adaptability for Efficiently Training Mixture of Experts
