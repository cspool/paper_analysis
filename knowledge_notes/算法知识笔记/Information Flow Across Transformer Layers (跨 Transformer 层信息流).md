## Information Flow Across Transformer Layers (跨 Transformer 层信息流)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

跨 Transformer 层信息流（Inter-layer Information Flow）是指 token 的隐藏状态表示在 LLM 层间传播时逐步聚合来自其他 token 信息的过程。在层 l，token i 通过自注意力从单层感受野中的 token 接收信息，经 FFN 转换后，更新后的表示 o_i^(l) 传播到层 l+1。这一过程使 token x 的表示不仅编码自身信息，还编码从其他 token "中继"（relay）来的信息。

PowerAttention 论文通过 probing 实验验证了信息流在 LLM 中的三个关键性质：(1) 信息流**本质上存在**——即使未训练的 Full Attention 也展示空间局部性和渐进式信息扩散；(2) 稀疏注意力**放大**了信息流的层次化特征——Sliding Window 展示线性扩展，PowerAttention 展示 phase transition 式跳跃扩展；(3) 通过 Continue Pretraining + Fine-tuning 可**激活**稀疏模式的信息流机制——训练后 PowerAttention 的 probing 精度从 56% 提升至 100%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**信息流 Probing 方法（PowerAttention Appendix C）**：

```
输入: 长序列（N=16K），包含随机 6 类 passkey（apple/banana/cherry/grape/kiwi/lemon）
      passkey 固定在 10% 位置，其余为无关文本

对每层 l ∈ {1, ..., 28} 和每 block b ∈ {1, ..., 64}:
    1. 收集所有样本在层 l、block b 位置的 hidden state h_{l,b}
    2. 对 h_{l,b} 做 average pooling（等间距采样）
    3. 训练 logistic regression classifier: h_{l,b} → 6-class passkey
    4. 计算 classification accuracy = 正确分类比例 / 总样本数
    // 如果 accuracy ≈ 1/6（random），则该位置不包含 passkey 信息
    // 如果 accuracy > 1/6，则该位置的 hidden state 编码了 passkey 信息

总计：28 layers × 64 blocks = 1792 个独立 classifier 训练
```

**信息流的数学建模（PowerAttention Section 3.1）**：

```
// 层 l 的信息聚合
o_i^(l) = Σ_{j ∈ A_i^(l)} softmax(q_i^(l) · k_j^(l) / √dk) · v_j^(l)

// 其中 A_i^(l) 是 token i 在层 l 的单层感受野
// h_j^(l-1) 已编码了前 l-1 层传播的聚合信息
// o_i^(l) 通过 FFN 后变为 h_i^(l)，继续传播到层 l+1

// 多层感受野 = 所有从 token i 出发，经过 ≤ d 步 DAG 路径可达的 token
R_d(i) = {j | 存在从 i 到 j 的路径，路径长度 ≤ d}
```

**信息流 probing 的关键发现**：

PowerAttention Figure 5 的可视化揭示了信息流的层间演变模式：
- Full Attention（未训练）：信息从 passkey 位置逐步向周围扩散，后期层覆盖全部位置
- Sliding Window：信息以线性速率逐层向前推进（每层约扩展 window_size 个 block）
- PowerAttention（未训练）：信息在特定层出现跳跃式扩展（phase transition），但精度仅 ~56%
- PowerAttention（训练后）：信息流边界更清晰聚焦，最终 token 的 probing 精度达 100%，展示了训练对信息流机制的激活效果

术语一般如何实现？如何使用？

信息流概念在 PowerAttention 中有三个核心用途：(1) 设计指导——以最大化多层可达性为目标设计稀疏注意力边集（DAG 最优化问题）；(2) 诊断工具——通过 probing 分析检测特定稀疏模式的覆盖盲区或信息传播瓶颈；(3) 训练验证——通过对比训练前后的 probing 精度，验证继续预训练+fine-tuning 是否成功激活了稀疏模式的信息流机制。

在更广泛的 LLM 研究中，信息流分析与 mechanistic interpretability（机制可解释性）和行为分析密切相关。具体分析手段包括：(1) Probing classifiers（线性/非线性分类器检测隐藏状态中的特定信息）；(2) Activation patching（替换特定位置的激活值，观察输出变化）；(3) Attention pattern visualization（可视化不同层的注意力分布热力图）。PowerAttention 的 probing 方法与这些技术互补，专注于量化感受野而非解释具体的注意力模式。

涉及论文标题：
- PowerAttention: Exponentially Scaling of Receptive Fields for Effective Sparse Attention
