## Graph Mixture-of-Experts (Graph MoE) for Dynamical Systems

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Graph Mixture-of-Experts（Graph MoE）是将 Mixture-of-Experts 架构应用于图神经网络以提升动态系统建模泛化能力的技术。与 LLM MoE（路由函数通常基于输入 token 的可学习 gate）不同，LEGO 提出的 Graph MoE 使用预训练 LLM 作为 context-aware routing function：K 个同构 GNN experts（如 EGNN）各自处理相同的输入图 G 和初始状态 X⁽⁰⁾，生成 K 个候选预测；LLM 基于环境上下文（系统参数、物体状态、连接关系）选择最合适的 expert。路由权重通过 one-hot + label smoothing（选中 expert α，其他 (1-α)/(K-1)）实现软性选择。Diversity-enhanced contrastive loss 确保不同 expert 学习互补的动力学模式。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
输入: 图 G=(V,E), 初始状态 X⁽⁰⁾, 环境上下文 C, K 个 GNN experts {f₁,...,f_K}
输出: 预测状态 X̂⁽ᵗ⁾

// 1. 所有 experts 并行预测
for k in 1..K:
    H^k = f_k(G, X⁽⁰⁾)                    // GNN 前向（L 层消息传递）
    X̂^k = Decoder(H^k)                     // 各 expert 独立预测

// 2. LLM Judge 选择 expert
prompt = HierarchicalPrompt(C, X⁽⁰⁾, E)   // 系统/物体/边 三层 prompt
chosen = LLM(prompt, {X̂^k}_{k=1}^K)       // LLM 评估并选择

// 3. Label Smoothing 权重
ω(k) = α           if k == chosen          // Eq.7
     = (1-α)/(K-1) otherwise

// 4. 加权组合（Eq.8）
for each node i:
    h̄_i = Σ_k ω(k) · h_i^k
    x̂_i⁽ᵗ⁾ = Decoder({h̄_i})
```

关键设计的独特之处：
- Routing 不是 learnable MLP gate（如标准 MoE），而是预训练 LLM 的 zero-shot 推理。LLM 不需要微调，利用世界知识理解环境语义
- 与环境无关的 MoE 路由仅依赖输入数据 → LEGO 的 LLM 路由额外利用了文本化的环境元信息
- Diversity loss 保证 expert specialization：不同 expert 专家的激活表征被推远（contrastive），同一 expert 的表征被拉近

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 开源：https://github.com/jdp22/LEGO.git
- expert 模型选择：EGNN（默认）、EGNO、Radial Field 均可作为基础 expert，仅需同构 GNN 架构
- 超参数：K=5 experts（默认，经验最优），α label smoothing 系数，τ contrastive loss temperature
- LLM：Llama 3.1 8B（大模型更优但小模型也可用），temperature=0 时推理阶段性能最好
- 训练：交替优化（每隔若干 epoch 更新 LLM routing weights，内部循环更新 expert 参数），Adam optimizer (lr=0.0005)
- 适用场景：动态系统预测、物理模拟、分子动力学、人体运动预测等含环境变化的图结构预测任务
- 与标准 LLM-MoE 的区别：LLM-MoE 中 gate 是小型可学习 MLP → LEGO 中 gate 是预训练 LLM，利用外部知识推理环境
- 局限：(1) LLM 推理成本较高（虽可通过交替优化降低调用频率）；(2) LLM 对专业科学领域（如分子动力学）的理解可能有限；(3) expert 数量过多时 LLM 判断困难

涉及论文标题：
- Marrying LLMs with Dynamic Forecasting A Graph Mixture-of-expert Perspective

---
