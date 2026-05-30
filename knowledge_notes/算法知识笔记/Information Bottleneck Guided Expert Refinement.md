## Information Bottleneck Guided Expert Refinement

术语解释
Information Bottleneck（信息瓶颈，IB）是一种平衡表示复杂度和预测能力的理论框架。MELD 将 IB 用于指导 expert 精炼和 router training，在 few-shot 场景防止过拟合。

术语是什么？
IB 原理：min I(X;Z) 且 max I(Y;Z)，X=输入，Y=标签，Z=表示。
- min I(X;Z)：压缩噪声、捕获高层特征
- max I(Y;Z)：保留足够的预测信息

从算法pipeline角度拆解术语。
Expert 训练优化函数：
```
arg min_{θ_M_RAG} max_{θ_M_G} I(M_G(X_i); M_G(RAG(X_i)))
```
- max θ_M_G：LoRA fine-tune，最大化输出与标签 Y_i 的互信息
- min θ_M_RAG：控制 RAG 采样和 meta-path，最小化原始数据与增强数据的互信息

Router 优化函数：
```
max Σ_{e_i∈N(q_u)} I(e_i(q_u^i); l_u^i)    // 相关性
min Σ_{e_i≠e_j} I(e_i(q_u^i); e_j(q_u^j))  // 多样性
```
实践中用对比学习近似实现，精炼迭代 σ=3 轮。

术语一般如何实现？如何使用？
- Expert 精炼在 LoRA fine-tune 基础上进行，不改 base model weights
- Router 用对比学习近似互信息计算
- IB 同时支撑 Theorem 2（MoE error bound）和 Theorem 3（router 收敛性）

涉及论文标题：
- Efficient Mixture of Experts based on Large Language Models for Low-Resource Data Preprocessing
