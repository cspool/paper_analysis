## Expert Collaboration Patterns (专家协作模式)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Collaboration Patterns（专家协作模式）是指在 MoE LLM 中，跨层（甚至同层）的不同专家之间存在的稳定的、可重复的共激活（co-activation）关系。与以往将每个专家视为独立实体的分析范式不同，协作模式视角认为：MoE 模型的最终输出并非由单个专家独立完成，而是由一组跨层的专家协同工作产生。例如，当处理数学推理任务时，可能同时出现 Layer 5 Expert 21（负责数值提取）和 Layer 6 Expert 3（负责逻辑运算）的频繁共激活，它们共同构成了一个"数学推理"的功能模块。

协作模式可以通过专家激活矩阵 X 的分析来发现：对 X 进行稀疏字典学习分解 $X \approx D \cdot R$，字典 D 的每个 atom（列向量）编码了一组共激活的专家集合（即一个协作模式），稀疏编码 R 控制各模式在不同输入样本上的参与度。实验验证：(1) 60% 的字典模式对应于穷举搜索中 top 10% 最高频的专家组合；(2) 语义相近领域（数学/物理/计算机科学）的协作模式分布相似度高，语义不同的领域（数学/法律）分布差异大；(3) 层级分解揭示从粗到细的语义层级——高层字典捕获"数学计算"等大类，深层字典细化为"日期识别"、"符号处理"等子任务。

从算法pipeline角度拆解术语：

协作模式发现的完整流程：

```
# 输入：MoE LLM（m 层，n 专家），数据集 S（N_s 个样本）
# 输出：协作模式字典 D 和稀疏编码 R

# Phase 1: 构建 Expert Activation Matrix
for each sample i in S:                  # i = 1..N_s
    for each token t in sample:
        alpha_t = Router(x_t)              # router 为每个 token 输出 n x m 个权重
    v_{i,j,k} = sum_t alpha(i)_{t,j,k}     # 句子级聚合，式(1)
X = stack(v)                             # X in R^{N_e x N_s}, N_e = m x n

# Phase 2: 层级稀疏字典学习 (HSDL)
D_1, R_1 = sparse_dict_learn(X, N_p1)    # Layer 1: X ≈ D_1 * R_1
for k in 2..K:
    D_k, R_k = sparse_dict_learn(D_{k-1}, N_pk)  # 递归分解
    L = L_sparse + lambda1*L_hier + lambda2*L_rec

# Phase 3: 结果解读
# D_1 的每个 atom: 一组粗粒度协作专家集合
# D_K 的每个 atom: 细粒度的子模式
# R_k 的每列: 各模式在不同样本上的激活强度
```

术语一般如何实现？如何使用？

实现方式：对每个输入样本前向传播，在 MoE 层的 router 输出位置插入 hook 记录激活权重，按句子求和得到激活矩阵 X。然后使用稀疏字典学习算法对 X 进行分解。该模式分析可用于：(1) 模型可解释性——可视化哪些专家协作处理何种语义任务；(2) 专家剪枝——识别并保留高贡献的协作模式；(3) 领域自适应——分析不同领域输入下的协作模式差异。

涉及论文标题：
- Unveiling Hidden Collaboration within Mixture-of-Experts in Large Language Models
