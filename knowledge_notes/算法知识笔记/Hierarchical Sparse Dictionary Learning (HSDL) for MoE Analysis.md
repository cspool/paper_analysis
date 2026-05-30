## Hierarchical Sparse Dictionary Learning (HSDL) for MoE Analysis

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Hierarchical Sparse Dictionary Learning (HSDL) 是一种专门用于从 MoE LLM 的专家激活数据中提取多粒度协作模式的层级无监督学习方法。它扩展了传统的单层稀疏字典学习，通过对字典矩阵递归分解来捕获从粗到细的专家协作层次结构：$D_{k-1} \approx D_k \cdot R_k$（式 3）。第一层从原始激活矩阵 X 开始分解 $X \approx D_1 \cdot R_1$，随后每一层对上一层字典进一步分解。

HSDL 引入三个关键约束：(1) 稀疏性约束 $L_{\text{sparse}} = \|R_{k,i,:}\|_{\infty}$——防止某些字典元素主导，确保稀疏激活；(2) 层间一致性约束 $L_{\text{hier}} = \sum_j \|R_{k+1,j}\|_1 \cdot \|R_{k,j}\|_1 / N$——控制跨层字典学习的影响传递；(3) 重构误差项 $L_{\text{rec}} = \sum_j \|D_{k,j} - (D_{k+1}R_{k+1})_j\|_1 \cdot \|R_{k,j}\|_1 / N$——保证层间关系一致。总损失 $L_{\text{total}} = L_{\text{sparse}} + \lambda_1 L_{\text{hier}} + \lambda_2 L_{\text{rec}}$（式 7）。

从算法pipeline角度拆解术语：

HSDL 的层级计算流程：

```
# Layer 1: 从原始激活矩阵开始
D_1, R_1 = argmin_{D,R} ||X - D*R||_F^2 + alpha*||R||_1   # 标准稀疏编码
s.t. ||D_j||_2 <= 1  for all columns j

# Layer k (k >= 2): 对上一层字典递归分解
D_k, R_k = argmin_{D,R} ||D_{k-1} - D*R||_F^2 + alpha*||R||_1

# 多目标联合优化（每层同时考虑三个损失）：
# L_sparse:  R_k 的每一行的 L_inf 范数，鼓励稀疏激活
# L_hier:    跨层 R 矩阵的 L1 范数乘积
# L_rec:     字典重构误差的加权 L1 范数

# 优化: 交替更新 D_k 和 R_k
# - 固定 D_k, 用 Lasso/CD 更新 R_k
# - 固定 R_k, 用 block-coordinate descent 更新 D_k
```

术语一般如何实现？如何使用？

可使用 scikit-learn 的 `MiniBatchDictionaryLearning` 或 SPAMS 库实现单层字典学习，HSDL 在此基础上增加递归分解和三个约束的联合优化。论文在 phi-moe 模型上用 MMLU-pro 数据集（2,812 样本，5 领域）验证——60% 的字典模式对应 top 10% 最高频穷举组合。HSDL 的应用场景：(1) MoE 模型可解释性——层级语义标注揭示模型如何从粗到细理解任务；(2) 专家剪枝的输入信号；(3) 领域特化分析。

涉及论文标题：
- Unveiling Hidden Collaboration within Mixture-of-Experts in Large Language Models
