## Cross-Polytope Hashing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Cross-Polytope Hashing (CPH) 是Locality-Sensitive Hashing (LSH)的一种具体hash函数族，专为角距离（angular distance / cosine similarity）设计。其核心思想是将输入向量通过随机旋转后映射到cross-polytope（交叉多面体）的最近顶点。Cross-polytope是一个在d维空间中由2d个顶点组成的几何体，顶点为各坐标轴正负方向上的单位向量：{±e_1, ±e_2, ..., ±e_d}。

数学表示：
$$LSH(\mathbf{x}) = \operatorname{argmax}_{i \in \{\pm 1, \pm 2, \dots, \pm d\}} |\mathbf{R}\mathbf{x}|_{i}$$

其中R是随机旋转矩阵（或使用Fast Johnson-Lindenstrauss Transform加速到O(d log d)），|Rx|_i是旋转后向量第i个分量的绝对值。hash结果为选中分量的索引i（带正负号），共2d个可能的bucket。

CPH的渐近最优敏感度（sensitivity）ρ = 1/c²（c为近似因子），在理论上是角距离下LSH能达到的最优值。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

在LSH-MoE训练pipeline中，Cross-Polytope Hashing的执行流程：

```
# 输入: X = [N, h]  — N个token，每个h维
# 参数: R = [d, h]  — 随机旋转矩阵（或Fast JL transform）

# Step 1: 随机旋转（矩阵乘法）
Rx = X @ R.T                            # [N, d], O(N·h·d)

# Step 2: 取每个分量绝对值，找最大值索引
abs_Rx = |Rx|                           # [N, d]
hash_indices = argmax(abs_Rx, dim=-1)   # [N], 值域 [0, 2d-1]
# argmax返回的是维度的索引，正负号由Rx值的符号隐含决定

# Step 3: 按hash_indices分组
# 相同hash_indices的token属于同一cluster
clusters = group_by(hash_indices)
```

CPH的计算复杂度为O(N·h·d)（旋转矩阵乘法），相比迭代聚类（如K-Means的O(N·k·h·iter)）在在线场景下效率显著更高。

LSH-MoE在消融实验中比较了Cross-Polytope Hashing与Spherical-Plane Hashing (SP)，发现CP在相同压缩率下达到更好的模型收敛质量。原因：CP基于n维cross-polytope编码数据，对多种复杂数据模式更有泛化能力；而SP依赖球面和平面之间的几何关系，更适合球面分布特征的数据。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- PyTorch实现：旋转矩阵R可以是一个随机初始化的固定矩阵（不需要学习/训练），LSH本身没有可训练参数
- Fast Cross-Polytope LSH (Kennedy & Ward, 2016): 使用subsampled randomized Hadamard transform将矩阵乘法从O(d²)加速到O(d log d)，适合极高维场景
- 在LSH-MoE中，R矩阵在训练开始时随机初始化并固定，不参与梯度计算
- CPH也用于Reformer（Kitaev et al., 2020）利用attention的稀疏性——使用LSH将query和key分桶，仅计算同一桶内的attention

涉及论文标题：
- LSH-MoE Communication-efficient MoE Training via Locality-Sensitive Hashing
