## Product Key Retrieval（产品密钥检索）

术语是什么？
Product Key Retrieval 是一种基于乘积量化（Product Quantization）思想的高效最近邻检索技术，由 Lample et al. (2019) 在 Product Key Memory (PKM) 中首次提出。核心思想是将 N 个 d 维 key 向量分解为两组子密钥的笛卡尔积：K = {[c; c'] | c ∈ C, c' ∈ C'}，其中 C, C' 各含 √N 个 d/2 维子密钥。查询向量 q 同样拆分为 q₁, q₂，分别在两组子密钥中做 top-k 检索，候选集大小为 k²，再从候选中选出最终 top-k。检索复杂度从朴素穷举的 O(Nd) 降至 O((√N + k²)d)，使得从百万级（10⁶）候选中检索成为可能。PEER 论文将 Product Key 用作 MoE router，从超过一百万个 singleton expert 中检索 top-k 专家。

从算法pipeline角度拆解术语：
Product Key Retrieval 在 PEER 中的路由 pipeline：
```
# 输入: query q ∈ R^d，子密钥组 C, C' 各含 √N 个向量
# Step 1: 拆分 query
q₁, q₂ = q[:d/2], q[d/2:]

# Step 2: 分别在两组子密钥中 top-k 检索
I_C = TopK({q₁^T c_i | c_i ∈ C}, k)   # k 个候选子密钥索引
I_C' = TopK({q₂^T c'_j | c'_j ∈ C'}, k)

# Step 3: 候选 product keys 集合 (k² 个)
K' = {(c_i, c'_j) | i ∈ I_C, j ∈ I_C'}

# Step 4: 在候选中最终 top-k（利用内积可加性）
scores = {q₁^T c_i + q₂^T c'_j | (c_i, c'_j) ∈ K'}
final_indices = TopK(scores, k)  # k 个最终 expert 索引
```
数学保证：K 中与 q 内积最大的 k 个 key 一定在候选集 K' 中。证明基础：内积的可加性——q^T [c; c'] = q₁^T c + q₂^T c'。

术语一般如何实现？
标准实现参考 Lample et al. (2021) 的 PKM-layer.ipynb（https://github.com/facebookresearch/XLM/blob/main/PKM-layer.ipynb）。子密钥存储为两组 Embedding 矩阵（各 √N × d/2），query 投影通过线性层映射到 d 维后拆分为两半。get_indices 函数执行上述两步 top-k 检索。PEER 中 expert 的 down/up projection 权重同样存储为 Embedding 层，通过检索到的索引进行 lookup。论文指出高效实现需要 specialized hardware kernels 加速 embedding lookup 与 einsum 的融合。当前 PEER 实现为 JAX 原型（内部代码库，未开源）。

涉及论文标题：
- Mixture of A Million Experts
