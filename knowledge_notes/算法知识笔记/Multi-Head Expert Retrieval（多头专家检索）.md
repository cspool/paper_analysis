## Multi-Head Expert Retrieval（多头专家检索）

术语是什么？
Multi-Head Expert Retrieval 是 PEER 层中的核心设计：使用 h 个独立的 query network（类似于 Transformer 的 multi-head attention 和 PKM 的 multi-head memory），每个 head 独立计算 query 向量并从共享的 N 个 singleton expert 池中检索 k 个 expert。不同 head 的检索结果直接求和：f(x) = Σ_{i=1}^h Σ_{j ∈ I^i} g_j(x) e_j(x)。Multi-head 设计的核心价值：(1) 增加模型表达能力——h 个 head × k 个 expert/head = hk 个 active expert，动态组装等效 hk 神经元 MLP；(2) 共享 expert pool 实现参数复用——不同 head 可检索相同或不同的 expert，隐式实现 hidden neuron 共享；(3) 每个 head 的 router 可学习不同的检索偏好，增强 expert 池的利用多样性。

从算法pipeline角度拆解术语：
Multi-Head Expert Retrieval 的计算过程（替换单头版）：
```
# 单头版本: h=1
q¹(x) = query_net₁(x)           # R^d
I¹ = ProductKeyRetrieve(q¹, C, C', k)  # k 个 expert 索引
output = Σ_{j∈I¹} gⱼ(x) · σ(uⱼ^T x) vⱼ

# 多头版本: h>1
for i in 1..h:
    qⁱ(x) = query_net_i(x)       # h 个独立 query network
    Iⁱ = ProductKeyRetrieve(qⁱ, C, C', k)  # 各自检索 k 个 expert
    # 共享相同的 N 个 expert 和 product keys C, C'
output = Σ_{i=1}^h Σ_{j∈Iⁱ} gⱼ(x) · σ(uⱼ^T x) vⱼ
```
等效关系：当 k=1 时，PEER 的 h 个 head 各检索 1 个 expert ≡ 1 个 h 神经元 MLP。

术语一般如何实现？
PEER 默认配置：h=8, k=16, hk=128 active experts。不同 head 的 query 通过 h 个独立线性层（或一个批量线性层）投影。get_indices 函数对每个 head 分别执行 product key 检索（当前实现未做 head 间检索共享优化）。每个 head 从共享的 N 个 expert（Embedding 层存储）中独立检索，检索到的索引可能重叠（不同 head 选到同一 expert）。Ablation 研究了 h 和 k 的最优组合：给定固定 hk，最优 h 随 hk 增大而增大。

涉及论文标题：
- Mixture of A Million Experts
