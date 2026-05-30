## SimHash

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

SimHash (Charikar, 2002) 是基于余弦相似度的LSH家族。对于向量x∈R^d，SimHash生成随机超平面w（从标准正态分布采样），返回Sign(w^T x)。两个向量x,y共享相同符号当且仅当随机投影不落在它们之间，概率为p = 1 - θ/π，其中θ = arccos(x·y/(|x|·|y|))。如果使用L张哈希表，每张K个随机哈希函数，y被查询x检索到的概率为1 - (1-p^K)^L。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**SimHash在LSH采样中的概率计算**：

```
// 给定: q∈R^d, k_i∈R^d, 随机投影矩阵W∈R^{d×(K×L)}
// q和k_i共享K-bit哈希码的碰撞概率:
cos_sim = (q·k_i) / (|q| * |k_i|)  // 余弦相似度
p_i = 1 - arccos(cos_sim) / π       // 单hash函数碰撞概率
// 至少2张哈希表碰撞的采样概率:
u_i = 1 - (1-p_i^K)^L - L·p_i^K·(1-p_i^K)^{L-1}
```

**关键性质**：u_i随q与k_i的余弦相似度单调递增——越相似的key越容易被采样，符合importance sampling要求。

术语一般如何实现？如何使用？

MagicPIG中，GPU侧对所有attention head共享随机投影矩阵W，K×L个随机向量，内存开销400KB~825KB（K=10,L=150时384KB）。SimHash的哈希函数计算（Sign(q@W)）是矩阵乘法和符号运算，适合GPU并行。数据预处理需要进行key centering——因为LLM中key向量集中在与query向量几乎相反的方向（attention sink几何），不centering则几乎所有key的碰撞概率都接近0。

涉及论文标题：
- MagicPIG: LSH Sampling for Efficient LLM Generation

---
