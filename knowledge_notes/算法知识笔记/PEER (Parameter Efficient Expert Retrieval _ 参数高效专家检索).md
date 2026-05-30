## PEER (Parameter Efficient Expert Retrieval / 参数高效专家检索)

术语是什么？
PEER 是 Google DeepMind 提出的新型 MoE 层设计，通过 Product Key Retrieval 实现从超过一百万（10⁶）个极小专家（单神经元 MLP）中高效稀疏检索。PEER 层由三部分组成：(1) N 个 singleton expert e_i(x) = σ(u_i^T x) v_i（每个仅一个隐藏神经元，参数为两个 d_model 维向量）；(2) N 个 product key（由两组各 √N 个 d/2 维子密钥的笛卡尔积构成）；(3) h 个独立 query network（multi-head retrieval），每个检索 k 个 expert。PEER 解耦了模型总参数 P 与每 token 激活参数 P_active：P = N × 2d_model 可扩展到百万级，而 P_active = hk × 2d_model 保持恒定。PEER 可直接替换 Transformer 中任意 FFW 层，在 isoFLOP 条件下显著优于 dense FFW、coarse-grained MoE 和 PKM。

从算法pipeline角度拆解术语：
PEER 层前向传播（基于论文 Algorithm 1）：
```
def peer_forward(self, x):                    # x: (batch, tokens, d_model)
    # 多 query 头投影
    queries = self.query_proj(x)               # (b, t, h, d)
    
    # Product Key 检索
    indices, scores = self.get_indices(queries, self.sub_keys, top_k=k)
    # indices: (b, t, h, k), scores: (b, t, h, k)
    
    # Embedding lookup 检索 expert 权重
    w_down = self.w_down_embed(indices)        # (b, t, h, k, d_model)
    w_up = self.w_up_embed(indices)            # (b, t, h, k, d_model)
    
    # Singleton expert 计算
    x = einsum('btd, bthkd -> bthk', x, w_down)  # u_i^T x
    x = activation(x)                              # σ(·)
    x = x * softmax(scores)                        # router 加权
    x = einsum('bthk, bthkd -> btd', x, w_up)     # 输出投影
    return x
```
等效解释：当 k=1 时，PEER 动态组装一个 h 神经元 MLP：f(x) = V σ(W^T x)，其中 W=[u₁,...,u_h], V=[v₁,...,v_h] 从共享 expert pool 中检索得到。

术语一般如何实现？
PEER 层可插入 Transformer backbone 中间（如 12 层 transformer 的第 6 层替换 FFW），也可替换全部 FFW 层。实现使用 JAX 的 Embedding 层存储 expert 权重（类似大词表），通过 einsum 操作执行批量内积计算。默认配置：N=1024²=1,048,576 experts, h=8 heads, k=16 experts/head, query BatchNorm 启用。可扩展至 GLU 变体（添加额外 linear gating 权重）。论文代码为 Google 内部代码库，未开源；参考实现可基于 facebookresearch/XLM 的 PKM-layer.ipynb 修改。

涉及论文标题：
- Mixture of A Million Experts
