## Singleton MLP Expert（单神经元 MLP 专家）

术语是什么？
Singleton MLP Expert 是 PEER 中使用的极小专家设计：每个 expert e_i(x) = σ(u_i^T x) v_i 仅有 1 个隐藏神经元（d_expert = 1），参数为两个 d_model 维向量 u_i（down projection）和 v_i（up projection），总计 2×d_model 个参数。这与传统 MoE 中每个 expert 为完整 FFW（hidden dimension 通常等于 d_ffn，如 4×d_model）形成鲜明对比。Singleton expert 将专家大小推到理论最小值，最大化 MoE 的粒度 G = P_active / P_expert = hk / 1 = hk，使得在固定 P_active 下总 expert 数量 N 达到 O(10⁶) 级别。由于不同 expert 共享 hidden neuron（通过 multi-head retrieval 动态组合），singleton expert 隐式实现知识共享和参数效率。

从算法pipeline角度拆解术语：
Singleton expert 的计算过程（单个 token x ∈ R^{d_model}）：
```
def singleton_expert(x, u_i, v_i):
    # u_i ∈ R^{d_model}, v_i ∈ R^{d_model}
    z = dot(u_i, x)          # 标量: 内积 → 单神经元激活输入
    a = σ(z)                  # 标量: 非线性激活 (ReLU/GELU/SwiGLU)
    output = a * v_i          # R^{d_model}: 标量 × 向量 = 缩放
    return output
```
与标准 MoE expert 的对比：标准 expert FFN(x) = W₂ σ(W₁ x)，W₁ ∈ R^{d_ffn×d_model}, W₂ ∈ R^{d_model×d_ffn}，参数量为 2×d_model×d_ffn。Singleton expert 参数量为 2×d_model（d_ffn = 1），减少 d_ffn 倍（通常为 4×d_model 倍，即减少数千倍）。

H 个 singleton expert 的聚合等价于一个 h 神经元 MLP：
```
output = Σ_{j=1}^{h} σ(u_j^T x) v_j = V σ(W^T x)
# 其中 W = [u₁,...,u_h] ∈ R^{d_model×h}
#      V = [v₁,...,v_h] ∈ R^{d_model×h}
```

术语一般如何实现？
Singleton expert 的权重 u_i, v_i 存储在 Embedding 层中（w_down_embed 和 w_up_embed），通过 product key 检索到的索引进行 lookup。计算通过 einsum 批量完成所有选中的 singleton expert。PEER 使用 h=8 heads × k=16 experts/head = 128 个 active singleton expert，等效于动态组装 128 神经元 MLP。可扩展至 GLU 变体（额外添加 linear gating 权重）。当前实现为 JAX 原型。

涉及论文标题：
- Mixture of A Million Experts
