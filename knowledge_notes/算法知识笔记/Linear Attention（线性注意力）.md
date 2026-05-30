## Linear Attention（线性注意力）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Linear attention 是 self-attention 的一种线性复杂度变体，由 Katharopoulos et al. (2020) 提出。核心思想是去除 self-attention 中的 softmax 非线性：z_t = Σ_{s=1}^t v_s k_s^T q_t = (Σ_{s=1}^t v_s k_s^T) q_t。由于去掉 softmax 后，计算可以重排为矩阵乘法的结合律形式，隐藏状态 M_t = Σ_{s=1}^t v_s k_s^T 可以通过 cumsum 递归更新，每个 token 的复杂度为 O(d²)（与 t 无关），总复杂度 O(T × d²)。与标准 self-attention 的 O(T² × d) 相比，在长序列下有渐近优势。TTT 论文证明（Theorem 1）：TTT-Linear + batch GD + W_0=0 + η=1/2 等价于 linear attention，即 TTT 框架可以严格推广 linear attention。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Linear attention 的递归形式和并行形式：

```
# 递归形式（类似 RNN，训练/推理均可用）
S_0 = 0  # d×d 矩阵
for t = 1,...,T:
    k_t = θ_K @ x_t   # key projection
    v_t = θ_V @ x_t   # value projection
    q_t = θ_Q @ x_t   # query projection
    S_t = S_{t-1} + v_t @ k_t^T    # 更新隐藏状态（外积）
    z_t = S_t @ q_t                 # 输出

# 并行形式（训练用，但也是 O(T²) 内存）
K = θ_K @ X, V = θ_V @ X, Q = θ_Q @ X
Z = V @ (K^T @ Q) * mask    # 等价于 attention without softmax
```

与 TTT 的关系：linear attention 的更新规则 `S_t = S_{t-1} + v_t k_t^T` 等价于 batch GD 的 TTT-Linear（G_t = -2 v_t k_t^T）。TTT 通过 mini-batch GD 打破了这一等价性，获得了更好的表达能力。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在实践中：
- linear attention 因其简单的实现被广泛使用于需要线性复杂度的场景
- 其表达能力被证明弱于 softmax attention，因为缺少非线性归一化
- TTT 框架将 linear attention 作为特例包含，并通过 mini-batch TTT 和可学习组件显著提升了性能
- Mamba-2（Dao & Gu, 2024）也基于类似 linear attention 的矩阵状态设计

涉及论文标题：
- Learning to (Learn at Test Time): RNNs with Expressive Hidden States
