## Fast Weights / Fast Weight Programmers（快速权重/快速权重编程器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fast weights 是一种神经网络概念，最早由 Hinton & Plaut (1987) 和 Schmidhuber (1992) 提出。核心思想：神经网络的权重分为两类——"slow weights"（在全部数据上缓慢更新，即常规训练的参数）和"fast weights"（仅在最近/最相关数据上快速更新）。Fast Weight Programmers (FWPs) 是 Schmidhuber 提出的延伸概念：使用一个学习到的"slow"网络来生成"fast"网络的权重更新规则。TTT 层可被看作 FWP 的特例：内循环权重 W 是"fast weights"（在单个序列上更新），外循环参数 θ 是"slow weights"（在数据集上更新），更新规则是显式的梯度下降。TTT 框架的独特之处在于将 fast weights 的更新形式化为一个显式的最优化问题（自监督学习），而非使用手工设计的更新规则。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Fast weights 在 TTT 中的具体体现：

```
# Slow weights (outer loop parameters, updated across datasets)
θ_K, θ_V, θ_Q    # reconstruction views
θ_init = W_0      # fast weight initialization
θ_lr              # learning rate for fast weight updates
θ_rest            # rest of the network

# Fast weights (inner loop state, updated per sequence)
W_0 = θ_init      # initialized from slow weight
for each token x_t in sequence:
    W_t = W_{t-1} - η(x_t) · ∇ℓ(W_{t-1}; x_t)   # fast update
    z_t = f(θ_Q x_t; W_t)                         # use fast weight
# After sequence ends, W_T is discarded; only θ_init is kept
```

与 DeltaNet 的关系：DeltaNet (Schlag et al., 2021) 等价于 TTT-Linear with b=1（online GD），但没有 LN 和残差连接。Gated DeltaNet (Yang et al., 2024) 添加了门控机制和数据依赖的衰减。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Fast weights/FWP 概念已催生了多条现代 RNN 研究线：
- **Linear attention** (Katharopoulos et al., 2020)：最简单的 FWP 形式
- **DeltaNet** (Schlag et al., 2021)：delta rule 更新
- **Mamba-2** (Dao & Gu, 2024)：结构化状态空间对偶，使用矩阵隐藏状态
- **Gated DeltaNet** (Yang et al., 2024)：添加门控的 delta rule
- **TTT layers** (Sun et al., 2024)：通用框架，支持任意神经网络作为隐藏状态

涉及论文标题：
- Learning to (Learn at Test Time): RNNs with Expressive Hidden States
