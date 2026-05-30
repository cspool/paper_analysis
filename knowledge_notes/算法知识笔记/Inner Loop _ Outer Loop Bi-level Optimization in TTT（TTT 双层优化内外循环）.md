## Inner Loop / Outer Loop Bi-level Optimization in TTT（TTT 双层优化内外循环）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TTT 将序列建模重构为一个双层优化（bi-level optimization）问题。**内循环（inner loop）**对每个序列执行：以 token 为数据点，最小化 self-supervised reconstruction loss ℓ，通过梯度下降更新隐藏状态 W。**外循环（outer loop）**在数据集层面执行：以序列为数据点，最小化 next-token prediction loss，优化网络参数 θ_rest 和内循环的超参数 θ_K, θ_V, θ_Q, θ_init, θ_lr。关键区别：常规的 learning to learn / meta-learning 中外循环是"高一层"的训练（需要多个数据集/task），而 TTT 中内循环是"低一层"的训练（每个序列是一个"数据集"），外循环与常规监督学习处于同一层级。这种设计使得外循环可以使用标准的大规模训练 recipe（如 Chinchilla），无需额外的 meta-learning 数据组织。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
双层优化的层次结构（Table 2）：

```
层级          | 内循环            | 外循环
-------------|-------------------|------------------
数据单元      | token x_t         | 序列 x_1,...,x_T
训练集        | 序列 x_1,...,x_T  | 数据集（如 Pile, Books3）
优化目标      | reconstruction ℓ  | next-token prediction
优化参数      | W（f 的权重）     | θ_rest, θ_K, θ_V, θ_Q, θ_init, θ_lr
优化器        | SGD (mini-batch)  | AdamW
```

计算图流程：
1. 外循环 forward：执行内循环（对序列 TTT），得到输出序列 z_1,...,z_T
2. 外循环 backward：计算 ∂L/∂θ_rest（通过内循环的梯度），更新外循环参数
3. 内循环涉及 "gradient of gradient"（二阶微分），因为外循环梯度需要穿过内循环的梯度步

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实践中：
- JAX/PyTorch 的自动微分原生支持 gradient-of-gradient，无需特殊处理
- 内循环使用 SGD（学习率 η 可学习），外循环使用 AdamW
- 梯度 checkpointing through time 用于节省内循环中间状态 W_1,...,W_T 的内存
- 内循环和外循环共享相同的 backbone 结构（Mamba backbone 或 Transformer backbone）

涉及论文标题：
- Learning to (Learn at Test Time): RNNs with Expressive Hidden States
