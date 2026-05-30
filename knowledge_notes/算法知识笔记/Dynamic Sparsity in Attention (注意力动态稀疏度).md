## Dynamic Sparsity in Attention (注意力动态稀疏度)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Dynamic Sparsity（动态稀疏度）指注意力计算中稀疏度（sparsity ratio / density）不是固定值，而是根据输入内容、序列长度、注意力头特性等因素自适应变化的策略。

在 XAttention 中，动态稀疏度通过 Threshold Block Selection 和 Minimum Threshold Prediction 两个机制共同实现：(1) Threshold-based 选择天然产生动态稀疏度——不同输入产生不同的反对角线分数分布，累计概率超过 τ 所需的 block 数自然不同；(2) 长序列的注意力天然更稀疏（信息分散在更多 token 上），阈值方法自动适配——128k 序列密度 ~6.89%，4k 序列密度 ~52.16%；(3) Per-head threshold optimization 进一步引入头间差异——不同功能头（retrieval head vs. streaming head）天然有不同的稀疏特性。

从算法pipeline角度拆解术语：

```
# 动态稀疏度的自适应行为

# 场景1: 短序列 (4k tokens)
# 注意力相对密集——信息集中，需更多 block 参与
# τ=0.9 时 density ≈ 52%，每个 query block 关注 ~33 个 key blocks

# 场景2: 长序列 (128k tokens)
# 注意力高度稀疏——信息分散，仅少数 block 含有效信息
# τ=0.9 时 density ≈ 6.89%，每个 query block 仅关注 ~141 个 key blocks

# 场景3: Per-head variation
# Head A (retrieval): 关注特定位置，稀疏度高 → τ_A=0.95
# Head B (streaming): 关注连续区域，稀疏度低 → τ_B=0.75
```

与固定稀疏度方法（Top-K: 固定 K 个 block；Top-Ratio: 固定比例）的对比（XAttention Table 8）：固定方法在短序列浪费计算（保留过多 block），在长序列丢失信息（保留不足），且无法适应不同输入内容。动态阈值按 attention mass 保留，自动匹配实际信息分布。

术语一般如何实现？如何使用？

实现方式：(a) 基于累积 softmax 概率的 Threshold Block Selection——无需预设任何稀疏度参数，仅需一个全局阈值 τ；(b) 离线 DP 搜索 per-head τ 值——一次性搜索后保存为配置文件，推理时零额外开销；(c) 也可以采用更简单的方案——所有头使用相同 τ（论文的默认 baseline，τ=0.9）。

与其他方法的动态稀疏度对比：MInference 通过 Kernel-Aware Search 为每个头分配固定稀疏模式但参数固定（k_v, k_s 不变）；FlexPrefill 使用 coverage α 参数控制动态 budget 分配但效果有限；XAttention 的 τ-based 方法是最直接的——累积概率超过阈值即停止，稀疏度完全由输入数据的注意力分布自然决定。

涉及论文标题：
- XAttention: Block Sparse Attention with Antidiagonal Scoring
