## Threshold Block Selection (阈值块选择)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Threshold Block Selection（阈值块选择）是 XAttention 论文提出的动态 block 选择算法，用于基于反对角线评分的 block-sparse attention。与 Top-K（固定保留 K 个 block）和 Top-Ratio（固定保留比例）不同，Threshold Block Selection 通过累积 softmax 概率阈值 τ 来自适应地决定选中的 block 数量——公式为：

$$\text{find\_blocks}(A, \tau) = \arg\min_{\mathcal{B}} \left\{ |\mathcal{B}| \ \Big| \ \sum_{b \in \mathcal{B}} \sum_{(i,j) \in b} A_{i,j} \geq \tau \right\}$$

其中 A 是近似注意力矩阵（由 Antidiagonal Scoring 产生），$\mathcal{B}$ 是选中的 block 集合。

从kernel调度角度拆解术语：

```
# Threshold Block Selection Kernel 伪代码
Input: antidiagonal scores ∈ R^{N_B}（每 block 一个分数）
       threshold τ ∈ [0, 1]
Output: selected_blocks list

# Step 1: Softmax normalize scores to probability distribution
probs = softmax(scores)  # [N_B], sum = 1.0

# Step 2: Sort blocks by descending probability
sorted_blocks = argsort(probs, descending=True)

# Step 3: Greedy cumulative accumulation
cumsum = 0
selected = []
for b in sorted_blocks:
    cumsum += probs[b]
    selected.append(b)
    if cumsum >= τ:
        break

# Step 4: Build sparse mask from selected blocks
# M[query_block, key_block] = 1 if key_block in selected else 0
# Only compute attention for M=1 positions
```

关键优势：自适应稀疏度。短序列时注意力密集（需保留更多 block，如 4k 序列密度 ~52%），长序列自动提高稀疏度（128k 序列密度 ~6.89%）。

与 Top-K/Top-Ratio 的消融对比（Table 8）：Top-K 和 Top-Ratio 无法适应不同输入序列长度——固定 K 或 ratio 在短序列浪费计算、在长序列丢失关键 block。Threshold-based 方法按累积概率自适应决定，在所有序列长度上取得最优准确率-计算平衡。

术语一般如何实现？如何使用？

Threshold Block Selection 在 GPU 上实现为轻量级 kernel：(1) 对 N_B 个 block scores 执行 softmax（parallel reduction via warp-level shuffle）；(2) 使用 bitonic sort 或 radix sort 按分数降序排列 block indices；(3) 对排序后概率执行 prefix sum（parallel scan），找到第一个 cumsum ≥ τ 的位置，保留该位置及之前的所有 blocks。整体 kernel 开销极小，远低于 MInference 的 vertical-slash index search（实测快 24.9×）。

τ 的选择：论文推荐 τ=0.9 作为默认值。更低的 τ（如 0.8，通过 Minimum Threshold Prediction 自动搜索得出）可进一步降低密度同时保持或提升准确率。

涉及论文标题：
- XAttention: Block Sparse Attention with Antidiagonal Scoring
