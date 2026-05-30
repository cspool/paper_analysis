## Vertical-Slash Attention Pattern (垂直-斜线注意力模式)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Vertical-Slash（VS）注意力模式是 MInference 论文识别的三种长上下文注意力稀疏模式之一，也是占比最高的模式（>90% 的 attention heads 被分配为 VS 模式）。VS 模式的注意力权重集中在：(1) **垂直列（Vertical lines）**——某些特定 token 被几乎所有位置的 query token 广泛关注（类似 "attention sink" 但不仅限于初始 token，可出现在序列中的任意位置）；(2) **斜线（Slash lines）**——token 以固定间隔关注序列中其他位置的 token，在注意力矩阵上形成对角线/斜线模式，是 RoPE 位置编码在长上下文下的典型表现。

VS 模式的关键特征：(1) **空间分布**：Dynamic structured——垂直列和斜线的**具体位置**随输入内容动态变化，但**模式类型**（即总是垂直+斜线组合）在同一个 head 上保持一致；(2) **GPU 延迟**：Medium——需要混合 block-level（斜线用 64×64 blocks）和 column-level（垂直线用 1×64 columns）两种稀疏格式；(3) **索引构建时间**：Small——仅使用最后 64 个 query 向量做估计，占 5-15% 总时间。

与 A-shape 的关键区别：VS 模式的垂直列可以出现在序列中任意位置（不仅是初始 token），因此能捕获分布在 prompt 中间位置的重要信息（如长文档中间的 key-value pairs、中间章节的主题句等）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Vertical-Slash Head 的动态稀疏索引构建与计算（Algorithm 2）
输入: Q, K, V ∈ R^{S×d_h}, k_v=30, k_s=2000, last_q=64

# Step 1: 估计注意力分布（仅用最后 64 个 query）
Â = softmax(Q[-64:] @ K^T / √d + m_causal)  # [64, S]

# Step 2: 提取垂直列索引（沿 query 维度求和→每列的全局重要性）
score_v = Â.sum(dim=0)                        # [S]
i_v = argtopk(score_v, k_v)                   # [30] — top-30 垂直列

# Step 3: 提取斜线索引（沿斜线方向求和→每条斜线的全局重要性）
score_s = Â 沿斜线方向求和                       # [S] 每条斜线 score
i_s = argtopk(score_s, k_s)                   # [2000] — top-2000 斜线

# Step 4: 构建混合稀疏格式
# 斜线用 64×64 blocks（slashes 在 block level 连续）
# 垂直线用 1×64 columns（vertical 是细粒度列级）
i_vs = PointRangeTwoWayMerge(i_v, i_s, block_size=64)

# Step 5: 稀疏注意力计算
A = softmax(sparse(Q @ K^T, i_vs) / √d)       # 仅计算 i_vs 索引区域
y = sparse(A @ V, i_vs)
```

**具体例子**（LLaMA-3-8B, 128K context, VS head, k_v=30, k_s=2000）：
- 计算量：~30 × 128K (垂直列) + 2000 × 64 × 64 (斜线 blocks) + 64 × 128K (估计)
- FLOPs: ~$2.0 \times 10^9$（vs dense 的 $2.2 \times 10^{11}$）
- 稀疏率: ~99%
- 1M context 下 kernel 级加速：13× vs FlashAttention

术语一般如何实现？如何使用？

实现需要两个定制 GPU kernel：
1. **Vertical-Slash Index Kernel**：使用 point-range two-way merge 算法——垂直列视为 points、斜线转换为每行对应的 column ranges，合并后输出两部分：block indexes（斜线的 64×64 blocks）+ column indexes（垂直的 1×64 columns）。GPU 上按行并行（N = S/B 行），每行时间复杂度 O(k_v + k_s)。

2. **Vertical-Slash FlashAttention Kernel**：混合 kernel——前半部分使用 Block-Sparse FlashAttention 处理斜线 blocks（标准 FlashAttention tiling），后半部分使用 PIT（Permutation Invariant Transformation）将非连续的 column data 加载到 dense compute blocks 处理垂直列。

使用场景：VS 模式是最通用的稀疏注意力模式，适用于绝大多数 attention head。能有效处理 retrieval（垂直列捕获关键 value token）、summarization（斜线捕获周期性结构）、QA（两者结合）等各类任务。需要注意的是 k_v 和 k_s 的配置需要通过 Kernel-Aware Search 离线确定以匹配 target FLOPs。

**Sparse Frontier 论文的补充发现**：VS 模式在 retrieval 任务（Low Scope, Low Dispersion）上表现优异，但需要根据任务类型选择近似窗口大小——retrieval-heavy 任务（Ruler NIAH、Story Retrieval）用 512 tokens 窗口，其他任务用 256 tokens。在 128K tokens 序列上，0.93 sparsity (1/15 budget) 的 VS pattern 仍保持在 Pareto 前沿上。FlexPrefill 在此基础上添加了 threshold-based 动态 budget 分配（由 coverage α 和 min_budget 控制），但论文发现在高压缩比下动态分配无效（回退到 α=0 均匀分配）。

涉及论文标题：
- MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention
- The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs
- XAttention: Block Sparse Attention with Antidiagonal Scoring
