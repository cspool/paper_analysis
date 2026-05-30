## Hierarchical Indexed Sparse Attention (HISA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

HISA 是一种免训练的即插即用式层级索引策略，用于替代 DSA 中 O(L²) 复杂度的 flat token scan indexer。HISA 将索引搜索路径从"全前缀逐 token 扫描"改写为"block 粗过滤 → token 精筛"两阶段层级过程：

- **Stage 1 (Block-level Coarse Filtering)**：将前缀分为 M = ⌈L/B⌉ 个连续 block，每 block 用 mean pooling 生成代表向量 k̃_b^I。Query 对所有 M 个 block 代表打分，选 top-m blocks。复杂度 O(L/B)。
- **Stage 2 (Token-level Refinement)**：在候选 block 内的至多 mB 个 token 上，使用与 DSA 相同的 token-level scoring 公式逐 token 打分，选最终 top-k tokens。复杂度 O(mB)。

总复杂度 per-query: O(L/B + mB)，per-layer: O(L²/B + LmB)。当 m ≪ M 且 B ≪ L 时（超长上下文 + 选择性粗过滤），缩减显著。

HISA 的三个关键设计决策：(1) **首尾 block 强制保留**：处理 attention sink 和局部上下文；(2) **候选池过采样**：mB > k（如 mB=8192, k=2048），保证精筛质量；(3) **输出同构**：T_t 格式与 DSA indexer 完全相同，Sparse MLA 无需任何修改。

从算法pipeline角度拆解术语。

**HISA 两阶段层级索引（Algorithm 1）**：

```
输入: indexing queries q_{t,j}^I, gating weights w_{t,j}^I,
      token indexing keys {k_s^I}_{s=1}^L, block size B, block budget m, token budget k

// Stage 0: Block 划分与 Pooling（增量维护在 KV cache 旁）
M = ceil(L / B)
for b = 1 to M:
    k̃_b^I = MeanPool({k_s^I | s ∈ B_b})

// 对每个 query position t
for t = 1 to L:
    // Stage 1: Block-level 粗过滤
    for b = 1 to M such that B_b precedes t:     // causal
        J_{t,b} = Σ_j w_{t,j}^I · ReLU(q_{t,j}^I · k̃_b^I)
    C_t = TopK(J_{t,:}, m) ∪ {first block, last block}
    Ω_t = ∪_{b ∈ C_t} B_b                         // |Ω_t| ≤ mB

    // Stage 2: Token-level 精筛（与 DSA 公式(1)相同）
    for s ∈ Ω_t:
        I_{t,s} = Σ_j w_{t,j}^I · ReLU(q_{t,j}^I · k_s^I)
    T_t^H = TopK({I_{t,s} | s ∈ Ω_t}, k)          // 同构输出

// T_t^H 直接送入 Sparse MLA（与 DSA 完全相同）
```

**三 regime 边界行为**：
- t ≤ k: 等价 dense attention（全选）
- k < t ≤ mB: 等价 DSA（粗过滤全选，精筛至 k）
- t > mB: HISA 层级优势激活（非平凡 block 剪枝）

默认超参数：B=128, m=64 (candidate 8192), k=2048。64K context 下 kernel 加速 2.16×-3.75× vs DSA indexer。

术语一般如何实现？如何使用？

HISA 作为 DSA indexer 的 drop-in replacement，直接替换 DeepSeek-V3.2 和 GLM-5 的 indexer 模块：
1. Block pooled keys 增量维护在 KV cache 旁，额外开销可忽略
2. Stage 1 在 TileLang 上实现为 block-level matmul kernel（M ≪ L）
3. Stage 2 在 TileLang 上实现为 token-level matmul kernel（仅在 Ω_t 上）
4. 输出 T_t^H 送入原始的 Sparse MLA 算子

代码仓库：https://github.com/MuLabPKU/TransArch（截至当前 HISA 代码标记为待发布）。

涉及论文标题：
- HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention
