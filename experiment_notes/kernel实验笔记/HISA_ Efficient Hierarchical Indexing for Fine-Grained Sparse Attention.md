## HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  使用 TileLang (Wang et al., 2025) 实现了 HISA 两个阶段的 GPU kernel，并对比 DSA 原始 TileLang indexer kernel 的延迟。HISA kernel 分解为两个阶段：(1) Block-level filtering kernel：对 ⌈L/B⌉ 个 pooled block 代表向量做 attention 式打分，选出 top-m blocks；(2) Token-level refinement kernel：仅对候选 block 内的 token（最多 mB 个）做 token-level indexer 计算，选出最终 top-k tokens。实验比较 HISA kernel vs DSA kernel (flat token scan) 在 8K-64K context length 下的 indexer latency，以及两种预算模式：(a) fixed block budget m=64 (B=128)，(b) fixed compression ratio M:m = 4:1。所有 kernel 在单张 NVIDIA A100 GPU 上测试，query lens=1024, k=2048。结果以 indexer kernel level 报告，不包含 Sparse MLA operator 和其他系统组件。

- 后端平台是什么，配置是什么。
  单张 NVIDIA A100 GPU。使用 TileLang (https://github.com/tile-ai/tilelang) 作为 kernel 编程语言。DSA baseline kernel 遵循 TileLang 官方参考实现 (https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32)。配置：query length=1024, final top-k=2048 tokens, block size B=128。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 TileLang 编写自定义 kernel。核心实现/修改：

  1. **Block-level Filtering Kernel**：
  - 输入：pooled block representative keys k̃_b^I ∈ R^{M×d}，query indexing representations q_{t,j}^I ∈ R^{H^I×d}，gating weights w_{t,j}^I
  - 过程：计算 J_{t,b} = Σ_j w_{t,j}^I · ReLU(q_{t,j}^I · k̃_b^I)，即 query 对所有 M = ⌈L/B⌉ 个 block 代表向量的 attention score
  - TopK 选出 m 个 block，同时强制包含首尾 block
  - TileLang tiling：沿 M 维分 tile 做 block-level matmul，M ≪ L（如 64K/B=128 → M=500），计算量远小于 token-level indexer
  - 输出：候选 block 索引集 C_t 和候选 token 集 Ω_t

  2. **Token-level Refinement Kernel**：
  - 输入：仅候选 token 集 Ω_t（≤ mB）的 indexing keys k_s^I，query representations q_{t,j}^I, gating weights w_{t,j}^I
  - 过程：使用与 DSA 相同的 scoring 公式 I_{t,s} = Σ_j w_{t,j}^I · ReLU(q_{t,j}^I · k_s^I)，但仅在 Ω_t 上计算
  - TopK 选出最终 k=2048 tokens
  - TileLang tiling：沿候选 token 维分 tile，|Ω_t| ≤ mB = 8192 (4:1 ratio) 或 2048 (fixed budget)，远小于全前缀 L
  - 在 fixed 8K budget 模式下，第二阶段输入输出长度均固定，计算图更易优化，进一步提速

  3. **与 DSA kernel 的差异**：
  - DSA kernel：对所有 L 个 token 执行一次完整 token-level indexer scan，复杂度 O(L)
  - HISA kernel：先对 M=⌈L/B⌉ 个 block 做轻量粗过滤 (O(L/B))，再对 mB 个候选 token 做精筛 (O(mB))
  - HISA 增加了 block filtering 阶段的开销，但该阶段仅在 pooled 摘要上操作（M ≪ L），代价远小于跳过大量不相关 token 带来的收益

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源地址：https://github.com/MuLabPKU/TransArch（论文声称仓库，HISA 代码标记为"Release HISA code ☐"尚未发布）。DSA 参考 TileLang kernel：https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32。

  **评估原理与 Kernel 执行全流程（以单层 HISA indexer 为例）**：

  ```
  输入: indexing query representations Q^I ∈ R^{H^I × d}（per query position）
        token indexing keys K^I ∈ R^{L × d}
        gating weights w^I ∈ R^{H^I}
        block pooled keys K̃^I ∈ R^{M × d}（M = ceil(L/B)，增量维护）
  输出: selected token indices T（size k=2048）

  Step 1: Block-level Filtering (TileLang kernel)
    // Grid: (H^I_heads, ceil(M / T_M))
    for each indexing head j:
        q_j = Q^I[j, :]                                // [d]
        for each block tile:
            K̃_tile = load K̃^I[tile] from HBM → SRAM    // [T_M, d]
            S_tile = q_j @ K̃_tile^T                     // [T_M], block scores
            // ReLU activation
            S_tile = ReLU(S_tile)
            // Gating weight multiply
            J[:, tile] += w^I[j] * S_tile              // accumulate across heads
    // Global TopK across all blocks
    C = TopK(J, m)                                      // m selected blocks
    C = C ∪ {0, M-1}                                   // force include first/last
    Ω = all token indices in selected blocks

  Step 2: Token-level Refinement (TileLang kernel)
    // Grid: (H^I_heads, ceil(|Ω| / T_tok))
    for each indexing head j:
        q_j = Q^I[j, :]
        for each candidate token tile in Ω:
            K_tile = gather_and_load K^I[Ω_tile]        // [T_tok, d], selective load
            S_tile = q_j @ K_tile^T                     // [T_tok]
            S_tile = ReLU(S_tile)
            I[:, Ω_tile] += w^I[j] * S_tile            // accumulate across heads
    // Global TopK across candidate tokens
    T = TopK(I, k)                                      // k=2048 final tokens

  Step 3: Sparse MLA（与 DSA 完全相同，不修改）
    u_t = SparseMLA(h_t, {c_s | s ∈ T})
  ```

  **评估指标与原理**：
  - Indexer kernel latency (ms)：纯 indexer 阶段的 wall-clock time
  - Speedup = latency_DSA / latency_HISA
  - 两种预算模式：
    - Fixed top-m=64 (B=128): 随 seq_len 增长，candidate pool 从 ~完全覆盖 变为 ~部分覆盖
    - Fixed compression ratio M:m=4:1: 随 seq_len 增长，m 自适应增长以保持恒定压缩比

  **关键性能数据（A100, query lens=1024, k=2048）**：
  | Context Len | DSA Indexer | HISA (4:1 ratio) | HISA (fixed 8K budget) |
  |-------------|-------------|-------------------|------------------------|
  | 8K          | ~0.7 ms     | ~0.5 ms           | ~0.5 ms                |
  | 64K         | ~5.6 ms     | ~2.6 ms (2.16×)   | ~1.5 ms (3.75×)        |

  Sparse MLA operator 自身约 1.6 ms（与 seq_len 无关），表明 DSA 的瓶颈在 indexer 而非 Sparse MLA。

  **Block filtering 开销分析**：HISA 增加了 block filtering 阶段，但该阶段仅在 M = ceil(L/B) 个 pooled 摘要上操作。以 64K context 为例：M = 64K/128 = 500，远小于 L = 64K。Block filtering 的额外开销约 0.2 ms，但节省了跳过 ~56K 个 token 的 token-level 计算，净收益显著。
