# <span id="page-23-1"></span>C Sparse Kinetics

We present additional results supporting the Sparse Kinetics across multiple tasks and demonstrate how these insights enable scalable test-time scaling with sparse attention.

### <span id="page-23-2"></span>C.1 Additional Benchmarks

Beyond AIME24, we evaluate our approach on LiveCodeBench [\(Jain et al.,](#page-14-9) [2024\)](#page-14-9) and AIME25 [\(MAA,](#page-15-7) [2025\)](#page-15-7). LiveCodeBench features complex programming problems from recent coding contests, while AIME25 consists of challenging math problems. In both cases, sparse attention—particularly oracle top-k—consistently outperforms dense attention. Block top-k attention, a tractable alternative, closely matches the performance of the oracle.

For LiveCodeBench, we sample 50 problems from the v5 subset (24 hard, 16 medium, 10 easy). As shown in Figure [21,](#page-24-0) oracle top-k attention can achieve ∼ 10× speedup in high-accuracy regimes and improves coverage by 40–50% in low-cost regimes. Conversely, the tractable alternative, Block top-k yields 5–6× speedup and 30–40% coverage gains. We further show how the benefits of sparse attention scale with problem difficulty (Figures [21g](#page-24-0) to [21i\)](#page-24-0). Figure [22](#page-25-1) confirms similar trends for AIME25, with substantial gains in both accuracy and efficiency under sparse attention.

We present the evaluations of MoE models (Qwen3-30B-A3B) in Figures [23a](#page-26-1) and [23b,](#page-26-1) where we draw consistent conclusions on the scalability of sparse attention.

### <span id="page-23-3"></span>C.2 Additional Analysis

Fixing a model (e.g., Qwen3-8B), we investigate the tradeoff between generating more tokens through Best-of-N and increasing the KV budget in Figures [24a](#page-26-2) to [24d.](#page-26-2) As the figures suggest, on AIME25, each doubling of total compute cost increases the optimal KV budget by 1.13×, while generated tokens grow by 1.67×; on LiveCodeBench, these factors are 1.14× and 1.89×, respectively. We find that although the concrete numbers depend on the types of tasks, the overall results confirm our suggestions in the main paper that allocating compute toward generating more responses is generally more effective than expanding KV budget, highlighting the scalability of sparse attention.

