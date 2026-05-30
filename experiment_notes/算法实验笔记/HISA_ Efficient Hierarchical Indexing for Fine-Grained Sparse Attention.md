## HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 HISA（Hierarchical Indexed Sparse Attention），一种免训练的即插即用式层级索引器，替代 DeepSeek Sparse Attention (DSA) 中 O(L²) 复杂度的 flat token scan indexer。核心设计为两阶段层级搜索：(1) **Block-level 粗过滤**：将前缀划分为大小为 B 的连续 block，对每个 block 内 indexing keys 做 mean pooling 得到代表向量，query 对所有 ⌈L/B⌉ 个 block 代表向量打分，保留 top-m 个 block；(2) **Token-level 精筛**：在保留的候选 block（最多 mB 个 token）内，使用与原始 DSA 相同的 token-level indexer 打分，选出最终 top-k token。HISA 输出与 DSA indexer 完全相同的数据结构（每个 query 的 k 个 token 索引集），下游 Sparse MLA 算子完全不变。每 query 索引复杂度从 O(L) 降至 O(L/B + mB)，每层从 O(L²) 降至 O(L²/B + LmB)。block size B 控制粗过滤粒度，top-m 控制候选池大小，满足 mB ≥ k 的可行性约束。首尾 block 强制保留以处理 attention sink 和局部上下文。实验比较三种索引策略：(a) DSA (原始 full-prefix token-level indexer)、(b) Block-Sparse (仅 Stage 1，无 token 精筛)、(c) HISA (完整两阶段)。在 kernel-level latency（8K-64K context）、Needle-in-a-Haystack (NIAH, 8K-648K context)、LongBench（6 类任务）上比较，以及 attention score 可视化和超参数敏感性分析。

- 硬件平台是什么，配置是什么。
  Kernel-level latency 测试：单张 NVIDIA A100 GPU，使用 TileLang kernel 实现。End-to-end 评测：vLLM online serving framework，FP8 精度部署 DeepSeek-V3.2 和 GLM-5。NIAH 评测使用基于 RULER (https://github.com/NVIDIA/RULER) 修改的评估代码库。LongBench 评测使用 lm-eval-harness framework (https://github.com/EleutherAI/lm-evaluation-harness)。对于 GLM-5，因 OOM 问题调整了部分任务的 concurrency（longbench_single concurrency=1，longbench_summary concurrency=2），默认 num_concurrent=20。

- 模型是什么。数据集和bench分别是什么。
  模型：DeepSeek-V3.2（采用 DSA + Sparse MLA，MQA mode）、GLM-5（同样采用 DSA 范式）。Benchmark：(1) Needle-in-a-Haystack (NIAH)——8K 至 648K tokens context，needle 深度 0%-100%，评估 retrieval accuracy，基于 RULER 修改的评估代码，不使用 chat template；(2) LongBench——bilingual multi-task long-context understanding benchmark，覆盖 6 类任务：Single-Doc QA (SQA)、Multi-Doc QA (MQA)、Summarization (Sum)、Few-shot Learning (FS)、Synthetic Retrieval (Syn)、Code Completion (Code)。所有评估均为 zero-shot 设置。DeepSeek-V3.2 使用标准 chat template，GLM-5 不使用 chat template。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/MuLabPKU/TransArch（论文声称的仓库地址，但截至当前仓库内 HISA 代码标记为"Release HISA code ☐"尚未发布）。DSA 的参考 TileLang kernel 实现在 https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32。算法 pipeline 如下：

  **HISA 两阶段层级索引（Algorithm 1 核心流程）**：
  ```
  输入: query indexing representations {q_{t,j}^I}, gating weights {w_{t,j}^I},
        token indexing keys {k_s^I}_{s=1}^L, block size B, block budget m, token budget k
  输出: 每 query 的 selected token set T_t (size k)

  // Stage 0: Block 划分与 Pooling
  M = ceil(L / B)
  for b = 1 to M:
      k̃_b^I = MeanPool({k_s^I | s ∈ B_b})

  // 对每个 query position t
  for each query position t:
      // Stage 1: Block-level 粗过滤（公式 5-7）
      for b = 1 to M (B_b causally precedes t):
          J_{t,b} = Σ_j w_{t,j}^I · ReLU(q_{t,j}^I · k̃_b^I)
      C_t = TopK(J_{t,:}, m) ∪ {first block, last block}
      Ω_t = ⋃_{b ∈ C_t} B_b            // 候选 token 集，|Ω_t| ≤ mB

      // Stage 2: Token-level 精筛（公式 8-9，与 DSA 相同机制）
      for s ∈ Ω_t:
          I_{t,s} = Σ_j w_{t,j}^I · ReLU(q_{t,j}^I · k_s^I)
      T_t = TopK({I_{t,s} | s ∈ Ω_t}, k)

  // T_t 送入 Sparse MLA（公式 3，与 DSA 完全相同）
  u_t = Attn(h_t, {c_s | s ∈ T_t})
  ```

  复杂度：per-query O(L/B + mB)，per-layer O(L²/B + LmB)。vs DSA indexer O(L²)。

  三 regime 边界行为：
  - t ≤ k: 等价 dense attention
  - k < t ≤ mB: 等价 DSA（粗过滤器全选）
  - t > mB: HISA 层级优势激活

  默认超参数：B=128, m=64 (candidate=8192), k=2048。也测试 (B=64,m=128) 和 (B=256,m=32)，均保持 mB=8192。Block-Sparse baseline: B=128, m=16 (candidate=2048, 无 token 精筛)。

  **关键性能数据**：
  - 64K context kernel speedup: 2.16× (4:1 ratio) ~ 3.75× (fixed 8K budget) vs DSA indexer
  - NIAH: HISA 接近 DSA 性能，远超 Block-Sparse（Block-Sparse 在 needle 位于中间位置时显著退化）
  - LongBench: DeepSeek-V3.2 Avg: DSA 51.05, HISA 50.78, Block 49.54; GLM-5 Avg: DSA 46.01, HISA 46.32, Block 42.67
  - 超参数敏感性：B=64/128 优于 B=256（更细粒度 block 精筛更准），所有 HISA 配置均远优于 Block-Sparse
