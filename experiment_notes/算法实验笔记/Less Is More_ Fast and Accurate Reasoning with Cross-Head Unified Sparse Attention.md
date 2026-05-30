## Less Is More: Fast and Accurate Reasoning with Cross-Head Unified Sparse Attention

- 属于算法pipeline的实现是什么？实验比较什么？
  LessIsMore 是一种免训练的稀疏注意力机制，专为长程推理设计。核心发现：推理模型中 token 重要性是全局属性而非 head-local 属性——(1) 跨 head 空间局部性：不同 attention head 的 token 重要性高度重叠；(2) 时间近邻局部性：最近 token 的高 attention 比例在 decoding 全程稳定。基于此提出 Cross-Head Unified Sparse Attention (CUSA)：各 head 独立提案 top-k token，通过 UnionFlatten 聚合为统一候选集后全局排名，保留 top K·(1-r) token 并固定比例 r=0.25 分配给近邻窗口。Token 选择仅在一层执行（token selection layer），产生的统一索引 ρ 跨后续所有层复用，摊销选择开销。实验比较 LessIsMore vs 免训练方法（TidalDecode、Quest、StreamingLLM）和需训练方法（SeerAttention-r）在 AIME-24/25、GPQA-Diamond、MATH500 上的推理准确率与生成长度，以及 LongBench 和 Needle-in-the-Haystack 上的长上下文能力。

- 硬件平台是什么，配置是什么。
  准确率评估：NVIDIA RTX A5000 GPU（HuggingFace 实现，32K token 生成需 >20 分钟）。效率评估：单张 NVIDIA A100 80GB GPU（DeepSeek-R1-Distill-Llama-8B，端到端 TBT 和 kernel 级延迟）。Serving 集成：单张 NVIDIA A5000 GPU（SGLang + FlashInfer）。

- 模型是什么。数据集和bench分别是什么。
  模型：DeepSeek-R1-Distill-Llama-8B（GQA）、Qwen3-4B/8B/14B（GQA）——四个推理模型；LongChat-7B-v1.5-32k（MHA）——长上下文模型；Llama-3-8B-Instruct-Gradient-1048k、Llama-3.1-8B-Instruct（non-reasoning 模型）。Benchmark：AIME-24/25（64 traces/problem）、MATH500（8 traces）、GPQA-Diamond（16 traces）、Needle-in-the-Haystack（10K/32K/100K）、LongBench（MultiFieldQA/Qasper/HotpotQA/TriviaQA/PassageRetrieval-en）。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/DerrickYLJ/LessIsMore

  **三种层类型**（Algorithm 1）：
  1. Full Attention Layers（Layer 0-1）：o = FullAttention(q, C[:])，确保早期上下文建模准确。
  2. Token Selection Layer（如 Layer 12）：计算 P = q·C.K^T，各 head 独立 TopK → 跨 head 统一聚合：
     ρ_head = TopKIndices(P[:, :-(K·r)], k=K·(1-r))
     ρ_unified = UnionFlatten(ρ_head)
     ρ = ρ_unified[:K·(1-r)] ∪ Recent(K·r)
  3. Sparse Attention Layers（其余层）：o = SparseAttention(q, C[ρ])，复用 token selection layer 生成的统一索引 ρ。

  **GQA 下的 CUSA 张量计算**（以 DeepSeek-R1-8B 为例，hq=32, hkv=8, r=4）：
  1. Q = hW_Q (32 query heads), K = hW_K (8 KV heads), V = hW_V (8 KV heads)
  2. 对每个 KV group g（4 query heads 共享 1 KV head）：
     P_g = Q[4g:4(g+1)] @ K_g^T / √d     # [4, 1, L_kv]
     按 query 维度 max pooling：P_g_agg = max(P_g, dim=0)
     每个 query head 独立 TopK：idx_h = TopKIndices(P_g_agg[h], k=K·(1-r))  for h=1..4
     (GQA 下 query head 独立选择但共享 KV head，这些 head 选择通过后续 UnionFlatten 跨 head 统一)
  3. 全局统一：idx_all = unique(flatten([idx_h for all 32 heads]))
     按 attention score 排序取 top：idx_hist = sort_by_score(idx_all)[:K·(1-r)]
  4. 近邻窗口：idx_recent = [L_kv-K·r, ..., L_kv-1]
  5. ρ = idx_hist ∪ idx_recent（所有 32 query heads 共享同一 ρ）

  **低频重选验证**（图 4）：LessIsMore 仅 Layer 2 选择 vs 每层都选，attention recall 几乎相同（~95% vs ~96%），而 head-to-head / randomized top-k 的方法从 ~96% 降至 ~65%/72%。因为 CUSA 的全局 token 重要性是跨层稳定的。

  **关键效果**：AIME-24 上 Qwen3-8B 以 2K token budget（87.5% sparsity）达 73.8% 准确率 vs Full Attention 74.5%，生成长度 15.8K vs Full Attention 14.8K——几乎无长度膨胀。SeerAttention-r 需 2K budget 仅达 58.2% 且生成 19.8K token。
