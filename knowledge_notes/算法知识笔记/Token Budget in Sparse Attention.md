## Token Budget in Sparse Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Token Budget（token 预算）是稀疏注意力方法中控制稀疏程度的核心超参数，表示参与 attention 计算的 token 数量上限。在 Quest 中，token budget = K × page_size，其中 K 是选中的关键 page 数量，page_size 是每 page 的 token 数（默认 16）。Token budget 直接决定了 memory load reduction 比例：加载量 = 完整 KV cache 的 (token_budget/seq_len + 1/page_size)。例如 32K context 下 token budget=2048 → 稀疏度 ~93.75%，memory load 减少 ~16×。

Token budget 是 accuracy-efficiency trade-off 的调控旋钮：较小 budget → 更高稀疏度/更快速度 but 可能遗漏关键 token → 精度下降；较大 budget → 更接近 full attention → 精度高 but 加速少。Quest 的实验显示：LongBench 六数据集上 1K budget 即达 full cache 可比性能，PG19 上 4096 budget (~1/8 of 32K) perplexity 与 full cache 几乎一致，Passkey retrieval 中 64-token budget (10K context) 即可 100% 准确。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// Token Budget 在不同稀疏注意力方法中的含义:

// Quest (page-level selection, non-eviction):
K_pages = token_budget / page_size  // e.g. 2048/16 = 128 pages
top_k_indices = TopK(criticality_scores, k=K_pages)
selected_tokens = gather(KV_cache[top_k_indices])  // 最多 token_budget 个 tokens
// KV cache 完整保留，仅本次 attention 不加载所有 tokens

// H2O/TOVA (token-level eviction):
K_tokens = token_budget
keep_indices = TopK(importance_scores, k=K_tokens)
KV_cache = KV_cache[keep_indices]  // 永久驱逐其余 tokens

// 内存加载量对比 (per decode step, per layer per head):
// Quest:   2 × d_head × (seq_len/page_size + token_budget) bytes
// Eviction: 2 × d_head × token_budget bytes
// Full:    2 × d_head × seq_len bytes
// Quest 比 eviction 多 metadata load，但保留了所有 token 的信息
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Token budget 的选择需要根据任务类型和 context length 确定。Quest 论文的指导：(a) 短依赖任务（语言建模）：可使用较大 budget（如 1/8 of context）；(b) 长依赖任务（passkey retrieval）：仅需极小 budget（64-1024 tokens for 10K-100K context）；(c) 通用 long-context benchmark（LongBench）：1K budget 通常足够。实际部署时，token budget 作为 serving 配置参数，根据 latency SLO 和 accuracy 要求动态调整。与 KV cache 量化正交——Quest 兼容 weight quantization（4-bit），两者可叠加。

在 SeerAttention-R 中，token budget 从 token 级别转换为 block 级别：block_budget = token_budget / block_size（block_size=64 为默认）。与 Quest 不同，SeerAttention-R 的 token budget 还对应两种 sparsification 策略对比：(1) Top-K 方法（固定 token budget）：对 AttnGate 输出的块分数排序取 top-k，保证每步计算量可控；(2) Threshold 方法（自适应）：分数超过阈值的块被激活，不同 head/step 可能有不同的稀疏比。Threshold 方法在实现上更简单（无需排序），且在 high sparsity 区域精度略优。SeerAttention-R 实验中 token budget 范围：AIME 用 2k-8k，MATH-500/GPQA 用 1k-6k。

涉及论文标题：
- Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference
- H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models
- SeerAttention-R: Sparse Attention Adaptation for Long Reasoning

---
