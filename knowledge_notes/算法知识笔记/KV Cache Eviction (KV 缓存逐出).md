## KV Cache Eviction (KV 缓存逐出)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

KV Cache Eviction 是一类 KV cache 压缩方法，通过选择性地逐出不重要 token 的 KV cache pairs 来减少显存占用，无需额外训练或微调。与量化（减少每个值的 bit 宽度）和低秩分解（从通道维度压缩）不同，eviction 方法的压缩来自减少保留的 token 数量。

Eviction 方法的核心在于两个决策：(1) Token 选择——如何判断哪些 token 的 KV pairs 重要/不重要；(2) 预算分配——每层/每 head 保留多少 token。Eviction 方法按评分依赖可分为两大类：(a) **Query-Aware**——重要性评分依赖当前 query 信息（如 SnapKV 的 observation window），压缩 cache 对初始 query 过拟合，多查询复用性能退化；(b) **Query-Agnostic**——评分仅依赖 context 自身（如 KVzip 的上下文重建），压缩后的 cache 可跨任意 query 复用。

所有 eviction 在 prefill 阶段完成后、decoding 阶段开始前执行。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**KV Cache Eviction 通用流程**：

```
# === Prefill 阶段 ===
Q, K, V = X @ W_Q, X @ W_K, X @ W_V
O_full = FlashAttention(Q, K, V)
K_cache = K; V_cache = V

# === Eviction 阶段（prefill 后） ===
importance_scores = compute_importance(Q, K, V)  # 各方法核心差异
budget = allocate_budget(layer_id, total_budget)  # 各方法核心差异
keep_indices = topk(importance_scores, budget)
K_cache = K_cache[keep_indices]
V_cache = V_cache[keep_indices]

# === Decoding 阶段 ===
for each new token:
    Q_new, K_new, V_new = x_new @ W_Q, x_new @ W_K, x_new @ W_V
    O_new = FlashAttention(Q_new, cat([K_cache, K_new]),
                           cat([V_cache, V_new]))
    K_cache.append(K_new); V_cache.append(V_new)
```

**各 Eviction 方法的 Token 重要性评估方式对比**：

| 方法 | 重要性评估 | Query依赖 | 预算分配 |
|------|-----------|----------|---------|
| StreamingLLM | 仅首+尾 token | Query-Agnostic | Uniform |
| H2O | 累积 attention 分数 | Query-Aware | Uniform |
| SnapKV | observation window attention clustering | Query-Aware | Uniform |
| PyramidKV | SnapKV attention clustering | Query-Aware | 金字塔形 |
| KVzip | 上下文重建 cross-attention max score | Query-Agnostic | Non-uniform head-budget |
| CAKE | attention entropy + variance + SnapKV | Query-Aware | attention 统计量 |
| CompressKV | SRH attention aggregation + pooling | Query-Aware | Error-aware (layer-level) |
| CoKV | SnapKV attention pooling | Query-Aware | Head-level Shapley |
| GemFilter | Filter layer last-query-key scores + pooling | Query-Aware | 单索引集（全局 uniform） |
| LOOK-M | 累积 attention + Text-Prior (文本优先) + KV pairs merging | Query-Aware (prefill attention) | Uniform (M+N recent+important) |
| StreamingLLM | Attention sink + sliding window (位置固定) | Query-Agnostic | Uniform |
| LaCache | Ladder-shaped cross-layer pattern (位置固定) | Query-Agnostic | Ladder (跨层错位) |

| TreeKV | 循环淘汰范围内相邻两 token 的平均 attention weight 比较 | Query-Aware (但淘汰范围均匀分布) | Uniform |
| LagKV | Channel-wise std after lag-normalize of K+V | Query-Agnostic | Uniform |
| LOCRET | Trained retaining head MLP predicts CIS | Causal (neither QA nor QG) | Uniform per-head |

注0：LOCRET 的评分既非 Query-Aware 也非 Query-Agnostic——它是 **causal** 的：CIS 仅依赖当前及之前 token 的 [Q, K, V]，不需要任何 query（prefill 阶段）也不需要完整序列。训练时 CIS 基于 answer tokens 的 attention score 为 ground truth，但推理时 retaining head 仅需 local context。LOCRET 是首个将 **trained** eviction scoring 与 chunked prefill 结合的方法，其 retaining head 开销 < 2% inference time。

注1：GemFilter 与传统 eviction 方法有本质区别——它在 prompt computation 阶段仅运行前 r 层处理全部 n 个 token（而非全部 m 层），因此 prompt 计算量从 Θ(mhn²d) 降至 Θ(rhn²d)。其余方法在 prompt computation 阶段仍处理全部 m 层。
注2：KVzip 是首个明确提出 query-agnostic 作为核心贡献的 eviction 方法。其重要性评分通过让 LLM 模拟重建原始上下文（"Repeat the previous context:" prompt + context forward pass），取每个 KV pair 在重建过程中收到的最大 cross-attention score。该评分与下游任务的 attention 模式高度重叠（Figure 6），证明重建关键 KV pairs 对各任务均重要。此外 KVzip 还支持 context-independent 模式：预计算 static head-level score，部署时零开销。

术语一般如何实现？如何使用？

Eviction 方法无需训练或微调，以即插即用方式集成到 HuggingFace Transformers 推理 pipeline——在每层 attention 计算后添加 eviction 步骤。所有 eviction 方法与 FlashAttention 兼容。CompressKV 额外包含一个自定义 CUDA kernel (`adakv`)。KVzip 通过 chunked scoring（m=2K chunk）将评分复杂度从 O(n_c²) 降至 O(m·n_c)，压缩开销约 2x prefill（仅执行一次）。Eviction 的压缩率取决于保留 token 数：如在 128K context 下仅保留 256 tokens（0.07% 容量），NIAH 准确率仍可达 90%（CompressKV 结果）。KVzip 在 30% budget（淘汰 70%）下保持接近无损性能，结合 4-bit 量化可将 16-bit 124K-token cache 从 16.3GB 降至 1.2GB。代码开源：https://github.com/TUDa-HWAI/CompressKV.git 和 https://github.com/snu-mllab/KVzip。

涉及论文标题：
- SnapKV: LLM Knows What You are Looking for Before Generation
- A2SF: Accumulative Attention Scoring with Forgetting Factor for Token Pruning in Transformer Decoder
- Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs
- Discovering the Gems in Early Layers: Accelerating Long-Context LLMs with 1000x Input Token Reduction
- Exploiting Sparsity for Long Context Inference: Million Token Contexts on Commodity GPUs
- KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction
- LOOK-M: Look-Once Optimization in KV Cache for Efficient Multimodal Long-Context Inference
- LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models
- LagKV: Lag-Relative Information of the KV Cache Tells Which Tokens Are Important
- LOCRET: Enhancing Eviction in Long-Context LLM Inference with Trained Retaining Heads on Consumer-Grade Devices
- CompressKV: Semantic Retrieval Heads Know What Tokens are Not Important Before Generation
- TriAttention: Efficient Long Reasoning with Trigonometric KV Compression （TriAttention 提出 pre-RoPE eviction：评分不依赖 attention scores 而依赖 Q/K 中心的三角函数级数预测距离偏好 + 自适应范数补充。区分于所有 post-RoPE 方法——TriAttention 回到 pre-RoPE 空间，利用 Q/K Concentration 跨位置稳定性绕过观察窗口限制。校准一次离线完成，推理时无需计算 attention。）
- WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference （WindowKV 提出 window 级 eviction——将 context 切分为 review windows，以 window 为粒度选择保留，而非逐 token eviction。引入任务自适应分类器决定每窗口内保留的 token 比例，解决 token 级 eviction 破坏语义连贯性的问题。同时提出 intra-group layer KV cache indices sharing 减少 window selection 开销。）

注：LOCRET 引入训练式 causal eviction——不同于所有其他方法依赖启发式统计量（attention weight/entropy/channel std），LOCRET 使用小型训练 MLP（retaining head）预测 Causal Importance Score (CIS)。训练开销 < 1 GPU hour，保留全模型权重冻结。CIS 为每个 (layer, head, token) 三元组独立打分，eviction 跨 head 独立进行。支持 MHA 和 GQA 架构。在 NVIDIA 4090 消费级 GPU 上实现 128K+ 长上下文推理，压缩比 up to 20×（<10% 性能损失），10M token 上下文评估（1747.6× 压缩比）达 100% 准确率。

注：LOOK-M 是首个专门针对多模态长上下文场景的 KV cache eviction 方法。其核心创新在于：(a) Text-Prior——在累积 attention score 基础上为文本 token 显式加 T_p = Max(A_s) 确保文本 KV pair 优先保留；(b) 对被 evicted 的 KV pair 不直接丢弃，而是通过 nearest-neighbor matching + 三种合并策略（averaged/pivotal/weighted）将其信息融入 conserved token。在多模态场景下，该 text-prior + merging 组合在 Needle-in-a-Haystack 任务上显著超越纯文本 eviction 方法（H2O/SnapKV/RoCo）。

注：LaCache 是首个提出跨层异质 KV 存储的 eviction 方法。其两点创新：(a) Ladder-Shaped Pattern——不同层缓存不同位置 token 的 KV cache（浅层存储早期 token、深层存储近期 token），在相同总 budget 下覆盖更长上下文。该 pattern 通过两个超参数控制——Span S（同一 token 被保留的连续层数）和 Overlap O（每层保留的 token 数），且经消融验证位于 PPL-cache size Pareto 最优边界（1500+ 随机 pattern 对比）；(b) Iterative Compaction——ladder pattern 可递归应用于已压缩 cache，实现 O(1) 内存复杂度的无限连续生成。LaCache 的一个重要设计决策是故意不使用 attention maps 进行 token 重要性评估（与 H2O/SnapKV 不同），因此与 FlashAttention 天然兼容，在 H200 实测中取得 score-throughput Pareto 最优。在 PG19 数据集上支持 10M+ token 连续生成（Full KV 在 160K token 即 OOM），NIAH 50% budget 下准确率 99.16% vs StreamingLLM 54.54%。

注：Exploiting Sparsity 论文提供了一种区别于 eviction 的方案——不做 token eviction，而是将完整 KV cache 存放在 CPU 内存中，通过 Faiss ANN search 在每次 decoding step 动态检索 top-k 个最相关的 KV pair 传输到 GPU。该方法在 1M token NIAH 测试中 k=1 即可 100% 成功，而 StreamingLLM（eviction-based）完全失败（被 evict 的 token 无法恢复）。这说明对于某些长上下文任务，per-query sparse retrieval 从根本上优于 static eviction。

注：Quest（ICML 2024）提供了另一种非 eviction 方法——完整保留 KV cache，不做任何 token 驱逐，而是在每步 decode 时基于当前 query 动态选择关键 page 加载到 attention。相比 eviction 方法的核心优势：(a) 不丢失信息——所有 token 始终保留，未来 query 始终可以访问任意 token；(b) query-aware 选择——同一 token 对 query="is" 关键对 query="D" 不关键（Fig. 2），eviction 无法处理这种动态性；(c) Passkey retrieval 证明——H2O/TOVA/StreamingLLM 在 10K-100K 下准确率 0-4%，Quest 64-1024 token budget 即 100%（Tab. 1）。代价是需额外存储 per-page metadata（~12.5% of KV cache）且不减少显存占用（仅减少 memory movement）。

涉及论文标题：
- Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference

---
