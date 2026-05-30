## SnapKV (基于注意力重要性的KV选择)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

SnapKV（Li et al., 2024）是一种静态 KV cache 压缩算法，在 prefill 阶段一次性选择重要的 KV pairs 供后续所有 decode step 复用。核心思想：利用 prefill 最后一层的 attention weight 来评估每个 KV pair 的重要性——一个 KV pair 如果被最后一层的 query 赋予高 attention score，说明它对后续生成也很重要。SnapKV 的处理流程：(1) 在 prefill 阶段计算最后一层的 attention weights；(2) 对 attention weights 沿 sequence length 维度做 average pooling（kernel_size=5）以平滑噪声；(3) 强制保留 observation window（最近 32 个 token 的 KV）；(4) 在剩余位置中选 Top-K 最高 attention score 的位置；(5) 将所有层的 KV cache 按选中的位置索引 gather 到压缩 KV cache 中。在 decode 阶段，draft model 仅对压缩 KV cache 做注意力计算。

在 MagicDec 论文中，SnapKV 被用作 self-speculation 的 draft KV 压缩算法，与 StreamingLLM 对比。SnapKV 的接受率远高于 StreamingLLM（Figure 4c），原因是 SnapKV 基于最后一层真实 attention 做重要性选择，而非 StreamingLLM 的固定 sink+window 策略。SnapKV 最佳 KV budget 为 2049（vs StreamingLLM 的 512），更大的 budget 带来更高接受率（>85%），且作为静态方法无搜索开销，使 SD speedup 达到 2.51x。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# SnapKV 的 KV 选择流程（prefill 阶段，一次完成）
输入: K_full, V_full ∈ [B, S, n_layers, n_heads, d_head]
       最后一层的 attention weights W_last ∈ [B, n_heads, S, S]

# Step 1: 沿序列维度 average pooling（平滑噪声）
W_pooled = AvgPool1d(W_last, kernel_size=5)  # [B, n_heads, S, S]

# Step 2: 取最后一行（最后一个 query 对所有 key 的 attention）
importance = W_pooled[:, :, -1, :]  # [B, n_heads, S]

# Step 3: 跨 head 聚合重要性（sum 或 max）
importance_agg = importance.sum(dim=1)  # [B, S]

# Step 4: 保留 observation window + Top-K 其余位置
obs_window = 32
obs_indices = [S-obs_window : S]  # 最近 32 token 必须保留
remaining = importance_agg[:, :S-obs_window]
top_k_indices = TopK(remaining, K - obs_window)

# Step 5: 合并索引并 gather 压缩 KV
draft_indices = sort(obs_indices ∪ top_k_indices)  # |draft_indices| = K
K_draft = gather(K_full, draft_indices, dim=2)  # [B, K, n_layers, n_heads, d_head]
V_draft = gather(V_full, draft_indices, dim=2)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

SnapKV 开源：https://github.com/FasterDecoding/SnapKV。实现依赖 HuggingFace Transformers，在模型 forward pass 中插入 KV selection 逻辑。使用方式：设置 `window_size=32`（observation window）、`kernel_size=5`（pooling 核大小）、`max_capacity_prompt=K`（目标 KV budget）。适用于长上下文 LLM 推理（32K+ tokens），尤其适合 retrieval 和 QA 任务（因基于真实 attention score，能保留关键信息 token）。MagicDec 中使用 SnapKV 作为 static KV selection 的推荐方案——接受率高且无 decode 阶段搜索开销。

**原始 SnapKV 论文的核心机制**（Li et al., 2024, NeurIPS）：

SnapKV 的核心创新在于两个关键发现：(1) prompt 末尾一个 "observation window" 内的 queries 对 prefix keys 的注意力分配模式与生成阶段高度一致（Fig. 2，overlap rates 验证）；(2) 这一注意力模式在生成过程中保持稳定（Fig. 3）。基于此，SnapKV 无需依赖最后一层特殊处理——每层独立使用自身的 observation window queries 计算对 prefix keys 的注意力权重，沿 query 维度求和得到投票分数 C_h = Σ_i W_obs[h, i, :]，再通过 1D pooling（kernel_size 可配）聚类保留上下文完整性，最后 TopK 选择保留的 prefix KV 位置。

与后续工作（MagicDec/R-KV）中描述的关键差异：(a) 原始 SnapKV 每层独立投票，而非仅依赖最后一层；(b) observation window 通常使用 prompt 末尾的直接 query tokens，而非专门训练的 head；(c) 投票后保留 observation window 的完整 KV（不做压缩），仅压缩 prefix 部分。

**SnapKV 压缩流程（原始论文全栈）**：
```
# Prefill 阶段，每层 attention 计算完成后：
Q_obs = Q[:, :, -L_obs:, :]                    # observation window queries
attn_weights = Q_obs @ K_prefix^T / sqrt(d)     # [H, L_obs, L_prefix]
vote = attn_weights.sum(dim=-2)                 # [H, L_prefix] 沿 query 维求和
pool_vote = MaxPool1d(vote, kernel_size, stride=1, padding=k//2)
k = max_capacity - L_obs
indices = TopK(pool_vote, k, dim=-1)            # 每 head 独立选 TopK
K_compress = K_prefix.gather(indices)            # 压缩后的 prefix KV
V_compress = V_prefix.gather(indices)
K_new = cat([K_compress, K_obs])                # 拼接完整 observation window
V_new = cat([V_compress, V_obs])
# Decode 阶段：直接使用 K_new, V_new，KV cache 大小恒定
```

涉及论文标题：
- SnapKV: LLM Knows What You are Looking for Before Generation
- MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding
- R-KV: Redundancy-aware KV Cache Compression for Training-Free Reasoning Models Acceleration
- The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs
- ZSMerge: Zero-Shot KV Cache Compression for Memory-Efficient Long-Context LLMs

**Sparse Frontier 论文中的 SnapKV 实现与评估**：使用 kernel_size=21（1D average pooling）平滑，observation window=128 tokens（原论文为 32），始终保留前 4 个 prefix token。近似窗口为 256 tokens（无显著任务依赖）。Paper 还评估了 Ada-SnapKV 变体——使用 max-aggregation（而非 mean）跨 query positions 和 heads 进行分数计算，结合动态 token budget 分配（每 head 最低 budget 20%）。在全量评测中，eviction-based 解码方法（SnapKV/Ada-SnapKV）在高稀疏度下普遍弱于 Quest 的 full-cache 方法，但 Ada-SnapKV（adaptive budget）始终优于 uniform SnapKV，尤其 multi-query 任务。

**R-KV 对 SnapKV 的扩展与改进**：R-KV 将 SnapKV 从 prefilling 阶段移植到 decoding 阶段，每 B_buffer=128 步触发一次压缩（而非仅 prefill 阶段一次性选择）。关键改进：(1) GQA 聚合方式从 mean-pooling 改为 max-pooling——R-KV 实验发现 max-pooling 更好地保留每个 query head 中最关键的 attention 信号；(2) 除 importance scoring 外额外引入 redundancy estimation（key vector 余弦相似度），通过 joint selection score Z = λ·I − (1−λ)·R 同时平衡重要性和去冗余性，解决 SnapKV 在推理模型（DeepSeek-R1）长 CoT 输出中因重复内容获得高 attention 而 over-retain 冗余 token 的问题。R-KV 使用的 λ=0.1 使 redundancy 项权重 (1−λ)=0.9 足以有效抑制冗余。在 AIME24 上，R-KV 以 10% KV cache budget 达到 lossless 压缩（SnapKV 同 budget 仅 ~20% pass@1）。
