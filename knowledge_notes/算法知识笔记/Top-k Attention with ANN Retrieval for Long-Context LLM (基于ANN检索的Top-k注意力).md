## Top-k Attention with ANN Retrieval for Long-Context LLM (基于ANN检索的Top-k注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Top-k Attention with ANN Retrieval 是一种将标准 dense attention 替换为仅对 top-k 个最相关 key-value pair 进行注意力计算的方法，通过 Approximate Nearest Neighbor (ANN) search 在 CPU 端的完整 KV cache 中检索与当前 query 最相关的 k 个 key，仅传输对应的 value 到 GPU 参与 attention 计算。核心逻辑链：(1) Attention scores 天然具有稀疏性——现代 LLM 中仅极少数 token 贡献了绝大多数 attention mass（图 3: 深层 layer 仅需少数 token 覆盖 75% 注意力质量）；(2) inner product attention score q·K^T 可直接用作向量相似度度量（dot product metric），因此 ANN search 可以代理 attention score computation；(3) 将完整 KV cache 存放在 CPU 内存中（便宜且充裕），仅将 k 个 value 向量从 CPU 传输到 GPU，将 GPU attention 计算的复杂度从 O(N·D) 降至 O(k·D)；(4) k 可以极小——2% of N 足以恢复 95% dense attention 性能，k=0.001% 即可完成 Needle In A Haystack 任务。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Top-k Attention Decoding Pipeline（Llama-3-8B, 1M token context, commodity GPU ~16GB）**：

```
# === Stage 1: Prefill & KV Cache Construction (一次性，高算力) ===
# 在 H100 GPU 上使用 FlashAttention 逐 chunk prefill
K_cache = []  # 存储在 CPU host memory
V_cache = []  # 存储在 CPU host memory
for chunk in split_into_chunks(input_tokens, chunk_size):
    Q, K, V = model.chunk_forward(chunk)
    K_cache.append(K.cpu())
    V_cache.append(V.cpu())

# === Stage 2: Build ANN Index (CPU side) ===
K_index = []  # Faiss indexes, one per head per layer
for ℓ in 1..L:
    K_index[ℓ] = []
    for h in 1..H:
        idx = faiss.IndexFlatIP(d_k)  # dot product = attention score
        idx.add(K_cache[ℓ][h])        # (N, d_k) -> index
        K_index[ℓ].append(idx)

# === Stage 3: Top-k Decoding Loop ===
K_gen = [[] * L]   # GPU-side: recently generated token keys
V_gen = [[] * L]   # GPU-side: recently generated token values

for step in 1..max_new_tokens:
    for ℓ in 1..L:
        # GPU: QKV projection
        q = x @ W_Q[ℓ]  # (1, d_k)
        k = x @ W_K[ℓ]; v = x @ W_V[ℓ]

        # CPU: ANN search for top-k context keys
        vals, I = K_index[ℓ][h].search(q.cpu(), k_per_head)
        # vals, I in R^k: top-k inner product scores + indices

        # GPU: Transfer selected V + scores
        V_sel = V_cache[ℓ][h][I].to_gpu()  # (k, d_v)
        vals_gpu = vals.to_gpu()

        # GPU: Attention over context (top-k)
        attn_ctx = softmax(vals_gpu / sqrt(d_k)) @ V_sel  # (1, d_v)

        # GPU: Attention over recent generated tokens (windowed)
        attn_gen = softmax(q @ K_gen[ℓ][h]^T / sqrt(d_k)) @ V_gen[ℓ][h]

        # Merge and continue
        attn_out = attn_ctx + attn_gen

        # Update GPU window cache
        K_gen[ℓ][h].append(k); V_gen[ℓ][h].append(v)

    x_new = sample(lm_head(attn_out))
```

术语一般如何实现？如何使用？

实现：(1) Faiss (Facebook AI Similarity Search) 作为核心向量检索引擎，支持 IndexFlatIP（exact inner product search）或 IndexHNSWFlat（approximate HNSW graph search）；(2) Prefill 阶段可使用 FlashAttention（单卡 H100）或 Ring Attention（分布式）构建完整 KV cache；(3) Decoding 阶段 GPU 侧维护一个小窗口的近期生成 token KV cache（windowed attention），CPU 侧维护完整 context KV cache 和 Faiss index；(4) CPU-GPU 数据传输仅涉及 k 个 value 向量（k ≪ N），避免 FlexGen 式全量 KV cache 往返搬移。

使用建议：(1) k 按 context length 的百分比设置——k = 2% of N 实现 >95% dense attention 性能，k = 0.001% 足以完成 NIAH；(2) 支持 layer-wise adaptive k budget——给定固定总 budget Σk_ℓ，按 linear increasing from first to last layer 分配获得更好的性能；(3) 适用于 "rent cloud for prefill once, query locally many times" 的使用模式；(4) 开源实现：https://github.com/ryansynk/topk-decoding。

涉及论文标题：
- Exploiting Sparsity for Long Context Inference: Million Token Contexts on Commodity GPUs

---
