## KV Cache (in SSM vs Transformer Context)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
KV Cache（Key-Value Cache）是Transformer自回归解码中缓存历史token的Key和Value投影结果以加速推理的标准技术。生成第t个token时，只需计算最新token的Q/K/V，利用已缓存的K_{1:t-1}和V_{1:t-1}完成attention计算，避免对历史token的重复投影。KV cache内存需求为O(num_layers × seq_len × 2 × num_heads × d_head × dtype_bytes)，随序列长度线性增长，是长上下文推理的主要内存瓶颈。在SSM（Mamba/Mamba-2）中，由于递归性质仅需维护固定大小的隐状态h_t ∈ R^{D×N}，完全不需要KV cache。但Hybrid模型因包含self-attention层需为这些层维护KV cache——论文中Mamba-2-Hybrid仅4/56层需要KV cache，且使用GQA(8 KV groups)进一步缩小KV cache。因此Hybrid模型的KV cache总量远小于同等Transformer（32层的32组KV heads）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
推理时KV Cache全流程（Hybrid模型）：
请求到达 → Tokenizer → Embedding → 
for layer in 1..56:
    if layer is Mamba-2:
        // 无KV cache: 使用128维recurrent state h_t
        // 每token内存开销: D × N × dtype = 4096 × 128 × 2 = 1MB (sum over all Mamba-2 layers)
    if layer is Self-Attention:
        // 需要KV cache
        // 第t个token: Q_t = W_Q(x_t), K_t = W_K(x_t), V_t = W_V(x_t)
        // KV_cache[layer].append(K_t, V_t)  // 仅4层需要
        // attention: Q_t @ [KV_cache[layer]].T
        // 每token内存开销: 4 × 2 × 8 × 128 × 2B = 16KB/token
// 对32K seq_len: 32K × 16KB = 512MB (Hybrid KV cache)
// 对比Transformer 32K: 32K × (32 × 2 × 32 × 128 × 2B) ≈ 16GB
```

术语一般如何实现？如何使用？
vLLM等推理框架通过PagedAttention管理KV cache内存，减少碎片。论文指出Hybrid模型的KV cache减少是推理加速的关键来源——更小的KV cache意味着更大的batch size或更长的上下文。在推理框架中使用时，Hybrid模型需识别哪些层需要KV cache（attention层）和哪些不需要（Mamba-2层），分别使用不同的内存管理策略。

涉及论文标题：
- An_Empirical_Study_of_Mamba-based_Language_Models
- RWKV-X__A_Linear_Complexity_Hybrid_Language_Model

### KV Cache Management (SnapKV-based Importance Eviction)

RWKV-X 采用了基于 SnapKV（Li et al., 2024, NeurIPS 2024）的 KV Cache 管理机制来实现解码阶段 O(1) 空间复杂度。核心思想：(1) 将 past cache 分为 earlier cached states (K_past, V_past) 和 recent observation window (K_obs, V_obs)；(2) 通过 softmax attention scores 计算 cumulative importance score: C = Σ_i softmax(Q_obs K_past^T/√d_k)[i,:]；(3) 选取 top-m 最重要的历史 entries（基于 C 分数）；(4) 压缩 cache = selected_entries || observation_window，总大小保持为 m + L_obs 的常数，不随生成序列长度增长。论文中 RWKV-X-3.6B 使用固定 64K KV cache budget，实现从 1K 到 1M tokens 的稳定解码延迟（Figure 4）。SnapKV 的 insight 是 attention head 在生成前已形成一致的关注模式，可从 observation window 的 queries 预判哪些 prefix KVs 最重要。

涉及论文标题（追加）：
- RWKV-X__A_Linear_Complexity_Hybrid_Language_Model

---
