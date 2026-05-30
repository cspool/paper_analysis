## KV Cache Pruning（KV Cache 剪枝）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
KV Cache Pruning 是一种通过评估历史 KV cache 中每个 token 的重要性，选择性丢弃低重要性 token 的 cache 来约束内存使用的技术。核心思路：并非所有历史 token 对当前/未来帧的 attention 计算同等重要——大量 token（尤其是视觉 patch tokens）携带冗余信息（spatial-temporal redundancy），可以在保持模型性能的同时被安全移除。

剪枝的关键要素：
- **重要性度量**：常用指标包括 attention scores（accumulated/current）、token saliency、QK 相似度、或基于 MLP 的 learned scorer。
- **剪枝粒度**：token-level（丢弃单个 token）、channel-level（丢弃特定 channel 的 KV 条目）、layer-level（不同层不同剪枝率）。
- **剪枝时机**：one-shot（prefill 后一次性剪枝，后续步骤用固定 cache）vs dynamic（每个 decoding step 重新评估和剪枝）。

XStreamVGGT 的剪枝机制特点：使用 query-guided 重要性评分（对 Query 分组池化后与 Key 计算内积），而非直接读取 attention scores，以保持与 FlashAttention 的兼容性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
XStreamVGGT 中的 query-guided KV cache pruning：

```
# 输入: Q_t^(ℓ) ∈ R^{(1+R+N)×C} (当前帧 Query)
#       Cache.K_{1:t}^(ℓ), Cache.V_{1:t}^(ℓ) (完整 KV cache)
#       L_max (缓存预算，如 2K tokens)
#       g = 16 (分组大小)

# 1. Query 分组池化
Q_special = Q_t^(ℓ)[:1+R, :]           # camera + register tokens
Q_normal  = Q_t^(ℓ)[1+R:, :]            # patch tokens
Q_pooled  = concat(Q_special, GroupAvg(Q_normal, g))
Q̄ = mean(Q_pooled, dim=heads)          # 跨 head 平均，shape: N_pooled × C

# 2. 提取中间帧 prunable keys（排除首帧和当前帧）
K̄_prunable = mean(K_{first+1 : t-1}, dim=heads)  # shape: T_prunable × C

# 3. 计算 token 重要性分数
S_matrix = Q̄ @ K̄_prunable^T              # QK 内积
S = mean(S_matrix, dim=query)             # 沿 query 维平均

# 4. Top-k 选择（保留首帧 + 当前帧 + 高分中间 token）
I_middle = TopK(S, k = L_max - T_first - T_current)
I_keep = {1..T_first} ∪ I_middle ∪ {T-T_current+1..T}

# 5. 同步剪枝 K 和 V
Cache.K = Cache.K[I_keep]
Cache.V = Cache.V[I_keep]
```

与文本 LLM 中的 KV pruning（如 H2O, SnapKV）的区别：后者通常基于 accumulated attention scores 选择 "heavy hitter" tokens，而 XStreamVGGT 使用 query-guided Q̄K̄^T 内积，适配 vision token 的 spatial-temporal 冗余特性，且 pooling 设计兼容 FlashAttention。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch 原生实现：在每层 attention 计算后调用 pruning 函数，返回 mask indices 后通过 `torch.index_select` 或直接索引裁剪 cache tensor。与 FlashAttention 的兼容性要求 pruning 不能依赖 attention scores 的中间结果——因此使用独立的 Q̄K̄^T 计算（额外开销小）或基于 hidden state 变化（如 L2 distance）的方法。kvpress（HuggingFace）库提供统一的 KV cache 压缩接口。XStreamVGGT 码：https://github.com/ywh187/XStreamVGGT/。

涉及论文标题：
- XStreamVGGT__Extremely_Memory-Efficient_Streaming_Vision_Geometry_Grounded_Transformer_with_KV_Cache_Compression
