## Sliding Window Attention (SWA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sliding Window Attention (SWA) 是一种稀疏注意力模式（Beltagy et al., 2020），每个 token 仅关注其前后固定窗口 w 内的 token，将注意力复杂度从 O(n²) 降至 O(n·w)。与全注意力不同，SWA 具有序列长度的平移不变性——任意长度序列的每 token 计算量恒定。在 SAMBA 中，SWA 窗口大小 w=2048，使用 FlashAttention 2 高效实现，配合 RoPE（base=10,000）编码相对位置。选择 w=2048 的关键原因：FlashAttention 2 在 seqlen=2048 时训练速度与 Mamba 的 selective parallel scan 相当（基于 Gu & Dao 2023 测量），使混合架构不会引入瓶颈层。训练序列长度设为 4096 = w×2（SAMBA 发现这是最优比，Table 9）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# SWA 层前向（Samba 中使用）
Input: X ∈ R^{n × d_m}, window_size=2048, RoPE base=10000

Q = X @ W_q   # [n, n_heads × d_head]
K = X @ W_k
V = X @ W_v
Q, K = RoPE(Q, K, base=10000)   # Rotary Position Embedding

# FlashAttention 2 with causal sliding window
# 对每个位置 i, attention 仅在 [max(0, i-w+1), i] 范围内
O_swa = FlashAttention2(Q, K, V, causal=True, window_size=(w, 0))
O = O_swa @ W_o   # output projection
```
关键特性：(1) 计算复杂度 O(n·w) 而非 O(n²)；(2) 平移不变性——模型在训练长度外的序列上仍表现良好；(3) 窗口内保留精确 softmax attention，可精确召回近期记忆；(4) 无法直接访问超出窗口的历史 token，需依赖 SSM 层的递归状态压缩来传递长程信息。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：(1) FlashAttention 2 原生支持 SWA 的 window_size 参数；(2) Hugging Face Transformers 中通过 `attention_mask` 或 `sliding_window` 配置参数支持。SAMBA 纯用 SWA（无 global attention tokens），依赖 Mamba 层处理超出窗口的长程依赖。SWA 适用于：(a) 需要 O(n) 复杂度的长文档处理；(b) 混合架构中作为精确检索组件；(c) 长度外推——训练长度 4K 的 SWA 模型在更长序列上 perplexity 自然下降（Table 3: Llama-2-SWA 在 16K 时 10.57 vs Llama-2 的 249.03）。

涉及论文标题：
- Samba__Simple_Hybrid_State_Space_Models_for_Efficient_Unlimited_Context_Language_Modeling

---
