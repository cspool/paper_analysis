## Multi-head Latent Attention (MLA / 多头潜在注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Multi-head Latent Attention (MLA) 是 DeepSeek-V2/V3 引入的 attention 机制创新。核心思想是将传统 Multi-Head Attention (MHA) 的 Key-Value (KV) cache 压缩为低维 latent vector，通过低秩分解在两个阶段工作：(1) Latent Space Encoding：将 K/V 投影到低维 latent 空间（如 d_model=5120 压缩到 latent_dim=512，约 10× 压缩），仅存储压缩后的 latent vector 而非每个 head 的完整 K/V；(2) Dynamic Decoding：注意力计算时，从 latent vector 动态上投影恢复各 head 的 K/V 表示。MLA 通过矩阵乘法结合律将解压矩阵与 Q 投影权重融合（"matrix fusion trick"），避免推理时额外计算开销。对于需要 Rotary Position Embedding (RoPE) 的部分维度，MLA 采用混合设计——部分维度带 RoPE（跨 head 共享）、部分不带 RoPE（允许 fusion trick）。MLA 在保持接近 MHA 表达能力的同时，将 KV cache 减少 87-92%。

从算法 pipeline 角度拆解术语，比如术语所在 pipeline 的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

MLA forward pass 伪代码（TileLang 约 70 行 Python 实现）：
```
输入: hidden_states [batch, seq_len, d_model]
输出: attention_output [batch, seq_len, d_model]

// Stage 1: Latent Compression (Down-Projection)
c_KV = W_down_KV × hidden_states        // [batch, seq_len, latent_dim]
c_Q  = W_down_Q  × hidden_states        // [batch, seq_len, latent_dim_Q]

// Stage 2: Up-Projection for K and V
K = W_up_K × c_KV                        // [batch, head, seq_len, dimqk]
V = W_up_V × c_KV                        // [batch, head, seq_len, dimv]

// Stage 3: RoPE Handling (hybrid design)
K_rope = RoPE(K[:, :, :d_rope])          // 部分维度带 RoPE，跨 head 共享
K_nope = K[:, :, d_rope:]                // 其余维度不带 RoPE
K_final = concat(K_rope, K_nope)

// Stage 4: Q with Matrix Fusion Trick
Q = W_up_Q × c_Q                         // 融合后的等效 Q

// Stage 5: Standard Attention (Parallel Pattern)
scores = Q × K_final^T                   // relevance scoring
scores = softmax(scores / sqrt(dimqk))   // RowNorm
output = scores × V                      // aggregation
// dimqk ≠ dimv (如 DeepSeek-V3: dimqk=576, dimv=512)
```

MLA 的关键特征：(1) KV cache 仅存 c_KV（latent vector，如 512 维），每个 head 不再存独立 K/V；(2) dimqk 和 dimv 通常不相等；(3) head 数远大于 head_kv（如 head=128, head_kv=1）；(4) query seqlen=1 的解码场景下，Q 只有一个 token。

TileLang 实现的 MLA kernel（图 18，~70 行 Python）使用 T.Pipelined loop over KV tiles，在 H100 上达 FlashMLA（手写 CUDA ~1.7k 行）的 98% 性能，在 MI300X 上达 AITER 手写 kernel 的 95% 性能。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MLA 有多个优化实现：(1) FlashMLA（DeepSeek 官方）——约 1.7k 行 CUDA，专门针对 H100/H800，使用 TMA + wgmma.mma_async + warp specialization；(2) TileLang 实现——约 70 行 Python，自动利用 TMA + WGMMA + warp specialization on H100，利用 HIP async copy on MI300X。开源：FlashMLA https://github.com/deepseek-ai/FlashMLA；TileLang MLA kernel 在 TileLang 仓库的 examples 中。

涉及论文标题：
- TileLang: A Composable Tiled Programming Model for AI Systems
- MetaAttention: A Unified and Performant Attention Framework Across Hardware Backends

---
