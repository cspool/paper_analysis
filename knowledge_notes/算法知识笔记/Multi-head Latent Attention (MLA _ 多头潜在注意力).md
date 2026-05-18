## Multi-head Latent Attention (MLA / 多头潜在注意力)

术语是什么？通过联网搜索让回答具体和精准。

Multi-head Latent Attention (MLA) 是 DeepSeek-V2/V3 引入的一种 attention 机制创新。其核心思想是将传统 Multi-Head Attention (MHA) 的 Key-Value (KV) cache 压缩为低维 latent vector，通过低秩分解在两个阶段工作：(1) Latent Space Encoding：将 K/V 投影到低维 latent 空间（如 d_model=5120 压缩到 latent_dim=512，约 10× 压缩），仅存储压缩后的 latent vector 而非每个 head 的完整 K/V；(2) Dynamic Decoding：注意力计算时，从 latent vector 动态上投影恢复各 head 的 K/V 表示。MLA 通过矩阵乘法结合律将解压矩阵与 Q 投影权重融合（"matrix fusion trick"），避免推理时额外计算开销。对于需要 Rotary Position Embedding (RoPE) 的部分维度，MLA 采用混合设计——部分维度带 RoPE（跨 head 共享）、部分不带 RoPE（允许 fusion trick），两部分分别计算后通过 dot product 组合。MLA 在保持接近 MHA 表达能力的同时，将 KV cache 减少 87-92%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

MLA forward pass 伪代码：
```
输入: hidden_states [batch, seq_len, d_model]

// Stage 1: Latent Compression (Down-Projection)
c_KV = W_down_KV × hidden_states        // [batch, seq_len, latent_dim], 压缩 KV 到 latent 空间
c_Q  = W_down_Q  × hidden_states        // [batch, seq_len, latent_dim_Q]

// Stage 2: Up-Projection for K and V
K = W_up_K × c_KV                        // [batch, head, seq_len, dimqk]
V = W_up_V × c_KV                        // [batch, head, seq_len, dimv]

// Stage 3: RoPE Handling (hybrid design)
K_rope = RoPE(K[:, :, :d_rope])          // 部分维度带 RoPE，跨 head 共享
K_nope = K[:, :, d_rope:]                // 其余维度不带 RoPE
K_final = concat(K_rope, K_nope)

// Stage 4: Q with Matrix Fusion Trick
// 将 W_up_Q 与 W_up_K 通过结合律预融合，避免显式 up-project K
Q = W_up_Q × c_Q                         // 融合后的等效 Q

// Stage 5: Standard Attention (in MetaAttention terms: Parallel Pattern)
scores = Q × K_final^T                   // relevance scoring
scores = softmax(scores / sqrt(dimqk))   // RowNorm
output = scores × V                      // aggregation
// dimqk ≠ dimv (如 DeepSeek-V3: dimqk=576, dimv=512)
```

MLA 的关键特征：(1) KV cache 仅存 c_KV（latent vector，如 512 维），每个 head 不再存独立 K/V；(2) dimqk 和 dimv 通常不相等（如 dimqk=576, dimv=512）；(3) head 数远大于 head_kv（如 head=128, head_kv=1）；(4) query seqlen=1 的解码场景（LLM inference）下，Q 只有一个 token，K/V 来自 KV cache。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MLA 的优化实现有多个版本：(1) FlashMLA（DeepSeek 官方）——约 1.7k 行 CUDA，专门针对 H100/H800 的 MLA 解码 kernel，使用 blockSize=64，利用 TMA 异步加载和 Tensor Cores MMA，支持 dimqk≠dimv 的非标准 shape；(2) MLA Triton kernel——Triton 实现的 MLA forward kernel，性能低于 FlashMLA；(3) MetaAttention 中的 MLA 支持——约 90 行代码，通过 Parallel Pattern + 自定义 Q/K/V shape（head=128, head_kv=1, dimqk=576, dimv=512, query seqlen=1）+ customizable functions 表达 MLA，性能接近 FlashMLA（within comparable），且自动受益于 MetaAttention 的跨后端支持（NVIDIA/AMD）。开源：FlashMLA https://github.com/deepseek-ai/FlashMLA。

涉及论文标题：
- MetaAttention: A Unified and Performant Attention Framework Across Hardware Backends

---

