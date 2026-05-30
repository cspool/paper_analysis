## Multi-Head Latent Attention (MLA / 多头潜在注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Multi-Head Latent Attention (MLA) 是 DeepSeek-V2/V3 引入的注意力机制，通过低秩分解将 KV cache 压缩为低维 latent vector，在保持模型质量的同时大幅降低推理时的内存和带宽开销。核心思想：将 Key 和 Value 的投影分解为两步——先将输入 x_t 通过下投影矩阵 W^{DKV} ∈ R^{r_kv × D} 映射到 r_kv 维 latent 空间，生成 c_t^{KV} = W^{DKV} x_t（仅缓存此 latent vector，而非完整 K/V）；再通过上投影矩阵 W^{UK}, W^{UV} ∈ R^{hd × r_kv} 将 latent vector 还原为每个 attention head 的 K/V。Query 同样做低秩分解（W^{DQ} + W^{UQ}）以减少训练激活内存。位置编码采用 Decoupled RoPE——额外使用独立的多头 query q_{t,i}^R 和共享 key k_t^R 携带 RoPE 位置信息，与 content 部分 [q_{t,i}^C; q_{t,i}^R] 和 [k_{t,i}^C; k_t^R] 拼接。

MLA 支持两种计算范式切换：(1) 训练/高计算阶段使用类 MHA 范式（Equation 9）——各 head 独立计算完整 K/V，计算开销略低于标准 MHA；(2) 推理/高通信阶段使用 Absorb 操作切换到类 MQA 范式（Equation 10）——将 W^{UK} 吸收进 query projection，所有 head 共享一个 latent KV head，仅需缓存 c_t^{KV}（r_kv 维），类似 MQA 的极致 KV cache 压缩。

TransMLA 论文理论证明了 MLA 的表达能力严格强于 GQA：相同 KV cache 大小下，GQA 仅是 MLA 的一个稀疏子集（W^{UK}/W^{UV} 必须是 block-selector 矩阵），而 MLA 的 dense 上投影矩阵允许跨 head 混合信息。GQA < MLA_Factorized < MQA。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**MLA 推理范式（Absorb 操作后，Equation 10）**：
```
// latent feature（缓存此项，r_kv 维）
c_t^{KV} = W^{DKV} @ x_t          // [r_kv]

// Absorb: W^{UK} 吸收进 query
q̂_{t,i} = [W_i^{UK}^T @ q_{t,i}^C; q_{t,i}^R]  // per head, [r_kv + d^R]
k̂_t = [c_t^{KV}; k_t^R]                         // shared, [r_kv + d^R]

// 所有 head 共享一个 KV head（类似 MQA）
ô_{t,i} = Σ_j softmax(q̂_{t,i}^T @ k̂_j / √(d+d^R)) @ c_j^{KV}  // [r_kv]

// W^{UV} 合并到 output projection
y_t = W^O @ [W_1^{UV} @ ô_{t,1}; ...; W_h^{UV} @ ô_{t,h}]
```

**KV Cache 对比（d=128, h=32, g=8, r_kv=512）**：
| 机制 | KV Cache 维/token | 相对于 GQA |
|------|-------------------|-----------|
| MHA | 2×32×128=8192 | 4× |
| GQA | 2×8×128=2048 | 1× |
| MLA (r_kv=512) | 512 | 0.25× |
| MLA (r_kv=144) | 144 | 0.07× (93% 压缩) |

术语一般如何实现？如何使用？

MLA 在 DeepSeek-V2/V3/R1 中全面部署，配合 FlashMLA kernel 实现高效推理。开源实现：DeepSeek 官方仓库（FlashMLA: https://github.com/deepseek-ai/FlashMLA），vLLM 和 SGlang 均有 MLA 优化支持。TransMLA 提供 GQA→MLA 转换工具（https://github.com/MuLabPKU/TransMLA），生成的 MLA checkpoint 可直接在 DeepSeek 生态中运行。一般使用流程：训练时使用 full MHA-like 范式（Equation 9）以利用 GPU 算力；推理时切换到 Absorb 范式（Equation 10）以减少 KV cache 内存和带宽。兼容 FP8 量化、Multi-Token Prediction 等 DeepSeek 优化。

涉及论文标题：
- TransMLA: Multi-Head Latent Attention Is All You Need
- xKV: Cross-Layer SVD for KV-Cache Compression

**xKV 对 MLA 架构的进一步压缩**：xKV 证明了其跨层 SVD 压缩方法可以直接应用于 MLA 架构的 latent KV-Cache，实现**已压缩 cache 的再压缩**。具体做法：对 MLA 的 non-RoPE latent representations（c_t^{KV}，已压缩到 r_kv 维）按组做跨层 SVD，解耦的 RoPE keys（k_t^R）不压缩。在 DeepSeek-Coder-V2-Lite-Instruct（16B MoE, 2.4B activated, MLA）上，xKV-4 在 RepoBench-P 上实现 3× 压缩率、LCC 上 3.5× 压缩率，均无精度损失。作为对比，MiniCache 和 Single SVD 在此 MLA 架构上连更低的压缩率都无法保持精度。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Decoupled RoPE 是 DeepSeek MLA 中处理位置编码的策略：将 RoPE 位置信息从 content 计算路径中分离出来，使 content 部分的 K/V 可以安全地做低秩分解和 Absorb 操作。核心问题：标准 RoPE 直接施加在 Q 和 K 上（旋转操作与 token 位置 t 相关），如果 K 通过 W^{UK} 从 latent c^{KV} 上投影得到，则 RoPE 旋转与矩阵乘法不满足交换律——(Rot(W^{UK} c)) 无法被吸收为 (W^{UK}_rot c)。Decoupled RoPE 的解决方案：content 部分 [q_{t,i}^C; k_{t,i}^C] 不施加 RoPE（可安全吸收/压缩），额外创建独立的 RoPE 通道——多头 query q_{t,i}^R = RoPE(W^{QR} c_t^Q, t) 和共享 key k_t^R = RoPE(W^{KR} x_t, t)，将位置信息编码在这些独立通道中。最终 attention score = content_score + position_score，其中 content_score = (q_{t,i}^C)^T k_{j,i}^C（可吸收），position_score = (q_{t,i}^R)^T k_j^R（MQA 结构）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Decoupled RoPE 的完整计算流程**：
```
// 输入 token x_t
c_t^{KV} = W^{DKV} @ x_t           // latent KV, content-only (无 RoPE)
c_t^Q = W^{DQ} @ x_t               // latent Q, content-only

// Content 通道（可安全吸收，因不含 RoPE）
k_{t,i}^C = W_i^{UK} @ c_t^{KV}    // per-head content key
q_{t,i}^C = W_i^{UQ} @ c_t^Q       // per-head content query

// Position 通道（独立 RoPE）
k_t^R = RoPE(W^{KR} @ x_t, t)      // shared position key
q_{t,i}^R = RoPE(W_i^{QR} @ c_t^Q, t)  // per-head position query

// 拼接后计算 attention
k_{t,i} = [k_{t,i}^C; k_t^R]       // content + shared position
q_{t,i} = [q_{t,i}^C; q_{t,i}^R]

// Attention score = content interaction + position interaction
score = (q_{t,i}^C)^T @ k_{j,i}^C + (q_{t,i}^R)^T @ k_j^R
```

**为何 Content 部分可吸收而 Position 部分不可**：
- Content: q_{t,i}^C = W_i^{UQ} @ W^{DQ} @ x_t, k_{j,i}^C = W_i^{UK} @ W^{DKV} @ x_j
  → (q_{t,i}^C)^T @ k_{j,i}^C = x_t^T @ (W^{DQ})^T @ (W_i^{UQ})^T @ W_i^{UK} @ W^{DKV} @ x_j
  → W_i^{UK} 可吸收进 query: q̂_{t,i} = (W_i^{UK})^T @ q_{t,i}^C, k̂_j = c_j^{KV}
- Position: 因 RoPE 旋转矩阵 R(t) 与 W 不交换: R(t) @ W ≠ W @ R(t)
  → 无法吸收，需独立处理

术语一般如何实现？如何使用？

Decoupled RoPE 在 DeepSeek-V2/V3 和所有 MLA-based 模型（包括 TransMLA 转换后的模型）中使用。实现时需额外分配 per-head dimension d^R 给 RoPE 通道（如 d^R = d/2 = 64），Content 维度为 d^C = d - d^R。总 KV cache 中的 content 部分可压缩（低秩 latent），position 部分 k_t^R（d^R 维，所有 head 共享）不可压缩但开销极小。TransMLA 的 RoRoPE 技术进一步将 GQA 模型中分散在各 head 的 RoPE 信息集中到第一 head，实现 Decoupled RoPE 的等价转换。

涉及论文标题：
- TransMLA: Multi-Head Latent Attention Is All You Need

---
