## Cross-Layer SVD for KV-Cache Compression (xKV / 跨层SVD)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

xKV（Cross-Layer SVD for KV-Cache Compression）是一种无需训练的 plug-and-play KV-Cache 压缩方法。其核心观察是：尽管相邻层 KV-Cache 的 token-wise cosine similarity 很低，但通过 Centered Kernel Alignment (CKA) 分析发现，不同层的**主导左奇异向量（dominant left singular vectors）高度对齐**。xKV 利用这一跨层奇异向量对齐特性，将多个相邻层的 KV-Cache 水平拼接后执行一次统一的 SVD，提取共享的低秩子空间基（A = U_r @ S_r），各层仅保留独立的低秩重构矩阵（B_ℓ_i），从而显著减少 KV-Cache 存储。

xKV 的关键公式：
$$
\left[\mathbf{X}_{\ell_1}, \dots, \mathbf{X}_{\ell_{|G|}}\right] \in \mathbb{R}^{L \times (|\mathcal{G}| \cdot d)} \approx \mathbf{U}_r \, \mathbf{S}_r \, \mathbf{V}_r^{\top} = \mathbf{A} \left[ \mathbf{B}_{\ell_1}, \dots, \mathbf{B}_{\ell_{|\mathcal{G}|}} \right]
$$
其中 A ∈ R^{L×r} 是被组内所有层共享的 left singular vectors（共享基），B_ℓ_i ∈ R^{r×d} 是层特定的重构矩阵。压缩后仅需存储 A 和 {B_ℓ_i}，原始存储 |G|·L·d → 压缩后 L·r + |G|·r·d。

从算法pipeline角度拆解术语：

```
// xKV Cross-Layer SVD 压缩算法
// 输入: 一组相邻层的 pre-RoPE key/value caches
// G: stride size (如 2, 4), L: sequence length, d: hidden dim, r: rank

def xkv_compress(group_layers, G, r_key, r_val):
    # 1. 水平拼接所有层的 KV-Cache
    K_cat = concat_horizontal([K_li for li in group_layers])  # [L, G*d]
    V_cat = concat_horizontal([V_li for li in group_layers])  # [L, G*d]
    
    # 2. 分别对 Key 和 Value 做跨层 SVD
    U_k, S_k, Vt_k = SVD(K_cat)
    U_v, S_v, Vt_v = SVD(V_cat)
    
    # 3. 保留 top-r 成分（key:value rank ratio = 1:1.5）
    A_k = U_k[:, :r_key] @ S_k[:r_key, :r_key]     # [L, r_key] 共享基
    A_v = U_v[:, :r_val] @ S_v[:r_val, :r_val]     # [L, r_val] 共享基
    
    B_k_li = [Vt_k[:r_key, i*d:(i+1)*d] for i in range(G)]  # 各层 key 重构矩阵
    B_v_li = [Vt_v[:r_val, i*d:(i+1)*d] for i in range(G)]  # 各层 value 重构矩阵
    
    # 4. 存储: A_k, A_v + {B_k_li, B_v_li}
    return (A_k, A_v, B_k_li, B_v_li)

# Decode 阶段: 重构并重新应用 RoPE
def xkv_decode(A_k, B_k_li, A_v, B_v_li, layer_idx):
    K_recon = A_k @ B_k_li[layer_idx]    # [L, d] 重构 key
    K_recon = apply_rope(K_recon)        # 重新施加 RoPE
    V_recon = A_v @ B_v_li[layer_idx]    # [L, d] 重构 value
    return attention(Q, K_recon, V_recon)
```

**Stride-based Grouping**：将 N 层 Transformer 按相邻层分组，每组大小 G（stride=G），共 N/G 组。论文实验验证 G=2 和 G=4。

**压缩率**：当 L >> r·d 时，压缩率 ≈ L/r（近似）。论文在 Llama-3.1-8B 上实现 2.5×-8× 压缩率，xKV-4 在 8× 压缩下仍保持 87.8% avg accuracy（vs Single SVD 的 35.3%）。

术语一般如何实现？如何使用？

通过 HuggingFace Transformers 实现，无需模型修改或微调。在 prefill 阶段按请求在线执行 SVD 分解（on-the-fly，非离线统计），更好地捕捉每个请求上下文的动态。SVD 开销在 128K context 下 <10% prefill time。keys 和 values 具有不同的压缩敏感度，固定 rank ratio(keys:values) = 1:1.5。对 pre-RoPE key states 执行 SVD，decode 时重新应用 RoPE。新生成的 tokens 不压缩（因长上下文场景下生成部分占比 <2%）。代码开源：https://github.com/abdelfattah-lab/xKV。

涉及论文标题：
- xKV: Cross-Layer SVD for KV-Cache Compression
