## Triton Fused Attention Kernel with Low-Rank KV Cache

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

ReCalKV 使用 Triton 实现了自定义 fused attention kernel，将低秩压缩的 Key 路径（含 HSR 在线置换）和 Value 路径（含离线 Matrix Fusion）整合到单一 kernel 中。该 kernel 支持 rotary position embedding (RoPE) 和 causal attention。核心优化：(1) 将 HSR 的 head inverse reordering 作为 kernel 内在线 permutation 步骤执行；(2) 利用离线预融合的 W_o_fused 跳过 Value 重建步骤；(3) 所有中间计算结果保持在 SRAM 内，减少 HBM 往返。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。

```
// Triton Fused Low-Rank KV Cache Attention Kernel (per token, per layer)
// Grid: tid = token_position

@triton.jit
def recalkv_fused_attention(Q_w, L_k, R_k, L_v, W_o_fused,
                             K_cache, V_latent_cache, ...):
    pid = tl.program_id(0)  // token position

    // 1. Load current hidden state
    x = tl.load(hidden_states + pid * d_model)  // [1, d_model]

    // 2. Q projection (standard, no compression)
    q = tl.dot(x, Q_w)  // [1, h * d_k]

    // 3. Key path with HSR
    // 3a. Low-rank projection + reconstruction per group
    for g in range(num_groups):
        z_g = tl.dot(x, L_k[g])              // shared latent [1, r_k]
        k_g = tl.dot(z_g, R_k[g])            // [1, s * d_k]

    // 3b. Inverse reordering: restore original head order
    k_reordered = inverse_permute(k_all, hsr_permutation)

    // 3c. Apply RoPE
    k_rope = apply_rotary_pos_emb(k_reordered, position_ids)

    // 4. Value path (no reconstruction needed — Matrix Fusion)
    v_latent = tl.dot(x, L_v)  // [1, r_v], store to KV cache

    // 5. Attention computation
    // Load full K_cache from HBM, compute scores
    scores = tl.dot(q, tl.trans(k_all_cached)) / sqrt(d_k)  // [1, cached_len]
    scores = causal_mask(scores)  // upper triangle → -inf
    attn_w = softmax(scores)                            // [1, cached_len]

    // 6. Fused output (no Value reconstruction)
    o = tl.dot(attn_w, V_latent_cache)  // [1, r_v]
    output = tl.dot(o, W_o_fused)       // [1, d_model]  (R_v fused into W_o)

    return output
```

关键数据流对比：
- **标准 Attention**: X → W_k (d×h·d_k) → K 存 cache → RoPE → QK^T; X → W_v (d×h·d_k) → V 存 cache; Attn·V → output → W_o
- **ReCalKV fused**: X → L_k (d×r_k) → R_k (r_k×s·d_k) → inverse reorder → RoPE → QK^T; X → L_v (d×r_v) → V_latent 存 cache; Attn·V_latent → fused W_o (r_v×d)

术语一般如何实现？如何使用？

Triton 实现，基于 `@triton.jit` 装饰器定义 kernel。关键实现细节：
- HSR permutation 作为预计算的 index mapping 数组在 kernel 内通过 gather/scatter 实现
- W_o_fused 作为 static weight 嵌入 kernel，编译时确定
- 支持 batch 维度上的并行（每个 token 一个 program）
- 延迟测量在 A800 GPU 上进行，100 次运行取平均

性能结果：70% 压缩率下, 4K/16K/65K prompt 分别加速 1.22×/1.59×/1.80×。加速随 prompt 长度和压缩率增长而增大，因为低秩压缩减少的 HBM 访问量在长序列下更显著。

代码：https://github.com/XIANGLONGYAN/ReCalKV

涉及论文标题：
- ReCalKV: Low-Rank KV Cache Compression via Head Reordering and Offline Calibration
