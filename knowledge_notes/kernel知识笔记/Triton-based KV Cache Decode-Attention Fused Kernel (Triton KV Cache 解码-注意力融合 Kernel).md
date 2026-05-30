## Triton-based KV Cache Decode-Attention Fused Kernel (Triton KV Cache 解码-注意力融合 Kernel)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Triton-based KV Cache Decode-Attention Fused Kernel 是 CommVQ 中用于高效执行量化 KV Cache 解码与 self-attention 计算的 GPU kernel。该 kernel 通过 Triton 语言实现，利用 RoPE-可交换码本的数学特性将 Key cache 的解码操作从独立的预处理步骤融合进 attention score 计算。核心优化包括：(1) 预计算复用：$(qR_t)C_K^T$ 在每 decoding step 仅计算一次并跨所有缓存 token 共享；(2) 稀疏旋转解码：利用 RoPE 矩阵的 2x2 块对角稀疏性，每个 token 仅需轻量的 $R_i^T s_i^T$ 旋转操作；(3) Value 重排乘法：先计算 Softmax(A) × S_V（小矩阵乘）再乘 C_V，将复杂度从 $O(d N_c N + dN)$ 降至 $O(N_c N + d N_c)$。这些优化使整体解码复杂度从原始方案的 $O(2d N_c N)$（是 self-attention 的 $N_c$ 倍）降至近似 $(R+1)/2$ 倍 self-attention 开销。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。

**Triton fused kernel 的伪代码（per-layer, per-head decoding step）**：

```
// Kernel 1: Query Precomputation (Triton gemm)
q = tl.load(query_ptr)                  // [d]
q_rope = apply_rope_2d(q, position)     // 逐 2D 子空间旋转
q_precomp = tl.dot(q_rope, C_K^T)       // [d] @ [d, K_dim] -> [K_dim]
store(q_precomp_buf, q_precomp)

// Kernel 2: Fused Key Decode + Attention Score (Triton)
// grid: (num_blocks_N,), block: (BLOCK_N,)
pid = tl.program_id(0)
offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)  // [BLOCK_N]

// 加载该 block 的量化 key cache
s_key = tl.load(S_K_ptr + offs_n * S_K_stride)   // [BLOCK_N, d/2, 2]

// 加载预计算结果（所有 block 共享）
q_pre = tl.load(q_precomp_buf)                     // [K_dim]

// 逐 2D 子空间累加 attention score
alpha = tl.zeros([BLOCK_N], dtype=tl.float32)
for j in range(d // 2):
    for n in range(BLOCK_N):
        s_val = s_key[n, j]                        // (idx_a, idx_b)
        alpha[n] += fused_rope_decode_dot(
            q_pre[j], s_val, positions[offs_n[n]]
        )

store(alpha_out + offs_n, alpha)

// Kernel 3: Value Decode Reordering (Triton)
temp = tl.dot(attn_weights, S_V)                   // [1, N_c]
output = tl.dot(temp, C_V)                          // [1, d]
store(O_ptr, output)
```

**复杂度对比**：
| 阶段 | Naive 实现 | Optimized (Triton fused) |
|------|-----------|-------------------------|
| Key 解码 | $O(2d N_c N)$ | $O(Rd N + d N_c + N_c N)$ |
| Value 解码 | $O(d N_c N + d N)$ | $O(N_c N + d N_c)$ |
| 128K ctx 延迟 | 36.6 ms/layer/token | 3.8 ms/layer/token |
| Speedup | 1x | 9.6x |

术语一般如何实现？如何使用？

使用 Triton 语言编写，利用 `tl.dot` 执行 Tensor Core 矩阵乘法，`tl.load`/`tl.store` 管理 HBM ↔ SRAM 数据传输。Kernel 在 LLaMA-3.1-8B 的每层每头上执行。Codebook 常驻 GPU 显存（仅 4.75-9.25 MB），kernel 通过指针传递引用。在 H100-80GB 上实现 1-bit 量化时 128K context 仅需约 20 GB（vs FP16 的约 60 GB），RTX 4090 上可运行 128K context。开源：https://github.com/UMass-Embodied-AGI/CommVQ。

涉及论文标题：
- CommVQ: Commutative Vector Quantization for KV Cache Compression

---
