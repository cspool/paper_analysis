## Group-Centric Data Loading for Sparse Attention Kernels

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Group-Centric Data Loading 是 NSA（Native Sparse Attention）提出的一种针对 GQA/MQA 架构的稀疏注意力 kernel 设计策略。传统 FlashAttention 的 tiling 策略是「按时间连续 query block 加载」——将时间上相邻的 query 加载到 SRAM，遍历 K/V tile 做 online softmax。但在稀疏注意力场景，不同 query position 的稀疏 KV block 索引 I_t 各不相同，导致 query block 内的多个 position 可能需要加载不相交的 KV block 集合，造成内存访问碎片化和冗余。

Group-Centric Data Loading 的核心创新是改变 query 的分组维度：不再按时间连续分组，而是**按 GQA group 分组**。对于每个 query 位置 t，将同一 GQA group 内所有 H 个 query head 的 Q ∈ R^{[H, d_k]} 一同加载到 SRAM。因为 GQA 架构下同一 group 的所有 query heads 共享完全相同的稀疏 KV block 索引 I_t，所以一次 KV block 加载即可服务所有 H 个 query heads，消除 H-1 倍的冗余 KV 传输。该设计将算术强度从接近 ~1 FLOP/byte（memory-bound）提升到 ~H × (2d_k+3d_v)/(d_k+d_v) ≈ 14 FLOP/byte（超过 A100 的 ~12.5 FLOP/byte critical arithmetic intensity），使 kernel 从 memory-bound 转为 compute-bound。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。

**NSA Selection Attention Kernel 伪代码**（Triton 实现）：

```
// Grid: 每个 program 处理一个 query 位置 t
// 假设 GQA group = 4, H = 16 heads per group
// 输入: Q_all ∈ R^{t×num_heads×d_k}, K/V_cache ∈ R^{t×num_kv_heads×d_k}
//       I_all ∈ R^{t×n} (每个 position 的 selected block indices)

@triton.jit
def nsa_selection_kernel(Q, K_cache, V_cache, I, Out, ...):
    t = tl.program_id(0)  // 当前 query 位置
    
    // ==== Group-Centric Loading ====
    // 加载同一 GQA group 内所有 H=16 heads 的 Q 到 SRAM
    q_heads = tl.load(Q + t * H * d_k)  // [H, d_k]
    
    // ==== 初始化 online softmax 状态 per head ====
    m = tl.zeros([H], dtype=tl.float32) - float('inf')
    l = tl.zeros([H], dtype=tl.float32)
    o = tl.zeros([H, d_v], dtype=tl.float32)
    
    // ==== Inner Loop: 遍历该 position 的 selected KV blocks ====
    for blk_idx in range(n):  // n=16 selected blocks
        blk_start = tl.load(I + t * n + blk_idx) * l'
        
        // Shared KV Fetching: 加载连续 KV block
        // 一次 HBM→SRAM 传输服务所有 H 个 heads
        for tile in range(0, l', B_k):  // B_k=128, l'=64
            K_blk = tl.load(K_cache + (blk_start+tile) * d_k)  // [B_k, d_k]
            V_blk = tl.load(V_cache + (blk_start+tile) * d_k)  // [B_k, d_v]
            
            // S = Q @ K^T: [H, d_k] @ [d_k, B_k] → [H, B_k]
            S = tl.dot(q_heads, tl.trans(K_blk)) / sqrt(d_k)
            
            // Online Softmax (per head)
            m_new = tl.maximum(m, tl.max(S, axis=1))  // [H]
            alpha = tl.exp(m - m_new)                  // [H]
            P = tl.exp(S - m_new[:, None])             // [H, B_k]
            l = alpha * l + tl.sum(P, axis=1)          // [H]
            o = alpha[:, None] * o + tl.dot(P, V_blk)  // [H, d_v]
            m = m_new
    
    // ==== 写出 ====
    o_final = o / l[:, None]  // [H, d_v]
    tl.store(Out + t * H * d_v, o_final)
```

**对比 FlashAttention 的 Tiling**：

| 维度 | FlashAttention | NSA Group-Centric |
|------|---------------|-------------------|
| Grid program 粒度 | 时间连续的 query block (Q tile) | 单个 query position + 其 GQA group |
| SRAM Q 内容 | [B_r, d_k]（时间连续） | [H, d_k]（同一 group 所有 heads） |
| KV block 索引 | 所有 position 相同（全量） | Per-position 不同（I_t） |
| KV 加载效率 | 每次服务 1 个 head | 每次服务 H=16 个 heads |
| 适用场景 | Dense attention | Sparse attention with GQA/MQA |

术语一般如何实现？如何使用？

Group-centric data loading 通过 Triton 的 grid scheduler 实现。关键实现要点：(1) Grid size = 总 query position 数，每个 block 处理一个 position，inner loop 长度几乎恒定（n × l'/B_k ≈ 8 iterations），确保 SM 间负载均衡；(2) 内循环按 I_t 升序加载 KV block，确保 HBM 读取连续；(3) 所有注意力计算在 SRAM 中完成（green blocks in Figure 3），Q 常驻 SRAM 供整个 inner loop 复用；(4) 与 FlashAttention 的 online softmax 融合在同一 kernel 中，避免写出中间 attention 矩阵。

使用场景：任何采用 GQA/MQA 架构且使用 query-aware sparse attention 的场景。不适用于 MHA（无 KV sharing）或固定稀疏模式（如 sliding window only）。该方法已被 DeepSeek V3.2-Exp 等生产模型采用。

涉及论文标题：
- Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention

---
