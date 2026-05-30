## Flash Attention 2 as Per-GPU Kernel in Distributed Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
在分布式注意力（如 Tree Attention 和 Ring Attention）中，Flash Attention 2（Dao, 2023）被用作每 GPU 的局部注意力计算 kernel。Flash Attention 2 通过 tiling（将 Q,K,V 分块加载到 GPU SRAM）和 online softmax rescaling 实现 IO-aware 的精确注意力计算，避免物化完整的 N×N 注意力矩阵（O(N²) memory → O(N)）。在分布式场景中，每 GPU 对其本地 K,V chunk（t = N/p tokens）调用 Flash Attention 2，计算局部注意力输出 o_i 和 logsumexp lse_i。

从kernel调度角度拆解术语。
Flash Attention 2 在 Tree Attention 每 GPU 上的执行流程：
```
# 输入每 GPU: q [1, d_h], K_i [t, d_h], V_i [t, d_h]
# Flash Attention 2 forward (简化):

Q_block_size = B_r, KV_block_size = B_c
num_kv_blocks = ceil(t / B_c)

# 初始化 online softmax 状态
o = zeros(1, d_h)    # 累加器
l = 0                # softmax 分母 (log space)
m = -inf             # running max

# 外层循环: Q blocks (解码时 Q 只有 1 个 token → 1 个 block)
for Q_block in Q_blocks:
    load Q_block to SRAM   # [B_r, d_h], B_r=1 for decode
    
    # 内层循环: KV blocks
    for j in 0..num_kv_blocks:
        load K_block, V_block to SRAM  # [B_c, d_h] each
        
        # Compute attention scores
        S = Q_block @ K_block^T / sqrt(d_h)  # [B_r, B_c]
        
        # Online softmax update
        m_new = max(m, rowmax(S))           # [B_r]
        P = exp(S - m_new)                   # [B_r, B_c]
        l_new = exp(m - m_new) * l + rowsum(P)  # [B_r]
        o_new = diag(exp(m - m_new)) @ o + P @ V_block  # [B_r, d_h]
        
        m = m_new; l = l_new; o = o_new
    
    # 归一化
    o = diag(1/l) @ o  # [B_r, d_h]

return o, log(l) + m  # (attention output, logsumexp)
```

Flash Attention 2 的关键特性使其成为分布式 attention 的理想 per-GPU kernel：
- **IO-aware**：KV 从 HBM 加载到 SRAM 的次数为 O(N²d²/M)，远低于 naive attention 的 O(N²)，M 为 SRAM 大小
- **Exact**：产出与 naive softmax attention 数值等价的结果（不是近似）
- **Work partitioning**：Flash Attention 2 改进了 thread block 的 work partitioning，减少 warp 间的同步和 shared memory 通信

术语一般如何实现？如何使用？
实现：通过 JAX binding（`flash_attn_jax.flash._flash_mha_vjp.fwd`）调用。在 Tree Attention 中，`_flash_mha_vjp.fwd` 返回 `(res, lse)` 元组，其中 `res` 是局部 attention 输出，`lse` 是局部 logsumexp——后者直接用于跨 GPU 的 AllReduce 归约。

解码场景的特殊性：解码时 Q 只有 1 个 token，Flash Attention 2 的 Q_blocks = 1，内层循环遍历所有 KV blocks。这时 attention 计算极快（~10μs for t=80K, d_h=128 on H100），远快于通信时间（~1ms for K,V P2P transfer），因此 Ring Attention 无法通过 overlap 隐藏通信。Tree Attention 通过仅传输标量级归约结果（而非完整 K,V chunks）解决此问题。

涉及论文标题：
- Tree Attention: Topology-aware Decoding for Long-Context Attention on GPU clusters
