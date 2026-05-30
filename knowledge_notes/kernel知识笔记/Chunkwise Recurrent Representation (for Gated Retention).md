## Chunkwise Recurrent Representation (for Gated Retention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Chunkwise Recurrent Representation（分块循环表示）是 Gated Retention 的第三种计算范式，统一了 Parallel 和 Recurrent 两种表示。它将序列划分为大小为 B 的 chunk（如 B=256），每个 chunk 内使用 Parallel 计算（利用 Tensor Core 的矩阵乘加速），chunk 间使用 Recurrent 计算（通过 state R_i 传递跨 chunk 的历史信息）。数学上输出分为 Inner-Chunk 部分（(Q_{[i]}K_{[i]}^T⊙D_{[i]})V_{[i]}，chunk 内标准并行计算）和 Cross-Chunk 部分（(Q_{[i]}R_{i-1})⊙β_{[i]}，利用上一 chunk 的 state）。此范式在 FLOPs 上优于 fully parallel（避免计算上三角全部元素）且在迭代数上优于 fully recurrent（B 倍减少）。在 YOCO 中，prefill 阶段使用 chunkwise（固定 chunk_size=256），decode 阶段切换到纯 recurrent。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 YOCO Triton kernel 中 chunkwise gated retention 的计算为例：

```python
def chunkwise_gated_retention(Q, K, V, gt, past_kv, chunk_size=256):
    """
    Q, K, V: [batch, heads, seq_len, dim]
    gt: [batch, heads, seq_len]  -- data-dependent gate γ
    past_kv: [batch, heads, dim, dim]  -- previous chunk's state R
    """
    B, H, N, D = Q.shape
    num_chunks = N // chunk_size
    
    # Compute decay in log-space for numerical stability
    log_gamma = F.logsigmoid(gt) / gate_logit_normalizer  # [B,H,N]
    
    output_chunks = []
    for i in range(num_chunks):
        start, end = i*chunk_size, (i+1)*chunk_size
        Q_c, K_c, V_c = Q[:,:,start:end], K[:,:,start:end], V[:,:,start:end]
        log_g_c = log_gamma[:,:,start:end]  # [B,H,chunk]
        
        # ---- Cross-Chunk (recurrent part) ----
        # cumulative decay over this chunk: β = exp(cumsum(log_gamma))
        cumdecay = log_g_c.cumsum(dim=-1)  # [B,H,chunk]
        beta = cumdecay.exp()               # multiplicative decay
        
        # Cross-chunk output: (Q @ past_kv) * exp(decay_from_start)
        cross_out = (Q_c @ past_kv) * beta.unsqueeze(-1)  # [B,H,chunk,D]
        
        # ---- Inner-Chunk (parallel part) ----
        # Causal decay mask within chunk
        decay_mask = (cumdecay.unsqueeze(-1) - cumdecay.unsqueeze(-2)).exp()
        causal_mask = torch.triu(torch.ones(chunk, chunk), diagonal=1) * -1e9
        D_c = decay_mask + causal_mask.to(Q.device)
        
        attn_scores = (Q_c @ K_c.transpose(-1,-2)) * D_c   # [B,H,chunk,chunk]
        inner_out = attn_scores @ V_c                       # [B,H,chunk,D]
        
        # ---- Combine & Output ----
        output_c = inner_out + cross_out
        output_c = group_norm(output_c)
        output_chunks.append(output_c)
        
        # ---- Update State for Next Chunk ----
        chunk_decay = beta[:,:,-1].unsqueeze(-1).unsqueeze(-1)  # [B,H,1,1]
        value_decay = (beta[:,:,-1].unsqueeze(-1) - cumdecay).exp()  # [B,H,chunk]
        current_kv = chunk_decay * past_kv + K_c.transpose(-1,-2) @ (V_c * value_decay.unsqueeze(-1))
        past_kv = current_kv
    
    return torch.cat(output_chunks, dim=2), past_kv
```

**Annotations**: chunk_size=256 是平衡点——太小则 recurrent 迭代次数多、太大则 parallel 部分 O(B²) 开销增大。gate_logit_normalizer 从训练数据统计得出，用于将 sigmoid 输出映射到合适的 decay 范围。past_kv ∈ R^{d×d} 是唯一跨 chunk 传输的状态矩阵（O(d²) 内存）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Chunkwise Recurrent 在 YOCO 中通过 Triton kernel 实现（基于 FLA 库：https://github.com/sustcsonglin/flash-linear-attention）。具体使用：(1) **Prefill 阶段**——输入长序列（可能 512K tokens），用 chunkwise 减少内存和 FLOPs；(2) **训练阶段**——长序列训练时用 chunkwise 代替 parallel 以降低峰值显存（避免存储完整 N×N attention matrix）；(3) **与 FlashAttention 类比**——chunkwise 的效果类似于分块 attention，但通过 recurrent state 保证 chunk 间信息无损传递。限制：chunkwise recurrent 在 chunk 边界处的精度依赖 gate_logit_normalizer 的校准；跨 chunk 的 state R 在长序列中可能积累数值误差。

涉及论文标题：
- Efficient implementations for emerging model architectures (YOCO: You Only Cache Once)
