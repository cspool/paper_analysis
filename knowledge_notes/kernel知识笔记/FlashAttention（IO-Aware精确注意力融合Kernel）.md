## FlashAttention（IO-Aware精确注意力融合Kernel）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FlashAttention 是 Dao et al.（2022, NeurIPS）提出的 IO-aware 精确注意力 CUDA kernel，通过 tiling 和 recomputation 技术避免将完整的 $N \times N$ 注意力中间矩阵（softmax 的输入和输出）写入 GPU HBM，将 HBM 访问量从 $O(N^2)$ 降至 $O(N)$。核心洞察：GPU 的 SRAM 带宽（~20TB/s on A100）远高于 HBM 带宽（~2TB/s），attention 的性能瓶颈在 HBM 访存而非计算——FlashAttention 将 Q/K/V 分块加载到 SRAM 中计算 softmax（online softmax with rescaling），直接输出 O 矩阵，再通过 backward 时 recompute S 和 P（不存储中间值）。结果：2-4× 加速，10-20× 内存节省，且输出与标准 attention 数值等价（非近似）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# FlashAttention: IO-Aware Tiled Attention (forward pass)
# Q, K, V: [N, d]; output O: [N, d]
# B_r, B_c: tile sizes for Q/O and K/V blocks

for j in range(0, N, B_c):           # iterate over K, V blocks
    load K_j = K[j:j+B_c] to SRAM    # [B_c, d]
    load V_j = V[j:j+B_c] to SRAM
    
    for i in range(0, N, B_r):       # iterate over Q, O blocks
        load Q_i = Q[i:i+B_r] to SRAM          # [B_r, d]
        load O_i = O[i:i+B_r] from HBM          # [B_r, d]
        load l_i, m_i (previous running stats)
        
        # On-chip computation (in SRAM)
        S_ij = Q_i @ K_j^T           # [B_r, B_c]
        m_ij = rowmax(S_ij)          # local max per row
        P_ij = exp(S_ij - m_ij)      # safe softmax numerator
        l_ij = rowsum(P_ij)          # local denominator
        
        # Online softmax rescaling (Algorithm 1)
        m_new = max(m_i, m_ij)
        l_new = exp(m_i - m_new) * l_i + exp(m_ij - m_new) * l_ij
        # Rescale existing O_i and add new contribution
        O_i = diag(exp(m_i - m_new)) * O_i + exp(m_ij - m_new) * (P_ij @ V_j)
        
        m_i, l_i = m_new, l_new
        store O_i, l_i, m_i to HBM   # write back

# Final: O_i = diag(1/l_i) * O_i (normalize by softmax denominator)
```

关键 IO 分析：标准 attention 每 forward pass 需读写 $O(N^2)$ 字节的 S+P 矩阵；FlashAttention 仅需读写 Q/K/V/O 块 $O(Nd)$ 字节 × 分块数。A100 HBM bandwidth 2TB/s, SRAM bandwidth 19.5TB/s → tiling 将 95% 的访存放至 SRAM。

FlashAttention-2 (2023)：改进 work partitioning（沿 seqlen 维度而非 batch/head 维度并行），减少非 MatMul FLOPs，增加 occupancy（更少的 register 使用），比 v1 再加速 2×。FlashAttention-3 (2024, H100)：利用 Hopper 的 WGMMA 异步指令和 TMA 实现 warp-specialized 双流水线（producer-consumer overlap 数据搬运与计算），H100 上达到 740 TFLOPS（75% 峰值利用率）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

开源实现：flash-attn 库（https://github.com/Dao-AILab/flash-attention），提供 PyTorch 接口（`flash_attn_func(q, k, v)`）。被 PyTorch 2.0+ 原生集成（`torch.nn.functional.scaled_dot_product_attention` 中自动 dispatch）。vLLM、HuggingFace Transformers、xFormers 等框架广泛使用。Flash-Decoding 针对 decoding phase（小 seqlen 大 batch）在 seqlen 维度额外并行化。FlashDecoding++ 进一步优化 softmax 和 flat GEMM 操作，并提供 AMD GPU 支持。对于长序列 >512，FlashAttention 具有显著的时延和内存优势。

涉及论文标题：
- A Survey of Resource-efficient LLM and Multimodal Foundation Models
