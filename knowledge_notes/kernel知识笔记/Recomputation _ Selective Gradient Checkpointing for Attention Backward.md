## Recomputation / Selective Gradient Checkpointing for Attention Backward

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Recomputation（重计算/选择性梯度检查点）是一种用计算换取内存的技术：前向pass不保存某类中间激活值（或仅保存其压缩表示），反向pass需要时重新计算这些值。标准gradient checkpointing（Chen et al., 2016）在深度学习中被用于减少训练峰值内存，但通常以牺牲速度为代价（recompute FLOPs > saved HBM reads）。FlashAttention的创新在于反向recomputation反而加速了训练：前向仅保存$O(N)$的softmax统计量（output O、running max m、running sum l），而非标准的$O(N^2)$ attention矩阵S和P。反向pass中，从HBM加载Q/K/V块到SRAM，利用保存的LogSumExp $L_i = m_i + \ln(\ell_i)$（每query row一个scalar）恢复softmax概率：$P_{ij} = \exp(S_{ij} - L_i)$，进而计算dQ/dK/dV。关键洞见：因为HBM带宽是真正的瓶颈（而非FLOPs），虽然重计算增加了FLOPs（GFLOPs从66.6→75.2，+13%），但消除了$O(N^2)$量的HBM读取（矩阵P, 2GB per 16 heads for N=1024），总wall-clock time反而从35.1ms降至11.7ms（3× faster）——与标准checkpointing的"speed-for-memory trade-off"完全相反（FlashAttention's recomputation is both faster AND more memory-efficient）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FlashAttention backward pass的recomputation流程：
```
// BWD kernel利用前向保存的O, m, l重计算P_ij（在SRAM中）
__global__ void flash_attn_bwd_kernel(Q, K, V, O, dO, m, l, dQ, dK, dV, ...) {
    // Step 1: 从保存的统计量恢复softmax概率（no HBM read for P!）
    for each (i,j) block pair in the backward traversal:
        S_ij = Q_i @ K_j.T              // Recompute scores in SRAM (Tensor core)
        L_i = m_i + logf(l_i);          // Retrieve LogSumExp (1 float per row, from HBM, O(N) total)
        P_ij = exp(S_ij - L_i)          // Recover softmax probs in SRAM (NO O(N²) HBM read!)
        
        // Step 2: 标准attention backward（所有中间值在SRAM中）
        dV_j += P_ij.T @ dO_i           // (B_c x B_r) @ (B_r x d) → (B_c x d)
        dP_ij = dO_i @ V_j.T            // (B_r x d) @ (d x B_c) → (B_r x B_c)
        dS_ij = P_ij * (dP_ij - rowsum(dP_ij * P_ij))  // Softmax backward: diag(p)-pp^T
        dQ_i += dS_ij @ K_j             // (B_r x B_c) @ (B_c x d) → (B_r x d)
        dK_j += dS_ij.T @ Q_i           // (B_c x B_r) @ (B_r x d) → (B_c x d)
}

// 对比标准attention backward:
//   需要从HBM读取整个P矩阵(N×N=2GB)来计算dS_ij = dP_ij ⊙ P - P ⊙ rowsum(dP_ij ⊙ P)
//   FlashAttention用~6KB的L_i (N floats = 4KB) + Q/K/V tile loads替代了2GB的HBM read
```
Memory对比：standard backward需要存储S（N×N, 2GB）+ P（N×N, 2GB）用于梯度计算；FlashAttention仅需存储O（Nd, 128KB）+ m（N, 4KB）+ l（N, 4KB）。Memory减少约500×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

通用gradient checkpointing：PyTorch的`torch.utils.checkpoint.checkpoint`标记不保存中间tensor的函数，反向时重新执行该函数。FlashAttention的recomputation将这一机制内化到单个CUDA kernel中，比框架级checkpointing更高效：(a) 仅需保存compressed stats而非重新执行整个forward；(b) recomputation与backward计算融合在同一kernel中，消除额外kernel launch和HBM IO。此技术启发了一系列工作：FlashInfer、xFormers的memory-efficient attention、以及FlashAttention-2/3/4中改进的backward pass。用户使用透明：`flash_attn_func`自动使用此机制。

涉及论文标题：
- FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness
