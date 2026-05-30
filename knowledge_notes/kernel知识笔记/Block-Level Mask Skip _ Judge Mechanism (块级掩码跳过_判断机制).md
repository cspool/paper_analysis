## Block-Level Mask Skip / Judge Mechanism (块级掩码跳过/判断机制)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Block-Level Mask Skip / Judge Mechanism 是 Dynamic Mask Attention (DMA) CUDA kernel 中的核心优化技术，在 FlashAttention 风格的 tiled 计算中通过 block 级别的 mask 预判来跳过不产生有效贡献的计算区域。核心思想：在 outer loop 加载每个 K/V block 之前，先加载对应的 mask block 并调用 Judge(M_j) 判断该 block 是否全为零（所有元素为 −∞），若是则直接 advance stream pointers 跳过整个 K/V block 的 HBM 加载和矩阵乘法，避免冗余内存访问和无效计算。

与 AdaSplash 的 Block Masking（基于 α-entmax τ 阈值构造 binary mask）不同，DMA 的 Judge 机制直接利用 DMA 生成的连续值 mask（−∞ / 有效分数），通过 block 级 all-zero 检测实现跳过。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

**Forward Pass with Judge（Algorithm 1 in paper）**：
```
Require: Q, K, V ∈ R^{N×d_h}, M ∈ R^N in HBM. Block size B.
Divide Q into T_r blocks, K/V into T_c blocks, M into T_c blocks.
O ← 0, ℓ ← 0, m ← (−∞) in HBM.

for 1 ≤ j ≤ T_c:                          // outer loop over K/V blocks
    Load M_j from HBM to SRAM.
    active_j = Judge(M_j)                 // 检查 block 是否全零
    if active_j == 0:                     // 全零 → 跳过
        Advance stream pointers for K_j, V_j.
        continue                          // 不加载 K_j, V_j!
    
    Load K_j, V_j from HBM to SRAM.       // 仅在 active 时加载
    for 1 ≤ i ≤ T_r:                      // inner loop over Q blocks
        Load Q_i, O_i, ℓ_i, m_i from HBM.
        S_ij = Q_i @ K_j^T × d_h^{−0.5} + M_j   // [B×B]
        m̃_ij = rowmax(S_ij)
        P̃_ij = exp(S_ij − m̃_ij)
        ℓ̃_ij = rowsum(P̃_ij)
        m_i^new = max(m_i, m̃_ij)
        ℓ_i^new = exp(m_i − m_i^new)·ℓ_i + exp(m̃_ij − m_i^new)·ℓ̃_ij
        O_i ← diag(ℓ_i^new)^{−1}(diag(ℓ_i)·exp(m_i−m_i^new)·O_i + exp(m̃_ij−m_i^new)·P̃_ij·V_j)
        Write O_i, ℓ_i, m_i to HBM.
Return O
```

**Backward Pass with Judge**：与 forward 共享统一 skip logic。在加载 K_j/V_j 前同样执行 Judge(M_j)，active=0 则跳过。关键优化——dM = dS（mask 梯度等于 score 梯度），kernel 只需局部重算 S 而无需额外存储中间 mask 梯度张量。

**Judge 实现细节**（来自 GitHub issues）：
- 使用 warp ballot 或 reduce 检测 mask block 是否全部为 −∞
- 可 bitpack mask block（128 bit per tile）做常数时间 all-zero 检测
- 可选实现：Phase A 扫描所有 mask tiles 生成 active bitmap → Phase B 仅迭代 active tiles

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

开源实现：https://github.com/HKUSTDial/flash-sparse-attention (kernel 实现在 `flash_fwd_kernel.h` / `flash_bwd_kernel.h`)，Triton 参考实现：`flash_dmattn_triton.py`。

核心文件结构：
- `mask.h`：动态 mask 计算逻辑（含 `apply_mask`）
- `utils.h`：稀疏 GEMM（`sparse_gemm`, `sparse_gemm_rs`）
- `flash_fwd_kernel.h`：前向 kernel（含 Judge + block skip）
- `flash_bwd_kernel.h`：反向 kernel（共享 skip logic）
- `flash_dmattn_triton.py`：Triton 参考实现

Judge 机制支持四种模式（via GitHub Issue #161）：
| Case | attn_mask | attn_bias | Behavior |
|------|-----------|-----------|----------|
| A | None | None | 全 dense 路径，无 block skip |
| B | Tensor | None | 用 mask 做 block skip |
| C | None | Tensor | 无 block skip（所有 block active） |
| D | Tensor | Tensor | mask skip + bias |

性能：在 A100-SXM4-80GB 上，DMA forward 相对 MHA (FlashAttention) 在 8192/16384/32768 token 长度分别提速约 26.1×/10.2×/21.5×；decode 在 65536/131072/262144 key 长度分别提速约 49.6×/92.7×/171.1×；backward 在 8192/16384/32768 分别提速约 2.5×/4.4×/7.9×。

优化技巧：shared memory aliasing（复用共享内存）、pipelined prefetching（预取下一个 block）、coalesced memory accesses（合并内存访问减少带宽压力）。
