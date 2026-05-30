## IO-Awareness / IO-Aware Algorithm

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

IO-Awareness（IO感知）是一种算法设计原则，要求算法明确考虑不同层级内存之间的读写（IO）开销，而不仅仅是算术运算（FLOPs）数量。该概念源于Aggarwal & Vitter (1988)的IO复杂度理论，FlashAttention论文将其引入深度学习attention计算领域。核心观察：现代GPU上计算速度已远超内存速度（A100 SRAM带宽~19TB/s vs HBM带宽~1.5-2.0TB/s，~10×差距），大多数Transformer操作是memory-bound而非compute-bound。IO-aware算法的目标是通过reorganizing computation来减少慢速内存（HBM）的访问次数，即使这意味着增加FLOPs，因为HBM带宽才是真正的瓶颈。FlashAttention通过tiling将$N \times N$ attention矩阵的HBM读写从$\Theta(N^2)$降至$\Theta(N^2d^2M^{-1})$（M为SRAM大小），实测HBM读写减少8×，整体加速3×，同时FLOPs反而从66.6增至75.2 GFLOPs。这一反直觉结果（FLOPs增但速度更快）正是IO-awareness的核心理念：memory access > compute。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

IO-awareness在attention计算中的应用（以FlashAttention为例）：
```python
# IO-unaware (standard attention): 3次独立操作，每次经HBM
S = Q @ K.T          # Step 1: GEMM, write S[N,N] to HBM
P = softmax(S)       # Step 2: softmax, read S from HBM, write P[N,N] to HBM
O = P @ V            # Step 3: GEMM, read P from HBM, write O to HBM
# HBM traffic = 2*Nd (Q,K input) + 2*N² (S write+read) + 2*N² (P write+read) + Nd (V input) + Nd (O output)
# ≈ 3Nd + 4N² elements

# IO-aware (FlashAttention): 单kernel，block-wise计算，无N×N矩阵在HBM
# Block sizes: B_c ≈ M/(4d), B_r = min(B_c, d)
for j in range(T_c):                    # outer loop: KV blocks in SRAM
    load K_j[ B_c x d ], V_j[ B_c x d ] from HBM to SRAM
    for i in range(T_r):                # inner loop: Q blocks
        load Q_i[ B_r x d ] from HBM to SRAM
        S_ij = Q_i @ K_j.T              # in SRAM: B_r x B_c
        m_new = max(m_i, rowmax(S_ij))  # online softmax stats
        l_new = exp(m_i-m_new)*l_i + sum(exp(S_ij - m_new))
        O_i = (l_i*exp(m_i-m_new)*O_i + exp(S_ij-m_new) @ V_j) / l_new
        save m_i, l_i, O_i to HBM       # only O(N) per write, NOT O(N²)
# HBM traffic = O(N²d²/M) << O(N²)
```
关键：中间S_ij和P_ij仅驻留SRAM，逻辑流程中的每个步骤都设计为最小化HBM交互。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

IO-awareness的实现方式：(1) **Tiling/Blocking**：将大数据分解为fit in fast memory的小块，分批处理；(2) **Kernel fusion**：将多个操作合并为单个kernel，消除中间结果的slow memory round-trip；(3) **Recomputation**：用compute换取memory——不存储中间结果而是重新计算；(4) **Memory hierarchy-aware scheduling**：根据各层内存的带宽/容量特性安排数据驻留位置。在FlashAttention中，这些技术组合使用：tiling确保每block fit in SRAM，kernel fusion消除kernel间HBM传输，recomputation消除backward的O(N²)存储需求。实际使用：`flash_attn_func(q, k, v)`作为PyTorch中标准attention的直接替代，自动在kernel内部应用所有IO-aware优化。IO-awareness理念已扩展到FFN（SRAMFFN/FlashMHF）、通信重叠（FlashOverlap）等场景。

涉及论文标题：
- FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness
