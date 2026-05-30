## GEMM-Softmax Pipelining（矩阵乘-Softmax流水线重叠）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GEMM-Softmax Pipelining 是 FlashAttention-3 提出的 intra-warpgroup 级别的异步流水线技术。在 tiled attention 的 inner loop 中，每个迭代天然存在串行依赖：WGMMA(QK^T) → softmax → WGMMA(PV)，但由于 Hopper WGMMA 是异步指令（发射后立即返回），可以通过跨迭代缓冲打破依赖。核心思想（2-stage）：将第 j+1 次迭代的 QK^T WGMMA 与第 j 次迭代的 softmax 重叠执行，同时将第 j 次迭代的 PV WGMMA 与第 j+1 次迭代的 softmax 重叠执行。具体地，在 iteration j 中：(1) 发射 WGMMA(QK^T) of iter j+1（存于寄存器 $\mathbf{S}_{\text{next}}$，commit but no wait）；(2) 发射 WGMMA(PV) of iter j（使用 $\tilde{\mathbf{P}}_{\text{cur}}$，commit but no wait）；(3) 等待 WGMMA(QK^T) 完成 → softmax on $\mathbf{S}_{\text{next}}$ → 生成 $\tilde{\mathbf{P}}_{\text{next}}$；(4) 等待 WGMMA(PV) 完成 → rescale $\mathbf{O}_i$。关键：步骤 (3) 的 softmax（CUDA core执行，仅3.9 TFLOPS）与步骤 (2) 的 WGMMA(PV)（Tensor Core异步执行，989 TFLOPS）在时间上重叠——Tensor Core 执行 PV 乘法的同时，CUDA Core 执行下一迭代的 exponential 和 rowmax。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。
FlashAttention-3 2-stage GEMM-Softmax Pipelining 的 consumer warpgroup 伪代码（Algorithm 2）：
```
// Prologue (j=0): no overlap, initialize pipeline
Wait for Q_i, K_0 in SMEM
S_cur = WGMMA(Q_i, K_0^T)             // SS-WGMMA, commit + wait
Release K stage 0
softmax_on(S_cur) → m_i, P̃_cur, ℓ_i    // standard softmax
rescale O_i

// Mainloop (j = 1..T_c-1): 2-stage overlap
for j in 1..T_c-1:
    Wait for K_j in SMEM
    S_next = WGMMA(Q_i, K_j^T)         // commit, NO WAIT — fires asynchronously
    Wait for V_{j-1} in SMEM
    O_i += P̃_cur × V_{j-1}              // RS-WGMMA, commit, NO WAIT
    Wait for S_next WGMMA completion    // barrier: sync on QK^T
    softmax_on(S_next) → m_i, P̃_next, ℓ_i  // overlaps with PV WGMMA above
    Wait for PV WGMMA completion        // barrier: sync on PV
    rescale O_i based on m_i
    release K/V stages
    Copy: S_cur ← S_next, P̃_cur ← P̃_next

// Epilogue (j = T_c-1): finish last iteration
Wait for V_{T_c-1} in SMEM
O_i += P̃_last × V_{T_c-1}               // final PV, commit + wait
O_i = O_i / ℓ_i, L_i = m_i + log(ℓ_i)
write O_i, L_i to HBM
```

SASS分析验证（Paper §B.2）：compiler将softmax指令（FMNMX, MUFU.EX2, FADD）重排到第二个WGMMA之前，第一个WGMMA（QK^T split为8个HGMMA）的前7个与FP32→FP16转换和softmax的rowsum交错发射，验证了compiler正确实现了overlap。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GEMM-Softmax pipelining 需要：(1) Hopper WGMMA 异步指令支持（通过 wgmma.commit_group + wgmma.wait_group 管理异步执行）；(2) 额外寄存器缓冲存储 S_next（大小为 $B_r \times B_c \times$ sizeof(float) per threadblock），增加register pressure——需与block size trade-off；(3) 编译器协同——NVCC可能重新排列指令顺序，需要验证SASS确保overlap确实发生。FlashAttention-3中该技术将FP16 forward从570 TFLOPS提升至620-640 TFLOPS（~12% gain）。3-stage变体（更多重叠但更大register压力）实测不如2-stage，因为register spilling抵消了overlap收益。

涉及论文标题：
- FlashAttention-3 Fast and Accurate Attention with Asynchrony and Low-precision
- FlashAttention-T: Towards Fully Tensorized Attention by Exploiting Tensor-Vector Parallelism
