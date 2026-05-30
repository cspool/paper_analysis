## FlashAttention-2 Tiling Strategy (FlashAttention-2 Tiling策略)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FlashAttention-2 Tiling Strategy 是 FlashAttention-2 (Dao, 2023) 提出的将 Attention 计算分解为小块（tiles）以消除 N×N 中间矩阵 HBM I/O 的算法-系统协同设计。核心思想：将 Q 沿 token 维分为 b_q 大小的 tiles {Q_i}，K,V 分为 b_kv 大小的 tiles {K_j},{V_j}，使用 online softmax 逐步累加 O_i，使得 N×N 的 S 和 P 矩阵永远不需要整体写入 HBM。FlashAttention-2 改进点：减少 non-matmul FLOPs、优化 parallelism（outer loop on Q 并行于 SMs、inner loop on KV 串行）、优化 warp partition。SageAttention 在 FlashAttention-2 tiling 基础上叠加 INT8 量化——Q,K tile 在加载到 SRAM 后先量化为 INT8，再执行 Tensor Core MMA。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# FlashAttention-2 Tiling (SageAttention adaptation)
# Grid: T_m = N/b_q SM blocks, each handles one Q tile
# Thread block i (runs on one SM):
Load Q̂_i_INT8 [b_q×d] + δ_Q[i] from HBM to SRAM     # outer tile
for j in 1..T_n:                                      # inner loop
    Load K̂_j_INT8 [b_kv×d] + V_j_FP16 [b_kv×d] + δ_K[j] to SRAM
    S_ij = INT8_MMA(Q̂_i, K̂_j^T) * δ_Q[i] * δ_K[j]    # [b_q×b_kv], SRAM resident
    O_i = online_softmax_update(O_i, S_ij, V_j)       # in SRAM/registers
Write O_i_FP16 [b_q×d] to HBM
```
SageAttention 的 block sizes: b_q=128, b_kv=64（比 FlashAttention-2 默认的 b_q=128 和 b_kv=64 保持一致）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FlashAttention-2 是当前 GPU attention 计算的事实标准。CUDA 实现（原版）：直接编写 CUDA C++ kernel，管理 shared memory allocation（Q_tile + K_tile: 2×b_q×d×2bytes + 2×b_kv×d×2bytes）。Triton 实现（SageAttention）：`tl.dot()` 自动管理 shared memory。与标准 FlashAttention-2 的差异：(1) SageAttention 在 K 加载后 fuse smooth K（减去 mean）和 INT8 量化；(2) QK MMA 从 FP16→INT8；(3) PV MMA 从 FP16+FP32 accum→FP16+FP16 accum。tiling pattern（outer Q loop + inner KV loop）保持不变。开源: FlashAttention-2 https://github.com/Dao-AILab/flash-attention, SageAttention https://github.com/thu-ml/SageAttention。

涉及论文标题：
- SageAttention2 Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization
