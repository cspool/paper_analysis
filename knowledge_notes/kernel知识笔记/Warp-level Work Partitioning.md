## Warp-level Work Partitioning

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Warp-level Work Partitioning（Warp级工作划分）是GPU kernel设计中决定thread block内各warp（32线程组）如何分配计算任务的策略。NVIDIA GPU的thread block由多个warp组成（通常4或8个warp，即128或256线程），warp是SIMT执行的最小调度单元。Warp间可通过shared memory通信（需`__syncthreads()` barrier）或shuffle指令通信（同一warp内线程间，warp shuffle不跨warp）。工作划分直接影响：(1) shared memory使用量——不当划分导致warp间需大量shared memory通信；(2) 寄存器压力——各warp持有不同数据片，需在寄存器容量（255 registers/thread on A100）和spilling间平衡；(3) 计算效率——划分需保证各warp计算量均衡，避免load imbalance。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FlashAttention-2 forward pass的warp划分（4 warps per thread block, B_r=128, B_c=128, d=128为例）：

```
Thread Block配置: 4 warps × 32 threads = 128 threads

Shared Memory布局（per thread block）:
  K_tile: [B_c=128, d=128] FP16 = 32KB
  V_tile: [B_c=128, d=128] FP16 = 32KB
  总计: ~64KB shared memory (fit 192KB A100 SM SRAM)

Register布局（per warp）:
  Warp 0: Q[0:31, 128]    = 32×128×2B = 8KB (in registers, 32 rows of Q)
  Warp 1: Q[32:63, 128]   = 8KB
  Warp 2: Q[64:95, 128]   = 8KB
  Warp 3: Q[96:127, 128]  = 8KB
  Each warp also holds: O_tilde[32,128], m[32], ℓ[32], S[32,128], P[32,128]

Per-warp execution (warp w, 完全独立, 无warp间通信):
  for j in 0..T_c-1:                          # 遍历所有KV blocks
      // 从shared memory读取K_j, V_j（所有warp同时读，无bank conflict）
      S = Q_w @ K_j.T                          # [32, 128] Tensor Core MMA
      m_new = max(m, rowmax(S, dim=1))         # CUDA core: 32×128 reduction
      m_rescale = exp2(m - m_new)              # MUFU.EX2, 32 values
      P = exp2(S - m_new[:, None])             # MUFU.EX2, 32×128 values
      ℓ_new = m_rescale * ℓ + rowsum(P, dim=1) # CUDA core: reduction + FMA
      O_w = diag(m_rescale) * O_w + P @ V_j    # Tensor Core MMA + elementwise
      m = m_new; ℓ = ℓ_new
  // 最终rescale:
  O_w = O_w / ℓ[:, None]                       # CUDA core: 32×128 divisions
  // Write O_w (32 rows) to HBM output
```

关键优势：相比FlashAttention v1的split-K，warp间零通信；(1) 无`__syncthreads()` barrier；(2) 无shared memory用于partial results交换；(3) 每个warp的work完全embarrassingly parallel。Shared memory仅用于存储K_j、V_j tile（所有warp只读共享）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FlashAttention-2在CUTLASS 3.x中通过`TiledMMA`和自定义`Collective`实现warp划分。具体地：(1) 使用`cutlass::gemm::GemmShape<128,128,128>`定义MMA tile size；(2) CuTe的`Layout`和`Tile`抽象定义各warp持有的Q slice映射（`make_tile(Layout<Shape<_32,_128>>)`）；(3) `TiledCopy`将Q从global memory分片加载到各warp的寄存器；(4) K/V通过`TiledCopy`加载到shared memory后，所有warp通过CuTe的thread-to-data映射访问。Block size和warp数量手动tune：head_dim=64时用{B_r=128, B_c=128}；head_dim=128时也适用{B_r=64, B_c=64}或{B_r=128, B_c=128}，取决于shared memory和register限制。

涉及论文标题：
- FlashAttention-2 Faster Attention with Better Parallelism and Work Partitioning
