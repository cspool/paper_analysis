## Partial Fusion (Kernel Fusion Strategy for Block Low-Rank Multiplications)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Partial Fusion是针对BLR（Block Low-Rank）矩阵乘法的GPU kernel融合策略：将BLR前向中的选定相邻操作（如permutation+bmm）融合为单个Triton kernel。解决两个极端的缺陷：(1) Full fusion——将整个BLR线性层所有操作融合为单kernel，因2-D output tiling导致冗余weight加载和中间结果重计算，且1-D tiling受限于shared memory容量（仅rank≤128可行）；(2) No fusion（PyTorch baseline）——每个操作独立kernel launch，中间张量通过global memory传递→memory-bound瓶颈。

从kernel调度角度拆解术语，给出具体例子。
Monarch ② partial fusion kernel伪代码（fuse b₂↔b₁ permutation + first bmm）：

```
// 输入: X ∈ R^{n×i}分块 [n, b₁, p]; V ∈ R^{b₁×p×(r'b₂)} 已重排布
// 输出: Z' ∈ R^{n×(b₂·b₁·r')} 直接写入permuted layout

parfor b_1 in 0..b₁-1, n_tile, r_tile:
  // 计算permutation目标索引
  b_2 = (r_start : r_end) // r'                    // 确定b₂
  r'_off = (r_start : r_end) % r' + b_1 * r'       // rank偏移

  acc = zeros(t_n, t_r)
  for p_tile in 0..ceil(p/t_p)-1:
    x = X[b_1, n_s:n_e, p_s:p_e]                   // load X tile
    v = V[b_1, p_s:p_e, r_s:r_e]                   // load V tile
    acc += dot(x, v)                                // Tensor Core MMA
  
  Z'[n_s:n_e, b_2 * n * r' + r'_off] = acc         // write to permuted position
```

BLAST ④ partial fusion（消除V→S中间物化，循环b₁维度做S-weighted累加）:
```
parfor n_tile, r_tile:
  z''= zeros(b₂, t_n, t_r)
  for b_1 in 0..b₁-1:                     // 循环而非并行b₁维度
    s = S[b_1, :, r_s:r_e].view(b₂, 1, t_r)
    z' = zeros(t_n, t_r)
    for p_tile:                            // 第一个bmm (tensor core)
      x = X[b_1, n_s:n_e, p_s:p_e]; v = V[b_1, p_s:p_e, r_s:r_e]
      z' += dot(x, v)
    z'' += s * z'                          // S-weighted累加 (CUDA core)
  Z''[:, n_s:n_e, r_s:r_e] = z''
```

术语一般如何实现？如何使用？
在Triton中实现，使用autotuner选择tile sizes（t_n, t_r, t_p ∈ {32,64,128,256}）。关键约束：(1) fused kernel shared memory ≤ SM capacity；(2) 保持足够并行度（grid dim ≥ SM数量）；(3) 优先保留tensor core利用率（BLAST ⑤ > ④的原因：⑤保持全部bmm在tensor core，④将第二bmm降级到CUDA core batched outer product）。适用所有BLR压缩模型的线性层（QKVproj, Oproj, gate/upproj, downproj, c_attn, c_fc等）。不适用单token推理（n=1, memory-bound下未融合已够快）。

涉及论文标题：
- Memory-Efficient Acceleration of Block Low-Rank Foundation Models on Resource Constrained GPUs

---
