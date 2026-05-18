## Scattered Co-location (散布共置)

术语是什么？

Scattered Co-location（散布共置）是μShare提出的替代stacked co-location的intra-SM kernel调度策略：通过将部分kernel的blocksize设置为略超半SM thread容量的值（half-plus blocksize），迫使同kernel的blocks散布到不同SM（因为同一SM无法容纳两个超过半容量的block），剩余threads可被其他kernel的小block占用，实现不同kernel的blocks在同一SM内的交叉分布（"scattered"）。与stacked co-location中同kernel blocks独占SM不同，scattered co-location使SM内同时运行不同kernel的blocks，各block使用不同的dominant hardware resource，互补利用SM内的多种hardware unit。

从kernel调度角度拆解术语：

Scattered co-location的调度流程（以μShare on A40, 1536 threads/SM, half-plus blocksize=800为例）：

```
// Kernel set O被分为X（half-plus shaped）和Y（time-shifted launch）
// X: vectorized_kernel (LDST dominant), blocksize=800 (half-plus)
// Y: roll_kernel (INT32 dominant), blocksize=512 (default)

// Dispatch决策：
// SM0空闲，剩余thread = 1536
//   T1: dispatch unit取vectorized_kernel block_0 (800 threads)
//       → 剩余thread = 1536 - 800 = 736
//   T2: dispatch unit取vectorized_kernel block_1 (800 threads)
//       → 800 > 736 → 无法放入SM0 → 分配到SM1
//   T3: dispatch unit取roll_kernel block_0 (512 threads)
//       → 512 ≤ 736 → 可以放入SM0
//   → SM0: vectorized block_0 (800 threads, LDST 58%) + roll block_0 (512 threads, INT32 33%)
//     共800+512=1312 threads ≤ 1536
//   → SM0内同时执行LDST-heavy和INT32-heavy计算
//   → 6种HW utilization: LDST ~58% + INT32 ~33% + SFU ~11%(vec) + ~25%(roll) + ...
//     相比stacked时只用LDST 58%或INT32 33%单独一种
//     avg low-level HW utilization: 15.10% (scattered) vs 10.90% (stacked INFless)
```

关键约束：
- Half-plus blocksize = thalf + α, 其中 thalf = SM_thread_capacity / 2 = 768 (A40), α最小为warp size (32)，所以最小800
- 对A800/A100/H200 (2048 threads/SM)，改用1/3-plus: blocksize = 2048/3 + α ≈ 704，允许同kernel两个1/3-plus block入1SM (704×2=1408, 剩余640)
- α动态调整: slack positive → α=32 (最小); slack negative → α逐步+32加速kernel

术语一般如何实现？如何使用？

Scattered co-location的实现需要：
1. **Kernel拦截**：通过LD_PRELOAD劫持CUDA kernel launch函数（cudaLaunchKernel/cublasSgemm），读取/修改blocksize参数
2. **Blocksize塑形**：根据SM thread capacity计算half-plus值 → 写入modified blocksize → 调用原始launch函数
3. **时移启动**：对unmodifiable kernel（cuDNN/cuBLAS/tiling kernel），检查其6种hardware resource utilization + shared memory/registers与当前SM中active kernel的combined utilization是否≤100% → 满足则立即launch → 不满足则delay β=10μs后重检
4. **资源互补判断**：基于offline profiled per-kernel 9-tuple {rFP32, rFP64, rINT32, rLDST, rSFU, rTensor, rmem, rreg, tLaunch}选择co-location pairing。实验显示：dominant resource不同的kernel配对时half-plus提升throughput 19.94%，相同>10.37%下降

涉及论文标题：
- μShare: Non-Intrusive Kernel Co-Locating on NVIDIA GPUs

