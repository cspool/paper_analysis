## Block Order Scheduling / L2 Cache Reuse (GPU)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Block Order Scheduling 是通过控制 GPU thread block 在 grid 中的执行顺序来最大化 L2 cache reuse 的调度策略。GPU 的 thread block 无法直接通信——数据共享必须通过 HBM（经过 L2 cache）。当相邻执行的 block 访问相同数据区域时，数据保持在 L2 cache（50MB, 12 TB/s）而非从 HBM（3 TB/s）重载。Block order 由 GPU 硬件调度器根据 blockIdx 分配决定，但 kernel 可通过 grid dimension 和 blockIdx→数据坐标的映射影响哪些 block 连续执行。TK 展示了 block order 对性能的惊人影响。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
GEMM M=N=K=16384 的对比：
```
策略 A: 3D Stride {8, N, M/8}
  row = 8*(task_id/super_repeat) + task_id%8
  col = (task_id/8) % N
  → 相邻 block 在 row 方向连续 → A 矩阵行数据 L2 reuse
  → HBM: 982 GB/s, 805 TFLOPS

策略 B: Row-Major {N, M}
  row = blockIdx.x / N, col = blockIdx.x % N
  → 相邻 block 遍历不同 row，cache 无法保存 B 矩阵所有列
  → HBM: 3070 GB/s (L2 miss!), 仅 392 TFLOPS
```
策略 B 的 HBM 带宽更高但性能减半——高带宽意味着 L2 cache miss，数据被迫从慢速 HBM 加载。策略 A 通过连续 block 复用相邻数据使大部分访问命中 L2。

Attention forward (d=128):
- 优化 order {N, H, B}（sequence 连续）: HBM 213 GB/s, 600 TFLOPS
- Naive order {B, H, N}（batch 连续）: HBM 2390 GB/s, 494 TFLOPS
优化 block order 提升 21% 性能，同时 HBM 带宽降低 91%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TK 在 common_setup 中通过 task_id→(row,col) 映射实现——用户选择 3D stride 维度（如 GEMM 用 SUPER_M 参数控制 cluster 行数），映射公式直接影响 L2 reuse。与 persistent grid 配合效果最佳：block 连续执行多个 task 时，L2 cache 可以在 task 之间保持数据。

涉及论文标题：
- ThunderKittens: Simple, Fast, and Adorable Kernels
