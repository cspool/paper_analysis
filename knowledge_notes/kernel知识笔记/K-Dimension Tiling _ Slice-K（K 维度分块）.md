## K-Dimension Tiling / Slice-K（K 维度分块）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
K-Dimension Tiling（也称 Slice-K）是 GPU GEMM 中将矩阵乘法的 inner dimension（K 维度）进一步切分为多个子 tile 以增加并行度、提升 shared memory 利用率的技术。在 MxMoE 的混合精度 Group-GEMM 中，不同精度的 micro-kernel 天然使用不同的 tile size：例如 W4A16 的 tile 比 W8A8 的 tile 显著更小。当这些 micro-kernel 水平融合时，必须统一 shared memory 分配（取最大值），导致小 tile 的 micro-kernel 出现 shared memory under-utilization。Slice-K 解决方案：对 W4A16 配置额外沿 K 维度切分 tile——将单个大 K-tile 分为多个子 K-tile，每个子 K-tile 增加 warp utilization 并更好利用分配的 shared memory。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
标准 Tile (无 Slice-K):
  W4A16 tile: [M=128, N=64, K=256] → 使用 shared memory = 128*64*2B = 16KB
  W8A8 tile:  [M=128, N=64, K=256] → 使用 shared memory = 128*64*4B = 32KB
  统一分配 32KB shared memory → W4A16 浪费 16KB

With Slice-K (K 切分为 2):
  W4A16 tile: [M=128, N=64, K=128] × 2 个子 tile
  每个子 tile:
    - shared memory = 128*64*2B = 16KB (×2 子 tile = 32KB 充分利用)
    - warp utilization 提升 (2 倍子 tile 并行计算)
    - 子 tile 结果累加得到最终输出
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Slice-K 通过调整 tile configuration y_{i,j,k,t} 实现。在 ILP 求解阶段，tile 配置已包含 K 切分选项，ILP 自动选择最优的分片数和 tile shape 组合。CUTLASS 原生不支持 slice-K 与混合精度的联合优化，MxMoE 通过 kernel generator 自动注入 K-splitting logic。

涉及论文标题：
- MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design
