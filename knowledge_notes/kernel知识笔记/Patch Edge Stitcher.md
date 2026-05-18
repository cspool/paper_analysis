## Patch Edge Stitcher

术语是什么？通过联网搜索让回答具体和精准。

Patch Edge Stitcher是MixFusion提出的一种fused CUDA kernel，将扩散模型中Convolution算子的跨patch边界stitching操作融合(fuse)到GroupNorm kernel中执行。核心观察是T2I扩散模型（SDXL等U-Net架构）中Convolution通常紧接GroupNorm操作——因此可在normalization过程中overlap boundary data exchange，消除独立的stitching kernel调用和额外的global memory round-trip。替代方案包括：(a) naive stitching——在Convolution前fetch所有邻接patch的boundary pixels并concat→额外的memory movement开销完全抵消patch parallelism收益（Figure 5）；(b) Ghost Zone——直接replicate边界像素（在科学计算stencil中常用），但扩散模型从noise生成图像，相邻patch间无natural locality→产生明显patch边界artifact（Figure 6, PSNR仅9.54/SSIM 0.45）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Patch Edge Stitcher fused with GroupNorm kernel的执行流程（Figure 10d）：

```
每个Thread Block (TB) 处理一个patch:
1. TB加载patch数据到registers/shared memory
2. 执行GroupNorm计算：
   - 计算patch内mean和variance
   - normalize: (x - mean) / sqrt(var + eps)
   - scale and shift: gamma * x_norm + beta
3. 同步检查依赖metadata（在patch splitting时预计算）：
   - 该patch的哪些boundary pixels被邻接patch需要？
   - 例如：P0的右边界被P1需要，下边界被P2需要
   - 例如：P3的左边界被P2需要，上边界被P1需要
4. 将被需要的boundary pixels写入TB的shared memory
5. __syncthreads() // 等待所有TB完成normalization
6. 根据metadata定位目标patch：
   - 遍历该TB需要写入boundary的目标patch
   - 将shared memory中的boundary data写回global memory对应位置
   - 处理position diversity：
       * row boundary: 连续内存访问（对齐memory layout）
       * column boundary: 通过shared memory中转（避免irregular global memory access）
7. 目标patch的TB可直接读取准确的boundary值用于Convolution
```

关键设计：(1) Shared memory作为boundary exchange的中间缓冲——column boundary的irregular memory access通过shared memory中转local化，row boundary直接高效对齐读写；(2) 消除额外synchronization——stitching与GroupNorm在同一个fused kernel内完成，不需要kernel launch overhead或global barrier；(3) Direction & Position Diversity处理——通过pre-computed dependency metadata统一处理不同位置patches的不同stitching方向（四个边界的子集）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Patch Edge Stitcher在MixFusion中以CUDA C++实现，作为PyTorch custom operator集成到diffusion pipeline的Convolution模块前。Dependency metadata在patch splitting阶段预计算（记录每个patch的四邻接patch ID及需要的boundary范围→生成TB间通信计划）。在SDXL上的evaluation显示：(a) latency overhead minimal——Figure 5中PES的latency接近理论optimal（无stitching overhead的patched execution latency）；(b) quality显著优于alternatives——Table 4中PES w/ 4 patches达到PSNR 28.82/SSIM 0.88 vs Ghost Zone 9.54/0.45, vs Distrifusion 10.96/0.49；(c) SD3模型无Convolution→自然100% accuracy（PSNR inf/SSIM 1.0）。

涉及论文标题：
- MixFusion: A Patch-Level Parallel Serving System for Mixed-Resolution Diffusion Models

