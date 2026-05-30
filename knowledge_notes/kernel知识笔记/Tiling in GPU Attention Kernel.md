## Tiling in GPU Attention Kernel

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Tiling（分块/tile化）在GPU kernel设计中指将大型矩阵/张量计算分解为较小tile（通常fit在on-chip SRAM/shared memory中）的技术。每个tile从HBM加载到SRAM，在SRAM中计算，累加中间结果，然后移到下一个tile。FlashAttention将tiling应用于attention计算——Q/K/V沿sequence length维度分块，双loop结构（outer: KV blocks加载到SRAM一次，inner: Q blocks迭代），每个(i,j) block pair的中间结果在SRAM中产生、消费并立即丢弃。核心权衡是block size选择：更大的block减少HBM passes（更好IO efficiency）但需要更多SRAM（可能降低SM occupancy）。FlashAttention的block size公式：$B_c = \lceil\frac{M}{4d}\rceil$，$B_r = \min(B_c, d)$。对A100（M=192KB SRAM, d=64, FP16）：B_c≈384, B_r=64（被d bound）。Block size消融（Figure 2 middle）验证了IO-aware设计的核心假设——随着B_c从64增至256，HBM accesses减少，runtime持续下降；B_c≥256后进入compute-bound regime。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FlashAttention的tiling scheme与每block pair的SRAM budget分析（A100, B_c=384, B_r=64, d=64, FP16）：
```
每block pair（i,j）的SRAM分配:
  K_j tile: B_c × d × 2B = 384 × 64 × 2 = 49,152 B ≈ 49KB
  V_j tile: B_c × d × 2B = 384 × 64 × 2 = 49,152 B ≈ 49KB
  Q_i tile: B_r × d × 2B = 64 × 64 × 2  =  8,192 B ≈  8KB
  S_ij / P_ij: B_r × B_c × 2B = 64 × 384 × 2 = 49,152 B ≈ 49KB
  Total: 49+49+8+49 = 155KB ≤ 192KB SRAM ✓

Tiling参数:
  T_r = ceil(N / B_r) = ceil(1024 / 64)  = 16 Q blocks
  T_c = ceil(N / B_c) = ceil(1024 / 384) = 3  KV blocks

HBM访问量（forward only）:
  - Q加载: N*d = 1024*64*2B = 128KB (分16次, 8KB/次)
  - K加载: T_r * (N*d) = 16 * 128KB = 2MB (每个Q block遍历所有K)
    实际更少——K_j在outer loop加载一次，被inner loop所有Q_i复用
    K加载 = T_c * (B_c*d) = 3 * 384*64*2B = 144KB
  - V加载: 同K = 144KB
  - O写入: N*d = 128KB
  - m/l写入: 2*N*4B = 8KB
  Total forward HBM traffic ≈ 144+144+128+128+8 KB ≈ 552KB
  vs standard: Q加载(128KB) + K加载(128KB) + S写(2MB) + S读(2MB) + P写(2MB) 
               + P读(2MB) + V加载(128KB) + O写(128KB) ≈ 8.5MB
  FlashAttention reduces HBM traffic by ~15× in forward pass
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Tiling的通用实现方式：(1) CUDA kernel中使用`__shared__` memory声明tile buffer，通过`__syncthreads()`协调tile加载和计算；(2) Triton的`tl.arange`和block pointer提供声明式tiling；(3) 编译器自动tiling（如TVM的split+reorder schedule primitives）。在FlashAttention中，tiling的block size选择通过sweep确定最优值（Figure 2 middle），需balance：(a) SRAM容量——总tile size ≤ M；(b) SM occupancy——更大的block占用更多SRAM per thread block，可能减少同时resident的block数；(c) HBM pass数——更大的block减少外循环迭代次数。实际使用中block size以power-of-2值（64/128/256）为佳，对齐GPU memory transaction sizes。

涉及论文标题：
- FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness
