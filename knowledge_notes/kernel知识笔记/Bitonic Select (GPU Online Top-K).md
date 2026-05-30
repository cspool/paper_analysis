## Bitonic Select (GPU Online Top-K)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Bitonic Select是一种基于bitonic sequence（双调序列）的在线top-K选择算法，DSV在fused critical KV estimation kernel中使用它进行per-query的在线top-K合并。Bitonic sequence是指先单调递增后单调递减（或其循环移位）的序列。Bitonic Sort/Merge利用这一性质，通过比较-交换（compare-and-swap）操作在log²N步内完成排序或合并。DSV使用Bitonic Select（而非完整的Bitonic Sort），因为只需保留K个最大元素，不需要完全排序。在线特性：每处理一个新的partial score tile，将当前top-K与新partial合并为大小为K+tile_size的bitonic sequence，然后通过bitonic merge保留最大K个。

从kernel调度角度拆解，Bitonic Select在GPU上的实现：
```
// Bitonic Select for online top-K merge (DSV kernel)
// 输入: current_topK [(score, idx)], size=K (已排序，descending)
//       new_scores [tile_size] (unsorted partial results)
// 输出: merged_topK [(score, idx)], size=K

// Step 1: 构建bitonic sequence
//   current_topK (descending) + sort(new_scores, descending) = bitonic
sorted_new = BitonicSort(new_scores)        // 局部排序tile_size个元素
bitonic = concat(current_topK, sorted_new)  // [K + tile_size] bitonic

// Step 2: Bitonic merge - 仅保留最大K个
// 使用compare-and-swap网络：
n = K + tile_size
for step in range(log2(n)):
    stride = 2^step
    for i in range(0, n, 2*stride):
        for j in range(stride):
            if stride >= K and j >= K - max(0, i+stride-K):
                continue  // 跳过已知不在top-K中的比较
            if bitonic[i+j] < bitonic[i+j+stride]:
                swap(bitonic[i+j], bitonic[i+j+stride])

merged_topK = bitonic[:K]
```

Bitonic Select的GPU优势：(1) compare-and-swap操作天然适合SIMD/SIMT架构（所有比较可parallel）；(2) 固定比较-交换网络（无分支），GPU warp内无divergence；(3) 可在寄存器中完成（K较小时），避免shared memory往返；(4) 在线特性使kernel可以在矩阵乘法累加过程中逐步合并，无需等待完整partial。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

DSV在CUDA kernel中实现Bitonic Select（CUDA cores上）。Bitonic Sort/Merge是经典并行排序算法（Batcher, 1968），在GPU编程中广泛使用。在DSV中的特殊用法：(1) 用于在线合并（而非完整排序）；(2) 利用已知sorted前缀（current_topK已降序）简化bitonic sequence构建；(3) K较小时（<512）完全在寄存器中完成，K较大时split为两阶段（先阈值后选择）。替代实现：Radix Select（更快但需要integer key）、Quick Select（有分支不适合GPU）、heap-based topK（sequential）。

涉及论文标题：
- DSV: Exploiting Dynamic Sparsity to Accelerate Large-Scale Video DiT Training
