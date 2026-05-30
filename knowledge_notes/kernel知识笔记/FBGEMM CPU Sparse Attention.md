## FBGEMM CPU Sparse Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

FBGEMM (Facebook GEneral Matrix Multiply, Khudia et al., 2021) 是Meta开源的CPU端高性能低精度深度学习推理库。MagicPIG使用FBGEMM在CPU上以bfloat16精度执行稀疏注意力计算——仅对LSH采样得到的key集合S（通常为全量KV cache的2%~5%）执行qK_S^T内积和weighted V_S求和。FBGEMM针对x86 CPU的AVX512指令集优化，提供高效的bfloat16矩阵乘法。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。

```
// CPU端稀疏注意力 (FBGEMM bfloat16)
// Input: q ∈ R^{1×d} (bfloat16), K_S ∈ R^{|S|×d}, V_S ∈ R^{|S|×d}
//        u ∈ R^{1×|S|} (采样概率)
// Output: ō ∈ R^{1×d}

// Step 1: 计算q与采样key的内积 (FBGEMM GEMV)
w_S = FBGEMM_GEMV(K_S, q^T)  // shape: [|S|, 1], bfloat16

// Step 2: 带采样概率修正的softmax (CPU scalar ops)
For i in 1..|S|:
  w_adj[i] = exp(w_S[i]/sqrt(d) - log(u[i] + eps))

// Step 3: 加权V求和 (FBGEMM GEMV)
Z = sum(w_adj)
ō = FBGEMM_GEMV(V_S^T, w_adj) / Z
```

**CPU kernel调度特点**：
- |S|通常为n的2%~5% → 稀疏计算量远小于全注意力
- FBGEMM利用AVX512指令集并行处理16个bfloat16值
- 结果通过PCIe传回GPU，通过recursive attention与GPU侧结果合并
- Intel Platinum 8480+ (A100搭配) / Intel 8563C (L20搭配)

术语一般如何实现？如何使用？

FBGEMM需要CPU支持AVX512指令集以获得最佳性能。MagicPIG中CPU attention是单线程/少量线程执行的（因为|S|较小），主要瓶颈是CPU DRAM带宽（100-200GB/s）而非计算——从DRAM加载K_S和V_S是主要时延来源。论文实测带宽约为150GB/s（GQA size=4时）。未来方向包括利用AVX512_BF16新特性进一步提升CPU端计算效率。

涉及论文标题：
- MagicPIG: LSH Sampling for Efficient LLM Generation
