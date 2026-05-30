## Expert-Specific Fused Kernel (ESFK)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert-Specific Fused Kernel (ESFK) 是 HEXA-MoE 中将 MoE 层反向传播中的三个 expert-specific 算子——ESS（expert-wise 求和）、ESTMM（expert-wise 转置矩阵乘法）、ESMM（expert-wise 矩阵乘法）——融合为单一 CUDA kernel 的技术。动机：三个算子在 backward pass 中计算不同梯度（ESS 计算 bias 梯度、ESTMM 计算 weight 梯度、ESMM 计算 input 梯度），独立启动 kernel 会产生多次 global memory 读写和 kernel launch overhead。ESFK 通过统一 thread-block shape 和扩展 thread-grid 维度将三者融合，使单 MoE 层 backward 仅需 2 个 fused kernels + 1 个 element-wise dot product。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

融合策略（Table 6）：
- 各算子原始 thread-block shape 统一为 (WARP, TIMES)
- ESS 二维 grid (E, ⌈D/(TIMES·BLK)⌉) 扩展第三维为 1
- ESMM 二维 grid (⌈N'/BLK⌉, ⌈D/(TIMES·BLK)⌉) 扩展第三维为 1
- ESTMM 三维 grid (E, ⌈D1/(TIMES·BLK)⌉, ⌈D2/(TIMES·BLK)⌉)
- ESFK 聚合 grid: dim-3 = ⌈N'/BLK⌉ + ⌈D2/BLK⌉ + ⌈D2/(TIMES·BLK)⌉

```
// ESFK 执行流程:
__global__ void ESFK(x, y1, ∂ℓ/∂y, W1, W2, R, v, idx, ...):
    gid_z = blockIdx.z
    
    if gid_z < ⌈N'/BLK⌉:
        // ESMM: 计算 ∂ℓ/∂x (input gradient)
        ESMM_block(x, ∂ℓ/∂y1, W1^T, R, v, ...)
    
    elif gid_z < ⌈N'/BLK⌉ + ⌈D2/BLK⌉:
        // ESS: 计算 ∂ℓ/∂b (bias gradient)
        ESS_block(∂ℓ/∂y, R, v, idx, ...)
    
    else:
        // ESTMM: 计算 ∂ℓ/∂W (weight gradient)
        ESTMM_block(y1, ∂ℓ/∂y, R, v, idx, ...)
```

一次 kernel launch 完成三种梯度计算，消除 kernel launch overhead 和中间结果的 global memory 往返。消融实验（Figure 9b）显示 ESFK 可有效减少 latency，且对 memory footprint 无影响。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要求三个算子的 thread-block shape 一致（统一为 WARP×TIMES），通过 shape transposing 或 dim expanding 对齐。CUDA 实现中通过 `blockIdx.z` 判断当前 thread-block 应执行哪个算子。开源实现：https://github.com/UNITES-Lab/HEXA-MoE。

涉及论文标题：
- HEXA-MoE: Efficient and Heterogeneous-aware MoE Acceleration with ZERO Computation Redundancy
