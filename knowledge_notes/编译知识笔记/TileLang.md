## TileLang

术语是什么？

TileLang (Wang et al., 2025) 是一种面向 AI 系统的可组合 Tiled 编程模型/框架。它允许开发者以 tile 为基本粒度描述计算，并将这些 tile 计算组合为完整的 GPU kernel。TileLang 的设计目标是简化具有复杂计算过程的 kernel 实现，以最小的工程代价充分利用显存带宽。

在 mHC 中，TileLang 被用于实现大部分 mHC kernel（除 Eq.14-15 的 MMA 融合核外），包括：
- Sinkhorn-Knopp 迭代 kernel（含自定义反向）
- 融合后处理 kernel（Eq.16-18）
- Pre/Post+Res 映射应用 kernel

TileLang 自动处理 tile 间的数据依赖、共享内存管理和线程调度，使开发者能聚焦于计算逻辑本身。

从编译框架角度拆解：

TileLang 的工作流程：
1. **输入**：开发者用 TileLang DSL 描述 tile 级计算逻辑（如矩阵乘法、逐元素操作、归约）
2. **Tile 组合**：TileLang 自动分析 tile 间的依赖关系，生成 tile 执行调度
3. **代码生成**：生成高效的 GPU kernel 代码（CUDA/PTX），包括共享内存分配、线程 block/warp 映射、寄存器分配
4. **执行**：生成的 kernel 直接在 GPU 上运行

在 mHC 中的具体使用：
```
# 开发者用 TileLang 描述 Sinkhorn-Knopp 迭代:
@tilelang.kernel
def sinkhorn_kernel(H_res_raw, H_res_out):
    M = tilelang.exp(H_res_raw)
    for t in range(20):
        M = tilelang.row_normalize(M)
        M = tilelang.col_normalize(M)
    H_res_out[:] = M

# TileLang 自动生成: 共享内存管理、线程映射、流水线
```

术语一般如何实现？如何使用？

TileLang 开源：https://github.com/anthropics/tilelang。它位于 Triton 之上或类似层次，但提供更高级的 tile 级抽象。适用于需要频繁优化 kernel 但又不想手动编写 CUDA 的场景。TileLang 处理如 RMSNorm 重排序优化、混合精度 cast 等底层细节。mHC 论文指出使用 TileLang 可以实现"以最小工程代价充分利用显存带宽"。

涉及论文标题：
- mHC Manifold-Constrained Hyper-Connections
