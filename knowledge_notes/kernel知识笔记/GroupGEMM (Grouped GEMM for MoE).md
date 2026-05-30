## GroupGEMM (Grouped GEMM for MoE)

术语解释
GroupGEMM 是针对 MoE 场景的批量矩阵乘法 kernel，将多个 expert 的独立 GEMM 操作合并为单次 kernel launch，避免 per-expert kernel launch overhead，同时利用 GPU 并行性提升小 expert 的计算效率。

术语是什么？
在 MoE 的 expert FFN 中，每个 expert 需要处理不同数量、不同来源的 tokens。传统实现为每个 expert 独立 launch GEMM kernel，当 expert 数量多且单个 expert token 少时产生大量 kernel launch overhead 和 GPU under-utilization。GroupGEMM 将所有 expert 的 GEMM 打包为单次调用，内部处理不同 expert 的不同输入形状（variable-sized batched GEMM）。Comet 基于 CUTLASS 的 GroupGEMM 模板生成高效率 kernel，并对 tile 调度顺序进行重排以实现通信-计算重叠。

从kernel调度角度拆解术语：

```
# GroupGEMM 在 MoE layer0 中的执行（Comet 版本）
# 输入: tokens 已按 expert 分组，各 expert token 数不同

# CUTLASS GroupGEMM 的 tile 调度（标准）:
for problem_idx in range(num_experts):
    m = token_counts[problem_idx]  # 该 expert 的 token 数（变化！）
    for tile_m in range(0, m, TILE_M):
        for tile_k in range(0, K, TILE_K):
            for tile_n in range(0, N, TILE_N):
                # 标准 tile 计算
                GEMM_tile(tile_m, tile_n, tile_k)

# CUTLASS GroupGEMM 的 tile 调度（Comet - 重排序）:
# 按数据依赖重排序: local token tiles 优先
tiles_sorted = sort_tiles_by_remote_dependency(
    all_tiles, 
    key=lambda t: count_remote_tokens(t),  # remote token 少的优先
    ascending=True
)
for tile in tiles_sorted:
    GEMM_tile(tile)
```

在 Hopper 架构上，CUTLASS GroupGEMM 内部使用 TMA (cp.async.bulk) 指令实现异步 global→shared memory 数据传输，producer warp 发起 TMA 请求后 consumer warp 在 tensor core 上执行 MMA，形成软件流水线。Comet 保持此流水线不变，仅在 tile 调度层面注入重排序逻辑。

术语一般如何实现？如何使用？
- CUTLASS 3.x 提供 `cutlass::gemm::kernel::Gemm` 的 Grouped GEMM 变体（`cutlass::gemm::grouped` 命名空间）
- NVIDIA 也提供专门的 grouped_gemm 库：https://github.com/fanshiqing/grouped_gemm
- Megatron-LM 默认使用 CUTLASS GroupGEMM（Megatron-Cutlass baseline）
- Comet 扩展了 GroupGEMM 的 tile 调度和 shared tensor 管理，增加 NVSHMEM 通信 TB 形成 fused kernel
- Triton 也支持 grouped GEMM（通过 `tl.dot` 的 block-pointer API）

涉及论文标题：
- Comet Fine-grained Computation-communication Overlapping for Mixture-of-Experts
- DualSparse-MoE: Coordinating Tensor/Neuron-Level Sparsity with Expert Partition and Reconstruction
- EPS-MoE: Expert Pipeline Scheduler for Cost-Efficient MoE Inference

**EPS-MoE 的 GroupGemm vs DenseGemm 动态切换**：
EPS-MoE 通过 profiling 发现：GroupGemm 和 DenseGemm 在不同输入规模下各有优势：
- **m < 2048**（如 decode 阶段）：GroupGemm 效率更高
- **m ≥ 4096**（如 prefill 阶段）：DenseGemm 效率反超 GroupGemm

关键洞察：(1) 对于 GroupGemm，当输入 size 达到一定阈值后，增加 group 数和 SM 数都不会进一步提高吞吐量（图5b,c）；(2) 通过水平切分输入按行+权重按专家切分，当 pipeline 数 N=E（专家数）时，GroupGemm 退化为 DenseGemm，可利用 cublas 的更高效率。

EPS-MoE 的 load-aware 自适应策略：根据输入 token 数 m 动态选择 GEMM 实现，在 Expert Pipeline Scheduler 中各 pipeline stage 可独立选择 GroupGemm 或 DenseGemm。
