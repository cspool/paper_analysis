## Shared Tensor Based Dependency Resolving

术语解释
Shared Tensor Based Dependency Resolving 是 Comet 提出的细粒度通信-计算重叠方法，通过识别 MoE layer 中通信和计算操作共享的缓冲区（shared tensor）并对其沿特定维度分解和重调度，将粗粒度 chunk 级 pipeline overlap 升级为 token/tile 级重叠，消除通信(token级)与计算(tile级)之间的粒度不匹配。

术语是什么？
在 MoE layer 中，两个 producer-consumer pipeline 各有一个 shared tensor——layer0 的 shared tensor 是 dispatch buffer [M×topk, N]（通信的输出、GEMM 的输入），layer1 的 shared tensor 是 GEMM 输出 buffer [M×topk, N]（GEMM 的输出、reduce+通信的输入）。Shared tensor 的分解和重调度基于两个原则：
1. **沿独立维度分解**：选择 consumer operator 数据独立的维度——layer0 沿 M（token）维度（每个 token 独立），layer1 沿 N（hidden）维度（各列独立）。不能沿 consumer 需要 reduce 的维度分解。
2. **重调度对齐 tile 粒度**：分解后的 sub-tensor 按原始 GEMM tile 粒度重组，调度策略优先处理 producer 侧立即可用的数据（最小化数据依赖等待）。

从kernel调度角度拆解术语：

```
# Comet Layer0 (Communication→Computation Pipeline) 的 Shared Tensor 流程
# shared_tensor = dispatch_buffer [M×topk, N]

# Step 1: 沿 M 维度分解 shared tensor
sub_tensors = decompose_along_dim(shared_tensor, dim=M)
# 每个 sub_tensor 对应一个或少量 token，token 粒度

# Step 2: 按 source_rank 排序 sub-tensor
sorted_tensors = sort_by_source_rank(sub_tensors)
# local tokens 聚集在前（无需通信），remote tokens 聚集在后（需 NVSHMEM get）

# Step 3: 重调度 GroupGEMM tile 计算顺序
for tile in GroupGEMM.tiles:
    if tile.only_contains_local_tokens():
        priority = HIGH     # 立即开始，无数据依赖
    elif tile.partial_remote_ready():
        priority = MEDIUM   # remote token 已通过 NVSHMEM 到达
    else:
        priority = LOW      # 等待 NVSHMEM 传输完成
    schedule(tile, priority)

# Layer1 (Computation→Communication Pipeline):
# 沿 N 维度分解 → column-wise GEMM
for col_block in range(0, N, T_N):        # T_N = GEMM tile N 维度大小
    for expert in local_experts:
        partial = GEMM_tile(expert, col_block)  # 只计算当前列块
    # T_N 列完成后立即 reduce + 通信
    topk_reduce(partial_results[:, :col_block * T_N])
    NVSHMEM_write_to_remote(ready_tokens)
    # 继续下一列块的计算（与 reduce/通信重叠）
```

在 Hopper GPU (132 SMs) 上，layer0 的 GEMM tile 重调度使 local token 的计算在 NVSHMEM 拉取 remote token 期间进行，实现了 token-tier 的重叠。layer1 的 column-wise 执行使 reduce+通信与后续列 GEMM 重叠，将传统 per-expert 串行改为列方向并行。

术语一般如何实现？如何使用？
- 依赖 NVSHMEM 的 Unified Virtual Address 实现 token 级 fine-grained remote I/O（替代 NCCL 粗粒度 all-to-all）
- Shared tensor buffer 由 NVSHMEM 分配（size = 2×M×N bytes），跨所有 MoE layers 和 experts 全局复用
- 需要修改 GroupGEMM 的 tile 调度顺序（layer0: remote-dependency 最小化优先；layer1: column-wise）
- 适配 expert parallelism + tensor parallelism 混合场景：TP 下 shared tensor 沿 N 维进一步分片，但分解策略不变
- Comet 代码 ~12k lines C++/CUDA + 2k lines Python，集成在 Megatron-LM 中

涉及论文标题：
- Comet Fine-grained Computation-communication Overlapping for Mixture-of-Experts
