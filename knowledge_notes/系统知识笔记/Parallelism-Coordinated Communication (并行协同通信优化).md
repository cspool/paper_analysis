## Parallelism-Coordinated Communication (并行协同通信优化)

术语解释
Parallelism-Coordinated Communication 是 DeepSpeed-MoE 推理系统提出的通信优化策略，当 Expert Parallelism 与 Tensor-Slicing 同时使用时，利用 Tensor-Slicing 的 all-reduce 造成的数据复制效应，将 Expert Parallelism 的 all-to-all 通信限定在同 tensor-slicing rank 的设备子集内，显著降低通信延迟。

术语是什么？
在 DeepSpeed-MoE 推理中，non-expert 参数使用 tensor-slicing（节点内），expert 参数使用 expert parallelism（跨节点）。两种并行各自需要通信：tensor-slicing 的 all-reduce 在 tensor parallel group 内复制数据，expert parallelism 的 all-to-all 在 expert parallel group 内交换 token。Naive 实现将它们独立处理导致通信开销叠加。

核心洞察：tensor-slicing 的 all-reduce 使得同一 tensor-slicing rank 的所有 GPU 持有相同数据副本。因此，expert parallel 的 all-to-all 不需要在所有 p 个设备间进行——仅需在共享同 tensor-slicing rank 的 p/L 个设备间进行（L = tensor-slicing degree）。

从系统架构角度拆解术语：
```
# 配置: 128 GPUs, TP=8 (tensor-slicing degree), EP=128 (expert parallelism)
# 每 8 GPU 为一组 tensor-slicing group

# Naive 方法（独立处理）
# Step 1: Tensor-slicing all-reduce（在每个 TP group 内）
for tp_group in tensor_parallel_groups:    # 16 个 TP group
    all_reduce(data, tp_group)              # 仅限 TP group 内
# Step 2: Expert parallel all-to-all（全 128 GPU）
all_to_all(tokens, all_128_gpus)           # O(128) communication hops

# Parallelism-Coordinated 方法
# Step 1: Tensor-slicing all-reduce（同上）
for tp_group in tensor_parallel_groups:
    all_reduce(data, tp_group)
# Step 2: Expert parallel all-to-all（仅在同 TP rank 的 GPU 间）
# 所有 TP rank=0 的 GPU 间 all-to-all（16 GPUs）
# 所有 TP rank=1 的 GPU 间 all-to-all（16 GPUs）
# ...
# 每个 all-to-all 仅涉及 128/8 = 16 GPUs → O(16) hops vs O(128)
for tp_rank in 0..7:
    subset = GPUs_with_tp_rank(tp_rank)    # 16 GPUs
    all_to_all(tokens, subset)             # 仅限同 rank 子集

# 延迟改进: O(p) → O(p/L) + O(L)
# 对于 128 GPUs, TP=8: 128C1 + C2 → 16C1 + C2
```

通信调度细节：
- Expert Params → Tensor-Slicing (EP before TP): all-to-all 在同 TP rank 子集内完成后，加入 all-gather 在 TP group 内复制数据（因 TP 需要全部数据）
- Tensor-Slicing → Expert Params (TP before EP): 数据已在 TP group 内复制，all-to-all 仅限同 TP rank 子集

术语一般如何实现？如何使用？
- 实现于 DeepSpeed-MoE 推理引擎（开源：https://github.com/microsoft/DeepSpeed）
- 基于 NCCL P2P 操作 + Microsoft SCCL 优化通信后端
- 需要推理系统感知并行拓扑并自动协调通信调度
- 扩展到 128+ GPUs 时特别有效（8-way tensor-slicing 提供额外 8x 通信 hops 减少）

涉及论文标题：
- DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale

---
