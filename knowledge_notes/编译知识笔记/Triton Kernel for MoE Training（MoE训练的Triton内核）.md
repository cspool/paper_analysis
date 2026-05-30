## Triton Kernel for MoE Training（MoE训练的Triton内核）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Triton (Tillet et al., 2019) 是 OpenAI 开发的一种中间语言和编译器，用于编写高性能 GPU kernel，编译为 PTX 代码在 NVIDIA GPU 上执行。在 MoE 训练场景中，Triton kernel 用于实现专家分组近似、token 重排、梯度聚合等操作，替代原生 PyTorch 操作以提升效率。本论文使用 Triton 实现 Expert Group Approximation 的核心计算——包括 "Router backward" kernel（计算传递给路由器的近似稠密梯度）、token 分组 kernel、以及梯度注入 kernel。

从编译框架角度拆解术语：

Triton kernel 在 Expert Group Approximation 中的工作流程：

```
# Triton kernel 伪代码: router_backward_kernel

@triton.jit
def router_backward_kernel(
    expert_outputs,    # [N, num_tokens, d_model] - 所有专家对所有 token 的输出
    routing_decisions, # [num_tokens, K] - 每个 token 的路由决策
    router_grad_out,   # [N, d_model] - 输出梯度
):
    pid = tl.program_id(0)
    
    # 每个 block 处理一个 (expert_i, expert_j) 对
    expert_i = pid // N
    expert_j = pid % N
    
    # 找到同时被路由到 expert_i 和 expert_j 的 tokens (X_{i,j,·})
    mask_i = routing_decisions_contains(expert_i)  # token 是否被路由到 i
    mask_j = routing_decisions_contains(expert_j)  # token 是否被路由到 j
    adjacent_mask = mask_i & mask_j
    
    # 对这些 token 的 expert_i 输出取平均
    if adjacent_mask.sum() > 0:
        approx = tl.sum(expert_outputs[expert_i] * adjacent_mask, axis=0) / adjacent_mask.sum()
    else:
        approx = 0
    
    # 累积到路由器梯度
    tl.atomic_add(router_grad[expert_i], approx * grad_scale / K)
```

Triton 将上述逻辑编译为高效的 GPU kernel，利用 shared memory 和 tiling 优化内存访问模式。

术语一般如何实现？如何使用？

Triton kernel 通过 `triton.jit` 装饰器定义，使用 `triton.autotune` 自动搜索最优的 block size 和 num_warps 配置。在 MoE 训练中，Triton kernel 插入到 GPT-NeoX + Megablocks 框架的 MoE 层反向传播路径中。本论文中，随 hidden size 增大（1024→4096），Triton kernel 的 CUDA 时间占比从 13.32% 降至 1.57%，因为 expert MLP matmul 时间占比随 hidden dim 平方增长而主导总时间。

涉及论文标题：
- TurboMoE Enhancing MoE Model Training with Smart Kernel-Fusion and Data Transformation
- X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms

Note: X-MoE 使用 Triton 实现 padding-free MoE pipeline 的 gather/scatter kernel（用于 dispatch 和 combine 阶段的 token 重排），以及通过 Triton 的跨平台特性（AMD ROCm + NVIDIA CUDA）实现硬件无关的 MoE 训练 kernel。
