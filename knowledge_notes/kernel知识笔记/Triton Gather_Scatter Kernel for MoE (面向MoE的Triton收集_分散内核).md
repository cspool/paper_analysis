## Triton Gather/Scatter Kernel for MoE (面向MoE的Triton收集/分散内核)

术语是什么？

X-MoE 为 padding-free MoE pipeline 实现了两个基于 Triton 的关键 kernel：Gather Kernel（dispatch 阶段，按 token_ids 从 gate_out 收集 token 到 dispatch buffer）和 Scatter Kernel（combine 阶段，按 token_ids 将 MLP 输出分散回原始序列位置并加权）。这两类 kernel 的核心挑战是嵌套索引访问（如 gate_out[token_ids[i], :]）导致的不规则内存访问模式。

从kernel调度角度拆解：

Gather Kernel 的实现策略（Triton）：

```
# 执行: dispatch_in[i, :] = gate_out[token_ids[i], :]
# 
# Launch: B 个 thread-blocks, 每 block 256 threads
# Block bi 负责复制第 bi 个token
#
# 伪代码 (per thread-block bi):
row_idx = token_ids[bi]           # 源token在序列中的位置
for j in range(0, H, 256):        # 沿hidden dimension循环
    thread_id = j + thread_idx    # 当前线程处理的hidden dim位置
    if thread_id < H:
        dispatch_in[bi, thread_id] = gate_out[row_idx, thread_id]
#
# Coalescing: 连续线程处理连续hidden dim位置
# 保证 gate_out[row_idx, :] 的读取是coalesced (同一row, 连续col)
```

Scatter Kernel 的实现策略：

```
# 执行: combine_out[token_ids[i], :] += mlp_out[i, :] * combine_weights[i]
#
# Scatter的不规则性在"写"端: 多个token可能写到同一行不同列
# 但每行不同列之间无依赖 → 仍可并行
#
# 伪代码 (per thread-block bi):
row_idx = token_ids[bi]
weight = combine_weights[bi]
for j in range(0, H, 256):
    thread_id = j + thread_idx
    if thread_id < H:
        # atomic add 或事先检查无冲突
        combine_out[row_idx, thread_id] += mlp_out[bi, thread_id] * weight
#
# Coalescing: 连续线程写入连续hidden dim位置 (同一row的连续列)
```

与 Megablocks 的对比：
- Megablocks 使用 block-sparse primitives，但仍需 padding 到固定 block size 的倍数
- X-MoE 的 Triton kernel 完全 padding-free，且跨平台（ROCm/CUDA 均支持）

术语一般如何实现？

使用 Triton 语言编写，编译为 AMD ROCm 或 NVIDIA CUDA 后端。关键优化：(1) 将线程映射到 model hidden dimension（外层维度），确保 coalesced memory access；(2) 每 token 一个 thread-block，B 个 block 并行。在 X-MoE 中 Gather kernel 加速 dispatch buffer 填充 35.7×（Small 模型 vs DeepSpeed-MoE 的 einsum dispatch）。

涉及论文标题：
- X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms
