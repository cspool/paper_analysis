## Uneven AlltoAll in MoE Training (MoE训练中的不等长全交换)

术语是什么？

Uneven AlltoAll（也称 alltoallv）是 MPI 风格的集合通信原语，允许每个参与方发送和接收不等量的数据。在 X-MoE 的 padding-free MoE pipeline 中，由于消除了 zero-padding，每个 expert 接收的 token 数量不同（由 tokens_per_expert ERI-array 描述），因此必须使用 uneven alltoall 替代传统 MoE 框架中的 even alltoall。

从kernel调度角度拆解：

传统 MoE 的 even alltoall vs X-MoE 的 uneven alltoall：

```
# Traditional (even alltoall):
# 所有expert的buffer固定大小 [E, C, H]
expert_buffers = alltoall(padded_buffers)  # 传输 E*C*H 个元素
# 大量zero-padding随通信传输

# X-MoE (uneven alltoall):
# Step 1: 先交换元数据
tokens_per_expert = alltoall(tokens_per_expert)  # [E] 整数，轻量
# 每rank据此计算inbound token数量 B_in = sum(tokens_per_expert[my_experts])

# Step 2: 交换实际数据
dispatch_out = alltoallv(dispatch_in, tokens_per_expert)  # 传输 B 个有效token
# dispatch_in: [B_out, H], dispatch_out: [B_in, H]
# 总通信量: B_out * H (无padding浪费)
```

通信量对比（Large 模型, EP=64, 256 GPU）：
- Even alltoall: 含大量 zero-padding，X-MoE 实测 alltoall 时间减少 50.7%
- Uneven alltoall: 仅传输有效 token，通信量 = 实际路由 token 数 × H

在 RBD 中进一步分层为 inter-node uneven alltoall（仅 pilot tokens）+ intra-node uneven alltoall（local replica）。

术语一般如何实现？

在 AMD ROCm 平台上通过 RCCL（ROCm Collective Communication Library）+ AWS-OFI-RCCL plugin（映射到 libfabric）实现。在 NVIDIA 平台上通过 NCCL 实现。X-MoE 使用 PyTorch 的 `torch.distributed.all_to_all_single` 配合 `split_sizes` 参数实现 uneven 传输。

涉及论文标题：
- X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms
