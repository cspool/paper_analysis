## Megatron-LM (NVIDIA 分布式训练框架)

术语解释
Megatron-LM 是 NVIDIA 开发的大规模 Transformer 模型分布式训练框架，支持 tensor parallelism (TP)、pipeline parallelism (PP)、data parallelism (DP) 和 expert parallelism (EP) 的组合使用。BigMac 论文使用 Megatron 作为主要训练评估平台。

术语是什么？
Megatron-LM 的核心是将模型参数和计算按多种并行策略分配到多 GPU/多节点：

- **Tensor Parallelism (TP)**：将单个 Transformer 层的权重矩阵按列/行切分，每个 GPU 计算部分结果后聚合。需要 All-Reduce 或 All-Gather + Reduce-Scatter 通信。
- **Expert Parallelism (EP)**：将 MoE 层的各 expert 分配到不同 GPU，token 通过 All-to-All 路由到对应 expert 设备。
- **Data Parallelism (DP)**：每 GPU 有完整模型副本，处理不同 micro-batch，梯度通过 All-Reduce 同步。
- **Pipeline Parallelism (PP)**：将模型按层切分为多个 stage，不同 GPU 负责不同 stage，batch 拆分为 micro-batches 流式处理。

BigMac 使用 Megatron 评估训练延迟（step time breakdown），配置 EP=1~32, TP=1~8 (ep×tp=32)。在纯 EP (ep=32, tp=1) 设置下，BigMac 端到端训练加速 2.37× (Top4) 和 2.95× (Top8)。

从系统架构角度拆解术语：
Megatron 在 MoE 层的训练流程（pre-training BigMac）：

```
# Megatron MoE layer training step
for each micro-batch:
    # Forward pass
    x = attention(x)                             # TP: All-Reduce after attention
    x_low = x @ W'_down                          # BigMac: descend
    x_routed = alltoall_scatter(x_low, gate_idx)  # EP: All-to-All dispatch
    for each expert on this GPU:
        out += expert_ffn(x_routed)
    x_combined = alltoall_gather(out)             # EP: All-to-All combine
    x = x_combined @ W'_up                        # BigMac: ascend

    # TP-SP communication in MoE layer:
    # All-to-All + All-Gather + Reduce-Scatter within each TP group
    # BigMac reduces all these communications to low dimension

    # Backward pass (compute gradients)
    # DP: gradient All-Reduce across data-parallel group
    # TP: gradient Reduce-Scatter / All-Gather within tensor-parallel group
```

术语一般如何实现？如何使用？
- 开源：https://github.com/NVIDIA/Megatron-LM
- 支持 GPT、BERT、T5 等模型的分布式训练
- 通过 mp_size（TP）、pp_size（PP）、dp_size（DP）参数配置
- BigMac 在 Megatron 上验证了 DCCA 策略与 TP-SP 通信的兼容性（TP-SP 通信也因低维化而减少 1.42-2.34×）

涉及论文标题：
- BigMac A Communication-Efficient Mixture-of-Experts Model Structure for Fast Training and Inference

---
