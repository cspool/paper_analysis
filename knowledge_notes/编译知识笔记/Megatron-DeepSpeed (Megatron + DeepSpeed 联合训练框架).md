## Megatron-DeepSpeed (Megatron + DeepSpeed 联合训练框架)

术语是什么？
Megatron-DeepSpeed 是 NVIDIA Megatron-LM 与 Microsoft DeepSpeed 的联合分布式训练框架，结合了两者优势：(1) Megatron-LM 提供 Tensor Parallelism (TP)、Pipeline Parallelism (PP)、Sequence Parallelism (SP) 等模型并行能力；(2) DeepSpeed 提供 ZeRO-1/2/3 内存优化、CPU/NVMe offload、通信优化。联合框架支持 4D 并行（TP+PP+SP+DP），可训练 10T+ 参数模型。Megatron-DeepSpeed 是目前训练 MoE 模型的主流框架之一，预集成 ZeRO-2 DP + EP 的混合并行策略。

从编译框架角度拆解术语：
Megatron-DeepSpeed 的 checkpoint 流程（MoE 训练）：
```
# Training loop with checkpoint (simplified)
for iteration in range(total_iterations):
    # Forward + Backward (F&B)
    loss = model.forward(batch)
    loss.backward()
    
    # Optimizer step (ZeRO-2: reduce-scatter gradients, update partitioned optimizer states)
    optimizer.step()
    
    # Checkpoint (if at save interval)
    if iteration % I_ckpt == 0:
        # Baseline: blocking checkpoint — all ranks save
        save_checkpoint(model, optimizer, iteration)
        # → GPU→CPU snapshot → CPU→Storage persist (can be async)

# save_checkpoint internal:
# 1. Collect model params (replicated across DP ranks → need sharding)
# 2. Collect optimizer states (already sharded by ZeRO-2)
# 3. Serialize and write to distributed filesystem
```

MoC-System 基于 Megatron-DeepSpeed 实现了三个层次修改：
- **Checkpoint 内容**：PEC 修改 `save_checkpoint` 中的 expert 参数收集逻辑，插入 sequential selection
- **Checkpoint 分布**：Fully Sharded Checkpointing 替换 original rank/ep_group 分配逻辑
- **Checkpoint 异步**：Triple Buffering + 代理线程管理替换原两阶段同步流

术语一般如何实现？如何使用？
- 安装：`git clone https://github.com/microsoft/Megatron-DeepSpeed`，配置 ZeRO + TP/PP 的 JSON 文件。
- MoE 训练典型配置：`--zero-stage 2` (ZeRO-2 DP)，`--moe-expert-parallel-size 8` (EP)，配合 `--num-experts 16`，`--topk 2`。
- Checkpoint 相关参数：`--save-interval 1000`（保存频率），`--async-checkpointing`（异步保存），DataStates-LLM backend 可用于更高性能的 I/O。

涉及论文标题：
- Partial Experts Checkpoint: Efficient Fault Tolerance for Sparse Mixture-of-Experts Model Training
- Toward Inference-optimal Mixture-of-Expert Large Language Models
