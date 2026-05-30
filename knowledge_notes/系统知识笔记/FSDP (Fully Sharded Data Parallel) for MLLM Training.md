## FSDP (Fully Sharded Data Parallel) for MLLM Training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FSDP (Fully Sharded Data Parallel) 是PyTorch提供的分布式训练策略（Zhao et al., 2023），它将模型参数、梯度和优化器状态在多个GPU之间分片（shard），而非像DDP那样在每个GPU上复制完整模型。FSDP的核心insight来自微软ZeRO（Zero Redundancy Optimizer）——在训练的不同阶段按需all-gather参数，计算完成后立即释放，从而显著降低每个GPU的内存占用。FSDP支持三种sharding策略：FULL_SHARD（参数+梯度+优化器状态全分片）、SHARD_GRAD_OP（仅分片梯度和优化器状态）、NO_SHARD（等价DDP）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
ML-Mamba训练中的FSDP流程（8×A100 80GB集群）：
```
// Step 1: FSDP初始化
model = MLMamba(vision_encoder, msc_connector, mlp_projector, mamba2_llm)
// 每个GPU分片获得完整模型参数的1/8

// Step 2: 每个training step的前向传播
for each FSDP unit (layer group):
    all_gather(params)      // 从各GPU收集完整参数到当前GPU
    forward_computation()   // 使用完整参数计算前向
    discard(params)         // 释放完整参数，仅保留分片

// Step 3: 反向传播
for each FSDP unit (reverse order):
    all_gather(params)      // 重新收集完整参数
    backward_computation()  // 计算梯度
    reduce_scatter(grads)   // 梯度分片归约，每个GPU保留自己的分片
    discard(params)

// Step 4: 优化器更新
optimizer.step()  // 仅更新本GPU分片中的参数
// 使用AdamW, lr=2e-5, cosine decay, warmup ratio=0.03

// 混合精度
with autocast(dtype=torch.bfloat16):  // 前向用BF16
    loss = model(input)
scaler.scale(loss).backward()  // 反向用FP32
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：`torch.distributed.fsdp.FullyShardedDataParallel`或`torch.distributed.fsdp.wrap`。ML-Mamba使用PyTorch FSDP + 自动混合精度（FP32 + BF16）在8×A100 80GB上训练约31小时（对齐1 epoch + 微调1 epoch, batch size=64）。FSDP相比DDP的优势：(1) 每个GPU内存占用显著降低（参数被分片而非复制），可训练更大模型；(2) 相比ZeRO-3更易配置（PyTorch原生集成）。FSDP在训练中的劣势：all-gather通信开销随层数增加。ML-Mamba选择FSDP而非DDP/DeepSpeed的原因可能包括：Mamba-2模型的层结构规整适合FSDP的分片粒度、PyTorch原生支持简化部署。

涉及论文标题：
- ML-Mamba__Efficient_Multi-Modal_Large_Language_Model_Utilizing_Mamba-2

---
