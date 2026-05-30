## Sharded Data Parallelism (ShardedDP)（分片数据并行）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sharded Data Parallelism (ShardedDP) 是一种分布式训练策略，将传统 Data Parallelism (DP) 中的优化器状态（optimizer states：模型参数、momentum、variance 等，通常为 FP32）在多个 GPU 间分片存储，每个 GPU 只持有 $1/P$ 的优化器状态，从而将每 GPU 的优化器状态内存开销降低 $P$ 倍（P 为 GPU 数量）。与 Naive DP（每 GPU 复制完整优化器状态）相比的核心区别在于通信模式的变化：(1) Forward pass 前需要 AllGather 完整模型权重（因为每个 GPU 只持有分片的 main weights）；(2) Backward pass 后需要 ReduceScatter 归约梯度（使每个 GPU 获得其分片对应的平均梯度后进行 optimizer step）。ShardedDP 在多个主流训练框架中以不同名称实现：Megatron-LM 的 Distributed Optimizer、DeepSpeed 的 ZeRO（ZeRO-1/2/3）、PyTorch 的 FSDP（Fully Sharded Data Parallelism）。关键差异在于权重释放策略：ZeRO-3 和 FSDP 在 forward/backward 后释放收集的权重（节省内存但 backward 需额外 all-gather weight）；ZeRO-2 和 Megatron-LM Distributed Optimizer 则保留收集的完整权重（无需 backward 再收集，但内存占用更高）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Megatron-LM Distributed Optimizer (ShardedDP 模式) 每轮迭代的通信-计算流程：
```
# 初始状态: 每个 GPU p 持有:
#   w_main[p] : 1/P 的 FP32 main weights (optimizer states)
#   w_model   : 完整 BF16 model weights (Megatron-LM 保留)
#   g_model^p : 本地计算的梯度

# === Forward Pass ===
# AllGather BF16 weights from main shards
w_model[p][0:shard_size] = w_main[p].to(BF16)
w_model = AllGather({w_model[p] for p in 0..P-1})  # 通信: O(W) BF16
output = ForwardPass(w_model, input)

# === Backward Pass ===
g_model^p = BackwardPass(w_model, output, labels)

# === Gradient Synchronization (ReduceScatter) ===
g_main[p] = ReduceScatter(g_model^p)  # 通信: O(G) FP32, P-1 rounds

# === Optimizer Step (仅本地分片) ===
w_main[p] = AdamW(g_main[p], w_main[p])  # 无需通信
```

关键特性：(1) 通信量与 GPU 数量几乎无关（AllGather 通信量 = W × (P-1)/P ≈ W，ReduceScatter 同理），意味着更多 GPU 时通信开销占总时间比例更大；(2) 可以与 Tensor Parallelism (TP) 和 Pipeline Parallelism (PP) 组合使用（3D parallelism）；(3) 通信带宽瓶颈在 inter-node（NVLink intra-node 带宽 >> InfiniBand inter-node 带宽），这正是 SDP4Bit 等通信压缩技术的优化目标。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Megatron-LM 的 Distributed Optimizer 通过 `--use-distributed-optimizer` 启用。关键实现细节：(1) 每个 GPU 维护一个 `param_shard_group`（由连续 GPU 组成），在此 group 内分片优化器状态；(2) DP group 通常跨所有节点，TP group 通常在一个节点内；(3) 数据并行通信（AllGather param / ReduceScatter grad）和数据并行参数分片在 `DistributedDataParallel` 类中协调。SDP4Bit 在此架构基础上插入量化步骤：AllGather 前量化权值差值（qWD），ReduceScatter 替换为两次 all-to-all 并各自量化（TLq-HS）。ZeRO-3/FSDP 则在每个 submodule 的 forward/backward 边界执行 all-gather/release 操作，实现内存和通信的更细粒度交换。

涉及论文标题：
- SDP4Bit: Toward 4-bit Communication Quantization in Sharded Data Parallelism for LLM Training
- ZeRO++: Extremely Efficient Collective Communication for Giant Model Training
- Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism

---
