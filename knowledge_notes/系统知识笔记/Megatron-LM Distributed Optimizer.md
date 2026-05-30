## Megatron-LM Distributed Optimizer

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Megatron-LM Distributed Optimizer 是 NVIDIA Megatron-LM 框架中实现的 ShardedDP 变体。与 ZeRO-3 的关键区别：在前向传播后保留收集到的完整 model weights（BF16）直到下次 AllGather，而非像 ZeRO-3 那样立即释放。这种"保留权重"的设计使 SDP4Bit 的 qWD 技术天然适配——可以复用 model weights buffer 计算权值差值，无需额外存储。Megatron-LM 还提供灵活的并行策略组合：(1) Tensor Parallelism (TP)：垂直切分模型层（每个 Transformer block 内的矩阵乘法被分布到 TP group），利用 NVLink 高带宽实现近零开销的 all-reduce；(2) Pipeline Parallelism (PP)：水平切分模型层序列，通过 micro-batch pipeline 隐藏通信延迟；(3) Data Parallelism (DP/ShardedDP)：数据并行训练且优化器状态分片。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Megatron-LM + SDP4Bit 的完整 3D 并行训练流程（GPT-18B, TP=4, PP=2, DP=16 on 128 GPUs）：
```
GPU (tp=0, pp=0, dp=0):
  # TP 通信: AllReduce within TP group (NVLink intra-node)
  # PP 通信: P2P send/recv activations/grads across PP stages
  # DP 通信: ShardedDP all-gather/reduce-scatter across all DP workers

  每个 micro-batch iteration:
  1. [DP] AllGather quantized weight diffs (qWD, INT4)
  2. [TP] Forward: matmul f w/ all-reduce within TP group (FP16)
  3. [PP] Forward: send activations to next PP stage
  4. [PP] Backward: receive grad from next PP stage; [TP] all-reduce grads
  5. [DP] TLq-HS gradient sync: intra-node INT8 all-to-all + inter-node INT4 all-to-all
  6. [DP] Optimizer step on local shard
  7. [DP] Compute weight differences for next iteration
```
Megatron-LM 的 global batch size 计算：`global_batch = micro_batch_size × accumulation_steps × DP_size`。当 accumulation_steps=1 时，DP 通信在每次 optimizer step 都执行，通信开销最显著。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Megatron-LM 通过 `pretrain_gpt.py` 脚本启动训练，关键配置参数：`--tensor-model-parallel-size`、`--pipeline-model-parallel-size`、`--use-distributed-optimizer`、`--micro-batch-size`、`--global-batch-size`。SDP4Bit 在此基础上增加 `--quantized-weights`、`--quantized-gradients`、`--hadamard-transform` 等量化参数。训练脚本通常由 `torchrun` 或 MPI 启动：`torchrun --nproc_per_node=8 --nnodes=16 pretrain_gpt.py <args>`。Distributed Optimizer 通过 NCCL 的 AllGather/ReduceScatter 原语实现通信。

涉及论文标题：
- SDP4Bit: Toward 4-bit Communication Quantization in Sharded Data Parallelism for LLM Training
