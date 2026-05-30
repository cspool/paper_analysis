## µ-queuing (Micro-queuing / 微队列 / 层粒度Token排队)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
µ-queuing（微队列）是 AMoE 系统中 AEP 范式的核心使能技术。传统 MoE serving 系统将请求整 batch 同步执行所有层，token 在全局 batch 中排队。µ-queuing 则将排队粒度细化到每个 decoding block 的每个 expert 层——每个 GPU Runtime 为每个 (block#, expert#) pair 维护独立的 token 队列（µ-queue）。Token 到达后按 LayerID 分离入队，而非合并到全局 batch。GPU 空闲时只从被选中的 µ-queue 中 drain tokens 组成 batch 执行。

从系统架构角度拆解术语，给出术语在系统架构中运转流程的具体例子。
以 8 experts × 32 blocks 的 Mixtral 为例，假设 GPU 持有 Expert 0 的所有 32 个 block 的层：

```
每个 GPU Runtime 的 µ-queue 结构：
Q[block=0][expert=0]: [token_1, token_5, token_12, ...]  ← 5 tokens
Q[block=0][expert=1]: []                                    ← 0 tokens (不在本 GPU)
Q[block=1][expert=0]: [token_3]                             ← 1 token
Q[block=1][expert=1]: []                                    ← 不在本 GPU
...
Q[block=31][expert=0]: [token_8, token_15, token_22, ...]  ← 8 tokens
```

Scheduler 对每个非空 µ-queue 计算 Score：
- Q[0][0]: LScore = (1/1 + 0/1 + ...)/1 × δ¹ + ... ≈ lookahead bonus, Q=5 → Score ≈ lookahead + 5
- Q[1][0]: Q=1 → Score 低 → 延迟执行，等待更多 token 积累
- Q[31][0]: Q=8 → Score 高 → 优先调度

关键效果：Cold expert tokens 积累到高效 batch size 前不被调度（≈128 时达到 linear throughput scaling），hot expert tokens 快速积累优先执行。这对应解决 EP 中 cold expert 小 batch 浪费 HBM 带宽的问题。

术语一般如何实现？如何使用？
在 AMoE 中，µ-queue 通过 C++ 数据结构实现（每个 Runtime 维护 N_B × N_E 的队列数组），Receptor（POSIX backend thread）负责 enqueue，Scheduler 负责 drain。Token metadata <RequestID, LayerID, Tensors[], ...> 在 CPU 上追踪，tensor 数据保留在 GPU memory 中避免 CPU-GPU 拷贝。µ-queuing 还支持 Top-K > 1 时的 token merge：receptor 维护 token pool，只有当 K 路 expert 输出全部到达时才将完整 token 移入目标 attention µ-queue。

涉及论文标题：
- Toward Cost-Efficient Serving of Mixture-of-Experts with Asynchrony
