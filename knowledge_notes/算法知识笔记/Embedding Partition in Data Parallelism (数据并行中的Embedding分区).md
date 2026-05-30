## Embedding Partition in Data Parallelism (数据并行中的Embedding分区)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Embedding Partition in Data Parallelism 是 MoESys 提出的在 Data Parallelism 框架下对大 vocabulary embedding table 做分布式存储和计算的技术。传统 tensor model parallelism 的 embedding 分区沿 vocabulary 维度切分（每个 GPU 持有 V/N 个 token 的 embedding），但在 Data Parallelism 下每个 GPU 处理不同的输入数据，沿 vocab 维度切分会导致部分 token 在本 GPU 没有对应 embedding。MoESys 的方法改为沿 embedding 的 hidden_size 维度做 column-wise partition：每个 worker 持有 [V, H/N] 的 embedding shard（完整 vocabulary 但部分 hidden dimension），这样每个 device 都能访问完整 vocabulary。计算时通过 3 次 AlltoAll 通信完成 embedding lookup：Forward 阶段 AlltoAll 交换 input data → 本地 lookup → AlltoAll 交换结果；Backward 阶段 AlltoAll 交换 gradients。

从算法pipeline角度拆解术语：
Embedding Partition in Data Parallelism 的计算过程（embedding table E[V, H]，N 个 devices）：
```
# 列切分: 每个 device i 持有 E_i[V, H/N]
# 即: E_i 是 E 的第 i 个列切片

# Forward:
# Step 1: AlltoAll 交换 input token IDs
# 每个 device 有 batch token IDs: ids_local[batch_size]
all_ids = AlltoAll(ids_local)  # 每个 device 获得所有 input token IDs

# Step 2: 本地 embedding lookup
for each token_id in all_ids:
    embed_partial = E_i[token_id]  # 形状 [H/N]，仅部分 hidden dim
    embed_results.append(embed_partial)

# Step 3: AlltoAll 交换 partial embedding 结果 (AlltoAll inverse)
# 将归属于原始 device 的 embedding 结果返回
final_embeddings = AlltoAll(embed_results)  # [batch_size, H]

# Backward:
# 只需 1 次 AlltoAll 交换 gradients of E
grads_E_i = backward_pass()
AlltoAll(grads_E_i)  # 交换梯度恢复完整 embedding table gradient
```

关键差异对比：
- 传统 tensor parallelism embedding partition: 沿 vocab 维度切分 → 每 GPU 有 V/N 个 token 的完整 embedding → 需要 AllReduce 同步
- MoESys DP embedding partition: 沿 hidden_size 维度切分 → 每 GPU 有全部 V 个 token 的部分 embedding → 需要 3 次 AlltoAll（替代 AllReduce）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 在 MoESys 实验中，embedding partition 在 vocab_size=50304 的 GPT MoE 模型上表现突出：4 experts, hidden=8192 时 memory 从 15.81GB 降至 8.63GB，speed 从 80421 tokens/s 升至 91687 tokens/s。
- 该方法的关键优势：在 DP 框架下（每个 device 处理不同 data），vocab 维度切分不可行（每个 device 的 input token 不同），hidden_size 维度切分避免了这一矛盾。
- 相比 EmbRace 的列切分（主要用于 tensor parallelism，针对的是通信均衡）和传统 DP 的 AllReduce embedding 同步，该方法的 3 次 AlltoAll 虽然引入更多通信次数，但避免了 AllReduce 的全局同步开销，在总通信量相近的情况下更灵活。

涉及论文标题：
- MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System for Internet Services
