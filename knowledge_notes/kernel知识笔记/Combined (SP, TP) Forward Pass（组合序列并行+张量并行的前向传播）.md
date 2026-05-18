## Combined (SP, TP) Forward Pass（组合序列并行+张量并行的前向传播）

术语是什么？

Combined (SP, TP) Forward Pass是Shift Parallelism中base configuration的核心执行路径（Algorithm 1），将Sequence Parallelism沿sequence维度的数据切分与Tensor Parallelism沿model weight维度的计算切分组合在同一forward pass中。SP负责沿sequence维度拆分batch中的token、通过all-to-all转换attention layout；TP负责沿weight维度拆分每层的QKV、O、MLP权重矩阵。关键约束：`SP × TP = P`（总GPU数），且SP group内执行all-to-all通信，TP group内执行all-reduce通信。

从kernel调度角度拆解术语：

Algorithm 1的kernel级执行流程（以SP=4, TP=2, 8 GPU, Llama-70B为例）：

1. **Sequence分片**（Line 1）：输入embedding `[n, d]` 被SP.slice沿sequence维度均分为4份`[n/4, d]`，每份分发到一个SP rank。

2. **QKV Projection**（Line 3）：每个GPU用本地sequence slice和本地TP weight shard `qkv[d, 3×h/2]`做矩阵乘法，产生`[n/4, 3×h/2]`的QKV heads。因为TP=2，每个GPU只持有half heads。

3. **SP All-to-All**（Line 4）：在SP group内执行fused all-to-all。对于SP=4，4个GPU互相交换sequence slices和head partitions。结果：每个GPU获得完整sequence length `n`但仅`3×h/8`（即3×h/(SP×TP)）个heads。通信复杂度O(n×h×d/SP)，不随SP degree增长。

4. **Head-Parallel Attention**（Line 5）：每个GPU对本地`h/8`个heads执行attention计算，使用本地KV cache。由于所有GPUs处理不同的attention heads，attention阶段无通信。

5. **SP All-to-All返回**（Line 6）：第二个all-to-all将结果从`[n, h/8]`转回`[n/4, h/2]`（sequence-parallel + TP head layout）。

6. **O Projection + TP All-Reduce**（Line 7-8）：`attn_o[n/4, h/2] * o[h/2, d]`得`embed[n/4, d]`，然后TP all-reduce在TP group内同步partial sums。因为TP=2，每对TP rank做all-reduce。

7. **MLP + TP All-Reduce**（Lines 9-11）：MLP up projection用TP column-parallel，MLP down用TP row-parallel，各产生一次all-reduce（总计每layer 2次TP all-reduce + 2次SP all-to-all）。

术语一般如何实现？如何使用？

在ArcticInference中，通过`--tensor-parallel-size TP --ulysses-sequence-parallel-size SP`同时指定。系统自动构建SP groups和TP groups，并编译对应CUDA graphs。基Config的约束是SP×TP=P，TP越小（SP越大）通信开销越低（TP的all-reduce随TP degree增高，SP的all-to-all不随SP增高），但TP需确保每GPU能fit模型权重+KV cache。论文中Llama-70B用(SP=4, TP=2)或(SP=8, TP=1)；Llama-17B-16E因109GB内存需TP=2配合SP=4。

涉及论文标题：
- Shift Parallelism: Low-Latency, High-Throughput LLM Inference for Dynamic Workloads

