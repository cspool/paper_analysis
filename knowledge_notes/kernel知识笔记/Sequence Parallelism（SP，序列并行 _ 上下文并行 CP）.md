## Sequence Parallelism（SP，序列并行 / 上下文并行 CP）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sequence Parallelism（SP）是把 transformer 输入序列沿 token 维度切分到多个 GPU 的分布式并行策略，用于满足长上下文请求的计算与内存需求（KV cache 与 attention 激活不放在单卡）。与 TP（沿 hidden/head 维切权重）、PP（沿层切）、DP（沿 batch 切）正交。两类主流实现：DeepSpeed-Ulysses 式（attention 前后两次 All-to-All 在 sequence layout 与 head layout 间切换）与 ring-attention 式（P2P 环形传递 KV，即 CP/context parallelism）。在 serving 中 SP 的核心优势：调整 SP 大小只重分 token、不需要重分片模型权重（TP 调整需 reshard 权重、挂起设备），且跨节点扩展优于 TP（TP all-reduce 对低带宽网络敏感）。局限：SP 环要求实例同步，大 SP 下短请求计算量不足、无法掩盖 ring 通信，性能反而劣于小 SP。Tetris（ISCA'26）在 prefill 阶段用 SP 池（TP=1 统一 SP 组）最大化资源分配灵活性，decoding 用大 TP 的 DP。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# ring-attention 式 SP（N 个实例，序列 S token）
# 每实例持有 S/N token 的 Q 分片 + 本地 K/V 分片
for step in 0..N-1:
    # 本地 partial attention（Flash Attention）与 KV ring 传输重叠
    o += FlashAttn(Q_local, K_cur, V_cur)
    K_cur,V_cur = ring_sendrecv(K_cur, V_cur)   # P2P
# 非 attention 算子（MLP/LN）各 GPU 只算自己的 token 分片，无通信
```
Annotations: 每实例计算量 S/N × d（attention）+ 自身 token 的 FFN；通信量每步 O(KV chunk)，总 O(N·KV)。
Tetris 的调度粒度：SP 大小（1/2/4/8/16，2 的幂）作为 CDSP 调度的候选，实例组扩展按"先节点内后跨节点"策略（GetGroup），prefill 统一 SP 池的 SP 大小在集群初始化（initialize_model_parallel）时显式配置建立 ring communicator。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Megatron-LM sequence_parallel（Korthikanti et al.，配合 TP 消除冗余 LN/Dropout）、DeepSpeed-Ulysses（All-to-All）、ring-flash-attention / Context Parallelism（Meta，P2P ring）、LoongServe 的 ESP（动态 SP）、Tetris 的 CDSP（chunk 级动态 SP）。使用：长上下文 serving/training 中作为与 TP/DP/PP 组合的 4D 并行之一；serving 里 SP 大小可随请求动态调整（无需权重重分片）。Web 证据：Context Parallelism 文档（Meta）与 DeepSpeed-Ulysses 论文确认两类实现与通信模式。

涉及论文标题：
- Tetris: Efficient Long-context LLM Serving with Chunkwise Dynamic Sequence Parallelism
