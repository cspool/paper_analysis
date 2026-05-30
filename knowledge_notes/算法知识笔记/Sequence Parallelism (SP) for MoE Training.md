## Sequence Parallelism (SP) for MoE Training

术语是什么？
Sequence Parallelism (SP) 是一种将 Transformer 层的激活张量沿序列长度（sequence length）维度切分到多个 GPU 的并行策略。与 Tensor Parallelism (TP) 沿 hidden dimension 切分不同，SP 使用 all-to-all 通信替代 TP 的 all-gather/reduce-scatter。MegaScale-MoE 将 DeepSpeed-Ulysses 的 SP attention 首次应用于大规模 MoE 训练场景——SP 将 Q/K/V 的 head 维度分片，通过 all-to-all 交换实现 attention 的分布式计算。SP attention 的通信量为 2bsh(n-1)/n × (2+2/m)/n，其中 m 为 GQA 的 query/key-value head 比。当 m=4 时，SP 通信量仅为 TP 的约 1/4。SP 复制 attention weights 而非切分（TP 切分），带来额外参数量，但在 MoE 中 expert 参数占绝对多数（>90%），内存开销仅 1.2-5.4%。

从算法pipeline角度拆解术语：
MegaScale-MoE 的单层 SP attention forward 计算流程：
```
输入: hidden [b, s/n, h]   // n-way SP, 每个GPU持有 s/n 长度的序列片段
// Step 1: QKV Projection (local)
qkv = MatMul(RMSNorm(hidden), qkv_weight)  // [b, s/n, h(1+2/m)]
q, k, v = split(qkv)
q_rope = RoPE(q)   // [b, s/n, h]
k_rope = RoPE(k)   // [b, s/n, h/m]

// Step 2: All-to-All 将 head 分片转为 sequence 分片
// 输入: 每个 GPU 持有所有 heads 但仅 s/n 的序列
// 输出: 每个 GPU 持有完整序列但仅 1/n 的 heads
qkv_a2a = All-to-All([q_rope, k_rope, v])  // [b, s, h(1+2/m)/n]

// Step 3: Self-Attention (每个 GPU 独立计算其 head 子集)
attn = FlashAttention(qkv_a2a)  // [b, s, h/n]

// Step 4: 反向 All-to-All 恢复为 sequence 分片
attn_a2a = All-to-All(attn)  // [b, s/n, h]

// Step 5: Output Projection (local)
attn_out = MatMul(attn_a2a, out_weight)  // [b, s/n, h]
```
SP 相比 Context Parallelism (CP) 的优势：CP 沿 sequence 切分所有激活，但由于 causal masking（每 token 仅能 attend 前面的 token），后部序列分片计算量少于前部分片，造成负载不均。SP 按 head 维度交换数据，每个 GPU 计算完整序列上的部分 head，天然负载均衡。

术语一般如何实现？如何使用？
- 实现来源：DeepSpeed-Ulysses (Jacobs et al., 2023) 提出，MegaScale-MoE 首次在 MoE 训练中大规模部署。
- 参数同步：SP 复制的 attention weights 需通过 hierarchical communication 同步——intra-node reduce-scatter → inter-node reduce-scatter → inter-node all-gather → intra-node all-gather，在 NVLink/NIC 带宽不对称场景下，inter-node 主导总延迟，与 TP 的参数同步时间差异仅 0.3-3.1%。
- 适用条件：(1) GQA 场景下 m>1 时 SP 通信优势扩大；(2) MoE 场景下 expert 参数占内存主体，SP 的额外参数开销可忽略；(3) 不适合 expert 数量少、attention 参数占比高的模型。

涉及论文标题：
- MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production
