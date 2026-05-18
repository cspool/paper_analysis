## Attention Fusion for Decode（解码阶段的注意力融合）

术语是什么？通过联网搜索让回答具体和精准。
Attention Fusion是MFS论文提出的一种GPU kernel级优化技术，用于提升multi-tier serving中小batch decode阶段的GPU并行度。在LLM decode阶段，每个请求每次只生成一个token，导致attention计算中query矩阵极小（batch_size=1或很小），GPU的SMs利用率低。MFS的Attention Fusion将来自group batch中不同请求的Q、K、V矩阵按batch维度拼接为一个更大的联合QKV矩阵，执行一次联合attention计算，以少量冗余attention计算（跨请求token之间本不需要计算的attention）换取更高的GPU SM利用率和计算吞吐。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Attention Fusion的kernel执行流（以3个请求的group batch为例）：

传统decode attention（无fusion）：
```
for each request in batch:
    Q = hidden[req] @ W_q  # shape: (1, num_heads, head_dim)
    K = hidden[req] @ W_k  # shape: (1, num_heads, head_dim)
    V = hidden[req] @ W_v  # shape: (1, num_heads, head_dim)
    # 小Q→低GPU occupancy
    attn_out = flash_attention(Q, K_cache[req], V_cache[req])
```

Attention Fusion（MFS方案）：
```
# Step 1: 分别计算各请求的QKV
Q_list, K_list, V_list = [], [], []
for req in group_batch:  # e.g., 3 requests
    Q_list.append(hidden[req] @ W_q)  # each (1, nh, hd)
    K_new = hidden[req] @ W_k
    V_new = hidden[req] @ W_v
    # 将新KV追加到各自cache
    K_cache[req] = concat([K_cache[req], K_new])
    V_cache[req] = concat([V_cache[req], V_new])

# Step 2: 拼接各请求的Q、K、V
Q_fused = concat(Q_list, dim=0)  # shape: (3, nh, hd)→GPU parallelism ↑
K_fused = concat([K_cache[r] for r in group_batch], dim=0)  # (3 * seq_len, nh, hd)
V_fused = concat([V_cache[r] for r in group_batch], dim=0)  # (3 * seq_len, nh, hd)

# Step 3: 单次联合attention（含冗余cross-request attention）
attn_fused = flash_attention(Q_fused, K_fused, V_fused)
# 注意：R1的query会attend to R2和R3的KV→冗余计算
# 论文认为GPU并行能力可掩盖这部分额外时间

# Step 4: 拆分结果回各请求
attn_out[req_1], attn_out[req_2], attn_out[req_3] = split(attn_fused, dim=0)
```

关键trade-off：
- Compute overhead：Attention Fusion引入了跨请求token之间的冗余attention计算（如R1的query与R2的KV做attention——这些结果会被丢弃或不影响最终输出）。在标准attention实现中无法mask掉这些跨请求attention（因为不同请求的sequence可能不等长且causal mask不同）。
- GPU parallelism gain：更大的QKV矩阵→更多thread blocks→更高SM occupancy→decode阶段GPU compute bound转变为更充分利用。论文认为在GPU compute能力充足时，parallelism gain > redundant compute cost。
- 适用场景：仅在小batch decode阶段有效（batch size小时GPU utilization低）。prefill阶段本身计算量已足够大，通常不需要fusion。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
论文未详细说明Attention Fusion的具体kernel实现。根据论文描述，Attention Fusion使用标准的attention实现（如FlashAttention-style kernel），但输入是拼接后的QKV矩阵。论文未提出新的CUDA kernel，而是在标准attention kernel的调用层面通过改变输入张量的组织方式来提升GPU利用率。该技术与flash attention的batch mode类似但增加了跨请求冗余计算。论文未开源实现代码。Attention Fusion在MFS系统中的典型收益：配合group batching，在speculative sampling场景中将GPU utilization从Orca的约23.9%提升到约59.8%（提升主要来自group batching + attention fusion的联合效果，论文未单独消融attention fusion的贡献）。

涉及论文标题：
- MFS: An Efficient Model Family Serving System for LLMs

