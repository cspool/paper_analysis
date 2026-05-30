## Group Query Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Group Query Attention (GQA，分组查询注意力) 是介于标准多头注意力(MHA)和Multi-Query Attention(MQA)之间的一种注意力机制变体，由Ainslie et al.在2023年提出。其核心思想是将query head分成g组（1 < g < h），每组内的多个query head共享同一个key-value head对。相比MHA（h个KV heads，每个query head独立使用一对KV），GQA仅需g个KV heads（g << h），大幅减少推理时需要缓存的Key和Value状态量。相比MQA（所有query head共享唯一一对KV），GQA通过保留多个KV组来维持更好的建模质量。GQA的KV Cache大小是MHA的(g/h)倍，在典型配置下（如h=32, g=8）可减少75%的KV Cache内存占用。

从算法pipeline角度拆解术语，给出术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
输入: 输入序列 x ∈ R^{seq_len × d_model}, 总head数 h, KV组数 g
参数: Q_proj ∈ R^{d_model × h·d_head}, K_proj ∈ R^{d_model × g·d_head}, V_proj ∈ R^{d_model × g·d_head}

# 1. 投影
Q = x @ Q_proj  # (seq_len, h*d_head)
K = x @ K_proj  # (seq_len, g*d_head)
V = x @ V_proj  # (seq_len, g*d_head)

# 2. 拆分head
Q = reshape(Q, [seq_len, h, d_head])  # (seq_len, h, d_head)
K = reshape(K, [seq_len, g, d_head])  # (seq_len, g, d_head)
V = reshape(V, [seq_len, g, d_head])  # (seq_len, g, d_head)

# 3. GQA计算: 每g组内的h/g个query共享同一个KV
heads_per_group = h // g
O = []
for group_idx in range(g):
    # 该组内的query heads
    Q_group = Q[:, group_idx*heads_per_group : (group_idx+1)*heads_per_group, :]  # (seq_len, heads_per_group, d_head)
    
    # 该组共享的单一KV head
    K_group = K[:, group_idx, :]  # (seq_len, d_head) → 广播为 (seq_len, 1, d_head)
    V_group = V[:, group_idx, :]  # (seq_len, d_head)
    
    # 标准Scaled Dot-Product Attention
    scores = Q_group @ K_group.unsqueeze(1).transpose(-2, -1) / sqrt(d_head)
    attn = softmax(scores, dim=-1)
    O_group = attn @ V_group.unsqueeze(1)  # (seq_len, heads_per_group, d_head)
    O.append(O_group)

O = concat(O, dim=1)  # (seq_len, h, d_head)
O = reshape(O, [seq_len, h*d_head])
output = O @ O_proj  # (seq_len, d_model)

# 4. KV Cache (推理时)
# MHA: Cache_size = 2 * h * seq_len * d_head  (无共享)
# GQA: Cache_size = 2 * g * seq_len * d_head  (每层有g对KV，减少(h/g)倍)
# MQA: Cache_size = 2 * 1 * seq_len * d_head  (最极端)
```

从MHA转换为GQA（本论文使用的方法）：
```python
# 通过mean-pooling将MHA的KV投影矩阵转换为GQA
# 输入: MHA模型有h个KV heads
# 输出: GQA模型有g个KV heads，每组mean-pool原h/g个heads
for group_idx in range(g):
    start = group_idx * heads_per_group
    end = start + heads_per_group
    K_GQA[group_idx] = mean(K_MHA[start:end], dim=0)  # 均值池化
    V_GQA[group_idx] = mean(V_MHA[start:end], dim=0)
# 然后用5%原始数据继续训练
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现和使用：
1. **生产模型中的GQA配置**：LLaMA 3.1系列（8B: h=32/g=8, 70B: h=64/g=8, 405B: h=128/g=8），Qwen 2.5 7B (h=28/g=4)，Mistral (h=32/g=8)，Gemma 3 12B (h=16/g=4)。
2. **KV Cache减少**：g=8时减少87.5%的KV Cache，对长上下文推理（32K-128K+）至关重要。
3. **从MHA转换**：可通过mean-pooling将MHA模型的KV heads转换为GQA，然后用~5%原始数据继续训练（本论文在PanGu-π-1.5B Pro上验证，GQA版本(1.4B参数)保持原MHA版本(1.5B参数)的~92-97%benchmark性能）。
4. **适用场景**：内存受限的边缘设备部署、长文本输入场景、需要降低推理延迟的在线服务。
5. **其他变体**：Cross-Layer Attention (CLA)进一步跨层共享KV；Cost-Optimal GQA根据上下文长度动态调整head数；QCQA通过进化算法实现非均匀分组。

涉及论文标题：
- PanGu-π Pro: Rethinking Optimization and Architecture for Tiny Language Models
