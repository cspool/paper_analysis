## Hierarchical Mixture of Experts (层次化混合专家)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hierarchical MoE 是 Shazeer et al. (2017) 提出的一种两级 MoE 结构，用于在 expert 总数极大时（数千至数万）降低计算 branching factor。在 flat MoE 中，Gate 网络为每个 token 从 n 个 expert 中选择 k 个，当 n 很大时 (如数万)，Gate 网络的计算量 (x@W_g: [d, n]) 本身成为一个瓶颈。Hierarchical MoE 将选择分为两级：第一级主 Gate (G_primary) 从 a 个 group 中选择 k1 个，第二级 Gate (G_i) 在每个 group 内的 b 个 expert 中选择 k2 个。总 expert 数 n = a × b。论文将第一级 branching factor 设为 GPU 数量，使次级 expert 无需跨设备通信。

从算法pipeline角度拆解术语：
Hierarchical MoE 的计算流程：

```
# 参数: a 个 group (第一级, 对齐 GPU 数), 每组 b 个 expert (第二级)
# 超参数: k1 (主 Gate 选的 group 数), k2 (次级 Gate 每组内选的 expert 数)

# Gate 1: 主 Gate
logits_primary = x @ W_g_primary        # [1, a]
H_primary = logits_primary + noise_1 * StandardNormal()
topk_vals_1, topk_idx_1 = KeepTopK(H_primary, k1)
G_primary = Softmax(topk_vals_1)        # [1, k1]

# Gate 2: 次级 Gate (在对应 GPU 上本地执行, 无跨设备通信!)
for i in topk_idx_1:
    logits_secondary = x @ W_g_secondary_i  # [1, b]
    H_secondary = logits_secondary + noise_2 * StandardNormal()
    topk_vals_2, topk_idx_2 = KeepTopK(H_secondary, k2)
    G_secondary_i = Softmax(topk_vals_2)    # [1, k2]
    
    for j in topk_idx_2:
        expert_out_{i,j} = Expert_{i,j}(x)  # 本地 FFN 计算

# 合并:
output = sum(G_primary[i] * G_secondary_i[j] * Expert_{i,j}(x) 
             for i in topk_idx_1 for j in topk_idx_2[i])
# 总激活 expert = k1 × k2 (如 2×2=4)
```

与 flat MoE 的关键区别：
- **Branching Factor**: flat 需 look up n 个 logits → Hierarchical 仅需 a + b (a+b << n)
- **通信优势**: 次级 expert 全在同一 GPU → 无跨设备通信
- **负载均衡扩展**:
  - Importance_H(X)_{i,j} = Σ_x G_primary(x)_i · G_i(x)_j
  - Load_H(X)_{i,j} = Load_primary(X)_i · Load_i(X^{(i)})_j / |X^{(i)}|

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 第一级 branching factor = GPU 数是有意设计——将 hierarchical 结构的物理意义与分布式拓扑绑定。
- 论文实验配置 (100B Word Corpus)：256×32, 1024×32, 4096×64, 16384×128, 65536×256, 131072×256 expert×branching factor。第一级 k1=2，第二级 k2=2。
- 论文指出"未发现需要更深层级"——两级已足够。
- 后续 MoE 模型（GShard, Switch Transformer, Mixtral）多采用 flat MoE，主要因现代 GPU 的 compute/memory ratio 更高且 flat 结构更简单，但 hierarchical 的设计思路影响了后续 Expert Parallelism 的拓扑感知设计。

涉及论文标题：
- Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer
