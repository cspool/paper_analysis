## Heterogeneous Mixture-of-Experts (Heterogeneous MoE)

术语解释
Heterogeneous MoE 是一种打破传统 MoE 均匀设计的架构范式，允许在 Transformer 的不同层中使用不同数量的专家（variable expert count）、不同大小的专家（variable expert FFN size），以及不同的专家放置策略（variable expert placement），形成非均匀、异质的 MoE 架构。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
传统 MoE（如 Switch Transformer, GShard）采用 homogeneous 设计：所有层中 expert 数量相同（如每层 4 个或每隔一层 4 个），所有 expert 的 FFN 尺寸相同（如 intermediate size = 3072），专家放置位置采用 ad-hoc 规则（每隔一层、每四层、或最后几层）。

Heterogeneous MoE 打破这些约束，搜索空间包含：
1. **每层可变 expert 数量**：第 i 层可有 1 到 M 个 expert，不同层可以不同
2. **每 expert 可变 FFN 尺寸**：同一层内的每个 expert 可有不同的 intermediate FFN size（如 1024, 2048, 3072），不同层的 expert 也可不同
3. **可变 decoder 层数**：对于 encoder-decoder 架构，decoder 层数可以少于 encoder 层数
4. **可变非 expert 模块**：attention heads 数量、hidden size、QKV dimension 也可变

这种异构设计使模型可以实现 **adaptive computation**：不同 token 通过 routing 自然分配到不同大小的 expert——简单 token 走小 expert（节省 FLOPs），复杂 token 走大 expert（保证质量）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
在一个 6-layer encoder-decoder Transformer 中，AutoMoE 搜索到的最优异构配置（WMT'14 En-De）：

```
# Encoder: 6 layers, 每层 expert 分布 [5, 1, 1, 1, 2, 1]
#   Layer 1: 5 experts, FFN sizes [3072, 3072, 3072, 2048, 3072]
#   Layer 2: 1 expert,  FFN size  [3072]
#   Layer 3: 1 expert,  FFN size  [3072]
#   Layer 4: 1 expert,  FFN size  [3072]
#   Layer 5: 2 experts, FFN sizes [3072, 2048]
#   Layer 6: 1 expert,  FFN size  [3072]

# Decoder: 4 layers, 每层 expert 分布 [1, 1, 1, 1]
#   每层: 1 expert, FFN size 3072

# Forward Pass (per token x):
for l in encoder_layers:
    logits = x @ W_router[l]              # W_router[l]: 每层不同维度
    # Layer 1 router: [d, 5]; Layer 2 router: [d, 1]; etc.
    routed_expert = argmax(logits)
    # Expert FFN: W_1 ∈ R^{ffn_size[e] × d}, W_2 ∈ R^{d × ffn_size[e]}
    # 不同 expert 的 ffn_size 可能不同
    h = ReLU(x @ W_1^T)
    out = h @ W_2^T                       # 输出维度始终为 d
```

异构设计的关键约束：所有 expert 的输入/输出维度保持相同（均为 d），仅中间 FFN 维度可变，因此 expert 输出可直接聚合。

术语一般如何实现？如何使用？
- 通过 NAS 自动搜索而非手动设计：在异构搜索空间中用演化算法找到最优配置
- AutoMoE 使用 Supernet（最大 MoE 配置）+ weight sharing 训练 + 演化搜索，在 latency constraint 下找到 Pareto 最优架构
- 搜索发现的一般规律：encoder 承担 71% 专家（中间层最多），decoder 首层 expert 最多、逐层递减
- 适用于 encoder-decoder Transformer（NMT），也可扩展到 decoder-only 架构
- 异构设计使 FLOPs 和 active parameters 大幅减少（AutoMoE: 4× FLOPs reduction vs dense Transformer）

涉及论文标题：
- AutoMoE: Heterogeneous Mixture-of-Experts with Adaptive Computation for Efficient Neural Machine Translation

---
