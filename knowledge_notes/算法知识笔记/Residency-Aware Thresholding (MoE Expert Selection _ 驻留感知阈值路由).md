## Residency-Aware Thresholding (MoE Expert Selection / 驻留感知阈值路由)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Residency-Aware Thresholding 是 MoE-ERAS 提出的第一种 residency-aware expert 选择算法。在标准 MoE gating 的 Softmax 之后、Top-K 选择之前，对已驻留在 GPU HBM（fast memory）的 expert 的概率值统一加上超参数 α，人工提升 on-chip expert 的激活竞争力。核心洞察：gating network 的输出并非总有"绝对赢家"——有时 top-1 expert 仅略微优于第二名。若 top-1 恰好在 CPU 中而第二名在 HBM 中，thresholding 的 α 偏置可以使 on-chip 的第二名在调整后的概率中超过 off-chip 的第一名，从而避免一次 costly 的 CPU→GPU 传输。

从算法pipeline角度拆解术语：
```
# 标准 MoE Gating
Logits = H_i @ W_exp               # [seq_len, num_experts]
Weights = Softmax(Logits)           # [seq_len, num_experts], sum=1 per token

# === Residency-Aware Thresholding ===
# residency[e] = True 表示 expert e 当前在 HBM 中
for e in range(num_experts):
    if residency[e]:
        Weights[:, e] += alpha       # on-chip expert 加 α 偏置

Selected = SelectTopK(Weights, k=2)  # 调整后概率的 Top-K
# 注意：Weights 加 α 后可能 sum > 1，但 Top-K 仅需相对排序，不影响选择
```
超参数 α ∈ {0.05, 0.15, 0.25}。α=0 退化为标准 Top-K routing。α 越大，on-chip expert 越容易被选中 → speedup 越大 → quality 下降越多。

从算法pipeline角度拆解术语，给出具体例子：
以 Mixtral-8x7B layer i，8 experts，当前 3 个 offloaded（expert 0, 3, 7 在 CPU，其余在 HBM）为例：
- Token hidden state 经 gating 得 logits = [2.1, 1.9, 1.5, 2.0, 0.8, 0.5, 1.2, 0.3]
- Softmax → weights = [0.22, 0.18, 0.12, 0.20, 0.06, 0.05, 0.10, 0.07]
- Standard Top-2: expert 0 (权重 0.22, CPU) + expert 3 (权重 0.20, CPU) → 两个都在 CPU！两次传输。
- Thresholding α=0.15: expert 0 在 CPU 不加 (0.22), expert 1 在 HBM 加至 0.33, expert 2 在 HBM 加至 0.27, expert 3 在 CPU 不加 (0.20), expert 4+α=0.21, expert 5+α=0.20, expert 6+α=0.25, expert 7 在 CPU 不加 (0.07)
- 调整后 Top-2: expert 1 (0.33, HBM) + expert 2 (0.27, HBM) → 零传输开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：在 serving 框架的 gating 输出后插入 residency lookup + 条件加法操作。residency table 由 LRU cache manager 维护。
- 使用：用户通过 α 控制 speedup-quality trade-off。论文实测 α=0.05→perplexity 几乎不变（C4-PPL 8.044→8.062），α=0.15→10-13% latency reduction，α=0.25→更大 speedup 但 PPL 退化至 8.522。
- 优势：实现极简（仅条件加法），零额外参数，推理时生效不改变模型权重。
- 局限：对 all experts 的 on-chip 概率加相同 α，不考虑 expert 热度差异——冷门 on-chip expert 和热门 on-chip expert 获得相同 boost。

涉及论文标题：
- MoE-ERAS: Expert Residency Aware Selection

---
