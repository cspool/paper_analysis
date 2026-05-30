## Complementary Contribution (互补贡献) in Cooperative Game Theory

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Complementary Contribution（互补贡献）是 Shapley Value 计算中的一种高效采样形式。给定 coalition S ⊆ N（N 为所有玩家集合），互补贡献定义为 U(S) - U(N\S)。与传统的 marginal contribution U(S∪{h_i}) - U(S) 不同，互补贡献的一次采样结果可同时用于更新 S 中所有玩家的 Shapley Value 估计：

对每个 h_i ∈ S，$SV_i = \mathbb{E}[\frac{U(S) - U(N \setminus S)}{|S|}]$

在 LLM 语境下，U(S) 是 coalition S 中 head 活跃（保留完整 KV cache）、N\S 中 head 被 mask（仅保留 local window KV）时的模型准确率。关键优势：marginal contribution 的每次采样（加入/不加入一人）只能更新一个 head 的估计值，而 complementary contribution 每次采样（对比 S vs N\S）可更新 |S| 个 heads。当 |S|≈n/2 时，效率提升约 n/2 倍。

CoKV 采用 complementary contribution 而非 traditional marginal contribution 正是利用了这一效率优势，使得在 8×3090 GPU 上约 21 小时即可完成 256 个 groups 的准确评估。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Complementary Contribution 的一次采样**：

```
// 一次 Complementary Contribution 采样
π = random_permutation([h_1, ..., h_n])
j = random_choice(H)                     // H = {32,64,96,128}
S = {π(1), ..., π(j)}                    // 前 j 个 heads

// 两次前向推理
acc_active = forward(model, mask=N\S)    // S 中 heads 活跃，其余 mask
acc_masked = forward(model, mask=S)      // S 中 heads mask，其余活跃

// 互补贡献
u = acc_active - acc_masked

// 同时更新 S 中所有 |S|=j 个 heads
for h in S:
    SV_{h, j}.accumulate(u)
    count_{h, j} += 1

// 对比：Traditional Marginal Contribution（一次仅更新 1 个 head）
// h_i = random_choice(N)
// S = random_subset(N \ {h_i})
// u = U(S ∪ {h_i}) - U(S)
// SV_i.accumulate(u)    ← 仅 1 个 head 受益
```

**效率对比**：
- Marginal Contribution：M 次采样 → 每个 head 期望 M/n 次更新
- Complementary Contribution：M 次采样 → 每个 coalition 平均 n/2 大小 → 每个 head 期望 M·|H|/2n 次更新 → 约 |H|/2 ≈ 2 倍加速（|H|=4, n=256）

术语一般如何实现？如何使用？

Complementary Contribution 源自 Zhang et al. (SIGMOD 2023) 和 Sun et al. (TKDE 2024) 的 Shapley Value 近似理论。在 LLM 场景中，每次采样需要 2 次模型前向推理（一次算 U(S)，一次算 U(N\S)）。CoKV 按 coalition size 并行化：每个 coalition size j∈H 分配独立 GPU 计算，8 卡服务器上 4 个 coalition size 各用 2 卡做独立采样。采样结果取平均后 MAE<1/n 即为收敛。代码开源：https://github.com/nawei1010/CoKV。

涉及论文标题：
- CoKV: Optimizing KV Cache Allocation via Cooperative Game

---
