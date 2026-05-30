## Power-Plus-Constant Scaling Law for GQA（GQA的幂加常数缩放定律）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Power-Plus-Constant Scaling Law 是本文发现的经验规律：(1) LM loss 与模型大小 N 之间满足 L(N) = (a/N)^b + E（用于 Step 2 拟合 scaling curves，R² > 0.999）；(2) LM loss 与 query head 数 n_h 之间也满足 L(n_h) = a·n_h^b + c（附录 I，R² > 0.999）。

关键特性：
- E/c 为"语言自然熵"——当 N→∞ 或 n_h→∞ 时 loss 收敛到的下界。在 470M 模型上拟合 c=1.53
- b < 0 → loss 随 N 或 n_h 增加而下降，但呈 diminishing returns
- n_h 的 power law 与 model size 和 context length 独立（在不同 N 和 T 下均成立）

从算法pipeline角度拆解术语：

**Step 2 Scaling Curve 计算**：
```
// 对每个 H=(nh, nkv)，训练 3M→1.2B 模型，拟合：
L(N) = (a/non_emb_params)^b + E

// 具体拟合值（以 H=(32,8) 为例）：
L(N) = (1.2×10^8 / N)^{0.12} + 2.615
// N=1.2B → L=2.615（与 Llama-3.2-1B 一致）

// 从 scaling curve 反求 N*:
N*(H) = a_H / (L* - E)^{1/b_H}
```

**n_h Scaling Law（附录 I）**：
```
// 470M model, T=1K:
L(n_h) = 0.579 · n_h^{-0.124} + 2.473    (R²>0.999)
// 680M model, T=1K:
L(n_h) = 0.398 · n_h^{-0.177} + 2.583    (R²>0.999)
// 1.2B model, T=1K:
L(n_h) = 0.301 · n_h^{-0.227} + 2.622    (R²>0.999)

// 不同 context length (470M):
L(n_h, T=1K)  = 1.513 · n_h^{-0.039} + 1.53
L(n_h, T=2K)  = 1.436 · n_h^{-0.041} + 1.53
L(n_h, T=8K)  = 1.356 · n_h^{-0.044} + 1.53
// 随 T 增大，c 收敛到相同值（1.53）→ n_h→∞ 时 loss 由 E 决定
```

术语一般如何实现？如何使用？

在 Step 2 中，对每个候选 GQA 配置 H 用 5-8 个不同 N 的训练数据进行非线性最小二乘拟合，获得 (a_H, b_H, E)。虽然 E 理论上由数据决定（跨 H 共享），但实践中为每个 H 独立拟合 E 可提升精度（因小模型下数据与模型间可能有交互）。该 law 经验上可外推至少一个数量级（如 3M→1.2B），类似 Llama-3 从 16B 预测 405B loss 的做法。代码开源：https://github.com/THUNLP/cost-optimal-gqa。

涉及论文标题：
- Cost-Optimal Grouped-Query Attention for Long-Context LLMs

---
