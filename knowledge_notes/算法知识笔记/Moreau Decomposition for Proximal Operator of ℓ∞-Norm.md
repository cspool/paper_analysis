## Moreau Decomposition for Proximal Operator of ℓ∞-Norm

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Moreau Decomposition（莫罗分解）是凸分析中的基本定理[29]，将任意范数的近端算子与其对偶范数的单位球投影联系起来。对于 ℓ∞-norm，其对偶范数为 ℓ₁-norm，Moreau 分解给出了恒等式：prox_{t‖·‖∞}(v) = v − t · proj_{‖·‖₁≤1}(v/t)，其中 proj_{‖·‖₁≤1} 是将向量投影到 ℓ₁ 单位球上的投影算子。这一分解的关键价值在于：ℓ∞-norm 不可微，其近端算子没有直接闭式解；但 ℓ₁-ball 投影有高效的直接算法（排序+软阈值，O(m log m)）。通过 Moreau 分解，将"困难的 ℓ∞ 近端算子"转化为"高效的 ℓ₁-ball 投影"，使整个 PGD 迭代可行且高效。MagR 在每次 PGD 迭代中使用此分解：梯度下降步 → V，然后对各列分别计算 prox_{ηα‖·‖∞}(V) = V − ηα · proj_{‖·‖₁≤1}(V/(ηα))。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# 已知: v ∈ R^m (梯度下降步结果的一列), t = ηα > 0

# Moreau 分解（理论等式）:
prox_{t‖·‖∞}(v) = v - t · proj_{‖·‖₁≤1}(v/t)

# 计算步骤:
# Step 1: 缩放
v_scaled = v / t

# Step 2: ℓ₁-ball 投影（Algorithm 3）
if ‖v_scaled‖₁ ≤ 1:
    p = v_scaled               # 已在球内
else:
    u = sort(|v_scaled|, descending)
    ρ = max{i: u_i > (Σ_{r=1}^i u_r - 1) / i}
    θ = (Σ_{r=1}^ρ u_r - 1) / ρ
    p = sign(v_scaled) ⊙ max(|v_scaled| - θ, 0)

# Step 3: Moreau 分解合成
result = v - t · p            # = prox_{t‖·‖∞}(v)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Moreau 分解是理论工具，在 MagR 代码中不单独存在，而是体现为 ℓ₁-ball 投影函数的调用。PGD 迭代中写为 `W_new = V - t * l1_ball_projection(V/t)`，这行代码隐式包含了 Moreau 分解。Moreau 分解不仅适用于 ℓ∞/ℓ₁ 对偶对，也适用于任何范数与其对偶范数之间（如 ℓ₂ 是对偶范数的自身），是 proximal algorithms 工具箱中的基础工具[32]。PyProximal 库（https://pyproximal.readthedocs.io）中的 `L1.prox` 和 `LInfinity.prox` 彼此通过 Moreau 分解互推。

涉及论文标题：
- MagR: Weight Magnitude Reduction for Enhancing Post-Training Quantization

---
