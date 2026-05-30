## Proximal Gradient Descent with ℓ₁-Ball Projection for PTQ Weight Preprocessing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Proximal Gradient Descent (PGD / 近端梯度下降) 是求解复合优化问题 min_x f(x) + g(x) 的一阶迭代算法，其中 f 是光滑凸函数（如最小二乘损失 ½‖XW − XŴ‖²_F），g 是非光滑凸函数（如 ℓ∞ 范数）。每轮迭代分两步：(1) Gradient Descent Step: v = x^k − η∇f(x^k)，沿负梯度方向下降；(2) Proximal Step: x^{k+1} = prox_{ηg}(v)，其中 prox_{ηg}(v) = argmin_x{½‖x−v‖² + ηg(x)} 是 g 的近端算子。在 MagR 中，梯度 ∇f(W) = XᵀX(W − Ŵ)，近端算子 prox_{t‖·‖∞} 通过 Moreau 分解转化为 ℓ₁-ball 投影。PGD 收敛保证：当步长 η ≤ 1/L（L 为 ∇f 的 Lipschitz 常数，即 λ_max(H)）时，PGD 以 O(1/k) 收敛到全局最优解。PGD 比 subgradient descent 快得多（对数收敛 vs 次线性收敛），且通过矩阵化实现支持列级并行处理整个权重矩阵。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MagR 中的 PGD 实现（Algorithm 1 + Algorithm 2）：

```
输入: Ŵ ∈ R^{m×n}, H = XᵀX ∈ R^{m×m}, K, η = 1/λ_max(H), α > 0
W⁰ = Ŵ

for k = 0 to K-1:
    # ---- Gradient Descent Step (matrix form, O(m²n)) ----
    G = H @ (W^k - Ŵ)              # ∈ R^{m×n}, 梯度
    V = W^k - η · G                # 梯度下降步

    # ---- Proximal Step (column-wise ℓ₁-ball projection) ----
    # Moreau: prox_{ηα‖·‖∞}(V) = V - ηα · proj_{‖·‖₁≤1}(V/(ηα))
    M = (‖V[:, j]‖₁ ≤ ηα for j=1..n)   # binary mask: 已在球内的列标记为 1
    U = sort(|V|, dim=0, descending)   # 列排序 → U ∈ R^{m×n}
    # 向量化找投影阈值索引 ρ_j (Algorithm 2, line 3)
    cumsum_U = cumsum(U, dim=0)        # 沿行累积和
    cond = U > (cumsum_U - ηα) / arange(1, m+1)  # 阈值条件矩阵
    ρ_j = max row index per column where cond is True
    # 软阈值参数 θ_j = (Σ_{r=1}^{ρ_j} U[r,j] - ηα) / ρ_j
    θ = gather_theta(cumsum_U, ρ_j)
    Θ = tile(θ, [m, 1])               # 广播到 m×n
    # 软阈值操作（仅对 M==0 的列）
    W^{k+1} = (1-M) ⊙ sign(V) ⊙ max(|V| - Θ, 0) + M ⊙ V

输出: W^K
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PGD 的 ℓ₁-ball 投影核心是排序 + 软阈值，O(m log m) per column。PyTorch 实现中使用 `torch.sort` + `torch.cumsum` + `torch.where` 进行向量化批量投影（Algorithm 2 的矩阵版本）。步长 η = 1/λ_max(H) 通过 power iteration 预计算。H = XᵀX 在预处理开始时计算一次即固定，后续仅需矩阵乘法。开源实现：https://github.com/AozhongZhang/MagR。K=150 经验上足够收敛。在 PyProximal（https://pyproximal.readthedocs.io）中也提供类似的 ℓ∞ proximal operator 实现。

涉及论文标题：
- MagR: Weight Magnitude Reduction for Enhancing Post-Training Quantization

---
