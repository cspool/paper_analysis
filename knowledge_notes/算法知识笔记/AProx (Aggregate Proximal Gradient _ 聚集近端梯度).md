## AProx (Aggregate Proximal Gradient / 聚集近端梯度)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
AProx（Aggregate Proximal Gradient，聚集近端梯度）是 PARQ 论文提出的随机优化算法，用于求解 PAR-正则化目标函数 minimize_w f(w)+λΨ(w)。算法形式：u^{t+1}=u^t-η_t ∇f(w^t,z^t)（隐变量累积纯梯度），w^{t+1}=prox_{γ_t λ Ψ}(u^{t+1})（用累积步长 γ_t=Ση_s 缩放正则化）。AProx 的核心创新在于使用累积步长 γ_t（而非 Prox-SGD 的单步步长 η_t）缩放近端正则化映射。由于 γ_t → ∞，proximal map 中 flat segments 长度不断增大，slanted segments 相对缩小，使软量化渐近收敛到硬量化。这与 Prox-SGD 的 diminishing regularization（η_t→0 导致正则化消失）恰好相反。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
AProx 与 Prox-SGD 的核心对比：
```
Prox-SGD: u^{t+1}=w^t-η_t g^t (w^t 已含过往 prox 贡献), w^{t+1}=prox_{η_t λ Ψ}(u^{t+1})
问题: η_t→0 → 正则化消失 → 无量化效果

AProx: u^{t+1}=u^t-η_t g^t (u^t 仅累加梯度), w^{t+1}=prox_{γ_t λ Ψ}(u^{t+1})
优势: γ_t→∞ → 正则化增强 → 软→硬量化渐进收敛
```
AProx 等价于 ProxConnect（Dockhorn et al. 2021），但从 RDA（Xiao 2010）推导而来，比 Fenchel-Rockafellar 对偶框架更直观。定理 3.2 证明了最后迭代收敛 O(ln(t)/√t)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PARQ 实现中，因训练迭代次数有限使得 γ_t 达不到无穷大，使用独立斜率 schedule ρ_t^{-1}（cosine decay 从 1→0）模拟 γ_t 的渐进效应。PARQ 算法将 AProx 的三个组件（LSBQ 在线估 Q、prox 软量化、逆斜率 schedule）组合成实用的 QAT pipeline。

涉及论文标题：
- PARQ Piecewise-Affine Regularized Quantization
