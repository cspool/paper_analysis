## ProxConnect（近端连接）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ProxConnect 是 Dockhorn et al.（2021, NeurIPS）提出的 QAT 泛化框架，将 BinaryConnect 的硬量化映射替换为任意单调递增 proximal map。推导自 Fenchel-Rockafellar 对偶和广义条件梯度（Yu et al. 2017）。PARQ 论文明确指出 AProx ≡ ProxConnect（仅有 γ_t 设置的微差异），但给出了一条更直观的推导路径（从 RDA/Xiao 2010 推广）和更强的理论结果。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ProxConnect/AProx 的统一形式：u^{t+1}=u^t-η_t g^t, w^{t+1}=P(u^{t+1})，P 为任意 monotone non-decreasing proximal map。关键区别：PARQ 构造的凸 PAR 给出了 P 的显式闭式解（式 7），而 Dockhorn et al. 虽然讨论了凸正则化的可能性但没有给出具体构造。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PARQ 的核心理论贡献：(1) 构建了具体的凸 PAR 实例（之前仅有非凸正则化的 W 形）；(2) 证明 ProxConnect/AProx 在凸 PAR 下具有最后迭代收敛——而 Dockhorn et al. 仅证明了平均迭代收敛（对 QAT 无实际意义，因为平均值通常不被量化）。

涉及论文标题：
- PARQ Piecewise-Affine Regularized Quantization
