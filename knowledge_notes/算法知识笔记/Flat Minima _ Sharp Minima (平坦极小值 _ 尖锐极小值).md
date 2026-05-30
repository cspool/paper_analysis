## Flat Minima / Sharp Minima (平坦极小值 / 尖锐极小值)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Flat Minima 和 Sharp Minima 是损失景观中两类不同性质的局部极小值。平坦极小值：权重空间中的大连通区域，区域内损失近乎恒定；尖锐极小值：损失在极小值点周围迅速上升的窄小区域。Hochreiter & Schmidhuber (1997) 定义平坦极小值为"误差保持近似恒定的大连通区域"。泛化关系（Keskar et al., ICLR 2017）：尖锐极小值对扰动敏感 → OOD 泛化差；平坦极小值对扰动鲁棒 → 泛化好。平坦度通过 F_γ(w) = E[E(w') - E(w)] 量化（w' 在 w 为中心半径 γ 的球面上 Monte-Carlo 采样），值越小越平坦。注意 Andriushchenko et al. (ICML 2023) 指出平坦极小值并非普遍保证更好泛化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QT-DoG 中量化噪声驱动平坦极小值的机制：二阶 Taylor 展开 L(w+Δ) ≈ L(w) + ∇LΔ + ½Δ^T H Δ。尖锐区域中 H 特征值大 → ½Δ^T H Δ 项使损失急剧增大 → 模型"逃离"；平坦区域中 H 特征值小 → 噪声影响小 → 模型稳定收敛。QT-DoG 训练平坦度超过 ERM、SAM、SWA，与 SWAD 相当但模型小 75%。寻找平坦极小值的主要方法：(1) SWA——平均训练轨迹上的 checkpoint；(2) SWAD——在验证最优区间内做权重平均；(3) SAM——最小化 max_{||ε||≤ρ} L(w+ε)；(4) 噪声注入——QT-DoG 的量化噪声、权重噪声等隐式正则化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
平坦度分析工具：Hessian 特征值谱、F_γ(w) Monte-Carlo 估计（100 次随机球面采样）、损失景观可视化（沿两个随机方向）。Rissanen (1978) 和 Hochreiter & Schmidhuber (1997) 证明平坦极小值对应低复杂度网络，需要更少的 bit 信息表示每个权重——这从信息论角度解释了为何降低权重比特精度（量化）能引导优化趋向平坦极小值。

涉及论文标题：
- QT-DoG Quantization-Aware Training for Domain Generalization
