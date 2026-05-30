## Expert Capacity / Adaptive Capacity in MoE

术语解释
Expert Capacity 是 MoE 路由中每个 expert 能处理的最大 token 数。传统方法用固定容量 C = capacity_factor × s/n。ETR 提出自适应容量策略，根据训练进度和亲和力分数动态调整 C。

术语是什么？
Expert capacity 决定每个 expert 一次前向中能处理多少 token。容量过低→过多 token drop (影响质量)，过高→浪费计算和通信 padding。ETR 证明自适应容量可将下界降低最多 40%:

$$C_{\min} = \frac{1}{n} \exp\left(\frac{d \cdot \delta_{\max}^2}{2 - \delta_{\max}^2}\right)$$

δ_max 为 gating weight 与 token 间的最大角度偏差。训练早期 δ_max 大 (token 分布分散)，C 需较大；后期 token 特征收敛，δ_max 减小，C 可显著降低。

术语一般如何实现？如何使用？
在 ETR 中，C 每 step 动态计算：统计当前 batch 的亲和力分数分布，计算 δ_max，代入 C_min 公式，取 max(C_min, s/n) 为最终容量。降低 C 直接减少 expert FFN 计算所需的中间 buffer，减少显存 4.57%-16.27%。

涉及论文标题：
- Expert-Token Resonance Redefining MoE Routing through Affinity-Driven Active Selection
