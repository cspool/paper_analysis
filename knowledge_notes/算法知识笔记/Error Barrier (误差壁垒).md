## Error Barrier (误差壁垒)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Error Barrier（误差壁垒）是 Frankle et al. (ICML 2020) 提出的概念，用于量化两个训练好的权重点之间沿线性插值路径的最大损失增量。给定两个收敛的权重点 θ₁ 和 θ₂，定义插值点 θ_λ = (1-λ)θ₁ + λθ₂ (λ∈[0,1])，Error Barrier 为：max_{λ∈[0,1]} [L(θ_λ) − ½(L(θ₁)+L(θ₂))]。Error Barrier 是模型合并可行性的核心指标：零 Error Barrier 意味着线性模式连通性（Linear Mode Connectivity），即两个模型位于同一平坦的损失盆地中，可以安全地通过简单权重平均进行合并。高 Error Barrier 则表示两模型处于不同盆地，线性插值路径会穿越高损失区域，合并质量差。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 HDRQ 论文中，Error Barrier 被扩展到量化噪声场景。量化误差 ε₁, ε₂ 被建模为均匀噪声 U[-s/2, s/2]，量化后 Error Barrier 为：max_{λ∈[0,1]} [L(θ_λ+ε_λ) − ½(L(θ₁+ε₁)+L(θ₂+ε₂))]。通过二阶 Taylor 展开分解为原始 Error Barrier + 量化相关项。HDRQ 的关键推导：通过 Hessian 的 M-Lipschitz 连续性可以边界化合并点的 Hessian：|∇²L(θ_λ) − (∇²L(θ₁)+∇²L(θ₂))/2| ≤ M||θ₂−θ₁||/2。这表明降低 Error Barrier 的两个方向：(1) 控制 Hessian 谱（降低曲率敏感性）→ Hessian Regularization；(2) 降低权重间距离 ||θ₂−θ₁|| → Distance Regularization。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Error Barrier 的计算通常涉及：对 λ 在 [0,1] 区间采样多个点（如 λ=0.0, 0.1, ..., 1.0），在每个插值点评估测试集上的损失，找到最大偏差。HDRQ 将其用作理论分析工具而非训练目标：通过分析出降低 Error Barrier 的两条路径（控制 Hessian 和控制距离），指导设计对应的正则化方法。在模型合并实践中，Error Barrier 概念也指导了合并策略（如 permutation alignment via Git Re-Basin）和合并条件（从同一预训练模型微调的模型处于同一 basin，因此可通过简单平均合并）。

涉及论文标题：
- Merge-Friendly Post-Training Quantization for Multi-Target Domain Adaptation
