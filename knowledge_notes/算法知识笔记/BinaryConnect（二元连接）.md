## BinaryConnect（二元连接）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BinaryConnect 是 Courbariaux et al.（2015, NeurIPS）提出的早期 QAT 方法，是 STE 的代表性实践。更新规则：u^{t+1}=u^t-η_t ∇f(Q(u^t),z^t), w^{t+1}=Q(u^{t+1})。全精度隐变量 u^t 累积在量化权重 w^t=Q(u^t) 处的梯度，硬量化映射 Q(·) 每次将隐变量投影到离散集 Q^d。Ste 在反向传播时替换 dQ/du=0 为 1。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
1-bit BinaryConnect 训练流程：
```
初始化: u^1 = w^1 (随机 FP32 权重)
for t=1 to T:
    w^t = sign(u^t) * q          // q = ‖u‖₁/d 或 q=1
    g^t = ∇f(w^t, minibatch_t)
    u^{t+1} = u^t - η_t * g^t    // STE: dQ/du 替换为 1
```
多 bit 推广：Q(·) 为投影到 Q^d (Q={0,±q_1,...,±q_m}) 的阶梯函数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PARQ 论文中，BinaryConnect 被统一进 AProx 框架：当 Ψ=δ_Q（indicator function）时，prox_{Ψ}=Q(·) 且尺度不变（任何缩放下不变），因此 BinaryConnect/STE 是 AProx 的特例。PARQ 揭示 STE 是 AProx 在 γ_t→∞ 下的渐近极限，赋予启发式 STE 严谨的优化理论基础。

涉及论文标题：
- PARQ Piecewise-Affine Regularized Quantization
