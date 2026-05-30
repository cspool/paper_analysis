## Whip Loss

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
Whip Loss 是 DartQuant 提出的旋转分布校准损失函数：`Whip = Σ_{i=1}^{C_in} exp(-|x_i|)`（x 为旋转后激活向量）。设计灵感来自 Laplace→Uniform 的 CDF 变换 U_X(x) = τ[exp(x/b)-1]（x≤0）/ τ[1-exp(-x/b)]（x>0）。由于 LLM 激活近似 Laplace(0,b) 分布（论文 Appendix G 统计分析验证），exp(-|x_i|) 在零附近梯度大、远离零时梯度趋近于零。配合旋转矩阵的 norm-invariance 约束（||Rx||=||x||），Whip 将小值"推开"增大 → outliers 被迫减小保持 L2 范数不变 → 激活分布趋向均匀 → 量化误差降低。图 6 直方图验证 Whip 优化后的分布最接近均匀；图 7a 验证 Whip 的量化误差下降曲线远优于量化 loss/方差/峰度。

从算法pipeline角度拆解术语，给出具体例子。
```
# Whip Loss 在旋转校准中的使用 (DartQuant Algorithm 1)
for k = 0 to T_max:
    R = QR(Z)                       # QR-Orth 获得正交矩阵
    O = X @ R                        # 激活旋转: O ∈ R^{T × C_in}
    L = sum(exp(-|O_ij|))           # Whip Loss（所有 token 和 channel）
    Z = Z - lr * ∂L/∂Z              # SGD 更新隐参数
```

量化误差降低机制（4 维示例）：x=[x1≈0, x2≈0, x3≈0, x4>>0]。Whip 优化后 x1,x2,x3 绝对值增大。由 norm-invariance：√(x̃1²+x̃2²+x̃3²+x̃4²)=√(x1²+x2²+x3²+x4²)，前三项增大 → x̃4 必须减小（outlier 被抑制）。

术语一般如何实现？如何使用？
PyTorch 实现：`loss = torch.exp(-torch.abs(rotated_acts)).sum()`。超参：SGD, lr=1e-3~1e-2, epoch=10, batch=64（论文表 23）。校准仅需 128 样本 token 前向收集激活（无需标签），70B 模型校准仅 0.91 GPU-hours（SpinQuant 的 1/47）。局限性：(1) 假设激活近似零均值，偏差大时效果下降；(2) 针对均匀整数格式设计，FP4 等非均匀格式待验证。

涉及论文标题：
- DartQuant Efficient Rotational Distribution Calibration for LLM Quantization

---
