## QR-Orth Optimization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
QR-Orth 是 DartQuant 提出的正交矩阵优化方案：引入隐参数 Z ∈ R^{n×n}（无约束实矩阵），通过 QR 分解 R = Q(Z) 获得正交旋转矩阵。优化时用标准 SGD/Adam 在欧几里得空间更新 Z，梯度通过 QR 分解自动微分反向传播至 Z，校准完成后丢弃 Z 仅保留 R。相比 Cayley SGD（需在 Stiefel 流形上投影，额外 6n³ 计算量），QR-Orth 仅需 Householder QR 分解约 4/3 n³。100 步 SGD 实测：QR-Orth 5.7h vs Cayley 8.2h（1.44×）。因 Whip Loss 配合下收敛更快，QR-Orth SGD 6 步即达 Cayley SGD 100 步效果（41× effective 加速）。

从算法pipeline角度拆解术语，给出具体例子。
```
Z = random_hadamard(n)                     # 隐参数初始化
optimizer = SGD([Z], lr=lr)
for step in range(max_steps):
    R, _ = torch.linalg.qr(Z)              # QR: Z = Q·R_upper, R ← Q
    O = X @ R
    loss = exp(-|O|).sum()                 # Whip Loss
    loss.backward()                        # 梯度通过 QR 反向传播至 Z
    optimizer.step()
R_final, _ = torch.linalg.qr(Z)  # 最终旋转矩阵
del Z  # 丢弃隐参数
```

术语一般如何实现？如何使用？
直接使用 PyTorch `torch.linalg.qr(Z)`（内部调用 cuSOLVER/LAPACK 优化的 Householder QR）。Householder QR 计算复杂度约 4/3 n³（论文 Appendix B 详细推导）。与任意 PyTorch 标准优化器（SGD/Adam/AdamW）兼容。对 n ≤ 16384（主流 LLM hidden dim）的矩阵，QR 分解在 GPU 上可高效完成。通用技术，适用于任何需要优化正交矩阵的场景。

涉及论文标题：
- DartQuant Efficient Rotational Distribution Calibration for LLM Quantization

---
