## Hardware Approximation Function (HAF, 硬件近似函数)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hardware Approximation Function (HAF) 是 LOGART 提出的将硬件乘法近似嵌入 PTQ 优化循环的机制。核心思想：对数量化中 base-√2 权重与激活相乘需要乘 √2，但 √2 不能用简单移位实现。HAF 用 K-term Signed Dyadic Expansion (SDE) 近似 √2（如 √2 ≈ 2⁰ + 2⁻¹），将乘法替换为 shift-add。关键是 HAF 作为附加模块插入量化前向传播中：在 LLR 优化期间，前向路径使用 SDE 近似计算 Ŵ，近似误差被梯度下降作为噪声吸收（而非后处理补偿）。这确保了量化后模型在真实硬件（使用 shift-add 而非精确 √2 乘法）上仍保持高精度。

从硬件架构角度拆解术语：
```
# HAF 在 LogART PTQ forward pass 中的位置
# 正常 dequant: Ŵ = S · sign(W) ⊙ B^{-Q_W}
# HAF 修改后:
M = (Q_W mod 2) ⊙ [B == √2]  # 识别需近似的位置
γ = SDE(√2, K) / √2          # 近似因子 (e.g., γ ≈ 1.414/1.414 = 1 for K→∞)
Ŵ' = Ŵ ⊙ (1 + (γ - 1)·M)     # 仅对 base-√2 的奇数 Q_W 元素注入近似
```
M 的含义：当 B=√2 且 Q_W 为奇数时，Ŵ 涉及 √2^{odd} = √2 · 2^{floor(odd/2)} = √2 · (容易移位实现的部分)。SDE 仅作用于孤立的 √2 因子。

术语一般如何实现？如何使用？
HAF 在 PyTorch 中作为额外前向操作实现：`W_hat_prime = W_hat * (1 + (gamma - 1) * mask)`。K=2 时 SDE: √2 ≈ 1 + 0.5（误差 0.058），K=3 时: √2 ≈ 1 + 0.5 + 0.03125（误差 0.0024）。LOGART 实验：HAF 导致 <0.2% accuracy degradation (vision) 和 <0.2 PPL (LLM)，同时 AE 面积从 95.8 µm² (BRECQ) 降至 53.2 µm²。

涉及论文标题：
- LOGART: PUSHING THE LIMIT OF EFFICIENT LOGARITHMIC POST-TRAINING QUANTIZATION

---
