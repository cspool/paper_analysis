## Taylor Expansion for State Discretization in SSM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Taylor Expansion for State Discretization 是 SSMLoRA 提出的一种替代 S4 标准 Zero-Order Hold (ZOH) 离散化的 SSM 状态更新方法。在标准 S4 中，连续状态空间方程 `h'(t) = Ah(t) + Bx(t)` 需通过 ZOH 将连续参数 A、B 离散化：`Ā = exp(ΔA)`、`B̄ = (ΔA)^{-1}(Ā - I)ΔB`，需矩阵指数和矩阵求逆。SSMLoRA 利用 Taylor 展开对状态直接一阶离散化：`h_t = h_{t-1} + h'_{t-1}·Δt`（取 Δt=1），即 `h_t = h_{t-1} + (h_{t-1}·W_c + x_new·W_d)`。核心优势：避免矩阵指数/求逆（仅需 O(r²) 矩阵乘法）、参数无需离散化保持可训练、h_t 可 detach 节省显存。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
# 对比：Taylor 展开 vs ZOH 离散化，均以 r=8 为例

# === SSMLoRA Taylor 展开（O(r²)，零额外开销）===
h_prime = h_prev @ W_c + x_new @ W_d    # [b,r] @ [r,r] + [b,r] @ [r,r] = [b,r]
h_new = h_prime + h_prev                # 一阶 Taylor: h += h'·1
# W_c,W_d: r×r=64 params each，共128 params（完全可忽略）

# === S4 ZOH 离散化（O(r³)，需要 matrix_exp + inverse）===
A_bar = torch.linalg.matrix_exp(delta * W_c)  # O(r³) expm
B_bar = torch.linalg.inv(W_c) @ (A_bar - I) @ W_d  # O(r³) inverse
h_new = A_bar @ h_prev + B_bar @ x_new
```
关键设计：h_t 在 SSMLoRA 中脱离计算图（detach），使 SSM 部分不参与反向传播——仅 W_c、W_d、W_a、W_b 接收梯度。结合零初始化策略（W_c、W_d 初始为零），训练初期 h' 为零，h 保持零向量，模型退化为稀疏 LoRA，逐渐学习非零状态转移。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SSMLoRA 中的具体实现：`self.W_c = nn.Linear(r, r, bias=False)`、`self.W_d = nn.Linear(r, r, bias=False)`（r=8 典型值），前向：`h_new = self.W_c(h_prev) + self.W_d(x_new) + h_prev`。ZOH 离散化（如 S4/Mamba 中使用）更适合精度敏感的长序列建模场景（严格数学推导保证数值稳定性）；Taylor 离散化更适合结合 LoRA——计算轻量、实现简单、小 rank 下精度损失可忽略。SSMLoRA 论文 Table 12 验证 LLaMA2-7B 推理开销与 LoRA 接近（4096 tokens: 2.740s vs LoRA 2.210s）。

涉及论文标题：
- SSMLoRA__Enhancing_Low-Rank_Adaptation_with_State_Space_Model

---
