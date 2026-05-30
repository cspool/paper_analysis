## Fisher Information Matrix (FIM) in PTQ

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FIM 在 PTQ 中作为 Hessian 矩阵近似用于量化质量评估。F = E[(∂log p/∂W)(∂log p/∂W)^T]，在模型完美拟合真实分布且使用负对数似然损失时 FIM = E[H]（Bartlett 第二恒等式）。BRECQ 将 Hessian 对角元近似为 H_i ≈ (∂L/∂O_i)²。APHQ-ViT 指出：实际模型拟合不完美时 FIM ≠ Hessian；蒸馏损失（KL 散度）不满足 Bartlett 恒等式条件；无法泛化到非分类任务。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# BRECQ: FIM 近似 Hessian
H ≈ F,  F_i ≈ (∂L/∂O_i)²                     # 两点近似
L_brecq = Σ (Ô_i - O_i)² * (∂L/∂O_i)²

# APHQ-ViT: 直接计算 Hessian 对角
H_i = (J⁺_i - J⁻_i) / (2 * 1e-6), J = ∂L/∂O  # 有限差分
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FIM 近似实现简单（只需一次 backward），被 BRECQ/PTQ4ViT 广泛采用。但 APHQ-ViT 实验显示在 ViT 量化中不稳定（Table 4: BH 在 ViT-B 上 66.62% vs MSE 73.79%）。APH 以一次额外 forward/backward 代价换取更准确估计。

GuidedQuant (ICML 2025) 将 FIM 近似从 diagonal 扩展到 block-diagonal：保留每个 output channel 对应的 d_in×d_in Fisher block `F_j = (1/n) Σ (∂ℓ_i/∂w_j)(∂ℓ_i/∂w_j)ᵀ`，捕获 channel 内跨权重交互；并通过 averaging approximation 将 d_out 个 block 按 g 组（2~4）平均，使 70B 级 LLM 的存储和计算可行。FIM block 的构建等价于计算 `H̄_k = XᵀDiag(avg_squared_grad)X`，核心信息来自单次 backward pass 的梯度。

SqueezeLLM (ICML 2024) 使用 Fisher 对角作为 sensitivity-based non-uniform quantization 的权重。优化目标从 `argmin ||W - W_Q||²` 变为 `argmin Σ F_ii (w_i - Q(w_i))²`，其中 `F_ii = (1/|D|) Σ_d g_{d,i}²` 来自 calibration 数据集（仅需 10-100 样本）的一次 backward pass。该 weighted k-means 目标源自 Optimal Brain Damage (OBD) 框架：假设模型已收敛（梯度 g≈0），对 loss 做 Taylor 展开得到 `L(W_Q) ≈ L(W) + ½(W-W_Q)ᵀH(W-W_Q)`，用 Fisher 对角近似 Hessian 对角（H ≈ diag(F)）。与 GPTQ 的 layer-wise objective（最小化 ||WX - W_QX||²，即每层输出 activation 扰动）相比，SqueezeLLM 的 final-loss-based objective 在所有 sparsity level 下 PPL 低约 0.3（D.4 消融实验，LLaMA-7B 3-bit C4）。Fisher 计算开销小（65B 模型仅需 2.5 分钟 on A100），但需要一次完整 forward+backward（内存峰值 7B=33GB, 65B=292GB）。

涉及论文标题：
- APHQ-ViT: Post-Training Quantization with Average Perturbation Hessian Based Reconstruction for Vision Transformers
- GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
- SqueezeLLM Dense-and-Sparse Quantization
