## Average Perturbation Hessian (APH) Loss

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Average Perturbation Hessian (APH) 是一种用于量化重建的损失函数，直接通过有限差分法计算输出 Hessian 矩阵的对角元，用以衡量每个输出元素对任务损失的重要性。推导流程：(1) 将量化视为对输出 O 的小扰动 ε，对蒸馏损失 L 进行 Taylor 展开：L(O+ε) - L(O) ≈ (1/2)·ε^T·H·ε，其中一阶项 J(O) 在 O 处为 0（蒸馏损失在 O=Ô 处取极小值）；(2) 对 block 输出施加微小扰动 ΔO=10^-6 得到 O⁺=O+ΔO 和 O⁻=O-ΔO；(3) 前向传播通过剩余 blocks 计算蒸馏损失；(4) 反向传播得到 Jacobian J⁺ 和 J⁻；(5) 用均值定理 H_i = (J⁺_i - J⁻_i) / (2·ΔO) 直接计算 Hessian 对角元；(6) 对所有校准样本取平均 H̄ = (1/N)·ΣH^(n)。最终 APH loss 为 L_APH = Σ_i (Ô_i - O_i)² · H̄_i。与 BRECQ 的 Hessian loss（Fisher Information Matrix + 梯度平方近似）相比，APH 直接从定义推导，避免 FIM 近似误差，理论上可泛化到检测、分割等多任务。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# APH 计算流程（APHQ-ViT Algorithm 1）
O = B.forward(X)                              # [N, L, D]
O_plus, O_minus = O + 1e-6, O - 1e-6
logit       = M.rest(O)                        # 通过剩余 blocks
logit_plus  = M.rest(O_plus)
logit_minus = M.rest(O_minus)
L_plus  = DistillLoss(logit, logit_plus)       # 分类: KL div
L_minus = DistillLoss(logit, logit_minus)      # 检测: KL + smooth L1
J_plus  = grad(L_plus,  O_plus)                # ∂L/∂O⁺
J_minus = grad(L_minus, O_minus)               # ∂L/∂O⁻
H_sample = (J_plus - J_minus) / (2e-6)         # 有限差分 Hessian 对角
H_bar = mean(H_sample, dim=0)                  # 跨样本平均

# APH loss 在量化重建中的使用
O_hat = B_quantized.forward(X)
L_APH = sum((O_hat - O)^2 * H_bar)             # APH 加权 MSE
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
APH loss 在 APHQ-ViT 开源代码中通过 PyTorch 自定义实现：(1) 在 block 重建前预先计算 H_bar（一次额外 forward+backward）；(2) H_bar 存储为张量在迭代中复用；(3) 蒸馏损失：分类用 nn.KLDivLoss，检测用 KLDivLoss + nn.SmoothL1Loss；(4) 定理 3.2 证明 APH 梯度方差更低（Var[∂L_APH/∂θ] ≤ Var[∂L_PH/∂θ]），训练更稳定。仅增加一次额外 forward/backward pass，训练复杂度不变。校准集：ImageNet 1024 张无标签图。

涉及论文标题：
- APHQ-ViT: Post-Training Quantization with Average Perturbation Hessian Based Reconstruction for Vision Transformers
