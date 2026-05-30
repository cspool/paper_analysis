## MLP Reconstruction (MR) — GELU-to-ReLU Replacement for ViT PTQ

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MLP Reconstruction (MR) 是 APHQ-ViT 提出的针对 ViT 中 post-GELU 激活量化困难的后训练重建方法。核心操作：(1) 将 ViT MLP 中的 GELU 激活函数替换为 ReLU；(2) 用 APH 加权的特征蒸馏损失重建 MLP，使 ReLU 版本逼近原始 GELU 版本输出。MR 同时解决 post-GELU 激活两个难题：负值集中在 [-0.17, 0] 造成的分布不平衡（GELU 密集负值被 ReLU 的精确零替代），以及层间激活范围变化大（通过 clamp loss 将正激活限制在 99% 分位数内）。因 MR 逐层单独重建（浅层网络），避免了深层 ReLU 的 dying ReLU 问题。额外收益：ReLU 可折叠进前层 FC，在 W8A8 CPU 推理中实现 1.49×-1.75× 加速。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# MR 伪代码
O_GELU = FC2(GELU(FC1(X)))                    # 原始 MLP 输出
MLP.activation = ReLU()                        # 替换 GELU → ReLU

for iter in range(max_iter):
    A_fc2 = ReLU(FC1(X))
    O_direct = FC2(A_fc2)                      # 无 clamp 输出
    thresh = quantile(A_fc2, p=0.99)           # 99% 分位数
    O_clamp = FC2(clamp(A_fc2, max=thresh))    # clamp 输出
    
    L_direct = mean((O_GELU - O_direct)^2 * H_bar)  # APH 加权
    L_clamp  = mean((O_GELU - O_clamp)^2  * H_bar)
    L_distill = L_direct + 2.0 * L_clamp       # α=2
    L_distill.backward(); optimizer.step()
```
注：L_Direct 不可省略——L_Clamp 对硬截断区域的梯度为 0，单独使用会导致梯度消失。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MR 在 APHQ-ViT 中作为 block 重建的第一阶段（在量化重建前）。参数：p=0.99, α=2, max_iter=20000, batch_size=32, lr=4e-5。单独使用 MR（不量化）精度损失 <0.5%，ViT-B 上甚至超全精度 baseline。局限性：仅在 ViT MLP 上验证，未在 LLM 等模型测试。

涉及论文标题：
- APHQ-ViT: Post-Training Quantization with Average Perturbation Hessian Based Reconstruction for Vision Transformers
