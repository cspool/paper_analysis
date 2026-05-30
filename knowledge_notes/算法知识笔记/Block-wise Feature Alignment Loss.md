## Block-wise Feature Alignment Loss

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Block-wise Feature Alignment Loss 是 SPR²Q 提出的细粒度特征对齐损失：L_feature = E_x [Σ_{l=1}^{L} ||φ_l(f_q(x)) - φ_l(f_FP(x))||_2²]，其中 φ_l(·) 是第 l 个计算 block 提取的中间特征图。不同于仅对最终输出做像素级 L1 损失的粗粒度监督，此损失在网络的每一层 block 输出级施加对齐约束，确保量化误差在每个 block 被局部补偿而非累积到输出端。属于 self-distillation 范畴（量化模型从自身 FP 版本学习），与知识蒸馏中的教师-学生 feature distillation 的区别在于对等 block 之间的对齐。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
L_feature = 0
for l in range(L):  # 遍历所有 L 个 Mamba block
    phi_q = intermediate_feature(quantized_model, X, block=l)
    phi_fp = intermediate_feature(fp_model, X, block=l)
    L_feature += ||phi_q - phi_fp||_2^2

L_pixel = ||f_q(X) - f_fp(X)||_1  # 像素级 L1
L_total = L_pixel + λ * L_feature  # Eq.6
```
SPR²Q 的总损失 = L_pixel + λ * L_feature，λ 为平衡超参数。此设计使模型同时关注全局 reconstruction fidelity (L_pixel) 和各 block 的局部 consistency (L_feature)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
通过在模型各 block 输出处注册前向 hook 提取中间特征。适用于 pixel-level fidelity 敏感任务（图像/视频超分辨率、图像恢复），这些任务中量化引起的中间层误差会逐步累积并在输出层表现为纹理模糊和细节丢失。SPR²Q 的消融验证仅使用 L_pixel 显著少于 L_pixel + L_feature。

涉及论文标题：
- SPR²Q: Static Priority-based Rectifier Routing Quantization for Image Super-Resolution
