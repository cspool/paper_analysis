## Distribution Matching Distillation (DMD) / 分布匹配蒸馏

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Distribution Matching Distillation (DMD) 由 Yin et al. (CVPR 2024, NeurIPS 2024) 提出，将多步扩散模型蒸馏为少步生成器的知识蒸馏方法。DMD 匹配 student 和 teacher 的输出分布（distribution matching）而非具体去噪值。通过辅助判别器（critic network）区分 student 和 teacher 生成样本的分布差异，以 adversarial loss 驱动 student 学习 teacher 的样本分布。LongLive 使用 DMD 将 Wan2.1-T2V-1.3B 的多步扩散生成器蒸馏为 few-step 因果 AR 生成器：student Gθ 从噪声通过单步预测 x̂_0，teacher (Wan2.1-T2V-14B) 对同样 noisy input 去噪，critic 计算分布距离。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
def DMD_loss(G_theta, x_real, p_text, G_teacher, D_phi):
    eps = randn_like(x_real); t ~ U(0,1)
    x_noisy = sqrt(a_t)*x_real + sqrt(1-a_t)*eps
    x_hat_s = G_theta(x_noisy, t, p_text)     # student single-step
    x_hat_t = G_teacher(x_noisy, t, p_text)    # teacher denoising
    L_distill = mean((D_phi(x_hat_t, t) - D_phi(x_hat_s, t)) ** 2)
    L_student = mean(D_phi(x_hat_s, t))
    return L_distill + L_student
```

Annotations: LongLive DMD config: student lr=1e-5 (β1=0.0, β2=0.999), critic lr=2e-6。EMA decay=0.99 from step 200。在 streaming long tuning 中仅应用到当前 5s clip（teacher 在自身能力范围内）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DMD 在 LongLive 中：(1) ODE initialization → 将 Wan2.1-T2V-1.3B 初始化为 causal AR；(2) DMD 训练 → short window + frame sink；(3) Streaming long tuning → 继续 DMD + KV-recache。适用于：(a) 扩散模型加速；(b) 自监督蒸馏（无需 ground truth）；(c) AR 模型因果化适配。论文：Yin et al. CVPR/NeurIPS 2024。LongLive 开源：https://github.com/NVlabs/LongLive。

涉及论文标题：
- LongLive__Real-time_Interactive_Long_Video_Generation
