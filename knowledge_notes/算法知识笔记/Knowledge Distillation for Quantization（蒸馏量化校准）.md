## Knowledge Distillation for Quantization（蒸馏量化校准）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
蒸馏量化校准（Distillation Quantization Calibration, DQC）是 2DQuant 提出的将知识蒸馏应用于 PTQ 第二阶段的技术。其核心思想是：将全精度（FP）模型作为教师网络，将量化后的模型作为学生网络（两者结构完全相同），通过最小化教师和学生之间在输出层面和中间特征层面上的差异，来微调量化器的 clip bounds。这种做法将量化参数优化从"最小化数值偏移（MSE）"提升到"面向任务目标的优化"，能更有效地保持量化模型的感知质量。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
2DQuant 的 DQC 阶段具体的损失函数与训练流程：
```
# 输出层蒸馏损失 (Eq.4)
L_O = (1/(C_O*H_O*W_O)) * ||O_fp - O_q||_1
# L1 Loss，促使量化模型的最终 SR 输出逼近 FP 模型

# 中间特征蒸馏损失 (Eq.5)
L_F = Σ_i (1/(C_i*H_i*W_i)) * ||F_i/||F_i||_2 - F_qi/||F_qi||_2||_2
# 对每层特征做 L2 归一化后计算 L2 距离，消除尺度差异影响

# 总损失 (Eq.6)
L = L_O + λ * L_F

# 训练配置：
optimizer = Adam(lr=1e-2, betas=(0.9, 0.999), weight_decay=0)
scheduler = CosineAnnealing
iterations = 3000
batch_size = 32
calibration_data = 32 random crops (3×64×64) from DF2K
```
关键设计：(1) 学生和教师共享相同网络结构，无需额外适配层；(2) L1 输出损失替代 L2（在 SR 任务中 L1 收敛性更好）；(3) 特征蒸馏使用 L2 归一化后的特征，消除量化引起的尺度变化干扰；(4) 仅更新量化器的 clip bounds（l, u），不修改模型权重本身。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
知识蒸馏用于量化的通用方式：(1) 输出蒸馏——最小化量化模型与 FP 模型最终输出的差异（L1/L2/KL 散度）；(2) 特征蒸馏——在中间层对齐特征表示，可使用 L2 距离、attention transfer 或 Gram 矩阵匹配；(3) Logit 蒸馏——对分类任务使用 soft targets（temperature-scaled softmax）。在 PyTorch 中实现类似 DQC 的蒸馏训练时，核心代码模式为：`loss = criterion(student_out, teacher_out.detach()) + lambda * feature_loss`，仅优化器和学生的量化参数被设为 requires_grad=True。

在 Task-Specific ZSQ for Object Detection 中，知识蒸馏被用于 QAT 阶段的三项联合蒸馏：(1) 预测匹配蒸馏 L_KD = (tau^2/N)*Sigma KL(z^F(x_i;theta), z^Q(x_i;theta'))，使用 KL 散度对齐 teacher 和 student 的输出预测分布；(2) 特征级蒸馏 L_feat = (1/(NL))*Sigma||f_l^F(x_i;theta) - f_l^Q(x_i;theta')||_2^2，MSE 对齐中间层特征图以稳定低比特训练、防止误差累积；(3) Task-specific 检测损失 L_detect（L_category + L_box + L_conf），直接利用合成标签训练 student 的检测能力。总损失 L^Q = beta_KL*L_KD + beta_feat*L_feat + beta_detect*L_detect。YOLOv5 超参 {beta_detect, beta_KL, beta_feat} = {0.04, 0.1, 1.0}。消融证明三项互补：同时移除 L_feat 和 L_KD 导致 mAP 下降 7.2pp（YOLOv5-s W4A4: 19.0% vs 11.8%）。

涉及论文标题：
- 2DQuant Low-bit Post-Training Quantization for Image Super-Resolution
- APHQ-ViT: Post-Training Quantization with Average Perturbation Hessian Based Reconstruction for Vision Transformers
- Squat (EdgeQAT): Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge
- PMQ-VE Progressive Multi-Frame Quantization for Video Enhancement
- Task-Specific Zero-shot Quantization-Aware Training for Object Detection

在 PMQ-VE 中，知识蒸馏扩展为多教师层次化蒸馏（PMTD）：训练低比特量化模型时，同时使用 FP 全精度教师和中间比特（INT8）教师进行监督，通过 α(t) 线性增长权重使监督信号从 INT8 逐步过渡到 FP。每个教师包含输出 L2 重建损失和中间特征 MSE 损失（λ=5），相比 2DQuant 的单教师 L1+归一化 L2 损失，PMTD 的多教师策略通过弥合容量差距更有效地提升低比特（4-bit/2-bit）模型的性能。

在 Squat 中，知识蒸馏被用于QAT训练：FP16教师模型通过软蒸馏（soft distillation）指导量化学生模型。蒸馏损失 L_distill = (1-γ)·L_CE + γ·τ²·L_KL。此外，Squat创新地在蒸馏中加入熵损失L_E（最大化量化query/key熵）和分布损失L_D（对齐量化与FP16注意力图余弦相似度），形成 L_total = L_distill + 0.5·L_E + 1.0·L_D 的复合蒸馏目标。

---
