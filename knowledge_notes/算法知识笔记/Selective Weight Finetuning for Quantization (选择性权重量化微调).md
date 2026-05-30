## Selective Weight Finetuning for Quantization (选择性权重量化微调)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
选择性权重量化微调是一种参数高效的低比特量化策略：仅微调模型中一小部分关键权重（而非全量参数），使量化模型输出对齐全精度模型。QuEST 的选择策略基于两类识别：(1) 时间嵌入层（需要准确传递时序信息）；(2) 注意力相关层（对位宽降低最敏感）。微调参数总量不足 7%（如 LDM-4 上 <7% 参数被更新），远超 EfficientDM（需训练 LoRA adapter）和 Full-finetune（100% 参数）的效率。选择性微调的理论依据来自 Theorem 3.2：在低比特下，激活扰动 Δ 太大导致泰勒展开不准，需要微调权重 w_n 使模型对扰动鲁棒——但只有与特定输入/功能强相关的层需要此处理，其他层的量化误差可通过全局损失间接优化其量化参数来缓解。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
选择性微调与全量微调的对比伪代码：
```
# 全量微调（Full-finetune）
for epoch in range(epochs):
    for t in sample_time_steps():
        output_q = Q_model(x_T, t)        # 全模型前向
        loss = MSE(output_fp, output_q)   # 仅全局损失
        loss.backward()                   # 所有 w 和 s 有梯度
        optimizer_full.step()             # 更新 ~100% 参数

# QuEST 选择性微调
frozen_layers = set(all_layers) \ (C_TE ∪ C_A)  # >93% 的层
for epoch in range(TLA_epochs):           # 阶段一：仅 TE 层
    output_q_TE = Q_model.TE_forward(t)
    loss = MSE(FP_TE, output_q_TE)        # L_TLA
    loss.backward()                       # 仅 w_TE, s_TE
    optimizer.step()

for epoch in range(CMA_epochs):           # 阶段二：仅 Attn 层
    output_q_all = Q_model(x_T, t)        # TE 层已量化冻结
    loss = L_CMA + 2*L_G                 # CMA + 全局
    loss.backward()                       # 仅 w_A, ŝ
    optimizer.step()
```
参数计数对比（LDM-4）：总参数 ≈ 400M，C_TE ≈ 0.5% ≈ 2M，C_A ≈ 5% ≈ 20M，微调参数 < 28M = < 7%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
QuEST 的选择性微调实现：(1) 权重量化参数固定（s_w 不参与训练），仅微调全精度权重和激活量化参数；(2) 微调无需引入额外参数（与 EfficientDM 的 LoRA 不同），直接修改原始权重；(3) 数据无关（校准集来自高斯噪声）；(4) 训练效率高——LDM-4 W4A8 仅需 0.45 GPU 小时（A6000），而 EfficientDM 需 2.60 小时，Full-finetune 需 0.85 小时但显存更高（15076MB vs 12178MB）；(5) 可扩展到 Stable Diffusion（单 48GB GPU 即可完成，而 Full-finetune 会 OOM）；(6) 集成 LoRA 反而降低性能（FID 增加 5.62），验证了直接微调原始层权重的有效性。

涉及论文标题：
- QuEST Low-bit Diffusion Model Quantization via Efficient Selective Finetuning
