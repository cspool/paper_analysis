## Critical Module Alignment (CMA, 关键模块对齐)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Critical Module Alignment（CMA）是 QuEST 提出的两种局部对齐损失之二，专门微调扩散模型中量化敏感的注意力相关层。其核心观察（Property ❷）是：不同类型层的激活对位宽降低的敏感度差异巨大——FeedForward 层激活在 6-bit 即导致生成失败，而所有其他线性层（虽多 5 倍）在 4-bit 才失败，卷积层（虽多 3 倍）也在 4-bit 才失败。CMA 通过最小化这些关键层的量化输出与全精度输出 MSE 来缓解敏感层的量化退化：L_CMA = Σ_{l∈C_A} E_t[||O(z_{t,l}; w_l) - Õ(z̃_{t,l}; w_l, ŝ)||²]，其中 C_A 为注意力相关层集合（包括 Q/K/V/O 投影和 FeedForward 层），z̃_{t,l} 为量化后的层输入，ŝ = s \ s_l (l∈C_TE) 表示除 TLA 已优化的量化参数外的所有参数。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CMA 的执行流程伪代码：
```
# 阶段二：CMA —— 微调注意力相关层（TLA 冻结后）
w_TE, s_TE = frozen()                    # TLA 结果已冻结
C_A = {Q_proj, K_proj, V_proj, O_proj, FeedForward_layers}
w_A = load_fp_weights()                  # 注意力相关层全精度权重
s = init_minmax_scales()                 # 所有激活量化参数（含 C_A 和其他层）
ŝ = s \ s_TE                             # 排除 TLA 已优化的参数
optimizer = Adam([w_A, ŝ], lr_w=1e-5, lr_s=1e-4)

for epoch in range(CMA_epochs):
    for t in sample_time_steps():
        x_T = randn(latent_shape)
        z_fp = get_calibration_activation(t, layer=l) # 预存的全精度各层激活
        z̃ = Q_model.forward(x_T, t)       # 量化模型前向（含 TLA 量化TE层）
        # 对每个 l ∈ C_A:
        O_fp_l = z_fp[l]                  # 全精度输出
        O_q_l = z̃[l]                      # 量化输出
        loss = sum(MSE(O_fp_l, O_q_l) for l in C_A)  # L_CMA
        loss.backward()                   # 仅 w_A, ŝ 有梯度
        optimizer.step()
# w_A 和 ŝ 得到更新，未选中的层仅 ŝ 得到优化（通过全局损失）
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CMA 的关键实现细节：(1) 每个模块使用不同的输入 z̃_{t,l} 进行优化（即已量化的前一层输出），以增强模块对输入扰动的鲁棒性；(2) CMA 在 TLA 之后执行（progressive alignment），因为时序信息独立于图像输入且在模型早期确定，先对齐时间嵌入为后续模块提供准确的时间步指导；(3) CMA 可显著改善 FID——在 TLA 基础上添加 CMA 使 FID 从 4.41 降至 3.26（W4A8, LSUN-Bedrooms）；(4) 如果没有全局损失 L_G，CMA 单独使用的 FID 为 8.99，添加 L_G 后降至 6.41（FID 提升 2.58），说明 CMA 需要全局监督来指导非敏感层的量化参数优化。CMA 的层选择策略是基于经验性的逐层位宽降低实验，而非自动搜索。

涉及论文标题：
- QuEST Low-bit Diffusion Model Quantization via Efficient Selective Finetuning
