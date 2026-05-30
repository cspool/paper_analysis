## Temporal Layer Alignment (TLA, 时序层对齐)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Temporal Layer Alignment（TLA）是 QuEST 提出的两种局部对齐损失之一，专门用于微调扩散模型中的时间嵌入（Time Embedding）层以改善量化性能。其核心观察（Property ❶）是：扩散模型在不同时间步承担不同功能（早期去噪 vs 晚期细化），因此准确的时间信息传递对量化至关重要——仅量化时间嵌入层就可使 FID 从 6.77 升至 7.58（W8A8），W4A8 下从 7.55 升至 8.59（相对提升约 15%）。TLA 通过最小化时间嵌入层量化输出与全精度输出的 MSE 来恢复时序精度：L_TLA = Σ_{l∈C_TE} E_t[||O(t; w_l) - Õ(t; w_l, s_l)||²]。关键设计：(1) 同时微调时间嵌入层权重 w_l 和该层的激活量化参数 s_l；(2) 使用单一量化参数集适配所有时间步（无需按时间步分别存储），提高时间效率和存储效率；(3) 时间嵌入层的微调权重和量化参数在后续 CMA 阶段冻结。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TLA 的执行流程伪代码：
```
# 阶段一：TLA —— 仅微调时间嵌入层
w_TE = load_fp_weights()           # 全精度时间嵌入层权重
s_TE = init_minmax_scales()        # 初始化激活量化参数
optimizer = Adam([w_TE, s_TE], lr_w=1e-5, lr_s=1e-4)

for epoch in range(TLA_epochs):
    for t in sample_time_steps():
        x_T = randn(latent_shape)
        O_fp = FP_model.time_embed(t; w_TE_fp)    # 全精度输出（来自校准集）
        w_TE_q = quantize(w_TE, s_w)               # 权重量化（冻结s_w）
        O_q = Q_model.time_embed(t; w_TE_q, s_TE)  # 量化输出
        loss = MSE(O_fp, O_q)                       # L_TLA
        loss.backward()                             # 仅 w_TE, s_TE 有梯度
        optimizer.step()
# TLA 完成后冻结 w_TE, s_TE
```
注意：时间嵌入 t 先经一个或两个线性层转换为 time embedding vector，然后在 UNet 各层注入（通过投影层与 latent image representation 合并）。TLA 微调覆盖这两类线性层。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TLA 是一个两阶段微调策略的第一阶段。实现要点：(1) 权重量化参数在 TLA 期间保持冻结（仅微调全精度权重，量化参数固定）；(2) 激活量化参数与权重联合优化；(3) 微调后的时间嵌入层甚至能超越全精度 baseline——由于同时优化了时间嵌入层和其量化参数，经过微调的量化模型可能输出比原始全精度模型更精确的时间信息（表 2：QuEST TLA 的 FID 5.61 vs FP 6.77）；(4) TLA 可独立使用，也可与 CMA 和全局损失组合获得最佳效果。

涉及论文标题：
- QuEST Low-bit Diffusion Model Quantization via Efficient Selective Finetuning
