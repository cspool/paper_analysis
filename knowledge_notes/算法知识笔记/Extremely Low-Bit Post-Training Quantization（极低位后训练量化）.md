## Extremely Low-Bit Post-Training Quantization（极低位后训练量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
极低位后训练量化（Extremely Low-Bit PTQ）指将 LLM 权重量化到 sub 2-bit（即有效位宽 <2 bit/权重）的后训练量化方法。与 4-bit/8-bit PTQ（如 GPTQ、AWQ、OmniQuant）不同，极低位 PTQ 面临的核心挑战是：量化误差在 1-2 bit 范围内急剧放大，二值化（1-bit）是其中最极端的形式。极低位 PTQ 的典型策略包括：(1) **混合精度**——部分显著权重保留较高位宽（4-bit/8-bit），其余二值化；(2) **结构化掩码**——用结构化方式标记显著权重以减少掩码存储开销；(3) **缩放因子优化**——通过可学习或迭代方式优化二值化缩放因子；(4) **预处理/后处理**——量化前后通过微调或变换改善权重分布。PTQ1.61 首次将有效位宽真正降至 1.61-bit（不含非结构化掩码开销），而此前 PB-LLM (2.7-bit) 和 BiLLM (2.1-bit) 虽声称 sub 2-bit，实际因非结构化掩码额外占用 ≥1 bit 而超出 2-bit。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 PTQ1.61 的极低位量化流程为例：
```
# 输入: 预训练 LLM 权重 W，校准数据 X_calib
# 目标: 量化权重 W_q，平均位宽 < 2 bit/权重

# Step 1 (可选): 量化预处理
W_preprocessed = W + LoRA(W, pretraining_data)

# Step 2: 结构化掩码识别显著权重
for each linear layer W_layer:
    X_act = forward(X_calib, layer)
    ch_mag = ||X_act|| per channel
    salient_channels = topk(ch_mag, 20%)  # 一维掩码
    mask = [1 if i in salient_channels else 0 for i in range(m)]

# Step 3: 混合精度量化
W_q[salient] = round(W[salient] / S_q) + Z_q   # 4-bit
alpha = learnable_init(||w_row||_1 / n_w)
W_q[non-salient] = alpha * sign(W[non-salient])  # 1-bit

# Step 4: 分块优化缩放因子
for epoch in range(20):
    loss = MSE(out_fp, out_q) + (-log(cos_sim(out_fp, out_q)))
    alpha = AdamW(loss, lr=5e-4).step()
```
位宽计算（4096x4096 权重矩阵）：权重位宽 = 0.8x1 + 0.2x4 = 1.6-bit；掩码 = 4096/(4096x4096x(0.8+0.2x4)) ≈ 0.0002-bit；量化参数 ≈ 0.008-bit；合计 1.61-bit。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
极低位 PTQ 目前主要在 PyTorch 上用 fake-quantization 做精度仿真，因为商用 GPU 尚不支持 sub 4-bit 整数推理。PB-LLM 和 BiLLM 使用逐元素非结构化掩码（需额外 1-bit bitmap），PTQ1.61 用一维结构化掩码（额外 0.0002-bit）。关键指标：WikiText2/C4 困惑度 + 推理 benchmark 零样本准确率（lm-evaluation-harness）。开源：https://github.com/zjq0455/PTQ1.61。

涉及论文标题：
- PTQ1.61 Push the Real Limit of Extremely Low-Bit Post-Training Quantization
- PB-LLM Partially Binarized Large Language Models
- BiLLM Pushing the Limit of Post-Training Quantization for LLMs

---
