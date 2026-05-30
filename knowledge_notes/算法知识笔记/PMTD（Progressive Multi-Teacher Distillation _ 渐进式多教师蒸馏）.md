## PMTD（Progressive Multi-Teacher Distillation / 渐进式多教师蒸馏）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

PMTD（Progressive Multi-Teacher Distillation）是 PMQ-VE 框架精阶段（fine stage）的核心方法，一种层次化知识蒸馏策略，用于恢复低比特量化模型的表示能力。与传统的单教师蒸馏（仅使用 FP32 教师监督低比特学生）不同，PMTD 引入中间比特教师（如 INT8）作为"桥梁"，使低比特学生（如 INT4/INT2）能够分阶段、逐步地从更容易的目标学习。

PMTD 的核心洞察在于：低比特模型（4-bit/2-bit）与全精度模型之间存在显著的"容量差距"（capacity gap），直接使用 FP 教师蒸馏使学生在有限容量下难以学习高质量映射。通过引入 INT8 中间教师——一个已经量化感知但在容量上更接近低比特学生的模型——PMTD 为学生提供了更可达的短期学习目标，再逐步过渡到 FP 教师的长远目标。

训练 4-bit 模型的 PMTD 流程：(1) 首先用 FP 教师蒸馏训练 8-bit 模型；(2) 训练 4-bit 模型时，同时使用 INT8 教师和 FP 教师进行监督。损失函数 `L_PMTD = (L_INT + α(t)·L_FP) / (1+α(t))`，其中 α(t) = min(1, t/T_warmup) 线性增长，使训练早期偏向 INT8 教师（学习更易达成的目标），后期逐步过渡到 FP 教师（追求更高精度）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

PMTD 训练 4-bit 模型的伪代码：

```
# 预训练 INT8 教师（阶段 1）
int8_model = quantize(fp_model, bits=8)
for iter in range(N1):
    x = batch_data()
    out_int8 = int8_model(x)
    out_fp = fp_model(x).detach()
    L = L2(out_int8, out_fp) + λ * MSE(feat_int8, feat_fp)
    L.backward(); update clippings via STE

# 训练 4-bit 学生（阶段 2，PMTD 核心）
int4_model = BMFQ_initialize(model, bits=4)   # 粗阶段初始化
for t in range(T):
    x = batch_data()
    out_4bit = int4_model(x)
    out_int8 = int8_model(x).detach()          # 中间教师
    out_fp = fp_model(x).detach()              # 全精度教师

    # 每个教师的损失 = 重建损失 + 特征匹配损失
    L_rec_int = ||out_4bit - out_int8||_2^2          # L2 输出损失
    L_feat_int = MSE(f_4bit, f_int8)                 # 中间特征 MSE
    L_INT = L_rec_int + λ * L_feat_int               # λ=5

    L_FP = ||out_4bit - out_fp||_2^2 + λ * MSE(f_4bit, f_fp)

    α(t) = min(1.0, t / T_warmup)                    # 线性增长权重
    L = (L_INT + α(t) * L_FP) / (1.0 + α(t))        # 加权组合

    L.backward()  # STE 梯度穿过量化
    optimizer.step()
```

关键超参数：λ=5（特征匹配权重），T_warmup 控制从 INT8 到 FP 监督的过渡速度，batch_size=2/GPU（因多教师增加显存）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

PMTD 在 PyTorch 中实现，利用 fake quantization + STE 进行端到端优化。多教师的前向传播可以批量并行处理（教师模型不需要梯度），仅学生模型的量化参数（clipping bounds）通过 STE 更新。对于 2-bit 量化，PMTD 会引入额外的中间教师（如 4-bit），形成 2bit←4bit←8bit←FP 的三级层次。代码开源：https://github.com/xiaoBIGfeng/PMQ-VE。

涉及论文标题：
- PMQ-VE Progressive Multi-Frame Quantization for Video Enhancement

---
