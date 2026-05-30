## Static Priority-Based Rectifier Routing (SPR²)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Static Priority-Based Rectifier Routing (SPR²) 是 SPR²Q 提出的静态 rectifier 路由机制。与 MoE 动态路由（推理时根据输入选择不同 expert，引入额外计算）不同，SPR² 将路由决策完全离线化：训练阶段用动态门控鼓励 rectifier group 专业化学习多种补偿策略，校准阶段通过梯度下降学习最优静态门控权重并预计算增量存入 SPR²Q Table，推理时直接查表融合——无动态门控、无额外 FLOPs。三阶段：(1) RGT（Rectifier Group Training）：动态门控 g_i 加权聚合 N 个 rectifier，12K iterations 训练；(2) OSRC（Offline Static Routing Calibration）：冻结参数后梯度下降 500 iterations 学习 ĉ = argmin_g L，构建 SPR²Q Table；(3) 推理：W_final = W + Table[l]，直接量化推理。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 阶段 1: RGT (12K iters)
N = 4  # rectifier group size
rectifiers = [(A_i, B_i) for i in range(N)]  # rank=8
gate = GatingNetwork()

for iter in range(12000):
    g = softmax(gate(X))  # 动态路由权重
    delta_W_fused = sum(g[i] * (B_i @ A_i) for i in range(N))
    W_q = Quantize(W + delta_W_fused)
    loss = L_pixel + λ * L_feature
    # STE 更新所有参数

# 阶段 2: OSRC (500 iters)
for each module l:
    g_hat = argmin_g L(W + Σ g_i * ΔW_i)  # Eq.12, 冻结 ΔW_i
    SPR2Q_Table[l] = Σ g_hat[i] * (B_i @ A_i)  # 预计算最优增量

# 阶段 3: 推理
for each module l:
    W_final = W + SPR2Q_Table[l]  # 离线融合
    W_q = Quantize(W_final)
    Y = X @ W_q  # 与原始模型计算图一致
```
消融：PQFR +0.24, +RGT +0.16, +OSRC +0.12 dB (Set5)。关键设计：N=4 够用（N=8 仅 +0.10 dB 边际增益），r=8 为 rank 饱和点。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
基于 PaddlePaddle + RTX 4090 实现。rectifier rank r=8, group size N=4。训练 12K+500 iterations (batch=8)。推理阶段：所有 rectifier 参数离线融合，模型结构与原始 MambaIRv2-light 完全一致，推理时无需任何额外的 gate 计算或动态路由分支判断。

涉及论文标题：
- SPR²Q: Static Priority-based Rectifier Routing Quantization for Image Super-Resolution

---
