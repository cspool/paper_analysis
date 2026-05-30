## DPO (Direct Preference Optimization，直接偏好优化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Direct Preference Optimization (DPO) 是 Rafailov et al. (NeurIPS 2023) 提出的一种 LLM 偏好对齐方法，直接在偏好数据集上优化策略模型，无需显式训练奖励模型。DPO 的核心思想：(1) 将 RLHF 的奖励函数 r(x,y) 重新参数化为策略模型 π 和参考模型 π_ref 的对数概率比：r(x,y) = β log(π(y|x)/π_ref(y|x))；(2) 将 Bradley-Terry 偏好模型 P(y_w ≻ y_l|x) = σ(r(x,y_w) - r(x,y_l)) 与重新参数化的奖励函数结合，直接推导出 DPO 损失函数：L_DPO = -E_{(x,y_w,y_l)} log σ(β log(π(y_w|x)/π_ref(y_w|x)) - β log(π(y_l|x)/π_ref(y_l|x)))。其中 β 控制偏离参考模型的程度（β 越小，策略越接近参考模型）。与 RLHF 相比，DPO 省去了 reward model 训练和 PPO 强化学习两个阶段，直接对偏好数据进行监督式优化，更稳定、更高效。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 Q-resafe 中的 DPO 用法（安全修补）：
```python
# 参考模型 = 量化后的 LLM π_Q⁰（安全能力受损）
# 策略模型 = 安全修补中的 LLM π_Q（正在被优化恢复安全）
# Q = Q⁰ + Quant(M_Q ⊙ AB)，仅安全关键权重参与更新

# DPO 损失计算
for batch in D_patch:  # D_patch = {(x, y_w, y_l)}
    # y_w: 全精度 LLM 生成的 preferred response（安全）
    # y_l: 量化 LLM 生成的 dispreferred response（不安全）

    log_pi_w = π_Q.log_prob(y_w | x)      # 策略模型对 preferred 的对数概率
    log_pi_l = π_Q.log_prob(y_l | x)      # 策略模型对 dispreferred 的对数概率
    log_ref_w = π_Q⁰.log_prob(y_w | x)    # 参考模型对 preferred 的对数概率
    log_ref_l = π_Q⁰.log_prob(y_l | x)    # 参考模型对 dispreferred 的对数概率

    # DPO 损失（Eq. 1 of Q-resafe）
    ratio_w = log_pi_w - log_ref_w        # preferred 的相对提升
    ratio_l = log_pi_l - log_ref_l        # dispreferred 的相对下降
    loss = -log(σ(β * (ratio_w - ratio_l)))

    loss.backward()
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：HuggingFace TRL 库的 DPOTrainer（https://github.com/huggingface/trl）。标准用法：准备偏好数据集（prompt, chosen, rejected），指定 reference model（通常为 SFT 模型本身或冻结副本），设置 β（默认 0.1）。TRL 中：`DPOTrainer(model=model, ref_model=ref_model, beta=0.1, train_dataset=dataset)`。Q-resafe 的特殊用法：参考模型设为量化后的不安全模型 π_Q⁰，让 DPO 的隐式正则化 '不要偏离 π_Q⁰ 太远' 变为 '不要偏离量化模型已保留的效用太远'。DPO β 在 Q-resafe 中设为 0.01（较小值，约束更紧）。

涉及论文标题：
- Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models
