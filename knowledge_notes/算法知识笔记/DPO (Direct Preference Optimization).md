## DPO (Direct Preference Optimization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Direct Preference Optimization (DPO) 是 Rafailov et al. (2024, NeurIPS) 提出的离线偏好对齐算法，被 Hunyuan-Large 用于 RLHF 阶段。DPO 直接从偏好数据中优化策略模型，无需显式训练 reward model，也无需在线采样：

$$\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}} \left[\log \sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

其中 y_w 是 chosen response，y_l 是 rejected response，π_ref 是参考模型（SFT 模型），β 控制偏离参考模型的程度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Hunyuan-Large 的 DPO 采用单阶段训练策略，结合离线与在线数据：

```
# Hunyuan-Large 的 DPO 训练流程
# 输入: SFT 模型 π_SFT, 偏好数据集 D_pref

for batch in training_data:
    # 1. 离线数据: 预编译的偏好对
    x, y_w, y_l = batch["offline"]
    loss_offline = DPO_loss(π_θ, π_SFT, x, y_w, y_l, β)
    
    # 2. 在线数据: 当前策略模型生成多个 response
    y_candidates = [π_θ.generate(x_i) for x_i in batch["online_prompts"]]
    y_w_online, y_l_online = reward_model.select_best_worst(y_candidates)
    loss_online = DPO_loss(π_θ, π_SFT, x, y_w_online, y_l_online, β)
    
    # 3. 添加 SFT loss 项 (stabilization)
    loss_sft = -log π_θ(y_w | x)  # 防止 chosen prob 下降
    
    # 4. Total loss
    loss = loss_offline + loss_online + λ * loss_sft
    
    # 5. EMA (exponential moving average) 防 reward hacking
    π_ema = EMA(π_θ, decay=0.999)
```

Hunyuan-Large 还使用 EMA（指数移动平均）策略减少 reward hacking 和 alignment tax。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

DPO 实现：(1) 需要 reference model（通常为 SFT 模型，frozen）；(2) 每步计算两个 forward pass（π_θ 在 chosen 和 rejected 上的 log prob）+ 一个 reference forward；(3) β 典型值 0.1-0.5。Hunyuan-Large 结合离线（预编译 preference data）和在线（当前策略生成+reward model 评分）数据。常见实现库：TRL (`DPOTrainer`)、HuggingFace TRL。DPO 相比 RLHF (PPO) 的优势：不需要 reward model 训练、不需要在线采样（纯离线）、训练更稳定。

涉及论文标题：
- Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent
