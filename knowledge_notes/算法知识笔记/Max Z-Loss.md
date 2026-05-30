## Max Z-Loss

术语解释
Max Z-Loss 是 MoE 训练中用于辅助稳定的正则化损失项，用于抑制 Router logits 的过大幅度，防止因 router 输出的 softmax 分数过于极端导致的训练不稳定。首次由 Zoph et al. (2022) 在 ST-MoE 中提出。

术语是什么？
Z-loss 惩罚 router logits 的最大值：
$$\text{max_z_loss} = \lambda_z \cdot \max_j(\text{router_logits})^2$$

router logits 过大 → softmax 后概率分布过于集中 → 负载严重不均 + 梯度不稳定。Max z-loss 与 load balancing loss 配合使用：load balancing loss 促进均匀分配，z-loss 防止 logit 过大。

在 AquilaMoE 的训练中，load balancing loss 乘以 0.001，max z-loss 乘以 0.01，共同施加于 LM loss 上：
$$L_{total} = L_{LM} + 0.001 \cdot L_{aux} + 0.01 \cdot L_{z-loss}$$

从算法pipeline角度拆解术语：
```
# MoE Router forward with Z-Loss
router_logits = x @ W_router  # [batch, num_experts]
router_probs = Softmax(router_logits)

# Z-Loss: 惩罚 logits 最大值
z_loss = max(router_logits, dim=-1)^2  # max over experts
L_z = 0.01 * z_loss.mean()

# 总 loss
L = L_lm + 0.001 * L_aux + L_z
```

术语一般如何实现？如何使用？
- ST-MoE (Zoph et al., 2022) 首次提出，称为 "router z-loss"
- PaLM (Chowdhery et al., 2023) 中也使用
- 通常与 load balancing loss 配合使用
- 系数需谨慎调优：AquilaMoE 使用 0.01，过大会过度约束 router 学习
- 是防止大规模 MoE 训练崩溃的关键技术之一

涉及论文标题：
- AquilaMoE Efficient Training for MoE Models with Scale-Up and Scale-Out Strategies
- Aria An Open Multimodal Native Mixture-of-Experts Model（ARIA 在 fine-grained MoE 训练中使用 z-loss 与 group-level load balancing loss 配合稳定训练）
- Continual Pre-training of MoEs How robust is your router（PBTk CPT 实验使用 z-loss coeff=0.001 与 aux loss coeff=0.01 组合，均与 DeepSeekMoE 和 ST-MoE 保持一致。Z-loss 在 CPT 分布偏移期间继续发挥作用，防止 router logits 因新分布而爆炸）
- CuMo: Scaling Multimodal LLM with Co-Upcycled Mixture-of-Experts（CuMo 使用 router z-loss α_z=0.01 与 load balancing loss α_b=0.1 组合，分别应用于 MLP connector、CLIP vision encoder 和 LLM 的每个 MoE 块，称为 "bzloss"）

---
