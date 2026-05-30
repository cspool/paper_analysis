## Load Balancing Loss for MoE (MoE 负载均衡损失)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Load Balancing Loss 是 MoE 训练中用于防止 Gate 网络收敛到仅激活少数几个 expert（expert collapse / routing collapse）的辅助损失函数。没有负载均衡损失时，Gate 网络会形成 self-reinforcing 循环：被频繁选中的 expert 梯度更新更多 → 更早学会 → Gate 更倾向选它 → 负载进一步集中。Shazeer et al. (2017) 提出两级负载均衡损失：L_importance 和 L_load，分别控制 expert 的重要性分布和负载分布。两者作为辅助损失加入总 loss：Total Loss = CrossEntropy + L_importance + L_load。

从算法pipeline角度拆解术语：
两级负载均衡损失的计算过程（Shazeer et al. 2017）：

```
# 给定 batch X，n 个 expert，gate 输出 G(x) [1, n] (稀疏)

# === Level 1: Importance Loss ===
# Importance(X)[i] = Σ_{x∈X} G(x)_i  # expert i 的 batch-wise gate sum
Importance = sum(G(x) for x in X)    # [n]

# CV (Coefficient of Variation) = σ / μ
CV_importance = std(Importance) / mean(Importance)

# Importance Loss: 鼓励所有 expert 的重要性相等
L_importance = w_importance * CV_importance^2

# === Level 2: Load Loss ===
# 问题: Importance 均衡不代表每个 expert 接收的样本数均衡
# (一个 expert 可能收少量大权重样本，另一个收大量小权重样本)

# Smooth Load Estimator (利用 noise 的可微性):
# P(x,i) = 概率(G(x)_i > 0 | 重新采样 expert i 的 noise, 保持其他 noise 固定)
P(x,i) = Φ((clean_logits_i - kth_excluding(H,k,i)) / Softplus(noise_std_i))
# Φ = 标准正态 CDF

# Load(X)[i] = Σ_{x∈X} P(x,i)
Load = sum(P(x, i) for x in X)  # [n]

CV_load = std(Load) / mean(Load)
L_load = w_load * CV_load^2

# === 组合 ===
Total_Loss = CrossEntropy + L_importance + L_load
```

关键设计要点：
- **Noise 的双重作用**：(1) 训练中提供探索随机性，防止过早收敛；(2) 利用 noise 分布构造平滑可微的 Load(X) 估计器，使负载均衡可反向传播。
- **初始化**：W_g 和 W_noise 初始化为全零 → 初始状态每个 expert 被均匀选中（仅有 noise 驱动选择），避免训练初期的 OOM。
- **权重调参**：w=0.1/0.1 可达到良好平衡（Test PPL 35.6 vs 无 loss 的 39.8），过大值(1.0/1.0) 不进一步改善质量但可进一步降低最大 expert 负载（max/mean ratio 1.07 vs 1.47）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 后续变体：(1) Switch Transformer — L_aux = E·Σ(f_e·P_e)，其中 f_e = fraction of tokens routed to expert e, P_e = mean routing probability；(2) GShard — auxiliary loss = α·Σ(f_e - 1/E)²；(3) Z-loss (ST-MoE) — 加在 logits 上的正则化项。
- 替代/补充方案：Capacity Factor (CF) — 硬限制每个 expert 最多处理 CF×(total_tokens/n) 个 token，超出部分丢弃并由 residual connection 绕过。CF 与辅助 loss 常联合使用。
- 工程要点：importance loss 和 load loss 在各 GPU 上的梯度同步方式需适配并行策略——在 Expert Parallelism 下每个 GPU 只计算本地 expert 的 loss 分量。

涉及论文标题：
- Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer
