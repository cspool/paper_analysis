## Gradient Penalty (1-Lipschitz Regularization for Neural Networks)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Gradient Penalty（梯度惩罚）是一种正则化技术，最早由 Gulrajani et al. (2017) 在 WGAN-GP 中提出。其核心思想是在损失函数中加入惩罚项 $\rho \cdot \mathbb{E}[(||\nabla_x f(x)||_2 - 1)^2]$，强制神经网络的梯度范数接近 1，从而实现 1-Lipschitz 连续性：$||f(x_1) - f(x_2)|| \le ||x_1 - x_2||$。在 WGAN 中，这是为了满足 Kantorovich-Rubinstein 对偶要求；在本论文中，用途完全不同——为下游 MPC 优化器提供平滑的优化景观。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

在本论文中，gradient penalty 被加入 critic 训练损失（Eq. 16）：
$$\hat{J}_c(\psi) = J_c(\psi) + \rho \cdot \mathbb{E}_{\mathcal{D}}[(1 - ||\nabla L_\psi(s,a,g)||_2)^2]$$

其中 $\nabla L_\psi$ 是 critic 输出对输入的梯度（通过 autograd 计算），$\rho \ge 0$ 控制惩罚强度。这使得 critic 在训练数据分布上近似 1-Lipschitz，从而其梯度 bounded：$||\nabla L_\psi||_2 \approx 1$。当这个 critic 被用作 MPC 的 cost function 时，SQP-RTI solver 在优化非凸 NN-based cost 时更稳定。

伪代码（Critic 训练时施加 Gradient Penalty）：
```
for batch (s, a, s') in replay_buffer:
    s.requires_grad = True
    L = critic(s, a, g)
    grad_L = autograd(L, s, create_graph=True)  # ∇_s L
    gp = ((grad_L.norm(2) - 1) ** 2).mean()     # gradient penalty
    
    L_target = cost + γ * target_critic(s', π(s'), g)
    critic_loss = F.mse(L, L_target) + ρ * gp
    critic_loss.backward()
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- PyTorch 实现：`torch.autograd.grad(outputs, inputs, create_graph=True)` 计算梯度，然后施加二次惩罚。
- 在 GAN 训练中（WGAN-GP），gradient penalty 作用于 discriminator 对插值样本的梯度。
- 在本论文的创新用途中：gradient penalty 使 critic 成为"MPC 友好的" cost function，解决了 AC4MPC (Reiter et al. 2024) 中观察到的"因 NN critic 高度非线性导致优化困难"问题。
- 与 Spectral Normalization 的关系：两者都实现 Lipschitz 约束，但 gradient penalty 是软约束（通过 loss），Spectral Normalization 是硬约束（通过权重归一化）。

涉及论文标题：
- Autonomous Wheel Loader Navigation Using Goal-Conditioned Actor-Critic MPC
