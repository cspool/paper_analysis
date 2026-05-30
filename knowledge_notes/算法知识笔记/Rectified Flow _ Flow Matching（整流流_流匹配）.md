## Rectified Flow / Flow Matching（整流流/流匹配）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Rectified Flow (Flow Matching) 是一种连续时间生成模型框架，通过学习一个 ODE 的速度场 v(x,t) 来在噪声分布和数据分布之间建立"直线"概率路径。与 DDPM/DDIM 的离散时间马尔可夫链不同，rectified flow 将生成过程建模为 dx/dt = v(x,t)，其中 x(0) 是纯噪声，x(T) 是数据。训练目标是学习神经网络预测速度场 v_theta(x_t, t) 匹配从噪声到数据的直线插值路径：x_t = (1-t) x x_0 + t x epsilon。损失函数为 L = E[||v_theta(x_t, t) - (epsilon - x_0)||^2]。相比 DDPM，rectified flow 允许更少的采样步数（路径更"直"）且训练更稳定。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 训练 rectified flow
def rectified_flow_training(video_data, text_condition):
    x_0 = vae_encode(video_data)
    epsilon = torch.randn_like(x_0)
    t = torch.rand(batch_size)
    x_t = (1 - t) * x_0 + t * epsilon  # 直线插值路径
    v_target = epsilon - x_0  # 目标速度
    v_pred = dit_model(x_t, t, text_condition)
    loss = F.mse_loss(v_pred, v_target)
    return loss

# 采样: Euler 积分 ODE
def rectified_flow_sampling(text_condition, num_steps=50):
    z = torch.randn(latent_shape)
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = step * dt
        v_pred = dit_model(z, t, text_condition)
        z = z + v_pred * dt  # Euler step
    return vae_decode(z)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Rectified flow 在视频生成中的关键点：(1) SD3 率先在大规模文生图中使用；(2) EasyAnimate 发现 rectified flow 下的梯度 norm 远小于 DDPM，因此 reward backpropagation 需 K=10（而非 DDPM 的 K=1）以保证训练稳定性；(3) 采样时可用 classifier-free guidance 和更高阶 ODE solver 加速。EasyAnimate 初始实验显示 rectified flow 效果优于 DDPM。

涉及论文标题：
- EasyAnimate__A_High-Performance_Long_Video_Generation_Method_based_on_Transformer_Architecture
