## Flow Matching

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Flow Matching（流匹配）是一种生成建模范式，替代传统的DDPM（Denoising Diffusion Probabilistic Models）用于训练扩散模型。与DDPM学习从噪声中预测原始数据或噪声不同，Flow Matching学习一个连续的velocity field（速度场）v(x, t)，该速度场定义了从简单分布（如标准高斯噪声）到数据分布的连续归一化流（Continuous Normalizing Flow, CNF）。训练时沿noise→data的线性/最优传输路径采样中间状态z_t = (1-t)·noise + t·data，模型学习预测速度v(z_t, t) = data - noise。推理时从噪声x_0 ~ N(0,I)开始，用ODE solver（如Euler method）沿学习的速度场逐步积分到达数据点：x_{t+Δt} = x_t + v(x_t, t)·Δt。Flow Matching的核心优势：(1) 比DDPM更简单的训练目标（直接预测速度而非噪声/原始数据，虽然数学上等价）；(2) 更灵活的前向过程（可用最优传输路径而非固定高斯扩散过程）；(3) 结合rectified flow可在少量ODE步骤（如<10步）内实现高质量采样。

从算法pipeline角度拆解，Flow Matching在Video DiT中的使用：
```
# Flow Matching training (DSV paper)
# 给定: latent video z ~ p_data, noise ε ~ N(0,I)

# Forward: 定义线性概率路径
t ~ Uniform(0, 1)
z_t = (1-t) * z + t * ε              # 最优传输路径
target_velocity = ε - z               # 速度场目标

# DiT预测速度
v_pred = DiT(z_t, t, text_emb)        # 模型输出=预测速度场

# Loss
L = MSE(v_pred, target_velocity)      # 简单MSE，无需noise schedule
```

Flow matching loss直接反映模型能力和训练进展（DSV论文和MovieGen论文均使用此特性评估收敛）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Flow Matching论文（Lipman et al., 2023, ICLR 2023; Tong et al., 2023, NeurIPS 2023）提供了理论基础。主流实现：Stable Diffusion 3 (Esser et al., 2024)使用rectified flow matching实现4-8步高质量采样；FLUX.1系列模型基于flow matching；Meta MovieGen (Polyak et al., 2024)使用flow matching训练视频DiT；SD3.5系列。PyTorch实现：扩散模型中简单的训练范式切换——将DDPM的noise prediction loss替换为velocity prediction loss，无需架构修改。推理时需ODE solver（如Euler, RK4, DPM-Solver）替代DDPM的DDIM采样。

涉及论文标题：
- DSV: Exploiting Dynamic Sparsity to Accelerate Large-Scale Video DiT Training
