## Flow Matching（流匹配）

术语是什么？通过联网搜索让回答具体和精准。
Flow Matching（流匹配）是由Lipman et al. (2023)提出的一种生成模型训练范式，作为Classical Diffusion (DDPM/DDIM)的替代方案。它在latent space中学习一个continuous velocity field v_Θ(z_t, t, y)，将噪声平滑地"流动"到数据分布。给定target latent z_0和噪声样本ε ∼ N(0,I)，通过线性插值 z_t = (1-t)z_0 + tε（t∈[0,1]），监督目标为velocity (ε-z_0)而非噪声ε：L_fm = ||(ε-z_0) - v_Θ(z_t,t,y)||²₂。与DDPM的离散马尔可夫链不同，Flow Matching使用连续时间ODE，消除了对固定noise schedule的依赖。代表性模型：Flux（Black Forest Labs, 2024）、Stable Diffusion 3（Esser et al., 2024）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Flow Matching训练pipeline（以Flux在latent space上的训练为例）：

```
// 训练阶段
1: x = real_image ∈ R^{H×W×3}
2: z_0 = VAE.encode(x) ∈ R^{C×h×w}          // 压缩到latent
3: ε ~ N(0, I)                                // 采样高斯噪声
4: t ~ Uniform(0, 1)                          // 连续时间采样
5: z_t = (1-t) * z_0 + t * ε                  // 线性插值
6: v_pred = v_Θ(z_t, t, text_conditioning)    // 预测velocity field
7: target = ε - z_0                           // ground truth velocity（从噪声指向数据）
8: loss = MSE(v_pred, target)                  // 回归velocity

// 推理阶段（从纯噪声生成）
1: z_1 ~ N(0, I)                              // 纯噪声（t=1）
2: for t = 1, 1-dt, ..., dt:                  // ODE求解
3:     z_{t-dt} = z_t - v_Θ(z_t, t, y) * dt  // Euler步或更高级ODE solver
4: image = VAE.decode(z_0)                    // latent→pixel
```

与Classical Diffusion的关键区别：
- DDPM学习预测噪声ε，Flow Matching学习预测velocity (ε-z_0)
- DDPM使用离散timestep t∈{1,...,T} + noise schedule β_t，Flow Matching使用连续t∈[0,1]
- Flow Matching的线性插值路径使得训练更稳定，采样可用更少的ODE步数

**x-prediction vs v-prediction parameterization（ELF论文的关键选择）：**
标准Flow Matching预测velocity v = x - ϵ，但也可reparameterize预测x（clean data）或ϵ（noise）。ELF选择x-prediction的原因：
- x-prediction在高维space（512/768/1024-dim）中保持稳定，v-prediction在高维退化，ϵ-prediction collapse
- x-prediction预测clean embeddings，与final-step token decoding（CE loss）目标一致，使shared-weight denoiser-decoder可行
- 训练loss等价转换：L_MSE = ||v_θ - v||² = ||(x_θ - x)/(1-t)||²（通过v = (x - z_t)/(1-t)关系转换）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Flow Matching已被主流扩散模型采用：Flux-1.dev、SD3、URAE等。LWD论文在Flow Matching基础上引入wavelet-based spatial masking：对latent z_t计算wavelet energy saliency map Awavelet，生成time-dependent mask M_t，最终loss变为 L_masked = ||M_t ⊙ [(ε-z_0) - v_Θ(z_t,t,y)]||²₂。训练后inference与原始Flow Matching完全相同。

ELF论文将Flow Matching扩展到语言建模领域（文本生成为"text-to-text" generation），使用x-prediction parameterization（预测clean embeddings x而非velocity v）替代标准v-prediction。x-prediction使shared-weight denoiser-decoder成为可能（denoising和decoding均predict clean embeddings），在高维embedding space（512/768/1024-dim）中比v-prediction/ϵ-prediction更稳定。ELF使用rectified flow linear interpolant: z_t = t·x + (1-t)·ϵ，训练目标 L_MSE = ||(x_θ(z_t,t) - x)/(1-t)||²。此外ELF还支持SDE-inspired sampler：在每个ODE step注入Gaussian noise（z_back = α·z + (1-α)·ε, α = 1-γ·dt），在perturbed state上重预测x̂，用原z更新。γ=0退化为ODE，γ>0引入stochasticity以纠正early denoising errors。

实现上，Flow Matching使用standard PyTorch/JAX training loop，与logit-normal time schedule（P_mean=-1.5, P_std=0.8）配合使用。in-context conditioning（prepend time/CFG/mode tokens）替代adaLN-Zero可减少参数量。

涉及论文标题：
- Latent Wavelet Diffusion for Ultra-High-Resolution Image Synthesis
- ELF: Embedded Language Flows

---

