# DDPM：NIPS20

## NIPS20：DDPM

NIPS20：Denoising Diffusion Probabilistic Models

DDPM扩散过程$q(x_T : x_1 | x_0)= Πq(x_t | x_{t-1})$ 定义了从$q(x_0)$采样生成$q(x_T)=N(0,1)$的映射方式。

相同扩散过程$q(x_T : x_1 | x_0)= q(x_T | x_0) \cdot Πq(x_{t-1} | x_{t}, x_0)$提供了从$q(x_T | x_0)=N(0,1)$采样生成$q(x_1 | x_0)$的方式，因此根据变分推断模型，定义$\{p_θ(x_{t-1} | x_t)=q(x_{t-1} | x_{t}, f_θ) | t=2...T \}$学习$\{q(x_{t-1} | x_{t}, x_θ) | t=2...T \}$，参数化并最大化$p_θ(x_0|x_1)$。

### Diffusion的动机

**VAE的概率模型是基于$p(Z)$的采样Z，按照$p(X | Z)$采样生成样本X**，隐变量Z的后验分布**$p(Z | X)$**将样本X分布$p(X)$映射到分布$p(Z)$。

VAE-Decoder中单步Guaasian采样会平均化图像的高频成分，导致图像可能**模糊**。

> **[图片提取文字 (image.png)]:**
> ## Before Diffusion, we had two main choices for generative AI, both with fatal flaws:
> 
> VAEs (Variational Autoencoders):
> 
> 1. Motivation: Why was Diffusion invented?
> 
> GANs (Generative Adversarial Networks):
> 
> Cons: Blurry images. The "One-Shot" decoder tries to approximate a complex reality
> with a single Gaussian, forcing it to average out details.
> 
> Pros: Mathematically grounded (Likelihood-based), stable training, good diversity.
> 
> - Pros: Sharp, realistic images.
> - Cons: Unstable training (Mode Collapse). It's a minimax game that is hard to balance.
> 
> based) but achieve the sharpness of GANs?"
> 
> The Motivation for Diffusion: "Can we keep the mathematical stability of VAEs (Likelihood-
> 
> The Solution: Instead of decoding in one giant jump (which causes blur), let's decode in 1,000 tiny steps. If the steps are small enough, the math becomes easy (Gaussian), and the details remain sharp.
![image.png](DDPM%EF%BC%9ANIPS20/image.png)

> **[图片提取文字 (image.png)]:**
> $\text{Loss} = \underbrace{-\log p_{\theta}(x|z)}_{\text{Reconstruction}} + \underbrace{KL(q(z|x)||p(z))}_{\text{Prior Match}}$ 
> 
> In standard VI (VAE), we maximize the ELBO (Evidence Lower Bound). Diffusion maximizes the
> 
> ## The Diffusion ELBO (Hierarchical)
> 
> The VAE ELBO
> 
> exact same bound, just split into many parts.
> 
> Since we have latent variables  $x_1, x_2, \ldots, x_T$ , the ELBO expands into a sum of transitions:
> 
> $$\text{Loss} = \sum_{t=1}^{T} \underbrace{KL(q(x_{t-1}|x_t, x_0)||p_{\theta}(x_{t-1}|x_t))}_{\text{Denoising Match at step } t} + \underbrace{KL(q(x_T|x_0)||p(x_T))}_{\text{Prior Match (Constant)}}$$
> 
> - Key Differences from Standard VI:
>   - 1. Fixed Encoder: In VAE, the encoder q(z | x) is a neural network we train. In Diffusion, the
> 
> Gaussian. This makes the math exact rather than an approximation.
> 
> - encoder  $q(x_t|x_{t-1})$  is a **fixed linear noise schedule**. We don't train it! 2. **Shared Decoder:** In VAE, the decoder is one network. In Diffusion, the "decoder"
> - transitions  $p_{\theta}(x_{t-1}|x_t)$  share the same U-Net weights for all time steps t.
> 
>   3. **Step Size:** Because the steps  $t \to t-1$  are tiny, the true posterior is extremely close to a
![image.png](DDPM%EF%BC%9ANIPS20/image%201.png)

VAE解码为了让ELBO中的重建项更大，会平均高频特征，但导致**blurry**。

比如**全局特征相似的两个局部pattern**会编码到同一个z，但单步高斯p(X|Z)在同一个z条件的均值下很难匹配不同pattern，因此解码时会平均化pattern后生成图像。

Diffusion将**编码采样映射设计为覆盖语义足够的扩散过程**（隐变量序列x1…xT），因此只需预测在解码映射的生成过程。VAE需要同时优化编码和解码的采样分布，要求编码采样能覆盖更多语义且保持在p(Z)范围，解码采样能正确还原样本。

> **[图片提取文字 (image.png)]:**
> ## To understand why VAEs blur, you must understand what happens when you average two high-
> 
> - Low Frequency (Structure): A face is "round." If you average a round face and a slightly different round face, you get a round face. (Constructive Interference).
> - High Frequency (Texture): Imagine a checkerboard pattern (black/white pixels).
>   - Pattern B: White-Black-White (Shifted by 1 pixel)
> 
> Pattern A: Black-White-Black
> 
> 1. The Core Physics: Why "Averaging" Kills High Frequencies
> 
> frequency patterns.
> 
> The Average: Gray-Gray-Gray. (Destructive Interference).
> 
> ## The VAE Problem: Since the VAE Decoder predicts the Mean of the distribution p(x|z), and the distribution of possible textures is multimodal (phase-shifted), the "Mean" of all possible textures is Gray.
![image.png](DDPM%EF%BC%9ANIPS20/image%202.png)

> **[图片提取文字 (image.png)]:**
> ## 3. Diffusion: The "Frequency Cascade" Diffusion Models solve this by separating frequencies across Time(t).
> 
> timesteps to specific frequency bands.
> 
> Phase A: High Noise ( $t=1000 \rightarrow 500$ ) = Low Frequency Generation
> 
> Instead of trying to generate all frequencies at once (like VAE), Diffusion dedicates specific
> 
> When the image is pure noise, the model cannot see fine details. It can only see coarse patterns.
> 
> - Task: "Create the blob of a head."
> - Frequency: Only Low Frequencies are generated here.
> - The model establishes the global structure.
> 
> Phase B: Low Noise ( $t=100 \to 0$ ) = High Frequency Generation This is the critical difference. At the end of the process, the "Structure" is already fixed (the
> 
> - face exists).
> - Task: "The face is there, but it's grainy. Polish the edges."
> - Frequency: The model focuses exclusively on High Frequencies. Why no blur? Because the model does not have to guess the structure anymore. It is
> 
> conditioned on the established structure  $x_t$ . It just needs to fill in the texture.
![image.png](DDPM%EF%BC%9ANIPS20/e7215f56-9562-4cd6-8a99-1cbb0a48790e.png)

*变分推断模型下的Diffusion*

$q(x_T : x_1 | x_0)= Πq(x_t | x_{t-1})$ 作为**编码器**，**编码**和理解图像$x_0$的分布$p(x_0)$。

$x_{0}→x_1→x_T$：**Markov扩散过程**$q(x_T : x_1 | x_0)= Πq(x_t | x_{t-1})$定义分布$p(x_0)$逐步采样（$x_{t-1}→x_t$）生成$x_1…x_T$序列的方式。

$q(x_0|x_1:x_T)$作为**解码器不可解**，**定义$p_θ(x_0:x_T)$来近似$q(x_0|x_1:x_T)$**。

$x_{T}→x_1  \color {grey}{→x_{0}}$：**相同扩散过程**$q(x_T : x_1 | x_0)= q(x_T | x_0) \cdot Πq(x_{t-1} | x_{t}, x_0)$定义分布$q(x_T | x_0)$ 逐步采样（$x_{t}→x_{t-1}$）生成序列$x_{T}…x_1$的方式，即$x_t$处采样时，**沿着$x_0$的编码路径逆向采样**。

$x_{T}→x_{1} \color {grey} →f_θ$：定义$\{p_θ(x_{t-1} | x_t)=q(x_{t-1} | x_{t}, f_θ) | t=2...T \}$，分布$p(x_T)$ 逐步采样（$x_{t}→x_{t-1}$）生成序列$x_{T}…x_1$的方式，即$x_t$处采样时，**沿着$x_0$预测值$f_θ(x_t)$的编码路径逆向采样**。

$x_{1}→f_θ$：定义$p_θ(x_0|x_1)$是$x_{1}$处采样得到$x_{0}$的**高斯采样**，$p_θ(x_0|x_1)$最大，是在修正$x_t$处的对$x_0$的预测值$f_θ(x_t)$。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 2: The directed graphical model considered in this work.
> 
> Gaussians too, allowing for a particularly simple neural network parameterization.
> 
> (which we will call a "diffusion model" for brevity) is a parameterized Markov chain trained using variational inference to produce samples matching the data after finite time. Transitions of this chain are learned to reverse a diffusion process, which is a Markov chain that gradually adds noise to the data in the opposite direction of sampling until signal is destroyed. When the diffusion consists of
> 
> small amounts of Gaussian noise, it is sufficient to set the sampling chain transitions to conditional
> 
> This paper presents progress in diffusion probabilistic models [53]. A diffusion probabilistic model
![image.png](DDPM%EF%BC%9ANIPS20/image%203.png)

$x_0$能逐步**扩散到标准高斯分布z**，是马尔可夫扩散过程$q(x_T : x_1 | x_0)$中静态参数$α_t$设置的结果。

> **[图片提取文字 (image.png)]:**
> $\frac{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0) := \prod_{t=1}^{T} q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})}{q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})}, \text{ where } q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1}) := \mathcal{N}\left(\sqrt{\frac{\alpha_t}{\alpha_{t-1}}}\boldsymbol{x}_{t-1}, \left(1 - \frac{\alpha_t}{\alpha_{t-1}}\right)\boldsymbol{I}\right) \quad (3)$ 
> 
> dimensional. For example, Ho et al. (2020) considered the following Markov chain with Gaussian
> 
> where the covariance matrix is ensured to have positive terms on its diagonal. This is called the
> 
> forward process due to the autoregressive nature of the sampling procedure (from  $x_0$  to  $x_T$ ). We
> 
> call the latent variable model  $p_{\theta}(\boldsymbol{x}_{0:T})$ , which is a Markov chain that samples from  $\boldsymbol{x}_T$  to  $\boldsymbol{x}_0$ , the generative process, since it approximates the intractable reverse process  $q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)$ . Intuitively, the forward process progressively adds noise to the observation  $\boldsymbol{x}_0$ , whereas the generative process progressively denoises a noisy observation (Figure 1, left).
> 
> A special property of the forward process is that  $q(\boldsymbol{x}_t|\boldsymbol{x}_0) := \int q(\boldsymbol{x}_{1:t}|\boldsymbol{x}_0) \mathrm{d}\boldsymbol{x}_{1:(t-1)} = \mathcal{N}(\boldsymbol{x}_t; \sqrt{\alpha_t}\boldsymbol{x}_0, (1-\alpha_t)\boldsymbol{I});$ 
> 
> transitions parameterized by a decreasing sequence  $\alpha_{1:T} \in (0,1]^T$ :
![image.png](DDPM%EF%BC%9ANIPS20/image%204.png)

> **[图片提取文字 (image.png)]:**
> $$x_t = \sqrt{\alpha_t} x_0 + \sqrt{1 - \alpha_t} \epsilon$$
> , where  $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ . (4)  
> When we set  $\alpha_T$  sufficiently close to  $0$ ,  $q(x_T | x_0)$  converges to a standard Gaussian for all  $x_0$ , so it
> 
> so we can express  $x_t$  as a linear combination of  $x_0$  and a noise variable  $\epsilon$ :
> 
> is natural to set  $p_{\theta}(x_T) := \mathcal{N}(0, I)$ . If all the conditionals are modeled as Gaussians with trainable mean functions and fixed variances, the objective in Eq. (2) can be simplified to<sup>1</sup>:
> 
> (4)
> 
> $$L_{\gamma}(\epsilon_{\theta}) := \sum_{t=0}^{T} \gamma_{t} \mathbb{E}_{\boldsymbol{x}_{0} \sim q(\boldsymbol{x}_{0}), \epsilon_{t} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})} \left[ \left\| \epsilon_{\theta}^{(t)} (\sqrt{\alpha_{t}} \boldsymbol{x}_{0} + \sqrt{1 - \alpha_{t}} \epsilon_{t}) - \epsilon_{t} \right\|_{2}^{2} \right]$$
![image.png](DDPM%EF%BC%9ANIPS20/image%205.png)

### Diffusion的数学原理

*Diffusion的Loss符合变分推断的优化方向*

变分推断认为隐变量序列$x_1…x_T$是样本$x_0$的原因，**近似分布为了$logp(x_0)$最大**：

> 后验分布$q(x_1 : x_T | x_0)$和**近似先验**分布$p_θ(x_T : x_1)$相似，但是扩散过程$q(x_T : x_1 | x_0)$是静态定义的。因此**$p_θ(x_{t-1} | x_t)$拟合$q(x_{t-1} | x_t , x_0)$**，对应$L_{t-1}$项。

> **最大化样本$x_0$的似然分布$p_θ(x_0| x_{T} : x_1)=p_θ(x_0|x_1)$** ，对应$L_0$项。

> **[图片提取文字 (image.png)]:**
> Diffusion models [53] are latent variable models of the form  $p_{\theta}(\mathbf{x}_0) \coloneqq \int p_{\theta}(\mathbf{x}_{0:T}) d\mathbf{x}_{1:T}$ , where  $\mathbf{x}_1, \dots, \mathbf{x}_T$  are latents of the same dimensionality as the data  $\mathbf{x}_0 \sim q(\mathbf{x}_0)$ . The joint distribution  $p_{\theta}(\mathbf{x}_{0:T})$  is called the *reverse process*, and it is defined as a Markov chain with learned Gaussian transitions starting at  $p(\mathbf{x}_T) = \mathcal{N}(\mathbf{x}_T; \mathbf{0}, \mathbf{I})$ :
> 
> $$p_{\theta}(\mathbf{x}_{0:T}) \coloneqq p(\mathbf{x}_{T}) \prod_{t=1}^{T} p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t}), \qquad p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t}) \coloneqq \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_{\theta}(\mathbf{x}_{t}, t), \boldsymbol{\Sigma}_{\theta}(\mathbf{x}_{t}, t))$$
> (1)
> 
> What distinguishes diffusion models from other types of latent variable models is that the approximate posterior  $q(\mathbf{x}_{1:T}|\mathbf{x}_0)$ , called the *forward process* or *diffusion process*, is fixed to a Markov chain that gradually adds Gaussian noise to the data according to a variance schedule  $\beta_1, \ldots, \beta_T$ :
> 
> $$q(\mathbf{x}_{1:T}|\mathbf{x}_0) \coloneqq \prod_{t=1}^{T} q(\mathbf{x}_t|\mathbf{x}_{t-1}), \qquad q(\mathbf{x}_t|\mathbf{x}_{t-1}) \coloneqq \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t \mathbf{I})$$
> (2)
> 
> Training is performed by optimizing the usual variational bound on negative log likelihood:
> 
> $$\mathbb{E}\left[-\log p_{\theta}(\mathbf{x}_{0})\right] \leq \mathbb{E}_{q}\left[-\log \frac{p_{\theta}(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_{0})}\right] = \mathbb{E}_{q}\left[-\log p(\mathbf{x}_{T}) - \sum_{t>1} \log \frac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{q(\mathbf{x}_{t}|\mathbf{x}_{t-1})}\right] =: L (3)$$
> 
> The forward process variances  $\beta_t$  can be learned by reparameterization [33] or held constant as hyperparameters, and expressiveness of the reverse process is ensured in part by the choice of Gaussian conditionals in  $p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t)$ , because both processes have the same functional form when  $\beta_t$  are small [53]. A notable property of the forward process is that it admits sampling  $\mathbf{x}_t$  at an arbitrary timestep t in closed form: using the notation  $\alpha_t := 1 - \beta_t$  and  $\bar{\alpha}_t := \prod_{s=1}^t \alpha_s$ , we have  $q(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})$  (4)
![image.png](DDPM%EF%BC%9ANIPS20/image%206.png)

> **[图片提取文字 (image.png)]:**
> Efficient training is therefore possible by optimizing random terms of L with stochastic gradient descent. Further improvements come from variance reduction by rewriting L (3) as:  $\mathbb{E}_{q}\left[\underbrace{D_{\mathrm{KL}}(q(\mathbf{x}_{T}|\mathbf{x}_{0}) \parallel p(\mathbf{x}_{T}))}_{L_{T}} + \sum_{t>1} \underbrace{D_{\mathrm{KL}}(q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0}) \parallel p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t}))}_{L_{t-1}} - \log p_{\theta}(\mathbf{x}_{0}|\mathbf{x}_{1})\right]$ (5)
> 
> (See Appendix A for details. The labels on the terms are used in Section 3.) Equation (5) uses KL divergence to directly compare 
> $$p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t)$$
>  against forward process posteriors, which are tractable when conditioned on  $\mathbf{x}_0$ :
> 
> $$q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I}),$$
> where  $\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) \coloneqq \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t$  and  $\tilde{\beta}_t \coloneqq \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$  (7)
> 
> Consequently, all KL divergences in Eq. (5) are comparisons between Gaussians, so they can be calculated in a Rao-Blackwellized fashion with closed form expressions instead of high variance Monte Carlo estimates.
![image.png](DDPM%EF%BC%9ANIPS20/image%207.png)

> **[图片提取文字 (image.png)]:**
> ## You are right: The definition of the diffusion process (the "Encoder") is strictly **Forward**:
> 
>  $q(x_t|x_{t-1})$ 
> 
> - Direction:  $t-1 \to t$
> - Meaning: "Add noise."
>   Status: Known. We defined this manually (Gaussian noise schedule).
> 
> 1. The Forward Process (What you are thinking of)
> 
> \_\_\_\_\_
> 
> 2. The True Reverse Posterior (What I was referring to)
> 
> The quote "We want to find 
> $$q(x_{t-1}|x_t)$$
> " refers to the **True Reverse Distribution**.
> 
>  $q(x_{t-1}|x_t)$ 
> 
> - Diversion 4 \ 4 1
> 
> impossible to calculate).
> 
> - Direction:  $t \to t-1$  Meaning: "Given a noisy image, what is the *mathematically correct* probability distribution
> - of the previous, slightly cleaner step?"
> 
> Status: Unknown / Intractable.
> 
> Why is it unknown? To calculate  $q(x_{t-1}|x_t)$  perfectly, you would need to know the distribution of every possible image in the universe. (Mathematically:  $q(x_{t-1}|x_t)$   $\propto$ 
> 
>  $q(x_t|x_{t-1})\cdot q(x_{t-1})$  . That last term  $q(x_{t-1})$  is the marginal probability of the data, which is
![image.png](DDPM%EF%BC%9ANIPS20/image%208.png)

### Loss转换成预测噪声εθ（Lt-1，L0）

*preview*

**分布参数化**：**$*x_t = a \cdot x_{t-1} + b \cdot ε$** → xt以xt-1为条件的条件分布是高斯分布，xt-1作为常量，ε是高斯变量，除去xT，x0~xt-1都不是高斯变量。*

**加高斯噪声**：$q(x_t|x_{t-1})=N(x_t; at \cdot x_{t-1}, b_t \cdot I)$的含义是**xt的条件分布**，xt-1影响条件分布的均值，即**$x_t = a \cdot x_{t-1} + b \cdot ε$**。

**去高斯噪声**：$pθ(x_{t-1}|x_t)=N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))$的含义是**xt-1的条件分布**，xt影响条件分布的均值和方差，定义**$x_{t-1} = μ_θ(x_t, t) + Σ_θ \cdot ε$**。

*Lt-1：让pθ(xt-1 | xt)拟合q(xt-1 | xt, x0)。*

**pθ(xt-1 | xt)拟合q(xt-1 | xt, x0)**，μt‘和βt’分别是q(xt-1 | xt, x0)的均值和方差。由于高斯分布的假设，**pθ(xt-1 | xt)的均值uθ拟合ut‘**，来指引逆向去噪过程的采样。

> **[图片提取文字 (image.png)]:**
> Efficient training is therefore possible by optimizing random terms of L with stochastic gradient descent. Further improvements come from variance reduction by rewriting L (3) as:  $\mathbb{E}_{q}\left[\underbrace{D_{\mathrm{KL}}(q(\mathbf{x}_{T}|\mathbf{x}_{0}) \parallel p(\mathbf{x}_{T}))}_{L_{T}} + \sum_{t>1} \underbrace{D_{\mathrm{KL}}(q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0}) \parallel p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t}))}_{L_{t-1}} - \log p_{\theta}(\mathbf{x}_{0}|\mathbf{x}_{1})\right]$ (5)
> 
> (See Appendix A for details. The labels on the terms are used in Section 3.) Equation (5) uses KL divergence to directly compare 
> $$p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t)$$
>  against forward process posteriors, which are tractable when conditioned on  $\mathbf{x}_0$ :
> 
> $$q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I}),$$
> where  $\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) \coloneqq \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t$  and  $\tilde{\beta}_t \coloneqq \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$  (7)
> 
> Consequently, all KL divergences in Eq. (5) are comparisons between Gaussians, so they can be calculated in a Rao-Blackwellized fashion with closed form expressions instead of high variance Monte Carlo estimates.
![image.png](DDPM%EF%BC%9ANIPS20/image%209.png)

> **[图片提取文字 (image.png)]:**
> $q(x_{t-1}|x_t,x_0)$  Using Bayes' Rule, we can actually calculate this distribution exactly using only the forward
> 
> knew the final answer  $x_0$ ?"
> 
> process definitions:
> 
> This is called the Forward Posterior:
> 
> Since we cannot calculate the true reverse  $q(x_{t-1}|x_t)$ , we cheat. We ask: "What if I already
> 
> If we condition on the original image  $x_0$ , the reverse step becomes **Tractable** (Calculable).
> 
> 7 1
> 
> $$q(x_{t-1}|x_t,x_0) = \frac{q(x_t|x_{t-1}) \cdot q(x_{t-1}|x_0)}{q(x_t|x_0)}$$
> 
> All three terms on the right are Gaussians we know!
![image.png](DDPM%EF%BC%9ANIPS20/image%2010.png)

**训练过程中，x0、t和xt(x0, εt‘)和参数θ无关**，因为他们是x0、静态βt和ε产生的样本数据。“*’”表示“*”是累计过程关联的变量，如 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \mathbf{\epsilon_t'}$，其中εt‘是噪声叠加后表达式中的标准高斯噪声。

> Lt-1 = pθ(xt-1 | xt)和q(xt-1 | xt, x0)分布期望**μθ和μt‘的KL距离——**关于q(xt-1 | xt, x0)的**均值**，让**μθ预测μt‘**，对应公式(9)**。**

> 生成过程x0未知，**将x0用xt和εt**表示，Lt-1中μθ的ground truth **$μ_t' = \frac{1}{\sqrt{α_t}} (x_t(x_0,ε_t') - \frac {β_t } {\sqrt{(1 - \bar{α_t})} } ε_t')$**，对应公式(10)**。**

> 用ground truth μt‘的形式定义μθ，并将μθ中参数转移到**εθ(xt，t)，$μ_\theta = \frac{1}{\sqrt{α_t}} (x_t(x_0,ε_\theta) - \frac {β_t } {\sqrt{(1 - \bar{α_t})} } ε_\theta)$，εθ(xt, t)预测**的ground truth是x0到xt过程的**叠加噪声εt‘**，对应公式(11)。

*Loss → 训练和推理的**kernel***

**训练**：pθ(xt-1 | xt)拟合q(xt-1 | xt, x0)的**训练数据是x0**，将xt表达为x0和εt‘的表达式，则：

**Lt-1** → **预测噪声εθ(xt(x0, εt’), t)**和x0加噪到xt（$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \mathbf{\epsilon_t'}$）的**实际噪声εt‘**的距离——关于x0、εt’的**均值**，对应公式(12)**。**

**推理**：基于$p_θ(x_{t-1} | x_t)=N(u_θ, σ_t^2I)$的**εθ**表达，得到**从xt采样xt-1的表达式$x_{t-1} = \color{blue}{\frac{1}{\sqrt{\alpha_t}}} \left( x_t - \color{red}{\frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}t}} \epsilon_\theta(x_t, t)} \right) + \color{green}{\sigma_t \mathbf{z}}$**，σt控制最终生成的x0的形式（N(0, 1)，确定点）。

> **[图片提取文字 (image.png)]:**
> ## 3 Diffusion models and denoising autoencoders
> 
> Diffusion models might appear to be a restricted class of latent variable models, but they allow a large number of degrees of freedom in implementation. One must choose the variances  $\beta_t$  of the forward process and the model architecture and Gaussian distribution parameterization of the reverse process. To guide our choices, we establish a new explicit connection between diffusion models and denoising score matching (Section 3.2) that leads to a simplified, weighted variational bound objective for diffusion models (Section 3.4). Ultimately, our model design is justified by simplicity and empirical results (Section 4). Our discussion is categorized by the terms of Eq. (5).
> 
> ## 3.1 Forward process and $L_T$
> 
> We ignore the fact that the forward process variances  $\beta_t$  are learnable by reparameterization and instead fix them to constants (see Section 4 for details). Thus, in our implementation, the approximate posterior q has no learnable parameters, so  $L_T$  is a constant during training and can be ignored.
> 
> ## **3.2** Reverse process and $L_{1:T-1}$
> 
> Now we discuss our choices in  $p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_{\theta}(\mathbf{x}_t, t), \boldsymbol{\Sigma}_{\theta}(\mathbf{x}_t, t))$  for  $1 < t \leq T$ . First, we set  $\boldsymbol{\Sigma}_{\theta}(\mathbf{x}_t, t) = \sigma_t^2 \mathbf{I}$  to untrained time dependent constants. Experimentally, both  $\sigma_t^2 = \beta_t$  and  $\sigma_t^2 = \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$  had similar results. The first choice is optimal for  $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ , and the second is optimal for  $\mathbf{x}_0$  deterministically set to one point. These are the two extreme choices corresponding to upper and lower bounds on reverse process entropy for data with coordinatewise unit variance [53].
> 
> Second, to represent the mean  $\mu_{\theta}(\mathbf{x}_t, t)$ , we propose a specific parameterization motivated by the following analysis of  $L_t$ . With  $p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_{\theta}(\mathbf{x}_t, t), \sigma_t^2 \mathbf{I})$ , we can write:
> 
> $$L_{t-1} = \mathbb{E}_q \left[ \frac{1}{2\sigma_t^2} \| \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) - \boldsymbol{\mu}_{\theta}(\mathbf{x}_t, t) \|^2 \right] + C$$
>  (8)
> 
> where C is a constant that does not depend on  $\theta$ . So, we see that the most straightforward parameterization of  $\mu_{\theta}$  is a model that predicts  $\tilde{\mu}_t$ , the forward process posterior mean. However, we can expand Eq. (8) further by reparameterizing Eq. (4) as  $\mathbf{x}_t(\mathbf{x}_0, \epsilon) = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon$  for  $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$  and applying the forward process posterior formula (7):
> 
> $$L_{t-1} - C = \mathbb{E}_{\mathbf{x}_0, \epsilon} \left[ \frac{1}{2\sigma_t^2} \left\| \tilde{\boldsymbol{\mu}}_t \left( \mathbf{x}_t(\mathbf{x}_0, \epsilon), \frac{1}{\sqrt{\bar{\alpha}_t}} (\mathbf{x}_t(\mathbf{x}_0, \epsilon) - \sqrt{1 - \bar{\alpha}_t} \epsilon) \right) - \boldsymbol{\mu}_{\theta}(\mathbf{x}_t(\mathbf{x}_0, \epsilon), t) \right\|^2 \right]$$
> (9)
> 
> $$= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \left| \frac{1}{2\sigma_t^2} \left\| \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t(\mathbf{x}_0, \boldsymbol{\epsilon}) - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon} \right) - \boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{x}_t(\mathbf{x}_0, \boldsymbol{\epsilon}), t) \right\|^2 \right|$$
> (10)
![image.png](DDPM%EF%BC%9ANIPS20/image%2011.png)

> **[图片提取文字 (image.png)]:**
> ## **Algorithm 1** Training
> 
> ## **Algorithm 2** Sampling
> 
> 2: 
> $$\mathbf{x}_0 \sim q(\mathbf{x}_0)$$
> 
> 3: 
> $$t \sim \text{Uniform}(\{1, \dots, T\})$$
> 
> 4: 
> $$\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
> 
> $$\nabla_{\theta} \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_{\theta} (\sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, t) \right\|^2$$
> 
> 1: 
> $$\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
>   
> 2: **for**  $t = T, \dots, 1$  **do**
> 
> 2: **for** 
> $$t = T, ..., 1$$
>  **do**
> 
> 3: 
> $$\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
>  if  $t > 1$ , else  $\mathbf{z} = \mathbf{0}$ 
> 
> 4: 
> $$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_{\theta}(\mathbf{x}_t, t) \right) + \sigma_t \mathbf{z}$$
> 
> 5: end for
> 
> 6: return  $\mathbf{x}_0$ 
> 
> Equation (10) reveals that  $\mu_{\theta}$  must predict  $\frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon \right)$  given  $\mathbf{x}_t$ . Since  $\mathbf{x}_t$  is available as input to the model, we may choose the parameterization
> 
> $$\boldsymbol{\mu}_{\theta}(\mathbf{x}_{t}, t) = \tilde{\boldsymbol{\mu}}_{t} \left( \mathbf{x}_{t}, \frac{1}{\sqrt{\bar{\alpha}_{t}}} (\mathbf{x}_{t} - \sqrt{1 - \bar{\alpha}_{t}} \boldsymbol{\epsilon}_{\theta}(\mathbf{x}_{t})) \right) = \frac{1}{\sqrt{\alpha_{t}}} \left( \mathbf{x}_{t} - \frac{\beta_{t}}{\sqrt{1 - \bar{\alpha}_{t}}} \boldsymbol{\epsilon}_{\theta}(\mathbf{x}_{t}, t) \right)$$
> (11)
> 
> where  $\epsilon_{\theta}$  is a function approximator intended to predict  $\epsilon$  from  $\mathbf{x}_t$ . To sample  $\mathbf{x}_{t-1} \sim p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t)$  is to compute  $\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_{\theta}(\mathbf{x}_t, t) \right) + \sigma_t \mathbf{z}$ , where  $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ . The complete sampling procedure, Algorithm 2 resembles Langevin dynamics with  $\epsilon_{\theta}$  as a learned gradient of the data density. Furthermore, with the parameterization (11), Eq. (10) simplifies to:
> 
> $$\mathbb{E}_{\mathbf{x}_0, \epsilon} \left[ \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)} \left\| \epsilon - \epsilon_\theta (\sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t) \right\|^2 \right]$$
>  (12)
> 
> which resembles denoising score matching over multiple noise scales indexed by t [55]. As Eq. (12) is equal to (one term of) the variational bound for the Langevin-like reverse process (11), we see that optimizing an objective resembling denoising score matching is equivalent to using variational inference to fit the finite-time marginal of a sampling chain resembling Langevin dynamics.
> 
> To summarize, we can train the reverse process mean function approximator  $\mu_{\theta}$  to predict  $\tilde{\mu}_{t}$ , or by modifying its parameterization, we can train it to predict  $\epsilon$ . (There is also the possibility of predicting  $\mathbf{x}_0$ , but we found this to lead to worse sample quality early in our experiments.) We have shown that the  $\epsilon$ -prediction parameterization both resembles Langevin dynamics and simplifies the diffusion model's variational bound to an objective that resembles denoising score matching. Nonetheless, it is just another parameterization of  $p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t)$ , so we verify its effectiveness in Section 4 in an ablation where we compare predicting  $\epsilon$  against predicting  $\tilde{\mu}_t$ .
![image.png](DDPM%EF%BC%9ANIPS20/image%2012.png)

*L0 = -log[pθ(x0|x1)]：pθ(x0|x1)拟合q(x0|x1,x0)=δ(x0)*

**高斯分布pθ(x0|x1)是x1到x0的解码器**，拟合ground truth是样本x0处的狄拉克分布q(x0|x1,x0)=δ(x0)，即**学习x0的条件分布**的均值uθ(x1，1)≈x0，类似于VAE-Decoder最大化pθ(x|z)。

计算pθi(x0i|x1)：x0i是离散点，uθi是连续值，将x0i量化到[-1,1]，δ为积分窗口函数，pθi(x0i|x1)的预测均值**uθi(x1，1)接近x0i的量化值**时，pθi最大。

假设每个x0i独立由x1生成，对D作累乘获得联合分布pθ(x0|x1)，要求**每个pixel x0i**都能被正确解码。

去除系数来简化Loss，称为**reweighted Loss**，让Diffusion聚焦于较大t值的去噪重建。

> **[图片提取文字 (image.png)]:**
> ## **3** Data scaling, reverse process decoder, and $L_0$
> 
> We assume that image data consists of integers in  $\{0, 1, \dots, 255\}$  scaled linearly to [-1, 1]. This ensures that the neural network reverse process operates on consistently scaled inputs starting from the standard normal prior  $p(\mathbf{x}_T)$ . To obtain discrete log likelihoods, we set the last term of the reverse process to an independent discrete decoder derived from the Gaussian  $\mathcal{N}(\mathbf{x}_0; \boldsymbol{\mu}_{\theta}(\mathbf{x}_1, 1), \sigma_1^2 \mathbf{I})$ :
> 
> $$p_{\theta}(\mathbf{x}_{0}|\mathbf{x}_{1}) = \prod_{i=1}^{D} \int_{\delta_{-}(x_{0}^{i})}^{\delta_{+}(x_{0}^{i})} \mathcal{N}(x; \mu_{\theta}^{i}(\mathbf{x}_{1}, 1), \sigma_{1}^{2}) dx$$
> 
> $$\delta_{+}(x) = \begin{cases} \infty & \text{if } x = 1\\ x + \frac{1}{255} & \text{if } x < 1 \end{cases} \qquad \delta_{-}(x) = \begin{cases} -\infty & \text{if } x = -1\\ x - \frac{1}{255} & \text{if } x > -1 \end{cases}$$
> (13)
> 
> where D is the data dimensionality and the i superscript indicates extraction of one coordinate. (It would be straightforward to instead incorporate a more powerful decoder like a conditional autoregressive model, but we leave that to future work.) Similar to the discretized continuous distributions used in VAE decoders and autoregressive models [34, 52], our choice here ensures that the variational bound is a lossless codelength of discrete data, without need of adding noise to the data or incorporating the Jacobian of the scaling operation into the log likelihood. At the end of sampling, we display  $\mu_{\theta}(\mathbf{x}_1, 1)$  noiselessly.
> 
> ## 3.4 Simplified training objective
> 
> With the reverse process and decoder defined above, the variational bound, consisting of terms derived from Eqs. (12) and (13), is clearly differentiable with respect to  $\theta$  and is ready to be employed for
![image.png](DDPM%EF%BC%9ANIPS20/image%2013.png)

> **[图片提取文字 (image.png)]:**
> training. However, we found it beneficial to sample quality (and simpler to implement) to train on the following variant of the variational bound:
> 
> $$L_{\text{simple}}(\theta) := \mathbb{E}_{t,\mathbf{x}_0,\epsilon} \left[ \left\| \epsilon - \epsilon_{\theta} (\sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t) \right\|^2 \right]$$
>  (14)
> 
> where t is uniform between 1 and T. The t=1 case corresponds to  $L_0$  with the integral in the discrete decoder definition (13) approximated by the Gaussian probability density function times the bin width, ignoring  $\sigma_1^2$  and edge effects. The t>1 cases correspond to an unweighted version of Eq. (12), analogous to the loss weighting used by the NCSN denoising score matching model [55]. ( $L_T$  does not appear because the forward process variances  $\beta_t$  are fixed.) Algorithm [1] displays the complete training procedure with this simplified objective.
> 
> Since our simplified objective (14) discards the weighting in Eq. (12), it is a weighted variational bound that emphasizes different aspects of reconstruction compared to the standard variational bound [18, 22]. In particular, our diffusion process setup in Section 4 causes the simplified objective to down-weight loss terms corresponding to small t. These terms train the network to denoise data with very small amounts of noise, so it is beneficial to down-weight them so that the network can focus on more difficult denoising tasks at larger t terms. We will see in our experiments that this reweighting leads to better sample quality.
![image.png](DDPM%EF%BC%9ANIPS20/image%2014.png)

> **[图片提取文字 (image.png)]:**
> ## 2. The Solution: Integration over a "Bucket"
> 
> probability mass that falls inside the 'bucket' of this pixel?"
> 
> This is what the paper means by "independent discrete decoder."
> 
> Instead of asking "What is the probability density at exactly x?", we ask: "What is the total
> 
> , .....
> 
> ## The Mechanism (The Integral)
> 
> Let's say the ground truth pixel value is  $x_{target}$  (e.g., the value 0.5 in the scaled range). This
> 
>  $[x_{target} - \frac{1}{255}, x_{target} + \frac{1}{255}].$ 
> 
> The model  $p_{\theta}(x_0|x_1)$  predicts a Gaussian distribution  $\mathcal{N}(\mu_{\theta}(x_1),\sigma^2)$ .
> 
> Gaussian within the bounds of that specific pixel.
> 
> $$p_{\theta}(x_0|x_1) = \int_{x_- \perp}^{x+\frac{1}{255}} \mathcal{N}(z;\mu_{\theta}(x_1),\sigma^2) \; dz$$
> 
> • If the model is confident: The Gaussian is narrow and centered on the correct pixel. The area under the curve in that bucket is close to 1.0 (Log Likelihood  $\approx$  0).
> 
> To get the likelihood  $L_0$ , we calculate the **Area Under the Curve (Integral)** of this predicted
> 
>  If the model is wrong: The Gaussian is centered somewhere else. The area under the curve in the correct bucket is tiny (Log Likelihood is huge).
![image.png](DDPM%EF%BC%9ANIPS20/image%2015.png)

*训练和推理的实现*

Diffusion**训练**的Loss是**x0和εt的函数均值**，因此训练的实现需要采样x0、t和ε，前向过程是：

> 采样样本x0，采样时间t，采样噪声ε。

> 计算x0经过t个噪声**叠加**的输出$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \mathbf ({\epsilon_t'}=\epsilon)$。

> BackBone输入xt和t后预测**εθ(xt,t)≈ε**，计算εθ(xt,t)和ε的距离来优化。

Diffusion**推理**需要从采样xT=z开始，**对t∈T:1执行t步去噪**，每个t的执行过程是：

> BackBone输入xt和t后预测εθ(xt,t)。

> 对xt去噪，计算$x_{t-1} = \color{blue}{\frac{1}{\sqrt{\alpha_t}}} \left( x_t - \color{red}{\frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}t}} \epsilon_\theta(x_t, t)} \right) + \color{green}{\sigma_t \mathbf{z}}$。

> t=1执行完后，输出x0。

> **[图片提取文字 (image.png)]:**
> ## Training a diffusion model is surprisingly simple. We do not need to generate full images during
> 
> ## Algorithm:
> 
> 5. The Training Process
> 
> training.
> 
> - Sample Data: Pick a real image x<sub>0</sub> from your dataset.
> - 2. Sample Time: Pick a random timestep  $t \in [1, 1000]$  (Uniformly).
> 
> 6. **Optimize:** Calculate Loss  $||\epsilon - \hat{\epsilon}||^2$  and Backpropagate.
> 
> - 3. Sample Noise: Pick random Gaussian noise  $\epsilon \sim \mathcal{N}(0,1)$ .
> - Corrupt: Create the noisy image x<sub>t</sub> using the "Nice Property" formula (mixing x<sub>0</sub> and ε).
> 
> - 5. **Predict:** Feed  $x_t$  and time t into the U-Net. The U-Net outputs  $\hat{\epsilon}$  (its guess of the noise).
![image.png](DDPM%EF%BC%9ANIPS20/image%2016.png)

> **[图片提取文字 (image.png)]:**
> This is the "Generation" phase. It is slow because it is iterative.
> 
> - 1. **Start:** Generate pure random noise  $x_T \sim \mathcal{N}(0, I)$ .
> - 2. **Loop:** For  $t = T, T 1, \dots, 1$ :
> 
> 6. The Inference Process (Sampling)
> 
> - . , , ,
>   - **Predict Noise:** Use the trained network to guess the noise  $\epsilon_{\theta}(x_t,t)$ .
> 
> the image collapses to a blurry deterministic average.
> 
> . . . . . . . . . . . . . . . . . . . .
> 
> Subtract Noise: Remove a fraction of that noise to estimate the "cleaner" image.
> 
> - Add Langevin Noise: (Crucial Step) Add a tiny bit of random noise back in.
> - Why? To simulate the randomness of the physical diffusion process. Without this,
> 
> - Result:  $x_{t-1}$ .
> - 3. **End:** The final  $x_0$  is your generated image.
![image.png](DDPM%EF%BC%9ANIPS20/image%2017.png)

### Diffusion实验和用途

Diffusion的x0扩散和z去噪轨迹是对x0的保守有损压缩，L1+…+LT是压缩率，L0是失真率，x1:xT是压缩表达，x0是原始信息，p(Y=x0 | x1)是x1对x0的还原度，因此-logp(Y= x0 | x1)是失真度。

> **[图片提取文字 (image.png)]:**
> ## 4.3 Progressive coding
> 
> Table 1 also shows the codelengths of our CIFAR10 models. The gap between train and test is at most 0.03 bits per dimension, which is comparable to the gaps reported with other likelihood-based models and indicates that our diffusion model is not overfitting (see Appendix D for nearest neighbor visualizations). Still, while our lossless codelengths are better than the large estimates reported for energy based models and score matching using annealed importance sampling [11], they are not competitive with other types of likelihood-based generative models [7].
> 
> Since our samples are nonetheless of high quality, we conclude that diffusion models have an inductive bias that makes them excellent lossy compressors. Treating the variational bound terms  $L_1 + \cdots + L_T$  as rate and  $L_0$  as distortion, our CIFAR10 model with the highest quality samples has a rate of 1.78 bits/dim and a distortion of 1.97 bits/dim, which amounts to a root mean squared error of 0.95 on a scale from 0 to 255. More than half of the lossless codelength describes imperceptible distortions.
![image.png](DDPM%EF%BC%9ANIPS20/image%2018.png)

> **[图片提取文字 (image.png)]:**
> ## 3. The Math: KL Divergence IS the "Correction Cost"
> 
> This "Correction Cost" is the exact definition of KL Divergence.
> 
> $$\text{Rate} = \underbrace{\mathbb{E}_q[-\log p_\theta(x)]}_{\text{Bits needed if we rely on Bob's guess}} - \underbrace{\mathbb{E}_q[-\log q(x)]}_{\text{Bits contained in the actual data}}$$
> 
> Rate =  $\mathbb{E}_q \left| \log \frac{q(x)}{p_\theta(x)} \right| = D_{KL}(q||p_\theta)$ 
> 
> • If Bob guesses perfectly (
> $$p_{\theta}=q$$
> ):
> 
> - $\log(1) = 0$ .
> - Rate = 0. Alice sends nothing. Bob generates the image on his own.
> - If Bob guesses wrong ( $p_{\theta} \neq q$ ):
> - Alice must transmit data to correct his distribution.
>   - Rate > 0. The worse Bob's guess, the higher the Rate (KL).
![image.png](DDPM%EF%BC%9ANIPS20/image%2019.png)

> **[图片提取文字 (image.png)]:**
> Rate  $(L_1 + \cdots + L_T)$ 
> 
> 2. The Math Definitions
> 
> . ...- -...
> 
> In a VAE or Diffusion model, "Rate" is the **KL Divergence**. It measures how much information the Latent Variables carry.
> 
> In Diffusion, the "Latents" are the entire trajectory of the reverse process.
> 
> Meaning: The model uses about 1.78 bits of information per pixel to describe the
> 
> - Score: 1.78 bits/dim.
> - structure and content of the image (shapes, textures, global composition).
> 
> ## Distortion ( $L_0$ )
> 
> Distortion is the **Reconstruction Loss** (Negative Log Likelihood).
> 
> - Distortion is the Records detail 2003 (Negative 200 Electricod).
> - In Diffusion, this is the very last step:  $p_{\theta}(x_0|x_1)$ .
>   Since digital images are discrete integers (0-255),  $L_0$  measures the error of rounding the
> - continuous values of  $x_1$  into specific colors.
> - Score: 1.97 bits/dim.
> - Meaning: The model loses about 1.97 bits of information per pixel just in the final "rounding" process.
![image.png](DDPM%EF%BC%9ANIPS20/image%2020.png)

> **[图片提取文字 (image.png)]:**
> ## 4. What this tells us about Diffusion
> 
> This quote proves that Diffusion Models focus on Semantic Content (Rate), not Pixel Perfection (Distortion).
> 
> - L1...LT (Rate): This is where the "Soul" of the image lives. The model spends 1.78 bits\nensuring the eyes are aligned, the fur looks real, and the lighting is correct.
> - L0 (Distortion): This is just imperceptible high-frequency noise. The model happily\nignores this (resulting in high distortion) because humans don't notice if a pixel is value
>   200 or 205, as long as it's part of a beautiful face.
> 
> Summary of the Quote: "We analyzed our loss function using Information Theory. We found that our model allocates ~1.78 bits to describing the image structure (Rate) and accepts ~1.97 bits of error in the final pixel values (Distortion). Even though this makes it a mediocre file
> 
> compressor mathematically, it produces the highest quality samples visually."
![image.png](DDPM%EF%BC%9ANIPS20/image%2021.png)

Diffusion的Loss来源于马尔库夫链，Diffusion和自回归模型的数学模型都是马尔可夫链，但**Diffusion和自回归生成图像的过程不同**。

自回归模型生成图像是“**重排数据坐标**”的，每一步基于之前生成位置的pixel或z，预测下一个位置的pixel，而预测起点的位置/坐标可以改变。 

Diffusion模型生成图像是“**位生成顺序**”的，即所有位置pixel的**数值位（如8bit，0-255）**随着去噪过程的每个t逐渐确定/生成。

> **[图片提取文字 (image.png)]:**
> $L = D_{\mathrm{KL}}(q(\mathbf{x}_T) \parallel p(\mathbf{x}_T)) + \mathbb{E}_q \left[ \sum_{t>1} D_{\mathrm{KL}}(q(\mathbf{x}_{t-1}|\mathbf{x}_t) \parallel p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t)) \right] + H(\mathbf{x}_0)$  (16)
> 
> Connection to autoregressive decoding Note that the variational bound (5) can be rewritten as:
> 
> (See Appendix  $\overline{A}$  for a derivation.) Now consider setting the diffusion process length T to the dimensionality of the data, defining the forward process so that  $q(\mathbf{x}_t|\mathbf{x}_0)$  places all probability mass on  $\mathbf{x}_0$  with the first t coordinates masked out (i.e.  $q(\mathbf{x}_t|\mathbf{x}_{t-1})$  masks out the  $t^{\text{th}}$  coordinate), setting  $p(\mathbf{x}_T)$  to place all mass on a blank image, and, for the sake of argument, taking  $p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t)$  to
![image.png](DDPM%EF%BC%9ANIPS20/image%2022.png)

> **[图片提取文字 (image.png)]:**
> a generalized bit ordering that cannot be expressed by reordering data coordinates. Prior work has shown that such reorderings introduce inductive biases that have an impact on sample quality [38], so we speculate that the Gaussian diffusion serves a similar purpose, perhaps to greater effect since Gaussian noise might be more natural to add to images compared to masking noise. Moreover, the Gaussian diffusion length is not restricted to equal the data dimension; for instance, we use T=1000, which is less than the dimension of the  $32 \times 32 \times 3$  or  $256 \times 256 \times 3$  images in our experiments.
> 
> be a fully expressive conditional distribution. With these choices,  $D_{\mathrm{KL}}(q(\mathbf{x}_T) \parallel p(\mathbf{x}_T)) = 0$ , and
> 
> minimizing  $D_{\mathrm{KL}}(q(\mathbf{x}_{t-1}|\mathbf{x}_t) \parallel p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t))$  trains  $p_{\theta}$  to copy coordinates  $t+1,\ldots,T$  unchanged
> 
> and to predict the  $t^{\text{th}}$  coordinate given  $t+1,\ldots,T$ . Thus, training  $p_{\theta}$  with this particular diffusion is
> 
> We can therefore interpret the Gaussian diffusion model (2) as a kind of autoregressive model with
> 
> Gaussian diffusions can be made shorter for fast sampling or longer for model expressiveness.
> 
> training an autoregressive model.
![image.png](DDPM%EF%BC%9ANIPS20/image%2023.png)

> **[图片提取文字 (image.png)]:**
> The following is an alternate version of L. It is not tractable to estimate, but it is useful for our discussion in Section 4.3
> 
> $$L = \mathbb{E}_q \left[ -\log p(\mathbf{x}_T) - \sum_{t \ge 1} \log \frac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_t|\mathbf{x}_{t-1})} \right]$$
> (23)
> 
> $$= \mathbb{E}_{q} \left[ -\log \frac{p(\mathbf{x}_{T})}{q(\mathbf{x}_{T})} - \sum_{t \geq 1} \log \frac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{q(\mathbf{x}_{t-1}|\mathbf{x}_{t})} - \log q(\mathbf{x}_{0}) \right]$$
> 
> $$= D_{\mathrm{KL}}(q(\mathbf{x}_{T}) \parallel p(\mathbf{x}_{T})) + \mathbb{E}_{q} \left[ \sum_{t \geq 1} D_{\mathrm{KL}}(q(\mathbf{x}_{t-1}|\mathbf{x}_{t}) \parallel p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})) \right] + H(\mathbf{x}_{0})$$
> (25)
> 
> $$= \mathbb{E}_{q} \left[ -\log p(\mathbf{x}_{T}) - \sum_{t \geq 1} \log \frac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{q(\mathbf{x}_{t-1}|\mathbf{x}_{t})} \cdot \frac{q(\mathbf{x}_{t-1})}{q(\mathbf{x}_{t})} \right]$$
> 
> $$= \mathbb{E}_{q} \left[ -\log \frac{p(\mathbf{x}_{T})}{q(\mathbf{x}_{t})} - \sum_{t \geq 1} \log \frac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{q(\mathbf{x}_{t})} - \log q(\mathbf{x}_{0}) \right]$$
> (24)
![image.png](DDPM%EF%BC%9ANIPS20/image%2024.png)

> **[图片提取文字 (image.png)]:**
> ## Think of an image as a sequence of pixels. A standard Autoregressive model (like PixelCNN) predicts them one by one.
> 
> The Process: It guesses Pixel 1 completely. Then it uses Pixel 1 to guess Pixel 2. Then uses
> 
> The Constraint: It must fully commit to the exact value (all 8 bits, Red=200, Green=50,
> 
> - The "Reordering": You can change the order (e.g., start in the middle, or do a spiral), but you are always locking in one spatial location at a time.
> - 2. The "Generalized" Bit Ordering (Diffusion)
> 
> 1 & 2 to guess Pixel 3.
> 
> 1. The "Standard" Bit Ordering (Autoregressive)
> 
> Blue=30) of the current pixel before moving to the next.
> 
> Diffusion doesn't solve the image pixel-by-pixel. It solves it layer-by-layer of detail.
> 
> (Information Importance). High-Order Bits (MSB): The general shape (Is it a cat or a dog? Is it light or dark?).
> 
> Low-Order Bits (LSB): The exact noise grain (The precise integer value 201 vs 202).
> 
> The quote suggests we view the image not as a grid of pixels, but as a stack of "Bit Planes"
> 
> - Middle-Order Bits: The texture (Fur, whiskers).
> - The Diffusion "Order":
> - Start (t = T): The canvas is blank.
> - 2. Early Steps (t = 900): The model predicts the High-Order Bits for all pixels
> 
> End Steps (t = 0): It predicts the Low-Order Bits (fine noise) for all pixels.
> 
> - simultaneously. It decides "This is a cat shape," but the pixels are still blurry.
> - 3. Middle Steps (t = 500): It predicts the Middle Bits for all pixels. "The cat is orange."
![image.png](DDPM%EF%BC%9ANIPS20/image%2025.png)

xT去噪生成图像x0过程中，**不同xt的含义**是优先生成大的轮廓，后生成局部的细节。

> **[图片提取文字 (image.png)]:**
> **Progressive generation** We also run a progressive unconditional generation process given by progressive decompression from random bits. In other words, we predict the result of the reverse process,  $\hat{\mathbf{x}}_0$ , while sampling from the reverse process using Algorithm 2. Figures 6 and 10 show the resulting sample quality of  $\hat{\mathbf{x}}_0$  over the course of the reverse process. Large scale image features appear first and details appear last. Figure 7 shows stochastic predictions  $\mathbf{x}_0 \sim p_{\theta}(\mathbf{x}_0|\mathbf{x}_t)$  with  $\mathbf{x}_t$  frozen for various t. When t is small, all but fine details are preserved, and when t is large, only large scale features are preserved. Perhaps these are hints of conceptual compression 18.
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Figure 6: Unconditional CIFAR10 progressive generation ( $\hat{\mathbf{x}}_0$  over time, from left to right). Extended samples and sample quality metrics over time in the appendix (Figs. 10 and 14).
> 
> ![](_page_0_Figure_3.jpeg)
> 
> Figure 7: When conditioned on the same latent, CelebA-HQ  $256 \times 256$  samples share high-level attributes. Bottom-right quadrants are  $\mathbf{x}_t$ , and other quadrants are samples from  $p_{\theta}(\mathbf{x}_0|\mathbf{x}_t)$ .
![image.png](DDPM%EF%BC%9ANIPS20/image%2026.png)

### DDPM Q & A

Q：为什么扩散参数选择是**sqrt(1-βt)和βt**？

A：前向扩散是q后验q(xt|xt-1)，由**静态参数βt**定义，扩散每一步结果xt的**方差为1**。

> **[图片提取文字 (image.png)]:**
> ## $\mathcal{N}(\dots)$ : This stands for a **Normal (Gaussian) Distribution**.
> 
> 1. The Symbols
> 
> step  $(x_t)$ ?"
> 
> The equation defines the new image  $x_t$  as a mix of **Signal** and **Noise**.
> 
>  $\beta_t$ : This is the **Variance Schedule** (a small number between 0 and 1). It controls "how
> 
> Example:  $\beta_t = 0.0001$  (Tiny noise addition).
> 
> 2. The Components (Signal vs. Noise)
> 
> much noise we add at step t."
> 
> ## A. The Mean: $\sqrt{1-\beta_t}x_{t-1}$ (Preserving the Signal)
> 
>  $\sqrt{0.9999}$ ). Meaning: We slightly fade out the original image. We shrink the signal towards zero to
> 
> This part takes the previous image  $x_{t-1}$  and multiplies it by a number slightly less than 1 (e.g.,
> 
>  $q(x_t|x_{t-1})$ : This is a **Conditional Probability Distribution**. It asks: "Given the image at
> 
> the previous step  $(x_{t-1})$ , what is the probability distribution for the image at the current
> 
> ## B. The Variance: $\beta_t \mathbf{I}$ (Injecting the Noise)
> 
> make room for the noise.
> 
> This part describes the random scatter added to the image.
> 
> **Meaning:** We add Gaussian noise with variance  $\beta_t$ .
![image.png](DDPM%EF%BC%9ANIPS20/image%2027.png)

> **[图片提取文字 (image.png)]:**
> ## In plain English, this equation says:
> 
> 3. The Physical Interpretation
> 
> "To get the image for step t: Take the image from step t-1,  $\dim$  it slightly (multiply by  $\sqrt{1-\beta}$ ), and then sprinkle some static on top (add variance  $\beta$ )."
> 
> ## 4. Why the Square Root ( $\sqrt{1-\beta_t}$ )?
> 
> You might wonder: "Why do we multiply by  $\sqrt{1-\beta_t}$ ? Why not just  $(1-\beta_t)$ ?"
> 
> The goal is to keep the **total energy (variance)** of the image constant at **1**.
> 
> - Variance of the signal part:  $(\sqrt{1-\beta_t})^2=1-\beta_t.$
> - Variance of the noise part:  $\beta_t$ .
> - Total Variance:  $(1 \beta_t) + \beta_t = 1$ .
> 
> throughout the process.
> 
> If we didn't use the square root, the pixel values would either explode to infinity or vanish to zero as we added noise 1,000 times. This scaling ensures the image stays numerically stable
![image.png](DDPM%EF%BC%9ANIPS20/image%2028.png)

Q：**VAE和Diffusion**的关联是什么？

VAE是x0单步编码到z的Encoder，z单步解码到x0的Decoder。

Diffusion是x0多步扩散到xT(z)的Diffusion，xT(z)多步去噪到x0的Denoise。

> **[图片提取文字 (image.png)]:**
> ## 1. Let's Align the Notation (N=1 case) Let's imagine a Single-Step Diffusion (T=1).
> 
> Forward (Fixed Encoder): Add huge noise to Image  $(x_0)$  to get Latent  $(x_1)$ .
> 
> • Reverse (Desired Decoder): Given Latent  $(x_1)$ , what was the Image  $(x_0)$ ?
> 
> Now compare this to a VAE:
> 
> **Target:** The True Posterior  $q(x_0|x_1)$ .
> 
> •  $q(x_1|x_0) = \mathcal{N}(x_1;...)$ 
> 
> • Model:  $p_{\theta}(x|z)$ .
> 
> the VAE Decoder target.
> 
> - Encoder (Learned): Map Image (x) to Latent (z).
> - Decoder (Learned): Map Latent (z) to Image (x).  $\mathscr D$
> - Your observation is correct: Structurally,  $q(x_0 | x_1)$  in Diffusion is exactly the counterpart of
![image.png](DDPM%EF%BC%9ANIPS20/image%2029.png)

> **[图片提取文字 (image.png)]:**
> 4. Direct Answer to your "Why that difference?"
> 
> Correction: In VAE, the **Model**  $(p_{\theta})$  is a Gaussian. But the **True Inverse** of the encoder (the
> 
> "In VAE decoder, the q(x0 | z) is a Gaussian and NN learn the mean and variance..."
> 
> - reality we are trying to model) is NOT a Gaussian.
>   VAE: We use a Simple Model to approximate a Complex Reality.
>   - Result: Mismatch (Blur).
> - **Diffusion:** We break the Reality into small pieces so that the Reality *becomes* Simple.
> - Result: Simple Model matches Simple Reality (Sharpness).
![image.png](DDPM%EF%BC%9ANIPS20/image%2030.png)

> **[图片提取文字 (image.png)]:**
> ## The reason we say $q(x_0|z)$ is "hard" in single-step diffusion (and VAE) is not that we can't
> 
> The "Hardness" comes from the Shape of the Distribution
> 
> The Scenario: "The Missing Details"
> 
> define a Gaussian model for it. We can! The problem is that the Gaussian model is WRONG.
> 
> ## Imagine your latent code & correspon
> 
> Imagine your latent code z corresponds to "A person with glasses." In reality, there are 1,000,000 different faces that fit this description.
> 
> - 1. The True Posterior ( $q(x_0|z)$ ):
>   - This distribution is **Multimodal** (has many peaks). It looks like a mountain range. There
>   - It is Complex.
>   - The VAE Decoder Assumption ( $p_{\theta}(x|z)$ ):
> 
>     We force this to be a **Unimodal Gaussian** (one single hill).
> 
> is a peak for "Bob," a peak for "Alice," a peak for "John."
> 
> • We assume  $p_{\theta}(x|z) = \mathcal{N}(\mu,\sigma)$ .
> 
> ## The Conflict (Why VAEs are Blurry)
> 
> multimodal reality of  $q(x_0|z)$ .
> 
> When you try to fit a **Single Gaussian Hill** (Model) over a **Mountain Range** (Truth):
> 
> - The Gaussian cannot cover all the peaks separately.
> - To minimize error, it positions itself right in the **middle** of all the peaks.
> - The Result: The "average" of Bob, Alice, and John is a blurry ghost face.
> 
> This is why we say it is "Hard": It is not computationally hard to run the neural network. It is representationally hard. A single Gaussian decoder (VAE style) fails to capture the complex,
![image.png](DDPM%EF%BC%9ANIPS20/image%2031.png)

Q：Diffusion-**denoising**和VAE-**decoder**的学习和训练有什么区别？

Diffusion和VAE都通过**变分推断近似未知分布，即最大化logp(X)**。Diffusion将编码映射**设计为覆盖语义足够**的扩散过程，因此只需预测在解码映射的生成过程。VAE需要同时优化编码和解码映射（分布），要求编码映射能覆盖更多语义，解码映射能正确还原样本。

Diffusion-Denoising预测xT→x0过程每个t的xt → xt-1的“中间路径”pθ(xt-1 | xt)。

> 但xt-1被xT~x0的“路径”q(xt | xt-1)定义，**q(x0)…q(xT-1)未知**，因此q(xt, xt-1)未知。

> **将x0作为条件**，让pθ(xt-1 | xt)拟合**q(xt-1 | xt, x0)**。当t=1时，pθ(x0 | x1)拟合到x0附近。

VAE-Decoder预测xT(z)→x0的“路径”**pθ(x0 | z)**，Encoder预测x0→xT(z)的“路径”**qφ(z | x0)**。

> **pθ(x0 | z)最大**，即denoising中pθ(x0 | x1)最大，让生成接近样本x0。

> qφ(z | x0)靠近先验分布p(z)，让编码z**覆盖**x0所需编码范围。

Q：pθ(xt-1 | xt)学习以x0为条件的**q(xt-1 | xt, x0)是否会损失效果？**

q(xt | xt-1)难以计算是因为**ground truth的xT~x0路径和未知的*xT‘~x0’*路径在xt→xt-1段存在重叠**，但Diffusion学习**多组xT~x0的途径**，包含多样的x0，就能获得多样的生成能力。

> **[图片提取文字 (image.png)]:**
> depends on an **impossible Marginal Probability**. To train a neural network  $p_{\theta}(x_{t-1}|x_t)$ , we need a target "ground truth" to learn from. Let's try to calculate the true target  $q(x_{t-1}|x_t)$  using Bayes' Rule:
> 
> Question 1: In the denoising phase, why not just assume  $q(x_{t-1}|x_t)$  is Gaussian and
> 
> The Answer: We do assume it is a Gaussian, but we cannot calculate it directly because it
> 
> $$q(x_{t-1}|x_t) = \frac{q(x_t|x_{t-1}) \cdot \mathbf{q}(\mathbf{x_{t-1}})}{\mathbf{q}(\mathbf{x_t})}$$
> 
> Here is where the "Conditional vs. Marginal" trap snaps shut:
> 
> maximize it directly?
> 
> - 1. The Conditional  $q(x_t | x_{t-1})$  is EASY:
> - This is just the noise schedule (add noise to an image). We know this formula
> - perfectly.
> - 2. The Marginals  $q(x_{t-1})$  and  $q(x_t)$  are IMPOSSIBLE:
>   - $\bullet$   $q(x_{t-1})$  asks: "What is the probability of this noisy pattern appearing, considering
>   - ALL possible images in the universe that could have created it?"
>     To know this, you would need the probability distribution of all real-world data, which is exactly what we are trying to learn in the first place!
>     - Because the **Marginal** is unknown, the **True Posterior**  $q(x_{t-1}|x_t)$  is mathematically intractable (unknown). We literally cannot calculate the target label for our network.
![image.png](DDPM%EF%BC%9ANIPS20/image%2032.png)

> **[图片提取文字 (image.png)]:**
> ## Since we can't solve the Marginal problem, VI gives us a cheat code. It says: "Don't try to
> 
> calculate the universal posterior. Just calculate the posterior given the training image  $x_0$ ."
> 
>  $q(x_{t-1}|x_t,\mathbf{x_0})$ 
> 
> ## By adding the condition $x_0$ :
> 
> • We no longer need to know the universe of all images  $(q(x_{t-1}))$ .
> 
> How VI Solves This (The "Conditioning" Trick)
> 
> unknown marginals.
> 
> - We only need to know the path from this specific image  $(x_0)$ .
> - This turns an Impossible Marginal Calculation into a Simple Gaussian Calculation.
> 
> ## **Summary for Q1:** We don't use direct maximization of $q(x_{t-1}|x_t)$ because calculating that term requires knowledge of the infinite data distribution (Marginals). We use VI to switch the target to $q(x_{t-1}|x_t,x_0)$ , which is calculable because it relies on fixed conditions rather than
![image.png](DDPM%EF%BC%9ANIPS20/image%2033.png)

> **[图片提取文字 (image.png)]:**
> ## Goal: Maximize $\log p_{\theta}(x|z)$ .
> 
> The VAE Decoder Scenario (We have the label)
> 
> - The Assumption: It is a Gaussian  $\mathcal{N}(x; \mu_{\theta}(z), \sigma^2)$ . The Process:
>   - Input z into network → Get predicted x̂.
> - Compare  $\hat{x}$  with **Real Image** x.
> - 3. Minimize Error  $||x \hat{x}||^2$ . Why it works: We have the Ground Truth x in our training dataset.
> - 2. The Diffusion Decoder Scenario (We miss the label)
> - Goal: Maximize  $\log p_{\theta}(x_{t-1}|x_t)$ .
> - The Assumption: It is a Gaussian  $\mathcal{N}(x_{t-1}; \mu_{\theta}(x_t), \sigma^2)$ .
> - The Process:
> - Input x<sub>t</sub> into network → Get predicted x̂<sub>t-1</sub>.
> - Compare \$\hat{x}\_{t-1}\$ with... Wait. What is the ground truth \$x\_{t-1}\$? The Problem: In our training set, we have  $x_0$  (clean image). We do **not** have the "correct"
> 
> slightly-less-noisy image  $x_{t-1}$ . There are infinite valid paths from  $x_0$  to  $x_t$ .
![image.png](DDPM%EF%BC%9ANIPS20/image%2034.png)

> **[图片提取文字 (image.png)]:**
> Below is a derivation of Eq. (5), the reduced variance variational bound for diffusion models. This material is from Sohl-Dickstein et al. [53]; we include it here only for completeness.
> 
> $$L = \mathbb{E}_{q} \left[ -\log \frac{p_{\theta}(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_{0})} \right]$$
> 
> $$= \mathbb{E}_{q} \left[ -\log p(\mathbf{x}_{T}) - \sum_{t \geq 1} \log \frac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{q(\mathbf{x}_{t}|\mathbf{x}_{t-1})} \right]$$
> (18)
> 
> $$= \mathbb{E}_{q} \left[ -\log p(\mathbf{x}_{T}) - \sum_{t>1} \log \frac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{q(\mathbf{x}_{t}|\mathbf{x}_{t-1})} - \log \frac{p_{\theta}(\mathbf{x}_{0}|\mathbf{x}_{1})}{q(\mathbf{x}_{1}|\mathbf{x}_{0})} \right]$$
> 
> $$= \mathbb{E} \left[ -\log p(\mathbf{x}_{T}) - \sum_{t>1} \log \frac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{q(\mathbf{x}_{t-1}|\mathbf{x}_{0})} - q(\mathbf{x}_{t-1}|\mathbf{x}_{0}) - \log \frac{p_{\theta}(\mathbf{x}_{0}|\mathbf{x}_{1})}{q(\mathbf{x}_{0}|\mathbf{x}_{1})} \right]$$
> (19)
> 
> $$= \mathbb{E}_q \left[ -\log p(\mathbf{x}_T) - \sum_{t>1} \log \frac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)} \cdot \frac{q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_t|\mathbf{x}_0)} - \log \frac{p_{\theta}(\mathbf{x}_0|\mathbf{x}_1)}{q(\mathbf{x}_1|\mathbf{x}_0)} \right]$$
> (20)
> 
> $$= \mathbb{E}_q \left[ -\log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T | \mathbf{x}_0)} - \sum_{t>1} \log \frac{p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_t)}{q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)} - \log p_{\theta}(\mathbf{x}_0 | \mathbf{x}_1) \right]$$
> (21)
![image.png](DDPM%EF%BC%9ANIPS20/image%2035.png)

> **[图片提取文字 (image.png)]:**
> $= \mathbb{E}_q \left[ D_{\mathrm{KL}}(q(\mathbf{x}_T | \mathbf{x}_0) \parallel p(\mathbf{x}_T)) + \sum_{t>1} D_{\mathrm{KL}}(q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) \parallel p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_t)) - \log p_{\theta}(\mathbf{x}_0 | \mathbf{x}_1) \right]$
![image.png](DDPM%EF%BC%9ANIPS20/image%2036.png)

Q：Diffusion的**扩散**过程和VAE的**Encoding**区别

Diffusion的扩散目的是将图像x0摧毁成纯噪声z，**摧毁方式是逐步加噪声（参数设置保证）**，**设置βt是为了防止方差爆炸**，是否动态选择参数不影响达成目的，因此静态设置βt来降低计算量。

VAE的Encoding目的是压缩数据维度，提取**存储价值更高的全局信息**到编码z中，而Diffusion计算过程不会改变数据维度。

> **[图片提取文字 (image.png)]:**
> ## The answer lies in the difference between **Compression** (VAE) and **Destruction** (Diffusion). 1. VAE Goal: Semantic Compression (Hard)
> 
> Task: Squeeze a 1,000,000-pixel cat image into a 512-dimension vector z without losing
> 
> Why don't we learn the noise schedule in Diffusion? Why don't we use a fixed encoder in VAE?
> 
> Question 2: Why does VAE use VI (Learned Encoder) while Diffusion uses Static Beta
> 
> (Fixed Encoder)?
> 
> the concept of "Cat".
> 
> Why Learning is required: This is an incredibly difficult cognitive task. You cannot write a simple math formula to extract "Cat-ness."
>  Fixed Encoder? If you used a fixed random encoder (like multiplying by a static random
> 
> matrix), you would destroy the semantic meaning. The decoder would essentially be trying
> 
> Result: The Encoder must be a Neural Network that learns the specific features of the
> dataset.
> 
> to reconstruct a cat from random hash codes—which is impossible.
![image.png](DDPM%EF%BC%9ANIPS20/image%2037.png)

> **[图片提取文字 (image.png)]:**
> ## Task: Turn a cat image into pure random noise.
> 
> 2. Diffusion Goal: Information Destruction (Easy)
> 
> - Why Learning is NOT required: We already know the formula for destroying information:
> - Static Beta? A simple fixed schedule (add 1% noise, then 2% noise...) does a perfect job of destroying the data until it matches the prior  $\mathcal{N}(0,I)$ . We don't need a neural network to learn how to break things.
> 
> Result: We use a Static Beta Schedule because it is computationally cheap,
> 
> mathematically stable, and sufficient for the task.
> 
> Add Noise.
> 
> ## 3. Stability: The "Anchor" Benefit
> 
> Using a fixed encoder gives Diffusion a massive engineering advantage over VAEs and GANs.
> 
> - using a fixed encoder gives Diffusion a massive engineering advantage over VAEs and GANs.
> 
>   VAE (Two Moving Targets): The Encoder learns  $\rightarrow$  The definition of z changes  $\rightarrow$  The
> - Decoder gets confused  $\to$  The Decoder changes  $\to$  The Encoder must adapt again. This "Co-adaptation" leads to training instability.
> - **Diffusion (One Anchor):** The Forward Process  $(x_0 \to x_t)$  is **Fixed in Stone**. The target never moves. The Decoder has a stable, supervised regression problem to solve.
![image.png](DDPM%EF%BC%9ANIPS20/image%2038.png)

 

Q：为什么VAE会优先将图像**全局信息**压缩到编码z？

全局信息会影响更大范围的数据，全局信息丢失对MSE造成很大损失，全局信息的编码连续而局部信息的编码离散且难以预测。

> **[图片提取文字 (image.png)]:**
> ## The Economy of Bits (The Bottleneck) Imagine you have to describe a 1,000,000-pixel image of a face using only 10 words (the
> 
> bottleneck z).
> 
> Strategy A (Local): "Pixel 1 is red. Pixel 2 is slightly darker red. Pixel 3 is..."
> 
> - Result: You describe 0.001% of the image. The rest is unknown. The reconstruction
> - fails.
> 
>    Strategy B (Global): "A smiling woman with blonde hair and glasses."
> - - - - - - - - - - -
> - Result: These 10 words allow the Decoder to guess the color of thousands of pixels at
>   once. If you say "Blonde," the decoder knows that 50,000 pixels on top are yellow.
> 
> Why Global wins: Global features (pose, shape, color scheme) have High Predictive Power. A
> 
> single bit of global information ("Day" vs "Night") predicts the value of every single pixel in the
> 
> image. To minimize reconstruction error with limited capacity, the encoder *must* latch onto these high-yield features first.
> 
> ## 2. The Loss Function Bias (MSE)
> 
> Most VAEs use **Mean Squared Error (MSE)** (or L2 loss) to measure x vs  $\hat{x}$ . MSE mathematically favors **Low Frequency (Global)** structures.  $\mathscr{D}$ 
> 
> - the skin pores.
> - Error: The pixels are "roughly" correct. The average error is small.
> 
> the z space to ensuring the eyes, nose, and mouth are in the right places.
> 
> - Scenario 2 (Missed Global): The model gets the skin texture perfect, but puts the left eye two inches to the right.
>   - Error: The pixels are wildly wrong (Black pupil where white skin should be). The error is
>      Massive.
> 
> Scenario 1 (Missed Local): The model gets the face shape right but misses the texture of
> 
> Conclusion: The optimizer learns that getting the "Big Stuff" (Global) wrong is penalized much more heavily than getting the "Small Stuff" (Local) wrong. Therefore, the Encoder dedicates
![image.png](DDPM%EF%BC%9ANIPS20/image%2039.png)

> **[图片提取文字 (image.png)]:**
> ## 3. The KL Divergence (Smoothness)
> 
> continuous manifold (The "Gaussian Prior"). It wants similar z vectors to decode into similar images.
> 
> • Global Features are Continuous:
> 
> This is the subtle geometric reason. The VAE tries to organize the latent space z into a smooth,
> 
> - If you slowly change z, you can smoothly morph a "Smile" into a "Frown." The
>   - intermediate images look valid.
>   - This fits the geometry of the Gaussian Prior perfectly.
>   - Local Features are Discontinuous/Chaotic:
>     - High-frequency details (like the exact pattern of static noise or precise edge coordinates) change unpredictably.
> 
> If z encoded "Pixel 45 is white," moving z slightly might make Pixel 45 black. This
> 
> creates a "jagged" latent space that violates the smoothness constraint of the KL
> 
> Divergence.
> 
> Result: The Encoder learns to ignore chaotic local details because trying to encode them
> 
> would break the smooth structure of the latent space.
> 
> ## Summary: The "Blurry" Consequence
> 
> so I will generate a yellow blur."
> 
> Because the Encoder prioritizes **Global Info** (to satisfy the Bottleneck and MSE) and ignores
> 
> Local Info (because it's expensive and discontinuous), the Decoder doesn't know exactly
> 
> where the fine details go.
> 
> So, the Decoder does the statistically safe thing: It outputs the average.
> 
> • "I know there is hair here (Global), but I don't know exactly where the strands are (Local),
> 
> (which don't rely on this bottleneck compression) can generate sharp local details.
> 
> This is why VAEs are famous for producing **coherent but blurry** images, while Diffusion models
![image.png](DDPM%EF%BC%9ANIPS20/image%2040.png)

Q：为什么BackBone称为UNet？

FCN中将不同层次特征**相加**后输出，而UNet中将不同层次特征**拼接**后输出，形状类似**U形**。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](DDPM%EF%BC%9ANIPS20/image%2041.png)