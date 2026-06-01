# DDIM、SDE/ODE Flow：21-23

BackBone UNet，新的采样方法

SDE/ODE模拟积分是指多次采样过程

## NIPS21、ICLR21：Score-Based

NIPS21：Score-based Generative Modeling in Latent Space

ICLR21：Score-based generative modeling through stochastic differential equations

扩散过程的正向采样过程是**正向SDE的模拟积分过程**，图像扩散的结果是正向SDE的积分结果。

扩散过程的逆向采样公式是**逆向SDE的模拟积分过程**，生成图像的结果是逆向SDE的积分结果。

> **[图片提取文字 (image.png)]:**
> Specifically, instead of perturbing data with a finite number of noise distributions, we consider a continuum of distributions that evolve over time according to a diffusion process. This process progressively diffuses a data point into random noise, and is given by a prescribed SDE that does not depend on the data and has no trainable parameters. By reversing this process, we can smoothly mold random noise into data for sample generation. Crucially, this reverse process satisfies a reverse-time SDE (Anderson, 1982), which can be derived from the forward SDE given the score of the marginal probability densities as a function of time. We can therefore approximate the reverse-time SDE by training a time-dependent neural network to estimate the scores, and then produce samples using numerical SDE solvers. Our key idea is summarized in Fig. 1.
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image.png)

**学习数据分布**，根据学到的数据分布生成数据。

扩散过程的逆向采样算法归约到学习**扰动分布p(ε)**的函数，即基于分数的生成式模型。

> **[图片提取文字 (image.png)]:**
> The long-standing goal of likelihood-based generative learning is to faithfully learn a data distribution, while also generating high-quality samples. Achieving these two goals simultaneously is a tremendous challenge, which has led to the development of a plethora of different generative models. Recently, score-based generative models (SGMs) demonstrated astonishing results in terms of both high sample quality and likelihood [1, 2]. These models define a forward diffusion process that maps data to noise by gradually perturbing the input data. Generation corresponds to a reverse process that synthesizes novel data via iterative denoising, starting from random noise. The problem then reduces to learning the score function—the gradient of the log-density—of the perturbed data [3]. In a seminal work, Song et al. [2] show how this modeling approach is described with a stochastic differential equation (SDE) framework which can be converted to maximum likelihood training [4]. Variants of SGMs
> 
> have been applied to images [1, 2, 5, 6], audio [7, 8, 9, 10], graphs [11] and point clouds [12, 13].
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%201.png)

### **随机微分方程SDE的积分求解**

**SDE的正向**$dx=f(x,t)dt+g(t)dw$（随机噪声**w**）是正向采样是，逆向SDE扩散过程被证明是$dx=[f(x,t)-g^2(t)\nabla_xlogp_t(x)]dt+g(t)d{\overline{w}}$。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> generative model. Transforming data to a simple noise distribution can be accomplished with a continuous-time SDE.
> 
> This SDE can be reversed if we know the score of the distribu-
> 
> tion at each intermediate time
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%202.png)

> **[图片提取文字 (image.png)]:**
> Figure 2: **Overview of score-based generative modeling through SDEs**. We can map data to a noise distribution (the prior) with an SDE (Section 3.1), and reverse this SDE for generative modeling (Section 3.2). We can also reverse the associated probability flow ODE (Section 4.3), which yields a
> 
> deterministic process that samples from the same distribution as the SDE. Both the reverse-time SDE
> 
> and probability flow ODE can be obtained by estimating the score  $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$  (Section 3.3).
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%203.png)

使用分数模型$s_{\theta}(x,t)$求解逆向采样过程中每个t的$p(x_t)$的梯度$*∇xlogp_t(x)*$。基于**SDE的求解原理构建模型（DDPM）**。

> **[图片提取文字 (image.png)]:**
> The score of a distribution can be estimated by training a score-based model on samples with score matching (Hyvärinen, 2005; Song et al., 2019a). To estimate  $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ , we can train a time-dependent score-based model  $\mathbf{s}_{\theta}(\mathbf{x},t)$  via a continuous generalization to Eqs. (1) and (3):
> 
> $$\boldsymbol{\theta}^* = \underset{\boldsymbol{\theta}}{\operatorname{arg min}} \mathbb{E}_t \Big\{ \lambda(t) \mathbb{E}_{\mathbf{x}(0)} \mathbb{E}_{\mathbf{x}(t)|\mathbf{x}(0)} \Big[ \left\| \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}(t), t) - \nabla_{\mathbf{x}(t)} \log p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0)) \right\|_2^2 \Big] \Big\}. \tag{7}$$
> Here,  $\mathbf{x} := [0, T]$ ,  $\mathbf{x} := \mathbf{x}$  is a positive weighting function,  $t$  is uniformly sampled over  $[0, T]$ .
> 
> Here  $\lambda: [0,T] \to \mathbb{R}_{>0}$  is a positive weighting function, t is uniformly sampled over [0,T],  $\mathbf{x}(0) \sim p_0(\mathbf{x})$  and  $\mathbf{x}(t) \sim p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0))$ . With sufficient data and model capacity, score matching
> 
> ensures that the optimal solution to Eq. (7), denoted by  $\mathbf{s}_{\theta^*}(\mathbf{x}, t)$ , equals  $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$  for almost all  $\mathbf{x}$  and t. As in SMLD and DDPM, we can typically choose  $\lambda \propto 1/\mathbb{E} \left[ \|\nabla_{\mathbf{x}(t)} \log p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0))\|_{2}^{2} \right]$ . Note that Eq. (7) uses denoising score matching, but other score matching objectives, such as sliced
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%204.png)

### **常微分方程ODE的积分求解**

逆向扩散过程的求解 → 常微分方程**ODE**的概率流模型，将随机噪声w改为**可学习的xt概率项$p_t(x)$**，分数模型预测$∇xlogp_t(x)$，即**Flow方法**。

> **[图片提取文字 (image.png)]:**
> diffusion processes, there exists a corresponding deterministic process whose trajectories share the same marginal probability densities  $\{p_t(\mathbf{x})\}_{t=0}^T$  as the SDE. This deterministic process satisfies an ODE (more details in Appendix D.1):
> 
> Score-based models enable another numerical method for solving the reverse-time SDE. For all
> 
> $$d\mathbf{x} = \left[\mathbf{f}(\mathbf{x}, t) - \frac{1}{2}g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right] dt, \tag{13}$$
>  which can be determined from the SDE once scores are known. We name the ODE in Eq. (13) the *probability flow ODE*. When the score function is approximated by the time-dependent score-based model, which is typically a neural network, this is an example of a neural ODE (Chen et al., 2018).
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%205.png)

## ICLR21：DDIM（高斯采样的ODE Flow：ε-prediction）

ICLR21：Denosing Diffusion Implicit Model

ref：[https://zhuanlan.zhihu.com/p/1900299484749072056](https://zhuanlan.zhihu.com/p/1900299484749072056)

### 和DDPM的关联

*DDPM的核心原理*

$x_{T}→x_1  \color {grey}{→x_{0}}$：构造$q(x_{t-1} | x_t , x_0)$，每步$x_t$采样时，**沿着$x_0$的编码路径$q(x_{t} | x_{t-1})$逆向采样。**

$x_{T}→x_{1} \color {grey} →f_θ$：定义$\{p_θ(x_{t-1} | x_t)=q(x_{t-1} | x_{t}, f_θ) | t=2...T \}$，在$x_t$处采样时，**沿着$x_0$预测值$f_θ(x_t)$的编码路径逆向采样**。

$x_{1}→f_θ$：$p_θ(x_0|x_1)$是$x_{1}$处采样得到$x_{0}$的**高斯采样**，$p_θ(x_0|x_1)$最大，是在修正$x_t$处的对$x_0$的预测值$f_θ(x_t)$。

*DDPM和DDIM的生成过程（左-DDPM，右-DDIM）*

DDPM和DDIM均定义生成采样分布**$p_θ(x_{t-1} | x_t)=q(x_{t-1} | x_t, f_θ)$**。**预测噪声值$ε_θ$**来近似从$x_0$到$x_t$（$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \mathbf{\epsilon_t'}$）的加噪噪声$ε_t'$，计算$f_θ$。

$q(x_{t-1} | x_t, x_0)$的定义不同

> **[图片提取文字 (image.png)]:**
> ence to directly compare  $p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t)$  against forward process posteriors, which are tractable onditioned on  $\mathbf{x}_0$ :
> 
> $$q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\boldsymbol{\beta}}_t \mathbf{I}), \tag{6}$$
> where  $\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) := \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0 + \frac{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t \text{ and } \tilde{\boldsymbol{\beta}}_t := \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \boldsymbol{\beta}_t \tag{7}$
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%206.png)

> **[图片提取文字 (image.png)]:**
> $$q_{\sigma}(\boldsymbol{x}_{T}|\boldsymbol{x}_{0}) = \mathcal{N}(\sqrt{\alpha_{T}}\boldsymbol{x}_{0}, (1 - \alpha_{T})\boldsymbol{I}) \text{ and for all } t > 1,$$
> 
> $$q_{\sigma}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t}, \boldsymbol{x}_{0}) = \mathcal{N}\left(\sqrt{\alpha_{t-1}}\boldsymbol{x}_{0} + \sqrt{1 - \alpha_{t-1} - \sigma_{t}^{2}} \cdot \frac{\boldsymbol{x}_{t} - \sqrt{\alpha_{t}}\boldsymbol{x}_{0}}{\sqrt{1 - \alpha_{t}}}, \sigma_{t}^{2}\boldsymbol{I}\right). \tag{7}$$
> The earn function is chosen to order to ensure that  $q_{\sigma}(\boldsymbol{x}_{t}|\boldsymbol{x}_{0}) = \mathcal{N}(\sqrt{\alpha_{t}}\boldsymbol{x}_{0}, (1 - \alpha_{t})\boldsymbol{I})$  for all
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%207.png)

$p_θ(x_{t-1} | x_t)$的定义不同

> **[图片提取文字 (image.png)]:**
> mean 
> $$\mu_{\theta}(\mathbf{x}_t, t)$$
> , we propose a specific parameterization motivated by the With  $p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_{\theta}(\mathbf{x}_t, t), \sigma_t^2 \mathbf{I})$ , we can write:
> 
>  $L_{t-1} = \mathbb{E}_q \left[ \frac{1}{2\sigma_t^2} \| \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) - \boldsymbol{\mu}_{\theta}(\mathbf{x}_t, t) \|^2 \right] + C$
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%208.png)

> **[图片提取文字 (image.png)]:**
> $$\begin{split} & f_{\theta}^{(t)}(\boldsymbol{x}_t) := (\boldsymbol{x}_t - \sqrt{1 - \alpha_t} \cdot \boldsymbol{\epsilon}_{\theta}^{(t)}(\boldsymbol{x}_t)) / \sqrt{\alpha_t}. \\ \text{e generative process with a fixed prior } & p_{\theta}(\boldsymbol{x}_T) = \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}) \text{ and} \\ & p_{\theta}^{(t)}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t) = \begin{cases} \mathcal{N}(f_{\theta}^{(1)}(\boldsymbol{x}_1), \sigma_1^2 \boldsymbol{I}) & \text{if } t = 1 \\ q_{\sigma}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, f_{\theta}^{(t)}(\boldsymbol{x}_t)) & \text{otherwise,} \end{cases} \end{split}$$
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%209.png)

根据$p_θ(x_{t-1} | x_t)$采样生成的公式不同

**DDIM中，$σ_t=0$的生成**：从$x_1$到$x_T$是**确定的双向映射**，$x_1$到$x_0$之间分别是基于分布$q(x_1 | x_0)$和$p_θ(x_{0} | x_1)$的**采样**。

> **[图片提取文字 (image.png)]:**
> a function approximator intended to predict  $\epsilon$  from  $\mathbf{x}_t$ . To sample
> 
>  $\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_{\theta}(\mathbf{x}_t, t) \right) + \sigma_t \mathbf{z}, \text{ where } \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}).$
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2010.png)

> **[图片提取文字 (image.png)]:**
> From 
> $$p_{\theta}(x_{1:T})$$
>  in Eq. (10), one can generate a sample  $x_{t-1}$  from a sample  $x_t$  via:
> 
> $$\underbrace{\frac{\sqrt{1-\alpha_t}\epsilon_{\theta}^{-}(\boldsymbol{x}_t)}{\sqrt{\alpha_t}}}_{\text{"predicted }\boldsymbol{x}_0"} + \underbrace{\sqrt{1-\alpha_{t-1}-\sigma_t^2\cdot\epsilon_{\theta}^{(t)}(\boldsymbol{x}_t)}}_{\text{"direction pointing to }\boldsymbol{x}_t"} + \underbrace{\sigma_t\epsilon_t}_{\text{random noi}}$$
> 
> $$\sqrt{\alpha_{t-1}} \left( \frac{\boldsymbol{x}_t - \sqrt{1 - \alpha_t} \epsilon_{\theta}^{(t)}(\boldsymbol{x}_t)}{\sqrt{\alpha_t}} \right) + \underbrace{\sqrt{1 - \alpha_{t-1} - \sigma_t^2} \cdot \epsilon_{\theta}^{(t)}(\boldsymbol{x}_t)}_{\text{random noise}} + \underbrace{\sigma_t \epsilon_t}_{\text{random noise}}$$
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2011.png)

*DDPM和DDIM中**$q(x_{t-1} | x_t, x_0)$**的构造区别*

DDPM定义扩散过程$q(x_T : x_1 | x_0)= Πq(x_t | x_{t-1})$。通过$Πq(x_t | x_{t-1}) = q(x_T | x_0) \cdot Πq(x_{t-1} | x_{t}, x_0)$**推导**$x_0$编码序列$x_{1}...x_{T}$的逆向采样方式$q(x_{t-1} | x_t, x_0)$和分布关系$q(x_t | x_0)$。

> 采样生成的方差$σ_t^2 \neq 0$和$x_{1}...x_{T}$的序列关系绑定，生成采样是SDE模拟积分，无法转为ODE模拟积分。

> **[图片提取文字 (image.png)]:**
> ence to directly compare  $p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t)$  against forward process posteriors, which are tractable onditioned on  $\mathbf{x}_0$ :
> 
> $$q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\boldsymbol{\beta}}_t \mathbf{I}), \tag{6}$$
> where  $\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) := \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0 + \frac{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t \text{ and } \tilde{\boldsymbol{\beta}}_t := \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \boldsymbol{\beta}_t \tag{7}$
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%206.png)

DDIM**定义并维持$q_σ(x_1 | x_0)...q_σ(x_T | x_0)$分布**，**加入$x_0$到$x_t$的加噪噪声$ε_t$来指引$x_{t-1}$的采样方向**，来**构造**$x_0$的编码序列$x_{1}...x_{T}$的逆向采样方式$q(x_{t-1} | x_{t}, x_0)$。

> 生成采样的方差$σ_t^2 = 0$不破坏$x_{1}...x_{T}$的序列关系。当$σ_t^2 = 0$时，采样转为ODE积分。

> **[图片提取文字 (image.png)]:**
> $$q_{\sigma}(\boldsymbol{x}_{T}|\boldsymbol{x}_{0}) = \mathcal{N}(\sqrt{\alpha_{T}}\boldsymbol{x}_{0}, (1 - \alpha_{T})\boldsymbol{I}) \text{ and for all } t > 1,$$
> 
> $$q_{\sigma}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t}, \boldsymbol{x}_{0}) = \mathcal{N}\left(\sqrt{\alpha_{t-1}}\boldsymbol{x}_{0} + \sqrt{1 - \alpha_{t-1} - \sigma_{t}^{2}} \cdot \frac{\boldsymbol{x}_{t} - \sqrt{\alpha_{t}}\boldsymbol{x}_{0}}{\sqrt{1 - \alpha_{t}}}, \sigma_{t}^{2}\boldsymbol{I}\right). \tag{7}$$
> The earn function is chosen to order to ensure that  $q_{\sigma}(\boldsymbol{x}_{t}|\boldsymbol{x}_{0}) = \mathcal{N}(\sqrt{\alpha_{t}}\boldsymbol{x}_{0}, (1 - \alpha_{t})\boldsymbol{I})$  for all
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%207.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 1: Graphical models for diffusion (left) and non-Markovian (right) inference models.
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2012.png)

### DDIM训练

分布簇$Q\{q_σ(x_{t-1} | x_t, x_0), …, q_σ(x_{T-1} | x_T, x_0), q_σ(x_T | x_0)\}$定义为生成过程的**Ground Truth**，分布簇Q的构造是$q_σ(x_t | x_0)$**和DDPM扩散过程$q(x_1 : x_T | x_0)$的marginal分布$q(x_t | x_0)$相同**。

> **$q_σ(x_T | x_0)$**是静态参数$α_T$定义的高斯分布，定义了**”起点“$x_T$分布和“终点“$x_0$分布**的关系，$x_T$到$x_0$的“路径”由$q_σ(x_{1} | x_2, x_0)...q_σ(x_{T-1} | x_T, x_0)$定义。

> 超参数**$σ_t$**提供逐步采样$q_σ(x_{t-1} | x_t, x_0)$中的不确定性，对应DDPM中$β_t$。

**训练Loss**

**> $ε_θ(x_t, t)$**预测$x_t$和$x_0$之间的差距**$ε_t$**（和DDPM中**$ε_θ$一致**）。

**>** 根据$x_t$和$ε_t$计算**$x_0$的预测值$f_θ(x_t)$**。

**>** 用$f_θ(x_t)$作为$x_0$，计算**未知$x_0$的**生成过程$p_θ(x_{t-1}| x_t) = q_σ(x_{t-1}| x_t, f_θ)$。

**>** $p_θ(x_{t-1}| x_t)$近似**已知x0时的扩散过程$q_σ(x_1:x_T | x_0)$的逆向采样$q_σ(x_{t-1}| x_t, x_0)$**。

**> 最大化**$p_θ(x_{0}| x_1)$，修正$x_t$处对$x_0$的预测值$f_θ(x_t)$。

**复用DDPM的参数**

**>** DDIM的训练损失J，**等价**于特定$γ_t$取值下DDPM的损失函数L，且和Q中$σ_t$取值无关。

**>** $γ_t$是DDPM中$α_t$和和$β_t$定义的静态参数，特定取值是$*γ_t = 1/(2d \cdot σ_t^2 \cdotα_t)$，*d是$x_0$的维度，据此反解$σ_t$值。

> **[图片提取文字 (image.png)]:**
> Because the generative model approximates the reverse of the inference process, we need to rethink the inference process in order to reduce the number of iterations required by the generative model. Our key observation is that the DDPM objective in the form of  $L_{\gamma}$  only depends on the marginals  $q(\boldsymbol{x}_{t}|\boldsymbol{x}_{0})$ , but not directly on the joint  $q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_{0})$ . Since there are many inference distributions (joints) with the same marginals, we explore alternative inference processes that are non-Markovian, which leads to new generative processes (Figure 1, right). These non-Markovian inference process lead to the same surrogate objective function as DDPM, as we will show below. In Appendix A, we show that the non-Markovian perspective also applies beyond the Gaussian case.
> 
> ## 3.1 Non-Markovian forward processes
> 
> Let us consider a family Q of inference distributions, indexed by a real vector  $\sigma \in \mathbb{R}^T_{\geq 0}$ :
> 
> $$q_{\sigma}(\mathbf{x}_{1:T}|\mathbf{x}_{0}) := q_{\sigma}(\mathbf{x}_{T}|\mathbf{x}_{0}) \prod_{t=2}^{T} q_{\sigma}(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})$$
> (6)
> 
> where  $q_{\sigma}(\boldsymbol{x}_T|\boldsymbol{x}_0) = \mathcal{N}(\sqrt{\alpha_T}\boldsymbol{x}_0, (1-\alpha_T)\boldsymbol{I})$  and for all t > 1,
> 
> $$q_{\sigma}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t},\boldsymbol{x}_{0}) = \mathcal{N}\left(\sqrt{\alpha_{t-1}}\boldsymbol{x}_{0} + \sqrt{1 - \alpha_{t-1} - \sigma_{t}^{2}} \cdot \frac{\boldsymbol{x}_{t} - \sqrt{\alpha_{t}}\boldsymbol{x}_{0}}{\sqrt{1 - \alpha_{t}}}, \sigma_{t}^{2}\boldsymbol{I}\right). \tag{7}$$
> 
> The mean function is chosen to order to ensure that  $q_{\sigma}(x_t|x_0) = \mathcal{N}(\sqrt{\alpha_t}x_0, (1-\alpha_t)I)$  for all t (see Lemma 1 of Appendix B), so that it defines a joint inference distribution that matches the "marginals" as desired. The forward process<sup>3</sup> can be derived from Bayes' rule:
> 
> $$\frac{q_{\sigma}(\mathbf{x}_{t}|\mathbf{x}_{t-1},\mathbf{x}_{0})}{q_{\sigma}(\mathbf{x}_{t-1}|\mathbf{x}_{0})} = \frac{q_{\sigma}(\mathbf{x}_{t-1}|\mathbf{x}_{0},\mathbf{x}_{0})q_{\sigma}(\mathbf{x}_{t}|\mathbf{x}_{0})}{q_{\sigma}(\mathbf{x}_{t-1}|\mathbf{x}_{0})},$$
> (8)
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2013.png)

> **[图片提取文字 (image.png)]:**
> which is also Gaussian (although we do not use this fact for the remainder of this paper). Unlike the diffusion process in Eq. (3), the forward process here is no longer Markovian, since each  $x_t$  could depend on both  $x_{t-1}$  and  $x_0$ . The magnitude of  $\sigma$  controls the how stochastic the forward process is; when  $\sigma \to 0$ , we reach an extreme case where as long as we observe  $x_0$  and  $x_t$  for some t, then  $x_{t-1}$  become known and fixed.
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2014.png)

> **[图片提取文字 (image.png)]:**
> Next, we define a trainable generative process  $p_{\theta}(x_{0:T})$  where each  $p_{\theta}^{(t)}(x_{t-1}|x_t)$  leverages knowledge of  $q_{\sigma}(x_{t-1}|x_t,x_0)$ . Intuitively, given a noisy observation  $x_t$ , we first make a prediction of the corresponding  $x_0$ , and then use it to obtain a sample  $x_{t-1}$  through the reverse conditional distribution  $q_{\sigma}(x_{t-1}|x_t,x_0)$ , which we have defined.
> 
> For some  $x_0 \sim q(x_0)$  and  $\epsilon_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ ,  $x_t$  can be obtained using Eq. (4). The model  $\epsilon_{\theta}^{(t)}(x_t)$  then attempts to predict  $\epsilon_t$  from  $x_t$ , without knowledge of  $x_0$ . By rewriting Eq. (4), one can then predict the *denoised observation*, which is a prediction of  $x_0$  given  $x_t$ :
> 
> $$\frac{\mathbf{f}_{\theta}^{(t)}(\mathbf{x}_t)}{\theta} := (\mathbf{x}_t - \sqrt{1 - \alpha_t} \cdot \epsilon_{\theta}^{(t)}(\mathbf{x}_t)) / \sqrt{\alpha_t}. \tag{9}$$
> 
> We can then define the generative process with a fixed prior  $p_{\theta}(x_T) = \mathcal{N}(0, I)$  and
> 
> $$p_{\theta}^{(t)}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t) = \begin{cases} \mathcal{N}(f_{\theta}^{(1)}(\boldsymbol{x}_1), \sigma_1^2 \boldsymbol{I}) & \text{if } t = 1\\ q_{\sigma}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, f_{\theta}^{(t)}(\boldsymbol{x}_t)) & \text{otherwise,} \end{cases}$$
> (10)
> 
> where  $q_{\sigma}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, f_{\theta}^{(t)}(\boldsymbol{x}_t))$  is defined as in Eq. (7) with  $\boldsymbol{x}_0$  replaced by  $f_{\theta}^{(t)}(\boldsymbol{x}_t)$ . We add some Gaussian noise (with covariance  $\sigma_1^2 \boldsymbol{I}$ ) for the case of t=1 to ensure that the generative process is supported everywhere.
> 
> We optimize  $\theta$  via the following variational inference objective (which is a functional over  $\epsilon_{\theta}$ ):
> 
> $$J_{\sigma}(\epsilon_{\theta}) := \mathbb{E}_{\boldsymbol{x}_{0:T} \sim q_{\sigma}(\boldsymbol{x}_{0:T})} [\log q_{\sigma}(\boldsymbol{x}_{1:T} | \boldsymbol{x}_{0}) - \log p_{\theta}(\boldsymbol{x}_{0:T})]$$
> 
> $$\tag{11}$$
> 
> $$= \mathbb{E}_{\boldsymbol{x}_{0:T} \sim q_{\sigma}(\boldsymbol{x}_{0:T})} \left[ \log q_{\sigma}(\boldsymbol{x}_{T} | \boldsymbol{x}_{0}) + \sum_{t=2}^{T} \log q_{\sigma}(\boldsymbol{x}_{t-1} | \boldsymbol{x}_{t}, \boldsymbol{x}_{0}) - \sum_{t=1}^{T} \log p_{\theta}^{(t)}(\boldsymbol{x}_{t-1} | \boldsymbol{x}_{t}) - \log p_{\theta}(\boldsymbol{x}_{T}) \right]$$
> 
> where we factorize  $q_{\sigma}(x_{1:T}|x_0)$  according to Eq. (6) and  $p_{\theta}(x_{0:T})$  according to Eq. (1).
> 
> From the definition of  $J_{\sigma}$ , it would appear that a different model has to be trained for every choice of  $\sigma$ , since it corresponds to a different variational objective (and a different generative process). However,  $J_{\sigma}$  is equivalent to  $L_{\gamma}$  for certain weights  $\gamma$ , as we show below.
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2015.png)

### DDIM推理和跳步

根据**$x_0$的预测值$f_θ$**计算的生成采样分布**$p_θ(x_{t-1} | x_t)$，**逐步采样，生成图像$x_0$。

$σ_t$仅用于训练时，生成时可以设置$σ_t=0$，**生成采样是确定的，即ODE积分**。

> **[图片提取文字 (image.png)]:**
> process considered in Sohl-Dickstein et al. (2015) and Ho et al. (2020), but also generative processes for many non-Markovian forward processes parametrized by  $\sigma$  that we have described. Therefore, we can essentially use pretrained DDPM models as the solutions to the new objectives, and focus on finding a generative process that is better at producing samples subject to our needs by changing  $\sigma$ .
> 
> With  $L_1$  as the objective, we are not only learning a generative process for the Markovian inference
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2016.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 2: Graphical model for accelerated generation, where  $\tau=[1,3].$
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2017.png)

> **[图片提取文字 (image.png)]:**
> ## 4.1 Denoising Diffusion Implicit Models
> 
> From  $p_{\theta}(x_{1:T})$  in Eq. (10), one can generate a sample  $x_{t-1}$  from a sample  $x_t$  via:
> 
> $$\boldsymbol{x}_{t-1} = \sqrt{\alpha_{t-1}} \underbrace{\left(\frac{\boldsymbol{x}_t - \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_{\theta}^{(t)}(\boldsymbol{x}_t)}{\sqrt{\alpha_t}}\right)}_{\text{"predicted } \boldsymbol{x}_0\text{"}} + \underbrace{\sqrt{1 - \alpha_{t-1} - \sigma_t^2} \cdot \boldsymbol{\epsilon}_{\theta}^{(t)}(\boldsymbol{x}_t)}_{\text{"direction pointing to } \boldsymbol{x}_t\text{"}} + \underbrace{\sigma_t \boldsymbol{\epsilon}_t}_{\text{random noise}}$$
> (12)
> 
> choices of  $\sigma$  values results in different generative processes, all while using the same model  $\epsilon_{\theta}$ , so re-training the model is unnecessary. When  $\sigma_{t} = \sqrt{(1-\alpha_{t-1})/(1-\alpha_{t})}\sqrt{1-\alpha_{t}/\alpha_{t-1}}$  for all t, the forward process becomes Markovian, and the generative process becomes a DDPM.
> 
> where  $\epsilon_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$  is standard Gaussian noise independent of  $x_t$ , and we define  $\alpha_0 := 1$ . Different
> 
> We note another special case when  $\sigma_t = 0$  for all  $t^5$ ; the forward process becomes deterministic given  $x_{t-1}$  and  $x_0$ , except for t=1; in the generative process, the coefficient before the random noise  $\epsilon_t$  becomes zero. The resulting model becomes an implicit probabilistic model (Mohamed & Lakshminarayanan, 2016), where samples are generated from latent variables with a fixed procedure (from  $x_T$  to  $x_0$ ). We name this the *denoising diffusion implicit model* (DDIM, pronounced /d:m/), because it is an implicit probabilistic model trained with the DDPM objective (despite the forward process no longer being a diffusion).
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2018.png)

**中间跳步采样**将T步缩短到S步

通过将T步分成**$τ\{S步采样生成\}$**和$\bar τ\{剩余步变分目标\}$两部分的方式，维持$q_σ(x_t | x_0)$和DDPM marginals一致，**重新定义T步扩散过程$q_σ(x_1:x_T | x_0)$和近似生成过程$p_θ(x_0: x_T)$**。

**>** logp(X)证明：**$τ$部分采样生成样本$x_0$（同上），$\bar τ$部分训练模型在$x_t$处对$x_0$的预测能力（$p_θ(x_t: x_0)$近似$q_σ(x_t | x_0)$）。**

**>** 和上文$q_σ(x_T : x_1 | x_0)= q_σ(x_T | x_0) Πq_σ(x_{t-1} | x_{t}, x_0)$的扩散过程**定义不同**。

**> 训练中T步采样**（$τ$和$\bar τ$两部分采样），但推理时能在中间跳步。

**>** S步-DDIM和DDPM的训练目标一致，可以**复用DDPM的训练好的参数**。

> **[图片提取文字 (image.png)]:**
> ## ACCELERATED SAMPLING PROCESSES
> 
> In the accelerated case, we can consider the inference process to be factored as:
> 
> $$q_{\sigma,\tau}(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0) = q_{\sigma,\tau}(\boldsymbol{x}_{\tau_S}|\boldsymbol{x}_0) \prod_{i=1}^{S} q_{\sigma,\tau}(\boldsymbol{x}_{\tau_{i-1}}|\boldsymbol{x}_{\tau_i},\boldsymbol{x}_0) \prod_{t \in \bar{\tau}} q_{\sigma,\tau}(\boldsymbol{x}_t|\boldsymbol{x}_0)$$
> (52)
> 
> where  $\tau$  is a sub-sequence of  $[1, \ldots, T]$  of length S with  $\tau_S = T$ , and let  $\bar{\tau} := \{1, \ldots, T\} \setminus \tau$  be its complement. Intuitively, the graphical model of  $\{x_{\tau_i}\}_{i=1}^S$  and  $x_0$  form a chain, whereas the graphical model of  $\{x_t\}_{t\in\bar{\tau}}$  and  $x_0$  forms a star graph. We define:
> 
> $$q_{\sigma,\tau}(\boldsymbol{x}_t|\boldsymbol{x}_0) = \mathcal{N}(\sqrt{\alpha_t}\boldsymbol{x}_0, (1-\alpha_t)\boldsymbol{I}) \quad \forall t \in \bar{\tau} \cup \{T\}$$
> (53)
> 
> $$q_{\sigma,\tau}(\boldsymbol{x}_{\tau_{i-1}}|\boldsymbol{x}_{\tau_i},\boldsymbol{x}_0) = \mathcal{N}\left(\sqrt{\alpha_{\tau_{i-1}}}\boldsymbol{x}_0 + \sqrt{1 - \alpha_{\tau_{i-1}} - \sigma_{\tau_i}^2} \cdot \frac{\boldsymbol{x}_{\tau_i} - \sqrt{\alpha_{\tau_i}}\boldsymbol{x}_0}{\sqrt{1 - \alpha_{\tau_i}}}, \sigma_{\tau_i}^2 \boldsymbol{I}\right) \ \forall i \in [S]$$
> 
> where the coefficients are chosen such that:
> 
> $$q_{\sigma,\tau}(\boldsymbol{x}_{\tau_i}|\boldsymbol{x}_0) = \mathcal{N}(\sqrt{\alpha_{\tau_i}}\boldsymbol{x}_0, (1-\alpha_{\tau_i})\boldsymbol{I}) \quad \forall i \in [S]$$
> (54)
> 
> i.e., the "marginals" match.
> 
> The corresponding "generative process" is defined as:
> 
> $$p_{\theta}(\boldsymbol{x}_{0:T}) := \underbrace{p_{\theta}(\boldsymbol{x}_{T}) \prod_{i=1}^{S} p_{\theta}^{(\tau_{i})}(\boldsymbol{x}_{\tau_{i-1}} | \boldsymbol{x}_{\tau_{i}})}_{\text{use to produce samples}} \times \underbrace{\prod_{t \in \bar{\tau}} p_{\theta}^{(t)}(\boldsymbol{x}_{0} | \boldsymbol{x}_{t})}_{\text{in variational objective}}$$
> (55)
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2019.png)

每个生成步骤$t_{τ_{i-1}}$，模型$ε_θ$预测$ε_{t_{τ_{i-1}}}$，计算**$x_0$的预测值$f_\theta$**，带入采样分布$p_\theta$进行采样。

> **[图片提取文字 (image.png)]:**
> where only part of the models are actually being used to produce samples. The conditionals are:  $p_{\theta}^{(\tau_i)}(\boldsymbol{x}_{\tau_{i-1}}|\boldsymbol{x}_{\tau_i}) = q_{\sigma,\tau}(\boldsymbol{x}_{\tau_{i-1}}|\boldsymbol{x}_{\tau_i}, \frac{f_{\theta}^{(\tau_i)}(\boldsymbol{x}_{\tau_{i-1}})}{f_{\theta}^{(\tau_i)}(\boldsymbol{x}_{\tau_{i-1}})})$  if  $i \in [S], i > 1$ (56)
> 
> $$p_{\hat{\boldsymbol{\theta}}} \wedge (\boldsymbol{x}_{\tau_{i-1}} | \boldsymbol{x}_{\tau_i}) = q_{\sigma,\tau}(\boldsymbol{x}_{\tau_{i-1}} | \boldsymbol{x}_{\tau_i}, \boldsymbol{f_{\hat{\boldsymbol{\theta}}}} \wedge (\boldsymbol{x}_{\tau_{i-1}})) \quad \text{if } i \in [S], i > 1$$
> 
> $$p_{\theta}^{(t)}(\boldsymbol{x}_0 | \boldsymbol{x}_t) = \mathcal{N}(f_{\theta}^{(t)}(\boldsymbol{x}_t), \sigma_t^2 \boldsymbol{I}) \quad \text{otherwise},$$
> 
> $$(57)$$
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2020.png)

> **[图片提取文字 (image.png)]:**
> subset  $\{x_{\tau_1}, \dots, x_{\tau_S}\}$ , where  $\tau$  is an increasing sub-sequence of  $[1, \dots, T]$  of length S. In particular, we define the sequential forward process over  $x_{\tau_1}, \dots, x_{\tau_S}$  such that  $q(x_{\tau_i}|x_0) = \mathcal{N}(\sqrt{\alpha_{\tau_i}x_0}, (1-\alpha_{\tau_i})I)$  matches the "marginals" (see Figure 2 for an illustration). The generative process now samples latent variables according to reversed $(\tau)$ , which we term (sampling) trajectory. When the length of the sampling trajectory is much smaller than T, we may achieve significant increases in computational efficiency due to the iterative nature of the sampling process.
> 
> Using a similar argument as in Section 3, we can justify using the model trained with the  $L_1$  objective, so no changes are needed in training. We show that only slight changes to the updates in Eq. (12) are needed to obtain the new, faster generative processes, which applies to DDPM, DDIM,
> 
> as well as all generative processes considered in Eq. (10). We include these details in Appendix C.1.
> 
> In principle, this means that we can train a model with an arbitrary number of forward steps but only
> 
> sample from some of them in the generative process. Therefore, the trained model could consider
> 
> many more steps than what is considered in (Ho et al., 2020) or even a continuous time variable t
> 
> (Chen et al., 2020). We leave empirical investigations of this aspect as future work.
> 
> In the previous sections, the generative process is considered as the approximation to the reverse
> 
> process; since of the forward process has T steps, the generative process is also forced to sample T
> 
> steps. However, as the denoising objective  $L_1$  does not depend on the specific forward procedure
> 
> as long as  $q_{\sigma}(x_t|x_0)$  is fixed, we may also consider forward processes with lengths smaller than T,
> 
> which accelerates the corresponding generative processes without having to train a different model.
> 
> Let us consider the forward process as defined not on all the latent variables  $x_{1:T}$ , but on a
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2021.png)

> **[图片提取文字 (image.png)]:**
> ## ACCELERATED SAMPLING PROCESSES
> 
> In the accelerated case, we can consider the inference process to be factored as:
> 
> $$q_{\sigma,\tau}(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0) = q_{\sigma,\tau}(\boldsymbol{x}_{\tau_S}|\boldsymbol{x}_0) \prod_{i=1}^{S} q_{\sigma,\tau}(\boldsymbol{x}_{\tau_{i-1}}|\boldsymbol{x}_{\tau_i},\boldsymbol{x}_0) \prod_{t \in \bar{\tau}} q_{\sigma,\tau}(\boldsymbol{x}_t|\boldsymbol{x}_0)$$
> (52)
> 
> where  $\tau$  is a sub-sequence of  $[1, \ldots, T]$  of length S with  $\tau_S = T$ , and let  $\bar{\tau} := \{1, \ldots, T\} \setminus \tau$  be its complement. Intuitively, the graphical model of  $\{x_{\tau_i}\}_{i=1}^S$  and  $x_0$  form a chain, whereas the graphical model of  $\{x_t\}_{t\in\bar{\tau}}$  and  $x_0$  forms a star graph. We define:
> 
> $$q_{\sigma,\tau}(\boldsymbol{x}_t|\boldsymbol{x}_0) = \mathcal{N}(\sqrt{\alpha_t}\boldsymbol{x}_0, (1-\alpha_t)\boldsymbol{I}) \quad \forall t \in \bar{\tau} \cup \{T\}$$
> (53)
> 
> $$q_{\sigma,\tau}(\boldsymbol{x}_{\tau_{i-1}}|\boldsymbol{x}_{\tau_i},\boldsymbol{x}_0) = \mathcal{N}\left(\sqrt{\alpha_{\tau_{i-1}}}\boldsymbol{x}_0 + \sqrt{1 - \alpha_{\tau_{i-1}} - \sigma_{\tau_i}^2} \cdot \frac{\boldsymbol{x}_{\tau_i} - \sqrt{\alpha_{\tau_i}}\boldsymbol{x}_0}{\sqrt{1 - \alpha_{\tau_i}}}, \sigma_{\tau_i}^2 \boldsymbol{I}\right) \ \forall i \in [S]$$
> 
> where the coefficients are chosen such that:
> 
> $$q_{\sigma,\tau}(\boldsymbol{x}_{\tau_i}|\boldsymbol{x}_0) = \mathcal{N}(\sqrt{\alpha_{\tau_i}}\boldsymbol{x}_0, (1-\alpha_{\tau_i})\boldsymbol{I}) \quad \forall i \in [S]$$
> (54)
> 
> i.e., the "marginals" match.
> 
> The corresponding "generative process" is defined as:
> 
> $$p_{\theta}(\boldsymbol{x}_{0:T}) := \underbrace{p_{\theta}(\boldsymbol{x}_{T}) \prod_{i=1}^{S} p_{\theta}^{(\tau_{i})}(\boldsymbol{x}_{\tau_{i-1}} | \boldsymbol{x}_{\tau_{i}})}_{\text{use to produce samples}} \times \underbrace{\prod_{t \in \bar{\tau}} p_{\theta}^{(t)}(\boldsymbol{x}_{0} | \boldsymbol{x}_{t})}_{\text{in variational objective}}$$
> (55)
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2019.png)

> **[图片提取文字 (image.png)]:**
> $p_{\theta}^{(\tau_i)}(\boldsymbol{x}_{\tau_{i-1}}|\boldsymbol{x}_{\tau_i}) = q_{\sigma,\tau}(\boldsymbol{x}_{\tau_{i-1}}|\boldsymbol{x}_{\tau_i}, \frac{\mathbf{f}_{\theta}^{(\tau_i)}(\boldsymbol{x}_{\tau_{i-1}})) \quad \text{if } i \in [S], i > 1$ 
> 
> (56)
> 
> (57)
> 
> where only part of the models are actually being used to produce samples. The conditionals are:
> 
> where we leverage 
> $$q_{\sigma,\tau}(x_{\tau_{i-1}}|x_{\tau_i},x_0)$$
>  as part of the inference process (similar to what we have done in Section 3). The resulting variational objective becomes (define  $x_{\tau_{i-1}} = \emptyset$  for conciseness):
> 
>  $p_{\boldsymbol{\alpha}}^{(t)}(\boldsymbol{x}_0|\boldsymbol{x}_t) = \mathcal{N}(f_{\boldsymbol{\alpha}}^{(t)}(\boldsymbol{x}_t), \sigma_t^2 \boldsymbol{I})$  otherwise,
> 
> in Section 3). The resulting variational objective becomes (define 
> $$\boldsymbol{x}_{\tau_{L+1}} = \varnothing$$
>  for conciseness):
> $$J(\epsilon_{\theta}) = \mathbb{E}_{\boldsymbol{x}_{0:T} \sim q_{\sigma,\tau}(\boldsymbol{x}_{0:T})}[\log q_{\sigma,\tau}(\boldsymbol{x}_{1:T}|\boldsymbol{x}_{0}) - \log p_{\theta}(\boldsymbol{x}_{0:T})]$$
> (58)
> 
> $$= \mathbb{E}_{\boldsymbol{x}_{0:T} \sim q_{\sigma,\tau}(\boldsymbol{x}_{0:T})} \left[ \sum_{t \in \bar{\boldsymbol{x}}} D_{\mathrm{KL}}(q_{\sigma,\tau}(\boldsymbol{x}_{t}|\boldsymbol{x}_{0}) || p_{\theta}^{(t)}(\boldsymbol{x}_{0}|\boldsymbol{x}_{t}) \right]$$
> 
> $$(59)$$
> 
> $$\begin{aligned} &\bigsplus_{t \in \bar{\tau}} \ &+ \sum_{i=1}^L D_{\text{KL}}(q_{\sigma,\tau}(\boldsymbol{x}_{\tau_{i-1}}|\boldsymbol{x}_{\tau_i},\boldsymbol{x}_0) \|p_{\theta}^{(\tau_i)}(\boldsymbol{x}_{\tau_{i-1}}|\boldsymbol{x}_{\tau_i}))) \end{bmatrix} \end{aligned}$$
> 
> where each KL divergence is between two Gaussians with variance independent of  $\theta$ . A similar argument to the proof used in Theorem 1 can show that the variational objective J can also be
> 
> converted to an objective of the form  $L_{\gamma}$ .
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2022.png)

### DDIM和ODE Flow的联系

DDIM定义的生成采样公式，是**ODE的从$x_T$到$x_0$的Euler积分**。由于ODE积分的**可逆性**，DDIM的**扩散过程（ODE从$x_0$到$x_T$的积分）可用于编码$x_T$**。

DDIM定义的生成过程，等价于使用**ODE积分**来求解基于分数的SDE积分。

> **[图片提取文字 (image.png)]:**
> ## 4.1 Denoising Diffusion Implicit Models
> 
> From  $p_{\theta}(x_{1:T})$  in Eq. (10), one can generate a sample  $x_{t-1}$  from a sample  $x_t$  via:
> 
> $$\boldsymbol{x}_{t-1} = \sqrt{\alpha_{t-1}} \underbrace{\left(\frac{\boldsymbol{x}_t - \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_{\theta}^{(t)}(\boldsymbol{x}_t)}{\sqrt{\alpha_t}}\right)}_{\text{"predicted } \boldsymbol{x}_0\text{"}} + \underbrace{\sqrt{1 - \alpha_{t-1} - \sigma_t^2} \cdot \boldsymbol{\epsilon}_{\theta}^{(t)}(\boldsymbol{x}_t)}_{\text{"direction pointing to } \boldsymbol{x}_t\text{"}} + \underbrace{\sigma_t \boldsymbol{\epsilon}_t}_{\text{random noise}}$$
> (12)
> 
> choices of  $\sigma$  values results in different generative processes, all while using the same model  $\epsilon_{\theta}$ , so re-training the model is unnecessary. When  $\sigma_{t} = \sqrt{(1-\alpha_{t-1})/(1-\alpha_{t})}\sqrt{1-\alpha_{t}/\alpha_{t-1}}$  for all t, the forward process becomes Markovian, and the generative process becomes a DDPM.
> 
> where  $\epsilon_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$  is standard Gaussian noise independent of  $x_t$ , and we define  $\alpha_0 := 1$ . Different
> 
> We note another special case when  $\sigma_t = 0$  for all  $t^5$ ; the forward process becomes deterministic given  $x_{t-1}$  and  $x_0$ , except for t=1; in the generative process, the coefficient before the random noise  $\epsilon_t$  becomes zero. The resulting model becomes an implicit probabilistic model (Mohamed & Lakshminarayanan, 2016), where samples are generated from latent variables with a fixed procedure (from  $x_T$  to  $x_0$ ). We name this the *denoising diffusion implicit model* (DDIM, pronounced /d:m/), because it is an implicit probabilistic model trained with the DDPM objective (despite the forward process no longer being a diffusion).
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2018.png)

> **[图片提取文字 (image.png)]:**
> Moreover, we can rewrite the DDIM iterate according to Eq. (12), and its similarity to Euler integration for solving ordinary differential equations (ODEs) becomes more apparent:
> 
> $$\frac{\boldsymbol{x}_{t-\Delta t}}{\sqrt{\alpha_{t-\Delta t}}} = \frac{\boldsymbol{x}_t}{\sqrt{\alpha_t}} + \left(\sqrt{\frac{1 - \alpha_{t-\Delta t}}{\alpha_{t-\Delta t}}} - \sqrt{\frac{1 - \alpha_t}{\alpha_t}}\right) \epsilon_{\theta}^{(t)}(\boldsymbol{x}_t)$$
> (13)
> 
> To derive the corresponding ODE, we can reparameterize  $(\sqrt{1-\alpha}/\sqrt{\alpha})$  with  $\sigma$  and  $(x/\sqrt{\alpha})$  with  $\bar{x}$ . In the continuous case,  $\sigma$  and x are functions of t, where  $\sigma: \mathbb{R}_{\geq 0} \to \mathbb{R}_{\geq 0}$  is continuous, increasing with  $\sigma(0) = 0$ . Equation (13) with can be treated as a Euler method over the following ODE:
> 
> $$d\bar{x}(t) = \epsilon_{\theta}^{(t)} \left( \frac{\bar{x}(t)}{\sqrt{\sigma^2 + 1}} \right) d\sigma(t), \tag{14}$$
> 
> where the initial conditions is  $x(T) \sim \mathcal{N}(0, \sigma(T))$  for a very large  $\sigma(T)$  (which corresponds to the case of  $\alpha \approx 0$ ). This suggests that with enough discretization steps, the we can also reverse the generation process (going from t=0 to T), which encodes  $x_0$  to  $x_T$  and simulates the reverse of the ODE in Eq. (14). This suggests that unlike DDPM, we can use DDIM to obtain encodings of the observations (as the form of  $x_T$ ), which might be useful for other downstream applications that requires latent representations of a model.
> 
> In a concurrent work, (Song et al., 2020) proposed a "probability flow ODE" that aims to recover the marginal densities of a stochastic differential equation (SDE) based on scores, from which a similar sampling schedule can be obtained. Here, we state that the our ODE is equivalent to a special case of theirs (which corresponds to a continuous-time analog of DDPM).
> 
> **Proposition 1.** The ODE in Eq. (14) with the optimal model  $\epsilon_{\theta}^{(t)}$  has an equivalent probability flow ODE corresponding to the "Variance-Exploding" SDE in Song et al. (2020).
> 
> We include the proof in Appendix B. While the ODEs are equivalent, the sampling procedures are not, since the Euler method for the probability flow ODE will make the following update:
> 
> $$\frac{\boldsymbol{x}_{t-\Delta t}}{\sqrt{\alpha_{t-\Delta t}}} = \frac{\boldsymbol{x}_t}{\sqrt{\alpha_t}} + \frac{1}{2} \left( \frac{1 - \alpha_{t-\Delta t}}{\alpha_{t-\Delta t}} - \frac{1 - \alpha_t}{\alpha_t} \right) \cdot \sqrt{\frac{\alpha_t}{1 - \alpha_t}} \cdot \epsilon_{\theta}^{(t)}(\boldsymbol{x}_t)$$
> (15)
> 
> which is equivalent to ours if  $\alpha_t$  and  $\alpha_{t-\Delta t}$  are close enough. In fewer sampling steps, however, these choices will make a difference; we take Euler steps with respect to  $d\sigma(t)$  (which depends less directly on the scaling of "time" t) whereas Song et al. (2020) take Euler steps with respect to dt.
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2023.png)

### 实验

配置：使用DDPM模型参数，T的采样步骤子集τ，Q的变分超参$σ_t$，每步一个$σ_t$。

**$σ_t$取非零特定值**$σ_{t1}$**时，DDIM的采样生成$q_σ(x_{t-1} | x_{t}, x_0)$和DDPM的$q(x_{t-1} | x_{t}, x_0)$相同。**

$σ_t$定义成$σ_{t1} * η$，来切换DDPM生成（η=1）和DDIM生成（η=0）过程。

> **[图片提取文字 (image.png)]:**
> ## 4.1 Denoising Diffusion Implicit Models
> 
> From  $p_{\theta}(x_{1:T})$  in Eq. (10), one can generate a sample  $x_{t-1}$  from a sample  $x_t$  via:
> 
> $$\boldsymbol{x}_{t-1} = \sqrt{\alpha_{t-1}} \underbrace{\left(\frac{\boldsymbol{x}_t - \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_{\theta}^{(t)}(\boldsymbol{x}_t)}{\sqrt{\alpha_t}}\right)}_{\text{"predicted } \boldsymbol{x}_0\text{"}} + \underbrace{\sqrt{1 - \alpha_{t-1} - \sigma_t^2} \cdot \boldsymbol{\epsilon}_{\theta}^{(t)}(\boldsymbol{x}_t)}_{\text{"direction pointing to } \boldsymbol{x}_t\text{"}} + \underbrace{\sigma_t \boldsymbol{\epsilon}_t}_{\text{random noise}}$$
> (12)
> 
> choices of  $\sigma$  values results in different generative processes, all while using the same model  $\epsilon_{\theta}$ , so re-training the model is unnecessary. When  $\sigma_{t} = \sqrt{(1-\alpha_{t-1})/(1-\alpha_{t})}\sqrt{1-\alpha_{t}/\alpha_{t-1}}$  for all t, the forward process becomes Markovian, and the generative process becomes a DDPM.
> 
> where  $\epsilon_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$  is standard Gaussian noise independent of  $x_t$ , and we define  $\alpha_0 := 1$ . Different
> 
> We note another special case when  $\sigma_t = 0$  for all  $t^5$ ; the forward process becomes deterministic given  $x_{t-1}$  and  $x_0$ , except for t=1; in the generative process, the coefficient before the random noise  $\epsilon_t$  becomes zero. The resulting model becomes an implicit probabilistic model (Mohamed & Lakshminarayanan, 2016), where samples are generated from latent variables with a fixed procedure (from  $x_T$  to  $x_0$ ). We name this the *denoising diffusion implicit model* (DDIM, pronounced /d:m/), because it is an implicit probabilistic model trained with the DDPM objective (despite the forward process no longer being a diffusion).
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2018.png)

> **[图片提取文字 (image.png)]:**
> In this section, we show that DDIMs outperform DDPMs in terms of image generation when fewer iterations are considered, giving speed ups of  $10\times$  to  $100\times$  over the original DDPM generation process. Moreover, unlike DDPMs, once the initial latent variables  $x_T$  are fixed, DDIMs retain high-level image features regardless of the generation trajectory, so they are able to perform interpolation directly from the latent space. DDIMs can also be used to encode samples that reconstruct them from the latent code, which DDPMs cannot do due to the stochastic sampling process.
> 
> from Eq. (5) with  $\gamma=1$ ; as we argued in Section 3, no changes are needed with regards to the training procedure. The only changes that we make is **how we produce samples from the model**; we achieve this by controlling  $\tau$  (which controls how fast the samples are obtained) and  $\sigma$  (which interpolates between the deterministic DDIM and the stochastic DDPM).
> 
> For each dataset, we use the same trained model with T=1000 and the objective being  $L_{\gamma}$ 
> 
> We consider different sub-sequences  $\tau$  of [1, ..., T] and different variance hyperparameters  $\sigma$  indexed by elements of  $\tau$ . To simplify comparisons, we consider  $\sigma$  with the form:
> 
> $$\sigma_{\tau_i}(\eta) = \eta \sqrt{(1 - \alpha_{\tau_{i-1}})/(1 - \alpha_{\tau_i})} \sqrt{1 - \alpha_{\tau_i}/\alpha_{\tau_{i-1}}},$$
> (16)
> 
> where  $\eta \in \mathbb{R}_{\geq 0}$  is a hyperparameter that we can directly control. This includes an original DDPM generative process when  $\eta=1$  and DDIM when  $\eta=0$ . We also consider DDPM where the random noise has a larger standard deviation than  $\sigma(1)$ , which we denote as  $\hat{\sigma}$ :  $\hat{\sigma}_{\tau_i}=\sqrt{1-\alpha_{\tau_i}/\alpha_{\tau_{i-1}}}$ . This is used by the implementation in Ho et al. (2020) only to obtain the CIFAR10 samples, but not samples of the other datasets. We include more details in Appendix D.
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2024.png)

生成采样ODE的逆向积分，用于编码图像。

> **[图片提取文字 (image.png)]:**
> be said for DDPMs due to their stochastic nature.
> 
> As DDIM is the Euler integration for a particular ODE, it would be interesting to see whether it can encode from  $x_0$  to  $x_T$  (reverse of Eq. (14)) and reconstruct  $x_0$  from the resulting  $x_T$  (forward of Eq. (14))<sup>7</sup>. We consider encoding and decoding on the CIFAR-10 test set with the CIFAR-10
> 
> model with S steps for both encoding and decoding; we report the per-dimension mean squared error (scaled to [0, 1]) in Table 2. Our results show that DDIMs have lower reconstruction error for larger S values and have properties similar to Neural ODEs and normalizing flows. The same cannot
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2025.png)

## ICLR23：ReFlow（直线插值采样的ODE Flow：v-prediction）

ICLR23：Flow straight and fast: Learning to generate and transfer data with rectified flow. 

[https://zhuanlan.zhihu.com/p/603740431](https://zhuanlan.zhihu.com/p/603740431)

### 动机

*扩散过程和生成过程本质是传输映射*

DDPM和DDIM中**$x_0$的扩散过程$q(x_{T} : x_1 | x_0)$将分布$q(x_0)$映射到分布$q(x_T)$，$x_0$的近似生成过程$p_θ(x_0 :x_T)$将分布$q(x_T)$映射到分布$q(x_0)$。**

DDPM和DDIM**人为构造$q(x_{t-1} | x_t , x_0)$来参数化定义可学习的$p_θ(x_{t-1} | x_t)$**，即$p_θ(x_{t-1} | x_t)=q(x_{t-1} | x_{t}, f_θ)$，模型预测$f_θ$或$ε_θ$，从$x_T$根据$p_θ(x_{t-1} | x_t)$迭代采样到$x_0$。

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
> ## **Generative Models**
> 
> ![](_page_0_Picture_1.jpeg)
> 
> ![](_page_0_Picture_2.jpeg)
> 
> ![](_page_0_Picture_3.jpeg)
> 
> 我们先定义好要解决的问题。无论是从噪声生成图片(generative modeling),还是将人脸转化为猫脸 (domain transfer),都可以这样概括成将一个分布转化成另一个分布的问题:
> 
> 给定从两个分布  $\pi_0$  和  $\pi_1$  中的采样,我们希望找到一个传输映射 T 使得,当  $Z_0\sim\pi_0$  时, $Z_1=T(Z_0)\sim\pi_1$ 。
> 
> 比如,在生成模型里,  $Z_0 \sim \pi_0$ 是高斯噪声分布,  $\pi_1$ 是数据的分布(比如图片),我们想找到一个方法,把噪声  $Z_0$ 映射成一个服从  $\pi_1$  的数据  $Z_1$ 。在数据迁移 (domain transfer)里,  $Z_0$ , $Z_1$ 分别是人脸和猫脸的图片。所以这个问题是生成模型和数据迁移的统一表述。
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2026.png)

*预测ODE函数：v-prediction，即$z_t$的差分预测*

DDPM/DDIM中，采样生成公式$p_θ(x_{t-1} |x_t)$定义了$x_t$关于t的微分公式。**$x_1$采样生成$x_0$，是$p_θ(x_{t-1} |x_t)$定义的ODE微分从0到T积分的模拟求解**。

ReFlow将**$p_θ(x_{t-1} | x_t)$抽象成可学习的$v(z_t, t)$，即$\frac{dZ_t}{dt}=v(Z_t,t)$**，将$x_T$根据$p_θ(x_{t-1} |x_t)$迭代采样到$x_0$的过程抽象成**$v$从0到1的积分，即$Z_{t+ε}=Z_t+εv(Z_t, t)$**，$Z_1$是图像$x_0$，$Z_0$是编码$x_T$。

> **[图片提取文字 (image.png)]:**
> Moreover, we can rewrite the DDIM iterate according to Eq. (12), and its similarity to Euler integration for solving ordinary differential equations (ODEs) becomes more apparent:
> 
> $$\frac{\boldsymbol{x}_{t-\Delta t}}{\sqrt{\alpha_{t-\Delta t}}} = \frac{\boldsymbol{x}_t}{\sqrt{\alpha_t}} + \left(\sqrt{\frac{1 - \alpha_{t-\Delta t}}{\alpha_{t-\Delta t}}} - \sqrt{\frac{1 - \alpha_t}{\alpha_t}}\right) \epsilon_{\theta}^{(t)}(\boldsymbol{x}_t)$$
> (13)
> 
> To derive the corresponding ODE, we can reparameterize  $(\sqrt{1-\alpha}/\sqrt{\alpha})$  with  $\sigma$  and  $(x/\sqrt{\alpha})$  with  $\bar{x}$ . In the continuous case,  $\sigma$  and x are functions of t, where  $\sigma: \mathbb{R}_{>0} \to \mathbb{R}_{>0}$  is continuous, increasing
> 
> with 
> $$\sigma(0) = 0$$
> . Equation (13) with can be treated as a Euler method over the following ODE: 
> $$d\bar{\boldsymbol{x}}(t) = \epsilon_{\theta}^{(t)} \left( \frac{\bar{\boldsymbol{x}}(t)}{\sqrt{\sigma^2 + 1}} \right) d\sigma(t), \tag{14}$$
>  where the initial conditions is  $\boldsymbol{x}(T) \sim \mathcal{N}(0, \sigma(T))$  for a very large  $\sigma(T)$  (which corresponds to the
> 
> case of  $\alpha \approx 0$ ). This suggests that with enough discretization steps, the we can also reverse the generation process (going from t=0 to T), which encodes  $x_0$  to  $x_T$  and simulates the reverse of the ODE in Eq. (14). This suggests that unlike DDPM, we can use DDIM to obtain encodings of
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2027.png)

> **[图片提取文字 (image.png)]:**
> 在我们的框架下,映射 T是通过以下连续运动系统,也就是一个常微分方程(ordinary differential equation (ODE)),或者叫流模型(flow),来隐式定义的:  $\frac{d}{dt}Z_t=v(Z_t,t),\;\;Z_0\sim\pi_0,\;\forall t\in[0,1].$ 我们可以想象从  $\pi_0$ 里采样出来的  $Z_0$ 是一个粒子。它从 t=0时刻开始连续运动,在 t 时刻 以  $v(Z_t,t)$  为速度。直到 t=1时刻得到  $Z_1$ 。我们希望  $Z_1$  服从分布  $\pi_1$  。这里我们假设  $v(Z_t,t)$ 是一个神经网络。我们的任务是从数据里学习出 $v(Z_t,t)$ 来达到 $Z_1 \sim \pi_1$ 的目的。
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2028.png)

*“直线”的传输映射*

DDPM和DDIM预测$x_t$处关于$x_0$的采样方向$ε_θ$，ODE/SDE积分的数值求解器在**生成时模拟积分**，完成**$x_{T}→x_{1} →f_θ$**的过程。

> DDPM的$v(x_t, t)$是$q(x_{t-1} | x_t , x_0)$的参数化表达，$q(x_{t-1} | x_t , x_0)$由马尔可夫过程$q(x_{t-1} | x_{t-1})$推导得到，采样步数长且无法跳步。

> DDIM的$v(x_t, t)$是$q(x_{t-1} | x_t , x_0)$的参数化表达，$q(x_{t-1} | x_t , x_0)$是根据预先定义$q(x_{t} | x_0)$后**人为构造**，**根据自定义的采样步数S，可以构造$v$的不同参数化表达**。

ReFlow希望将**$v(Z_t, t)$**构造成从样本$Z_0$到样本$Z_1$的**直线映射**，而非DDIM/DDPM中的多步采样完成的映射。$Z*$表示直线采样方式，$X*$表示不一定直线但是有效的采样方法。

> **[图片提取文字 (image.png)]:**
> ## 走直线,走得快
> 
>  $Z_{t+\epsilon} = Z_t + \epsilon v(Z_t, t),$ 
> 
> 除了希望  $Z_1 \sim \pi_1$ ,我们还希望这个连续运动系统能够在计算机里快速地模拟出来。注意到,在实际计算过程中,上面的连续系统通常是用 Euler 法(或其变种)在离散化的时间上近似:
> 
> 这里  $\epsilon$  是一个步长参数。我们需要适当的选择  $\epsilon$  来平衡速度和精度:  $\epsilon$ 需要足够小来保证近
> 
> 似的精度,但同时小的  $\epsilon$ 意味着我们从 t=0到 t=1要跑很多步,速度就慢。
> 
> 那么问题来了,什么样的系统能最快地用 Euler 法来模拟呢?也就是说,什么样的体系能允许我们在用较大的步长  $\epsilon$ 的同时还能得到很好的精度呢?
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2029.png)

> **[图片提取文字 (image.png)]:**
> 答案是"走直线"。如下图所示,如果粒子的运动轨迹是弯曲的,我们需要很细的离散化来 得到很好的结果。如果粒子的轨迹是直线,那么即使我们取最大的步长( $\epsilon=1$ ),只用一步走 到 t=1 时刻, 还是能得到正确的结果! 所以,我们希望我们学习出来的速度模型v既能保证  $Z_1 \sim \pi_1$ ,又能给出尽量直的轨迹。怎么同时实现这两个目的在数学上是一个非常不简单 (non-trivial)的问题,涉及最优传输(optimal transport)的一些深刻理论。但是我们发现 其实可以用一个非常简单的方法来解决这个问题。
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2030.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2031.png)

### Rectified Flow方法

**配对Coupling，类似DDPM中图像$x_0$和$x_0$用$q(x_{t} | x_{t-1})$采样得到的编码$x_T$，或者编码$z_T$和$z_T$用$q(z_{t-1} | z_{t}, z_0)$采样生成的图像$z_0$。因为$z_T→z_0$的采样“路径”$q(z_{t-1} | z_{t}, z_0)$是$z_0→z_T$的采样路径逆向构造的，$z_t$的采样几乎不可能“交叉”，和$v(Z_t, t)$对类似$q(z_{t-1} | z_{t}, z_0)$路径的近似程度相关。**

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
>  $X_t = tX_1 + (1-t)X_0$  induced by  $(X_0, X_1)$   $Z_t = tZ_1 + (1-t)Z_0$  induced by  $(Z_0, Z_1)$ Figure 2: (a) Linear interpolation of data input  $(X_0, X_1) \sim \pi_0 \times \pi_1$ . (b) The rectified flow  $Z_t$  induced by  $(X_0, X_1)$ ; the trajectories are "rewired" at the intersection points to avoid the crossing. (c) The linear interpolation of the end
> 
> points  $(Z_0, Z_1)$  of flow  $Z_t$ . (d) The rectified flow induced from  $(Z_0, Z_1)$ , which follows straight paths.
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2032.png)

*1-ReFlow：$v(X_t, t)$*

**希望任意步迭代采样的结果$X_t$是$Π_0$的样本$X_0$到$Π_1$的样本$X_1$所定义的直线上的点**，定义$X_t = tX_1 +(1-t)X_0$， 则$\frac{dX_t}{dt} = X_1 -X_0$。但$X_1 -X_0$在生成时不可解，因为$X_1$是$X_0$采样的目标结果，不可能提前知道。

类似DDIM中，$q(x_0 | x_{T} : x_1 )$不可解，定义$p_θ(x_{0} : x_T)$近似$q(x_0 | x_{T} : x_1 )$，将$p_θ(x_{t-1} | x_t)$参数化为$x_0$的编码序列$x_{1}...x_{T}$的逆向采样方式$q(x_{t-1} | x_{t}, f_0)$并学习$q(x_{t-1} | x_{t}, x_0)$。

对已知的$Π_0$分布的样本$X_0$和$Π_1$分布的样本$X_1$，定义$v(X_t, t)$学习$X_1 -X_0$，其中，$X_t = tX_1 +(1-t)X_0$。

*2-ReFlow：$v(Z_t, t)$*

$v(X_t, t)$的采样方式近似了$X_1 -X_0$的采样方式，从而学到了分布$Π_0$到分布$Π_1$的**“不交叉”的采样方式**，类似于学到了$q(x_{t-1} | x_{t}, x_0)$。

令$Z_0$是分布$Π_0$的采样，$Z_1 = Flow(v(X_t, t), Z_0)$是生成的图像，则$Z_0$和$Z_1$是建立了**“不交叉”采样的配对样本，类似DDPM中$q(x_{t-1} | x_{t}, x_0)$和$q(x_{t} | x_{t-1})$配对的编码$x_T$和图像$x_0$**。

定义$v(Z_t, t)$学习$Z_1 -Z_0$，其中，$Z_t = tZ_1 +(1-t)Z_0$。

> **[图片提取文字 (image.png)]:**
> ## Rectified Flow-基于直线 ODE 学习生成模型
> 
> 假设我们有从两个分布中的采样  $X_0 \sim \pi_0$ ,  $X_1 \sim \pi_1$  (比如  $X_0$  是从  $\pi_0$  里出来的随机噪声,  $X_1$ 是一个随机的数据(服从  $\pi_1$ ))。我们把  $X_0$  和  $X_1$  用一个线性插值连接起来,得到
> 
> $$X_t = tX_1 + (1-t)X_0, \ \ t \in [0,1].$$
> 
> 这里  $X_0$  和  $X_1$  是随机,或者说,以任意方式配对的。你也许觉得  $X_0$  和  $X_1$ 应该用一种有意义的方式配对好,这样能够得到更好的效果。我们先忽略这个问题,待会回来解决它。
> 
> 现在,如果我们拿  $X_t$  对时间 t 求导,我们其实已经可以得到一个能够将数据从  $X_0 \sim \pi_0$  传输到  $X_1 \sim \pi_1$ 的"ODE"了,
> 
> $$\frac{d}{dt}X_t=X_1-X_0, ~~\forall t\in [0,1].$$
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2033.png)

> **[图片提取文字 (image.png)]:**
> 但是,这个"ODE"并不实用而且很奇怪,所以要打个引号:它**不是**一个"因果" (causal),或者"可前向模拟" (forward simulatable)的系统,因为要计算  $X_t$  在 t 时刻的速度  $(X_1-X_0)$ 需要提前(在 t<1时)知道 ODE 轨迹的终点  $X_1$ 。如果我们都已经知道  $X_1$ 了,那其实也就没有必要模拟 ODE 了。
> 
> 那么我们能不能学习 v ,使得我们想要的"可前向模拟"的 ODE  $\frac{d}{dt}Z_t=v(Z_t,t)$ 能尽可能逼近刚才这个"不可前向模拟"的过程呢?最简单的方法就是优化 v 来最小化这两个系统的速度函数(分别是 v 和  $X_1-X_0$  )之间的平方误差:
> 
> $$\min_v \int_0^1 \mathbb{E}_{X_0 \sim \pi_0, X_1 \sim \pi_1} \left[ ||(X_1 - X_0) - v(X_t, t)||^2 \right] dt, \;\; \text{where} \;\; X_t = t X_1 \ + (1 - t) X_0.$$
> 
> 这是一个标准的优化任务。我们可以将v设置成一个神经网络,并用随机梯度下降或者Adam来优化,进而得到我们的可模拟ODE模型。
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2034.png)

> **[图片提取文字 (image.png)]:**
> **图(a)**: 在我们用直线连接  $X_0$  和  $X_1$  时,有些线会在中间的地方相交,这是导致  $\frac{d}{dt}X_t=X_1-X_0$  非因果的原因(在交叉点, $X_t$  既可以沿蓝线走,也可以沿绿线走,因此 粒子不知该向岔路的哪边走)。 **图(b)**:我们学习出的ODE因为必须是因果的,所以不能出现道路相交的情况,它会在原来相 交的地方把道路交换成不交叉的形式。这样,我们学习出来的ODE仍然保留了原来的基本路 径,但是做了一个重组来避免相交的情况。这样的结果是,图(a)和图(b)里的系统在每个时刻 *t*的边际分布是一样的,即使总体的路径不一样。
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2035.png)

> **[图片提取文字 (image.png)]:**
> ## Reflow-拉直轨迹,一步生成
> 
> 因为Rectified Flow要在直线轨迹的交叉点做路径重组,所以上面的ODE模型(或者说 flow)的轨迹仍然可能是弯曲的 (如上面的图(b)),不能达到一步生成。我们提出一个"**Reflow**"方法,将ODE的轨迹进一步变直。
> 
> 具体的做法非常简单: 假设我们从  $\pi_0$  里采样出一批  $X_0$  。然后,从  $X_0$  出发,我们模拟上面学出的flow(叫它1-Rectified Flow),得到  $X_1=\mathrm{Flow}_1(X_0)$  。我们用这样得到的  $(X_0,X_1)$  对来学一个新的"2-Rectified Flow":
> 
> $$\min_v \int_0^1 \mathbb{E}_{X_0 \sim \pi_0, X_1 = \mathrm{Flow}_1(X_0)} \left[ \left| \left| (X_1 - X_0) - v(X_t, t) \right| \right|^2 \right] dt, \quad \text{with} \quad X_t = t X_1 \ + (1 - t) X_0.$$
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2036.png)

> **[图片提取文字 (image.png)]:**
> ## **Algorithm 1** Rectified Flow: Main Algorithm
> 
> **Procedure**:  $Z = RectFlow((X_0, X_1))$ :
> 
> *Inputs*: Draws from a coupling  $(X_0, X_1)$  of  $\pi_0$  and  $\pi_1$ ; velocity model  $v_\theta \colon \mathbb{R}^d \to \mathbb{R}^d$  with parameter  $\theta$ .
> 
> Training: 
> $$\hat{\theta} = \underset{\alpha}{\operatorname{arg \, min}} \mathbb{E} \left[ \|X_1 - X_0 - v(tX_1 + (1-t)X_0, t)\|^2 \right]$$
> , with  $t \sim \operatorname{Uniform}([0, 1])$ .
> 
> Sampling: Draw  $(Z_0, Z_1)$  following  $dZ_t = v_{\hat{\theta}}(Z_t, t) dt$  starting from  $Z_0 \sim \pi_0$  (or backwardly  $Z_1 \sim \pi_1$ ). Return:  $\mathbf{Z} = \{Z_t : t \in [0, 1]\}$ .
> 
> **Reflow** (optional):  $\mathbf{Z}^{k+1} = \text{RectFlow}((Z_0^k, Z_1^k))$ , starting from  $(Z_0^0, Z_1^0) = (X_0, X_1)$ .
> 
> **Distill** (optional): Learn a neural network  $\hat{T}$  to distill the k-rectified flow, such that  $Z_1^k \approx \hat{T}(Z_0^k)$ .
> 
> First, for a given input coupling  $(X_0, X_1)$ , it is easy to see that the exact minimum of (1) is achieved if
> 
> $$v^{X}(x,t) = \mathbb{E}[X_1 - X_0 \mid X_t = x], \tag{2}$$
> 
> which is the expectation of the line directions  $X_1 - X_0$  that pass through x at time t. We discuss below the property of rectified flow  $dZ_t = v^X(Z_t, t)dt$  with  $Z_0 \sim \pi_0$ , assuming that the ODE has an unique solution.
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2037.png)

ReFlow的作用是将编码分布$X_0$和样本分布$X_1$之间**寻找直线映射**，从而一步或少数步采样生成。

Distillation在现有的采样方式基础上，学习更少步数的近似采样方式。

> **[图片提取文字 (image.png)]:**
> ## Reflow与Distillation
> 
> 给定一个配对  $(X_0,X_1)$  ,要想实现一步生成,也就是  $X_1 \approx X_0 + v(X_0,0)$  , 我们好像也可以通过优化下面的平方误差来直接"蒸馏(distillation)"出一个一步模型 :
> 
> $$\min_v \mathbb{E}\left[\left|\left|X_1-X_0-v(X_0,0)\right|\right|^2\right].$$
> 
> 这个目标函数和上面的 Reflow 的目标函数很像,只是把所有的时间 t 都设成 t=0 了。
> 
> 尽管如此,Distillation和 Reflow是**有本质的区别**的。Distillation试图—五一十地复现  $(X_0,X_1)$  配对的关系。但是,如果  $(X_0,X_1)$  的配对是随机的,Distillation最多只能得到  $X_1$  在给定  $X_0$  时的条件平均,也就是  $\mathbb{E}[X_1|X_0] \approx X_0 + v(X_0,0)$  ,并不能成功地完全匹配  $X_1 \sim \pi_1$ 。即使  $(X_0,X_1)$  有确定的——对应关系,他们的配对关系也可能很复杂,导致直接蒸馏很困难。
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2038.png)

> **[图片提取文字 (image.png)]:**
> Reflow解决了Distillation的这些困难。它的意义在于:
> 
> - 1) 给定任何 $(X_0,X_1)$ 配对,就算是随机的配对,他都能学出一个给出正确边际分布 (marginal distribution)的 flow。Reflow不会去试图完全复现  $(X_0,X_1)$ 的配对关系,而只注重于得到正确的边际分布。
> - 2) 从 Reflow 出的 ODE 里采样,我们还可以得到一个更好的配对  $(X_0,X_1')$  ,从而给出更好的 flow。 重复这个过程可以最终得到保证一步生成的直线 ODE。
> 
> 形象地来讲,如果  $(X_0,X_1)$  太复杂,Reflow会"拒绝"完全复现  $(X_0,X_1)$ ,转而给出一个新的,更简单的,但仍然满足  $X_1'\sim\pi_1$  的配对  $(X_0,X_1')$ 。 所以,Distillation 更像"模仿者",只会机械地模仿,就算问题无解也要"硬做"。Reflow 更像"创造者",懂得变通,发现新方法来解决问题。
> 
> 当然,Reflow和 Distillation 也可以组合使用:先用 Reflow 得到比较好的配对,最后再用已经很好的配对进行 Distillation。我们在论文里发现,这个结合的策略确实有用。
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2039.png)

> **[图片提取文字 (image.png)]:**
> 下面,我们进一步基于具体例子解释一下Reflow对配对的提高效果。如果一个配对  $(X_0,X_1)$ 是好的,那么从这个配对里随机产生的两条直线  $X_t=tX_1+(1-t)X_0$  就不 会相交。在我们的论文里,这种直线不相交的配对我们叫做"Straight Coupling"。我们的 Reflow过程就是在不停地降低这个相交概率的过程。下图我们展示随着 Reflow的不断进行, 配对的直线交叉数确实逐渐降低。在图中,对每种配对方法,我们随机选择两个配对,分别 用直线段连接它们,然后若它们相交,就用红色点标出这两条直线段的交点。 对于这种交叉 的配对, Reflow就有可能改善它们。我们重复10000次并统计交叉的概率。我们发现: (1) 每次Reflow都降低了交叉的概率和L2传输代价(2)即使2-Rectified Flow在肉眼观 察时已经很直,但它的交叉概率仍不为0,更多的Reflow次数就可能进一步使它变直并降低 传输代价。相比之下,单纯的蒸馏是不能改善配对的,这是Reflow与蒸馏的本质区别。
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2040.png)

> **[图片提取文字 (image.png)]:**
> 2-Rectified Flow配对: 1-Rectified Flow配对: 随机配对:  $X_0 \sim \pi_0, X_1 = \text{Flow}_1(X_0) \sim \pi_1$  $X_0 \sim \pi_0, X_1 = \text{Flow}_2(X_0) \sim \pi_1$  $X_0 \sim \pi_0, X_1 \sim \pi_1$ 80 80 00 BB 0 交叉概率:46.39% 交叉概率:0.76% 交叉概率 [: 0.14% CLiu] 传输代价(L2): 20.14 专输代价(L2): 21.20 传输代价(L2): 20.16 图中,每个红点代表一次两随机的直线交叉的事件。随着 reflow,交叉的概率逐渐降低,对应的 ODE的轨迹也越来越直。
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2041.png)

### 实验

递归获取1-ReFlow…K-ReFlow，蒸馏K-ReFlow，使用K-ReFlow单步生成$Z_1$。

> **[图片提取文字 (image.png)]:**
> **Algorithm** We follow the procedure in Algorithm 1. We start with drawing  $(X_0, X_1) \sim \pi_0 \times \pi_1$  and
> 
> use it to get the first rectified flow  $Z^1$  by minimizing (1). The second rectified flow  $Z^2$  is obtained by the
> 
> same procedure except with the data replaced by the draws from  $(Z_0^1, Z_1^1)$ , obtained by simulating the first
> 
> rectified flow  $Z^1$ . This process is repeated for k times to get the k-rectified flow  $Z^k$ . Finally, we can further
> 
> distill the k-rectified flow  $\mathbb{Z}^k$  into a one step model  $z_1 = z_0 + v(z_0, 0)$  by fitting it on draws from  $(\mathbb{Z}_0^k, \mathbb{Z}_1^k)$ .
> 
> By default, the ODEs are simulated using the vanilla Euler method with constant step size 1/N for N steps,
> 
> that is,  $\hat{Z}_{t+1/N} = \hat{Z}_t + v(\hat{Z}_t, t)/N$  for  $t \in \{0, \dots, N\}/N$ . We use the Runge-Kutta method of order 5(4)
> 
> from Scipy [86], denoted as RK45, which adaptively decide the step size and number of steps N based on
> 
> user-specified relative and absolute tolerances. In our experiments, we stick to the same parameters as [73].
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2042.png)

Example

> **[图片提取文字 (image.png)]:**
> ## **Toy Examples**
> 
> To accurately illustrate the theoretical properties, we use the non-parametric estimator  $v^{X,h}(z,t)$  in (5) in the toy examples in Figure 2, 3, 4, 5. In practice, we approximate the expectation in (5) an nearest neighbor
> 
> estimator: given a sample  $\{x_0^{(i)}, x_1^{(i)}\}_i$  drawn from  $(X_0, X_1)$ , we estimate  $v^X$  by  $v^{X,h}(z,t) \approx \sum_{i \in \text{knn}(z,m)} \frac{x_1^{(i)} - z}{1 - t} \omega_h(x_t^{(i)}, z) \ / \sum_{i \in \text{knn}(z,m)} \omega_h(x_t^{(i)}, z), \qquad x_t^{(i)} = t x_1^{(i)} + (1 - t) x_0^{(i)},$ 
> 
> $$i \in \text{knn}(z,m)$$
>   $1-t$   $i \in \text{knn}(z,m)$  where  $\text{knn}(z,m)$  denotes the top  $m$  nearest neighbors of  $z$  in  $\{x_t^{(i)}\}_i$ . We find that the results are not
> 
> where knn(z, m) denotes the top m nearest neighbors of z in  $\{x_t^{(i)}\}_i$ . We find that the results are not sensitive to the choice of m and the bandwidth h (see Figure 7). We use h = 1 and m = 100 by default.
> 
> The flows are simulated using Euler method with a constant step size of 1/N for N steps. We use N=100 steps unless otherwise specified.
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2043.png)

> **[图片提取文字 (image.png)]:**
> ## **5.2** Unconditioned Image Generation
> 
> reference. More detailed settings can be found in the Appendix.
> 
> datasets. The methods are evaluated by the quality of generated images by Fréchet inception distance (FID) and inception score (IS), and the diversity of the generated images by the recall score following [38].
> 
> We test rectified flow for unconditioned image generation on CIAFR-10 and a number of high resolution
> 
> **Experiment settings** For the purpose of generative modeling, we set  $\pi_0$  to be the standard Gaussian distribution and  $\pi_1$  the data distribution. Our implementation of rectified flow is modified upon the open-
> 
> source code of [73]. We adopt the U-Net architecture of DDPM++ [73] for representing the drift  $v^X$ , and report in Table 1 (a) and Figure 8 the results of our method and the (sub)-VP ODE from [73] using the same architecture. Other recent results using different network architectures are shown in Table 1 (b) for
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2044.png)

## ICLR22：DDIM Distilling（DDIM Teacher，x-prediction）

ICLR22：Progressive distillation for fast sampling of diffusion models 

**distilling策略**

标准DDIM作为Teacher在生成过程中，每步$z_t$预测$x_0$的值进行采样，得到$z_{t-\epsilon}$。

Student DDIM在生成过程中，每步$z_t'$直接学习Teacher连续两步采样的结果$z_{t-2\epsilon}$。

Student DDIM蒸馏后极限情况，由$z_1$直接生成$x_0$。

> **[图片提取文字 (image.png)]:**
> To make diffusion models more efficient at sampling time, we propose *progressive distillation*: an algorithm that iteratively halves the number of required sampling steps by distilling a slow teacher diffusion model into a faster student model. Our implementation of progressive distillation stays very close to the implementation for training the original diffusion model, as described by e.g. Ho et al. (2020). Algorithm 1 and Algorithm 2 present diffusion model training and progressive distillation side-by-side, with the relative changes in progressive distillation highlighted in green. We start the progressive distillation procedure with a teacher diffusion model that is obtained by
> 
> training in the standard way. At every iteration of progressive distillation, we then initialize the student model with a copy of the teacher, using both the same parameters and same model definition. Like in standard training, we then sample data from the training set and add noise to it, before forming the training loss by applying the student denoising model to this noisy data  $z_t$ . The main difference in progressive distillation is in how we set the target for the denoising model: instead of the original data x, we have the student model denoise towards a target  $\tilde{x}$  that makes a single student DDIM step match 2 teacher DDIM steps. We calculate this target value by running 2 DDIM sampling steps using the teacher, starting from  $\mathbf{z}_t$  and ending at  $\mathbf{z}_{t-1/N}$ , with N being the number of student sampling steps. By inverting a single step of DDIM, we then calculate the value the student model would need to predict in order to move from  $\mathbf{z}_t$  to  $\mathbf{z}_{t-1/N}$  in a single step, as we show in detail in Appendix G. The resulting target value  $\tilde{\mathbf{x}}(\mathbf{z}_t)$  is fully determined given the teacher model and starting point  $\mathbf{z}_t$ , which allows the student model to make a sharp prediction when evaluated at  $z_t$ . In contrast, the original data point x is not fully determined given  $z_t$ , since multiple different data points x can produce the same noisy data  $z_t$ : this means that the original denoising model is
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2045.png)

> **[图片提取文字 (image.png)]:**
> sharper predictions, the student model can make faster progress during sampling. After running distillation to learn a student model taking N sampling steps, we can repeat the procedure with N/2 steps: The student model then becomes the new teacher, and a new student model
> 
> is initialized by making a copy of this model.
> 
> predicting a weighted average of possible x values, which produces a blurry prediction. By making
> 
> Unlike our procedure for training the original model, we always run progressive distillation in discrete time: we sample this discrete time such that the highest time index corresponds to a signal-to-noise ratio of zero, i.e.  $\alpha_1 = 0$ , which exactly matches the distribution of input noise  $\mathbf{z}_1 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 
> 
> noise ratio of zero, i.e.  $\alpha_1=0$ , which exactly matches the distribution of input noise  $\mathbf{z}_1 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$  that is used at test time. We found this to work slightly better than starting from a non-zero signal-to-noise ratio as used by e.g. Ho et al. (2020), both for training the original model as well as when performing progressive distillation.
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2046.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 1: A visualization of two iterations of our proposed *progressive distillation* algorithm. A sampler  $f(\mathbf{z}; \eta)$ , mapping random noise  $\epsilon$  to samples  $\mathbf{x}$  in 4 deterministic steps, is distilled into a new sampler  $f(\mathbf{z}; \theta)$  taking only a single step. The original sampler is derived by approximately integrating the *probability flow ODE* for a learned diffusion model, and distillation can thus be understood as learning to integrate in fewer steps, or *amortizing* this integration into the new sampler.
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2047.png)

**distilling实现**

> **[图片提取文字 (image.png)]:**
> 的,在加噪的时候,有  $z_t=\alpha_t x+\sigma_t \epsilon$  ,在训练的时候,输入为  $z_t$  ,可以预测 x 为  $\hat{x}_{\theta}(z_t)$  或者预测  $\epsilon$  为  $\hat{\epsilon}_{\theta}(z_t)$  );
> 
> 在标准 $\operatorname{diffusin}$ 训练的时候, $\operatorname{target}$ 是真实的图片 x ,(预测真实图片和预测噪声是等价
> 
> 在蒸馏的时候,取当前时刻为 t ,则 teacher 的上一个时刻为 t'=t-0.5/N ,上两个时刻为 t''=t-1/N .
> 
> 假设在 t 时刻,student的输入  $z_t$  ,输出为  $\tilde{x}$  ,那么在 t'' 时刻采样结果为  $\tilde{z}_{t''}=\alpha_{t''}\tilde{x}+\frac{\sigma_{t''}}{\sigma_t}(z_t-\alpha_t\tilde{x})$  ,他应该等于算法截图里面的  $z_{t''}$  ,于是就计算出了  $\tilde{x}$  的 表达式,该表达式也就是student的模型 target。这里用采样结果相等作为优化目标,而不用  $\hat{x}_{\eta}(z_{t'})$  作为优化目标,是因为我们在循环采样的时候,是以 $z_{t''}$ 作为下一步的网络的输入。
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2048.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2049.png)

## ICLR23：Flow Matching（最优传输计划ODE Flow：v-prediction）

ICLR23：Flow matching for generative modeling    

[https://zhuanlan.zhihu.com/p/685921518](https://zhuanlan.zhihu.com/p/685921518)

类似ReFlow方法，模型直接学习ODE本身（v-prediction），而不是人为构造ODE的参数化形式。

用CNF理论模型建模和解释ODE积分的求解方式，比如DDPM、DDIM等。

最优传输计划OT是直线映射方式。

### 连续规整流CNF

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig.1 Illustration of a Flow-based model
> 
> Flow-based Model 是一种基于 Normalizing Flows (NFs)的生成模型,它通过一系列概率密度函数的变量变换,将复杂的概率分布转换为简单的概率分布,并通过逆变换生成新的数据样本。而 Continuous Normalizing Flows (CNFs)是 Normalizing Flows 的扩展,它使用常微分方程 (ODE)来表示连续的变换过程,用于建模概率分布。
> 
> Flow Matching(FM)是一种训练 Continuous Normalizing Flows的方法,它通过学习与概率路径相关的向量场(Vector Field)来训练模型,并使用 ODE 求解器来生成新样本。
> 
> 扩散模型是 Flow Matching的一个应用特例,使用FM可以提高其训练的稳定性。此外,使用最优传输(Optimal Transport\*)技术构建概率路径可以进一步加快训练速度,并提高模型的泛化能力。
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2050.png)

> **[图片提取文字 (image.png)]:**
> Normalizing Flows (CNFs), allowing us to train CNFs at unprecedented scale. Specifically, we present the notion of Flow Matching (FM), a simulation-free approach for training CNFs based on regressing vector fields of fixed conditional probability paths. Flow Matching is compatible with a general family of Gaussian probability paths for transforming between noise and data samples—which subsumes existing diffusion paths as specific instances. Interestingly, we find that employing FM with diffusion paths results in a more robust and stable alternative for training diffusion models. Furthermore, Flow Matching opens the door to training CNFs with other, non-diffusion probability paths. An instance of particular interest is using Optimal Transport (OT) displacement interpolation to define the conditional probability paths. These paths are more efficient than diffusion paths, provide faster training and sampling, and result in better generalization. Training CNFs using Flow Matching on ImageNet leads to consistently better performance than alternative diffusion-based methods in terms of both likelihood and sample quality, and allows fast and reliable sample generation using off-the-shelf numerical ODE solvers.
> 
> We introduce a new paradigm for generative modeling built on Continuous
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2051.png)

### CNF理论例子：VP、VE、OT

CNF理论解释不同的扩散过程：方差爆炸VE和方差保护VP

> **[图片提取文字 (image.png)]:**
> 程。在这种模型中,随着时间的推移,数据样本会逐渐变得更加嘈杂,即方差会不断增大, 直到达到一个稳定的状态。VE过程的一个特点是,它允许模型在生成数据时探索更广泛的潜
> 
> (1) Variance Exploding (VE): VE扩散模型是一种在生成过程中增加数据方差的扩散过
> 
> 在空间,这有助于生成多样化的样本。
> 
> ## VE的条件概率路径为:
> 
> 其中,  $\sigma_t$  是递增函数,  $\sigma_0=0,\sigma_1\gg 1$  ,对应均值和标准差为
> 
> .\_\_.
> 
> ## 根据Theorem 3 可以计算条件向量场为:
> 
> $$u_t\left(x\mid x_1\right) = -\frac{\sigma_{1-t}'}{\sigma_{1-t}}(x-x_1)$$
> 
>  $p_t\left(x\mid x_1\right) = \mathcal{N}\left(x\mid x_1, \sigma_{1-t}^2 I\right)$ 
> 
>  $\mu_t(x_1) = x_1, \sigma_t(x_1) = \sigma_{1-t}$ .
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2052.png)

> **[图片提取文字 (image.png)]:**
> 散过程。在这种模型中,数据样本在生成过程中的方差保持恒定,这意味着模型在引入噪声的同时,也会以某种方式减少噪声,以保持数据的整体方差不变。VP模型通常用于那些需要 目录 3分布稳定性的应用场景,例如在图像生成中保持图像的清晰度和结构特征。
> 
>  $p_t\left(x\mid x_1\right)=\mathcal{N}\left(x\mid \alpha_{1-t}x_1,\left(1-\alpha_{1-t}^2\right)I\right), \text{ where } \alpha_t=e^{-\frac{1}{2}T(t)},T(t)=$ 
> 
> (2) Variance Preserving (VP): VP扩散模型是一种在生成过程中保持数据方差不变的扩
> 
> ## VP的条件概率路径为:
> 
>  $\int_0^t \beta(s) ds$ 
> 
> 其中, 
> $$\alpha,\beta$$
>  为噪声策略函数,对应均值和标准差 $\mu_t\left(x_1\right)=\alpha_{1-t}x_1,\sigma_t\left(x_1\right)=\sqrt{1-\alpha_{1-t}^2}$ 。
> 
> ## 根据Theorem 3 可以计算条件向量场为:
> 
>  $u_t\left(x\mid x_1\right) = \frac{\alpha_{1-t}'}{1-\alpha_{1-t}^2} (\alpha_{1-t}x-x_1) = -\frac{T'(1-t)}{2} \left\lceil \frac{e^{-T(1-t)}x-e^{-\frac{1}{2}T(1-t)}x_1}{1-e^{-T(1-t)}}\right\rceil$ 
> 
> **结论**:实验发现,将扩散模型条件向量场与Flow Matching目标结合起来优化,相比于现有的Score Matching方法,训练更加稳定。
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2053.png)

> **[图片提取文字 (image.png)]:**
> **Example I: Diffusion conditional VFs.** Diffusion models start with data points and gradually add noise until it approximates pure noise. These can be formulated as stochastic processes, which have strict requirements in order to obtain closed form representation at arbitrary times t, resulting in Gaussian conditional probability paths  $p_t(x|x_1)$  with specific choices of mean  $\mu_t(x_1)$  and std  $\sigma_t(x_1)$  (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song et al., 2020b). For example, the reversed (noise  $\rightarrow$ data) Variance Exploding (VE) path has the form
> 
> $$p_t(x) = \mathcal{N}(x|x_1, \sigma_{1-t}^2 I),$$
>  (16)
> 
> where  $\sigma_t$  is an increasing function,  $\sigma_0 = 0$ , and  $\sigma_1 \gg 1$ . Next, equation 16 provides the choices of  $\mu_t(x_1) = x_1$  and  $\sigma_t(x_1) = \sigma_{1-t}$ . Plugging these into equation 15 of Theorem 3 we get
> 
> $$u_t(x|x_1) = -\frac{\sigma'_{1-t}}{\sigma_{1-t}}(x - x_1). \tag{17}$$
> 
> The reversed (noise→data) Variance Preserving (VP) diffusion path has the form
> 
> $$p_t(x|x_1) = \mathcal{N}(x \mid \alpha_{1-t}x_1, (1 - \alpha_{1-t}^2)I), \text{ where } \alpha_t = e^{-\frac{1}{2}T(t)}, T(t) = \int_0^t \beta(s)ds,$$
>  (18)
> 
> and  $\beta$  is the noise scale function. Equation 18 provides the choices of  $\mu_t(x_1) = \alpha_{1-t}x_1$  and  $\sigma_t(x_1) = \sqrt{1 - \alpha_{1-t}^2}$ . Plugging these into equation 15 of Theorem 3 we get
> 
> $$u_t(x|x_1) = \frac{\alpha'_{1-t}}{1 - \alpha_{1-t}^2} \left(\alpha_{1-t}x - x_1\right) = -\frac{T'(1-t)}{2} \left[ \frac{e^{-T(1-t)}x - e^{-\frac{1}{2}T(1-t)}x_1}{1 - e^{-T(1-t)}} \right]. \tag{19}$$
> 
> Our construction of the conditional VF  $u_t(x|x_1)$  does in fact coincide with the vector field previously used in the deterministic probability flow (Song et al. (2020b), equation 13) when restricted to these conditional diffusion processes; see details in Appendix D. Nevertheless, combining the diffusion conditional VF with the Flow Matching objective offers an attractive training alternative—which we find to be more stable and robust in our experiments—to existing score matching approaches.
> 
> Another important observation is that, as these probability paths were previously derived as solutions of diffusion processes, they do not actually reach a true noise distribution in finite time. In practice,  $p_0(x)$  is simply approximated by a suitable Gaussian distribution for sampling and likelihood evaluation. Instead, our construction provides full control over the probability path, and we can just directly set  $\mu_t$  and  $\sigma_t$ , as we will do next.
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2054.png)

**CNF理论设计OT Flow**

> **[图片提取文字 (image.png)]:**
> 最优传输(Optimal Transport,简称OT)选择定义条件概率路径的均值和标准差为简单的时间线性函数,当时间  $t:0\to 1$  ,对应概率密度路径从 $p(x)=\mathcal{N}(x\mid 0,I)$ 到  $p_1(x\mid x_1)$  ,均值和标准差定义为:
> 
>  $\mu_t(x) = tx_1$ , and  $\sigma_t(x) = 1 - (1 - \sigma_{\min}) t$ 
> 
> $$u_t(x) = tx_1$$
> , and  $\sigma_t(x) = 1 - (1 - \sigma_{\min})t$ 
> 
> 那么可得其对应 Flow Map为:
> 
> $$\psi_t(x) = \left(1 - \left(1 - \sigma_{\min}\right)t\right)x + tx_1$$
> 
> 根据Theorem 3可计算条件向量场的封闭解为:
> 
>  $u_{r}(x \mid x_{1}) = \frac{x_{1} - (1 - \sigma_{\min})x}{x_{1}}$ 
> 
>  $u_t\left(x\mid x_1\right)=\frac{x_1-(1-\sigma_{\min})x}{1-(1-\sigma_{\min})t}$ **结论:** 最优传输路径轨迹为直线,而扩散路径轨迹为曲线,因而可以得到更快的训练速度和 生成速度,以及更好的性能表现。
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2055.png)

> **[图片提取文字 (image.png)]:**
> ## Example II: Optimal Transport conditional VFs. An arguably more natural choice for conditional probability paths is to define the mean and the std to simply change linearly in time, i.e., $\mu_t(x) = tx_1$ , and $\sigma_t(x) = 1 - (1 - \sigma_{\min})t$ .
> 
> According to Theorem 3 this path is generated by the VF 
> $$x_1 - (1 - \sigma_{\min})x$$
> 
> $$u_t(x|x_1) = \frac{x_1 - (1 - \sigma_{\min})x}{x_1 - (1 - \sigma_{\min})x}$$
> 
> $$u_t(x|x_1) = \frac{x_1 - (1 - \sigma_{\min})x}{1 - (1 - \sigma_{\min})t},$$
> (21)
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2056.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 2: Compared to the diffusion path's conditional score function, the OT path's conditional vector field has constant direction in time and is arguably simpler to fit with a parametric model. Note the blue color denotes larger magnitude while red color denotes smaller magnitude.
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2057.png)

### 实验

![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2058.png)

> **[图片提取文字 (image.png)]:**
> models can also be sampled through an SDE formulation, this can be highly inefficient and many methods that propose fast samplers (e.g., Song et al. (2020a); Zhang & Chen (2022)) directly make use of the ODE perspective (see Appendix D). In part, this is due to ODE solvers being much more efficient—yielding lower error at similar computational costs (Kloeden et al., 2012)—and the multitude of available ODE solver schemes. When compared to our ablation models, we find that models trained using Flow Matching with the OT path always result in the most efficient sampler, regardless of ODE solver, as demonstrated next.
![image.png](DDIM%E3%80%81SDE%20ODE%20Flow%EF%BC%9A21-23/image%2059.png)

> **[图片提取文字 (image.png)]:**
> we use in this paper are: the probability density path  $p:[0,1]\times\mathbb{R}^d\to\mathbb{R}_{>0}$ , which is a time dependent probability density function, i.e.,  $\int p_t(x)dx = 1$ , and a time-dependent vector field,  $v:[0,1]\times\mathbb{R}^d\to\mathbb{R}^d$ . A vector field  $v_t$  can be used to construct a time-dependent diffeomorphic map, called a flow,  $\phi:[0,1]\times\mathbb{R}^d\to\mathbb{R}^d$ , defined via the ordinary differential equation (ODE):
> 
>  $\frac{d}{dt}\phi_t(x) = v_t(\phi_t(x))$ 
> 
>  $\phi_0(x) = x$ 
> 
> Let  $\mathbb{R}^d$  denote the data space with data points  $x=(x^1,\ldots,x^d)\in\mathbb{R}^d$ . Two important objects
![image.png](DiT%E3%80%81JiT%EF%BC%9A23%E3%80%8125/image%202.png)