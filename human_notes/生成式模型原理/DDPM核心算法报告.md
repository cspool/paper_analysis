# 从 DDPM 到 ODE Flow 与 JiT 的生成算法研究

## 摘要
本文以 DDPM 为主线，研究扩散生成如何由随机反向过程发展为 ODE 确定性传输，并讨论 JiT 对预测目标、损失函数与采样函数的重新组合。结果表明，三种表示在数学上可相互换算，但有限容量网络对其学习难度并不相同。

## 1. 论文基本信息
*Denoising Diffusion Probabilistic Models* 由 Jonathan Ho、Ajay Jain、Pieter Abbeel 发表于 NeurIPS 2020，研究扩散概率生成及其与去噪分数匹配、Langevin 动力学的联系，链接为 <https://arxiv.org/abs/2006.11239>。JiT 代表文献 *Back to Basics: Let Denoising Generative Models Denoise* 由 Tianhong Li、Kaiming He 于 2025 年公开，研究基于流形假设的像素空间生成，链接为 <https://arxiv.org/abs/2511.13720>。ODE 代表文献为 ICLR 2021 *Denoising Diffusion Implicit Models*；线性 Flow 代表文献为 ICLR 2023 *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*，链接分别为 <https://arxiv.org/abs/2010.02502>、<https://arxiv.org/abs/2209.03003>。

## 2. 研究背景与问题定义
VAE 的单步解码容易平滑细节，GAN 的对抗训练又可能不稳定，因而需要兼具概率解释与生成质量的方法。任务是学习从简单高斯噪声分布到真实图像分布的映射。设真实图像为 $x$，噪声为 $\epsilon\sim\mathcal N(0,I)$，中间状态为 $x_t$ 或 $z_t$。训练输入是图像、噪声与时间，输出是噪声、速度或干净图像的预测；推理输入是随机噪声，输出是生成图像。该问题的重要性在于，高维图像分布难以直接估计，而逐步去噪或连续传输可将复杂生成拆成可监督的局部回归。

## 3. 核心算法思想
DDPM 首先用固定扩散过程定义图像的编码路径：$x_0$ 按 $q(x_t|x_{t-1})$ 逐步采样为近似标准高斯的 $x_T$。同一扩散过程又给出已知 $x_0$ 时沿编码路径逆向采样的方式 $q(x_{t-1}|x_t,x_0)$。生成时真实 $x_0$ 未知，因此定义 $p_\theta(x_{t-1}|x_t)=q(x_{t-1}|x_t,f_\theta(x_t))$，让模型预测图像或噪声，并沿预测出的编码路径从 $x_T$ 逐步生成 $x_0$。

ODE Flow 将上述逐步采样抽象为可学习速度 $v(z_t,t)$ 的积分，并用 Euler 法模拟从噪声分布到图像分布的传输。JiT 沿用线性 Flow，但组合选择“预测 $x$、使用 $v$-Loss、通过 $v$ 采样”：网络直接恢复干净图像，再换算速度执行 ODE。其创新是分离预测空间、损失空间和采样函数。

## 4. 数学原理与公式推导
DDPM 扩散过程定义了从 $q(x_0)$ 采样生成 $q(x_T)=N(0,1)$ 的映射方式：
$$
q(x_{1:T}|x_0)=\prod_{t=1}^{T}q(x_t|x_{t-1})
=q(x_T|x_0)\prod_{t=2}^{T}q(x_{t-1}|x_t,x_0).
$$
前一分解作为编码器，编码和理解图像 $x_0$ 的分布；后一分解定义在 $x_t$ 处沿着 $x_0$ 的编码路径逆向采样。生成时定义：
$$
p_\theta(x_{t-1}|x_t)=q(x_{t-1}|x_t,f_\theta(x_t)),
$$
即沿着 $x_0$ 预测值 $f_\theta(x_t)$ 的编码路径逆向采样。任意时刻可直接采样：
$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon .
$$
变分推断认为隐变量序列 $x_1\ldots x_T$ 是样本 $x_0$ 的原因。为了使 $\log p(x_0)$ 最大，$p_\theta(x_{t-1}|x_t)$ 拟合 $q(x_{t-1}|x_t,x_0)$，并最大化 $p_\theta(x_0|x_1)$。高斯分布间的 KL 距离可转化为均值距离，进一步转为噪声预测：
$$
L_{\rm DDPM}=\mathbb E\|\epsilon-\epsilon_\theta(x_t,t)\|^2 .
$$

扩散过程的逆向采样公式是逆向 SDE 的模拟积分过程，生成图像是逆向 SDE 的积分结果。概率流 ODE 将随机噪声改为可学习的概率项，采用：
$$
{dz_t\over dt}=v_\theta(z_t,t),\qquad
z_{t+\Delta t}=z_t+\Delta t\,v_\theta(z_t,t).
$$
后一式是 Euler 积分。JiT 沿用直线 Flow，$v_\theta$ 预测 $v$，计算 v-Loss 来优化 $v_\theta$：
$$
z_t=tx+(1-t)\epsilon,\quad v=x-\epsilon,\quad
x_\theta={\rm net}_\theta(z_t,t),\quad
v_\theta={x_\theta-z_t\over1-t}.
$$
JiT 发现，$x_\theta$ 预测 $x$，代入 ODE 采样方程，能计算 $v_\theta$ 和 $\epsilon_\theta$；因此可分别选择预测空间、损失空间与生成方式。采用 v-Loss 时：
$$
L_{\rm JiT}=\mathbb E\|v_\theta-v\|^2
=\mathbb E{\|x_\theta-x\|^2\over(1-t)^2}.
$$
即训练 $x_\theta$ 预测 $x$，使用 v-Loss 优化，使用 $v_\theta$ 采样生成：
$$
t\sim p(t),\quad \epsilon\sim N(0,I),\quad
z_t=tx+(1-t)\epsilon,\quad
\theta\leftarrow\arg\min_\theta\|v_\theta-(x-\epsilon)\|^2 .
$$
训练时先构造 $z_t$，再由网络预测 $x_\theta$ 并换算 $v_\theta$；生成时从 $z_0\sim N(0,I)$ 出发，每步计算 $v_\theta$，按 $z_{t_{k+1}}=z_{t_k}+(t_{k+1}-t_k)v_\theta$ 作 Euler 更新，直至得到 $z_1$。
后续 DDIM 在 $\sigma_t=0$ 时将采样转为 ODE 积分并允许中间跳步；Score-SDE 给出概率流 ODE：
$$
d x=\left[f(x,t)-\frac12g(t)^2\nabla_x\log p_t(x)\right]dt .
$$
Reflow 和 CNF 则定义直线 Flow 来求解积分，达到加速采样的目的。

## 5. 实验设计与结果分析
DDPM 在 CIFAR-10 上进行无条件生成，并以 CelebA-HQ、LSUN 验证高分辨率生成。主要指标为：
$$
{\rm FID}=\|\mu_r-\mu_g\|^2+
{\rm Tr}(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2}),
$$
$$
{\rm IS}=\exp\left(\mathbb E_xD_{\rm KL}(p(y|x)\|p(y))\right).
$$
FID 衡量特征分布差异，越低越好；IS 反映可辨识性与多样性，越高越好。DDPM 在 CIFAR-10 上取得 FID 3.17、IS 9.46；渐进实验中大尺度特征先出现、细节后出现，支持渐进有损压缩解释，但其似然仍不及其他似然模型。

DDIM 复用 DDPM 参数，以采样子序列将 $T$ 步缩为 $S$ 步。CIFAR-10 编码解码实验显示，$S$ 越大，重建误差越低，验证确定性 ODE 可用于编码和重建，也说明少步 Euler 积分存在更大误差。

ReFlow 实验采用 Euler 法，以 FID、IS 与 recall 评价生成。重复 ReFlow 使配对直线交叉概率从 46.39% 降至 0.76% 和 0.14%，同时降低 L2 传输代价，支持拉直轨迹以减少采样步数。

JiT 在 ImageNet 高维 patch 上采用 v-Loss 时，预测图像、噪声和速度的 FID 分别为 8.62、372.38、96.53；低维 patch 中差距很小，支持高维输入下直接预测干净图像。

## 6. 总结
DDPM 的主要贡献是用固定编码路径监督可学习的逆向生成；ODE Flow 提供确定性积分解释；JiT 证明预测、损失与采样可以组合设计。其优点是训练稳定且理论统一，缺点是多步采样成本较高，线性训练路径也不保证生成轨迹笔直。

从生成时的概率模型看，模型并非一次计算“最可能图像”，而是在每个状态预测下一步条件分布或采样方向，再将局部决策组合为完整样本。随机 DDPM 中，初始噪声和逐步随机项共同选择样本；确定性 ODE 中，初始噪声固定后轨迹唯一，但不同噪声仍映射到不同图像。因此，模型学习的是从先验分布到数据分布的采样规则，而不是某一张标准答案。可进一步联合优化路径、时间权重与求解器。该研究联系模式识别中的概率建模、贝叶斯推断、流形假设与数值积分。
