# meeting-25/12/11（DiT、Diffusion、GPU虚拟化）

# 想法

DiT的不同step数，是不是又多种Mapping方式？     

数据的多个用处，重排 内存 schedule

冷模型推理加载热模型的重排？

GPU的pipeline推理过程：从硬盘加载参数到host memory，之后加载到GPU memory，因此loading pipeline component（的进度条）是分为两个阶段推进达到100%；

img2col conv、gemm能提供介于thread和instr之间的抢占粒度吗？  大于instr，小于layer，比如row-wise，顺便支撑attention，row-wise或channel-wise？；

DiT、Z-image：**context parallel**、tensor parallel，conv主要用于decode或encode的前后处理，而不是BiT的Back；？？decoder和encoder中还有UNet？

LDM中denoise stage比encode和decode stage慢很多，所以或许能够拆分端到端的request？但设计统一架构？

加速器可以用在云端，类似TPUi；

1 gemm，n conv，cpu non- leaner

row-wise更方便重排列？IW和OC的并行容易产生channel和width的layout；

云的服务模型；云边端；llm拆；

云边端拆LLM：llm的中间数据在边缘cache，多用户共享中间结果和参数；

共享方式：batch；很早batch，底层多样，高层共性没有暴露和共享；

后端SLM生成的结果共享；

隐私放到小模型；

单纯卷性能已经有范式，优化重点在于应用场景的复杂多变；

im2col模块能增大llm、conv调度的选择？llm和conv的同卡调度会有好处吗？不会，不同机器应调度不同类型模型；

冷热LLM在多卡的调度场景，结合调度方法，作单卡体系结构层面的优化（互联、kernel）；

MaaS的两种调度：静态绑定GPU+动态分配GPU，多任务共享GPU；

layer之间的抢占、token之间的抢占，动态DNN；

边缘场景的多任务，自动驾驶的chain of thought，能引入DNN，conv在边缘侧推理；

IoT：每个核的PCIe；

自动驾驶、VR的多任务，体系结构优化？

slam；

多算法内核；

资源受限，DFG不一定最优，引入抢占；

DiT的DFG的输入token不同，动态；

输入的长度动态，资源受限；

MoE的动态FFN；

不同layout下的concate+shortcut（residual），多分支的数据依赖；

# 方法/Acc

## NV MPS、MIG、vGPU

GPU VIRTUALISATION

VIRTUALIZING HARDWARE PROCESSINGRESOURCES INA PROCESSOR

## **DiT**

**DiT**将Difuusion的backbone中convolution-UNet替换为Transformer，但decoder和encoder依然是conv-based UNet（Resnet、conv、upsample等）；

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](meeting-25%2012%2011%EF%BC%88DiT%E3%80%81Diffusion%E3%80%81GPU%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image.png)

> **[图片提取文字 (image.png)]:**
> Diffusion [19, 54] and score-based generative models [22, 56] have been particularly successful as generative models of images [35,46,48,50], in many cases outperforming generative adversarial networks (GANs) [12] which had previously been state-of-the-art. Improvements in DDPMs over the past two years have largely been driven by improved sampling techniques [19, 27, 55], most notably classifierfree guidance [21], reformulating diffusion models to predict noise instead of pixels [19] and using cascaded DDPM pipelines where low-resolution base diffusion models are trained in parallel with upsamplers [9, 20]. For all the diffusion models listed above, convolutional U-Nets [49] are the de-facto choice of backbone architecture. Concurrent
> 
> work [24] introduced a novel, efficient architecture based
> 
> on attention for DDPMs; we explore pure transformers.
> 
> Denoising diffusion probabilistic models (DDPMs).
![image.png](meeting-25%2012%2011%EF%BC%88DiT%E3%80%81Diffusion%E3%80%81GPU%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%201.png)

> **[图片提取文字 (image.png)]:**
> rectly in high-resolution pixel space can be computationally prohibitive. Latent diffusion models (LDMs) [48] tackle this issue with a two-stage approach: (1) learn an autoencoder that compresses images into smaller spatial representations with a learned encoder E; (2) train a diffusion model of representations z = E(x) instead of a diffusion model of
> 
> images x (E is frozen). New images can then be generated
> 
> by sampling a representation z from the diffusion model
> 
> and subsequently decoding it to an image with the learned
> 
> decoder x = D(z).
> 
> **Latent diffusion models.** Training diffusion models di-
![image.png](meeting-25%2012%2011%EF%BC%88DiT%E3%80%81Diffusion%E3%80%81GPU%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%202.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 3. **The Diffusion Transformer (DiT) architecture.** *Left:* We train conditional latent DiT models. The input latent is decomposed into patches and processed by several DiT blocks. *Right:* Details of our DiT blocks. We experiment with variants of standard transformer blocks that incorporate conditioning via adaptive layer norm, cross-attention and extra input tokens. Adaptive layer norm works best.
![image.png](meeting-25%2012%2011%EF%BC%88DiT%E3%80%81Diffusion%E3%80%81GPU%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%203.png)

## Zimage：Diffusion的蒸馏方法

Diffusion蒸馏→减少扩散的时间步；一般方法是分布匹配蒸馏DMD来**训练**学生模型；

论文发现：在短时间步的扩散中，DMD过程的CFG（无分类引导）是效果好的核心；

论文设计：将**DMD拆解成core engine+regularizer**，regularzer为了生成稳定、去除非自然），core是CFA，而regularizer替换成更简单的规则约束或GAN；

实现：noise scheduler→ engine + regularizer，**两种latent**而非一种（推理过程不同）；

> **[图片提取文字 (image.png)]:**
> efficient few-step and single-step generators. Among these, Distribution Matching Distillation (DMD) and its variants stand out for their impressive performance, which is widely attributed to their core mechanism of matching the student's output distribution to that of a pre-trained teacher model. In this work, we challenge this conventional understanding. Through a rigorous decomposition of the DMD training objective, we reveal that in complex tasks like text-to-image generation, where CFG is typically required for desirable few-step performance, the primary driver of few-step distillation is not distribution matching, but a previously overlooked component we identify as *CFG Augmentation* (CA). We demonstrate that this term acts as the core "engine" of distillation, while the Distribution Matching (DM) term functions as a "regularizer" that ensures training stability and mitigates artifacts. We further validate this decoupling by demonstrating that while the DM term is a highly effective regularizer, it is not unique; simpler non-parametric constraints or GAN-based objectives can serve the same stabilizing function, albeit with different trade-offs. This decoupling of labor motivates a more principled analysis of the properties of both terms, leading to a more systematic and in-depth understanding. This new understanding further enables us to propose principled modifications to the distillation process, such as decoupling the noise schedules for the engine and the regularizer, leading to further performance gains. Notably, our method has been adopted by the **Z-Image** project to develop a top-tier 8-step image generation model, empirically validating the generalization and robustness of our findings.
> 
> Diffusion model distillation has emerged as a powerful technique for creating
![image.png](meeting-25%2012%2011%EF%BC%88DiT%E3%80%81Diffusion%E3%80%81GPU%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%204.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 1: Two perspectives on the DMD algorithm. (a) The conventional view, which treats the use of CFG as a heuristic relaxation of the theoretical framework, with the algorithm's success solely attributed to this (relaxed) distribution matching mechanism. (b) Our proposed decoupled view, where the objective is a combination of two distinct mechanisms: a CFG Augmentation (CA) engine that drives the few-step conversion, and a Distribution Matching (DM) regularizer—which strictly adheres to the theoretical derivation (Eq. 1)—that ensures training stability.
![image.png](meeting-25%2012%2011%EF%BC%88DiT%E3%80%81Diffusion%E3%80%81GPU%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%205.png)

diffusion推理的Model**蒸馏方法**：轨迹匹配、GAN蒸馏、 得分蒸馏（**DMD**）；

> **[图片提取文字 (image.png)]:**
> & Song, 2024; Wang et al., 2024; Ren et al., 2024). Another prominent direction is *GAN-based distillation* (Sauer et al., 2024b;a; Lin et al., 2024), which leverages an adversarial objective to match the student's output distribution with the teacher's or with real data.
> 
> Score-based Distillation was initially proposed for 3D generation (Poole et al., 2022; Wang et al., 2023). Diff-Instruct (Luo et al., 2023b) pioneered its application in few-step diffusion distillation, and DMD (Yin et al., 2024b) was among the first to successfully apply this principle to large-scale text-to-image models. Following works have explored different distribution metrics (Zhou et al., 2024b;a) or combining this principle with other distillation paradigms (Yin et al., 2024a; Chadebec
> 
> et al., 2025; Luo et al., 2024). Notably, the adoption of CFG in real score is a common practice
> 
> among these works, but this choice is rarely officially discussed. An exception is (Luo, 2024), which
> 
> models the CFG term as an extra reward function after distillation. We are the first to decouple the
> 
> role of this CFG term during distillation and to reveal its dominance in multi-to-few-step conversion.
> 
> **Few-Step Diffusion Distillation** aims to reduce the inference cost of diffusion models. *Trajectory-*
> 
> matching approaches train a student model to replicate the teacher's sampling path in fewer steps (Liu
> 
> et al., 2023; Zhu et al., 2024; Kim et al., 2024; Frans et al., 2024; Salimans & Ho, 2022; Meng et al.,
> 
> 2023), with consistency distillation as a renowned branch (Song et al., 2023; Kim et al., 2023; Lu
![image.png](meeting-25%2012%2011%EF%BC%88DiT%E3%80%81Diffusion%E3%80%81GPU%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%206.png)

**DMD方法**

**Zt**是t时刻的隐变量，Z0是随机噪声，Zt是t时刻的某种分布；

**G（Zt）**是Zt基于text生成的图像（Diffusion推理），G（Zt）加噪声扩散得到**Xr**，Xr用于计算老师模型和蒸馏模型的分数S-real-cond和S-fake-cond，最小化分数差来更新参数，cond表示基于text输入的条件；

实际应用中，对老师模型计算分数**S-real-cfg**=**a** * S-real-cond - **（a-1）*** S-real-uncond，a是CFG优化scale；

> **[图片提取文字 (image.png)]:**
> The goal of Distribution Matching Distillation (DMD) is to train a student generator, denoted as  $G_{\theta}$ , to emulate the output distribution of a pre-trained, frozen teacher diffusion model in a few-step or even single-step inference process. The training is guided by minimizing a loss function, Eq. 1, whose gradient with respect to the generator's parameters  $\theta$  can be estimated by:
> 
> $$\nabla_{\theta} \mathcal{L}_{\text{DMD-theory}} = \mathbb{E}_{z_t, \tau, \mathbf{x}_{\tau}} \left[ -\left( s_{\text{cond}}^{\text{real}}(\mathbf{x}_{\tau}) - s_{\text{cond}}^{\text{fake}}(\mathbf{x}_{\tau}) \right) \frac{\partial G_{\theta}(z_t)}{\partial \theta} \right]. \tag{2}$$
> 
> In this paper, we follow the flow matching notations (Lipman et al., 2022) and define t=0 with pure noise and t=1 with clean data.  $z_t$  denotes the prepared generator input at noise level t. For single-step generation, t is 0 and  $z_t$  is random noise; for few-step generation,  $z_t$  can be obtained by going through the previous steps, a technique called "backward simulation" (Yin et al., 2024a). The generator  $G_{\theta}$  takes  $z_t$  and makes the image prediction  $G_{\theta}(z_t)$ , which is then renoised to  $x_{\tau}$  with a sampled noise level  $\tau$ . After renoising,  $x_{\tau}$  would be fed to two diffusion models for score estimates:  $s_{\text{cond}}^{\text{real}}$ , the "real score" estimated by the original multi-step teacher model; and  $s_{\text{cond}}^{\text{fake}}$ , the "fake" score estimate from an auxiliary "fake" model that is trained concurrently on the generator's outputs. The subscript "cond" indicates the score is conditioned on a text input. Pseudo-code provided in Sec. B
> 
> However, Eq. 2 usually leads to poor performance in practice, and a subtle modification is involved:
> 
> $$\nabla_{\theta} \mathcal{L}_{\text{DMD}} = \mathbb{E}_{z_t, \tau, \mathbf{x}_{\tau}} \left[ -\left( s_{\text{cfg}}^{\text{real}}(\mathbf{x}_{\tau}) - s_{\text{cond}}^{\text{fake}}(\mathbf{x}_{\tau}) \right) \frac{\partial G_{\theta}(z_t)}{\partial \theta} \right]. \tag{3}$$
> 
> The only difference between Eq. 2 and Eq. 3 is that the real score  $s_{\text{cond}}^{\text{real}}$  is replaced with  $s_{\text{cfg}}^{\text{real}}$ , where
> 
> $$s_{\text{cfg}}^{\text{real}}(\mathbf{x}_{\tau}) = s_{\text{uncond}}^{\text{real}}(\mathbf{x}_{\tau}) + \alpha \left( s_{\text{cond}}^{\text{real}}(\mathbf{x}_{\tau}) - s_{\text{uncond}}^{\text{real}}(\mathbf{x}_{\tau}) \right). \tag{4}$$
> 
>  $s_{\rm cond}^{\rm real}$  and  $s_{\rm uncond}^{\rm real}$  are the conditional and unconditional scores from the real model, respectively, and  $\alpha$  is the CFG guidance scale (typically  $\alpha>1$ ). Despite the introduction of discrepancy between theory and practice, this modification empirically yields substantially better results. Interestingly, this substitution has been largely overlooked in prior literature, often dismissed as a mere implementation detail rather than a fundamental deviation from the original theory. **However, we will show**
![image.png](meeting-25%2012%2011%EF%BC%88DiT%E3%80%81Diffusion%E3%80%81GPU%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%207.png)

> **[图片提取文字 (image.png)]:**
> To scrutinize the underlying mechanisms of the DMD algorithm, we begin by substituting the definition of Classifier-Free Guidance (Eq. 4) into the DMD gradient formula (Eq. 3):
> 
> $$\nabla_{\theta} \mathcal{L}_{\text{DMD}} = \mathbb{E} \left[ -\left[ s_{\text{uncond}}^{\text{real}}(\mathbf{x}_{\tau}) + \alpha \left( s_{\text{cond}}^{\text{real}}(\mathbf{x}_{\tau}) - s_{\text{uncond}}^{\text{real}}(\mathbf{x}_{\tau}) \right) - s_{\text{cond}}^{\text{fake}}(\mathbf{x}_{\tau}) \right] \frac{\partial G_{\theta}(z_{t})}{\partial \theta} \right]. \tag{5}$$
> 
> With simple rearrangement, we can decompose Eq. 5 into two distinct components:
> 
> $$\nabla_{\theta} \mathcal{L}_{\text{DMD}} = \mathbb{E} \left[ - \left( \underbrace{\left( s_{\text{cond}}^{\text{real}}(\mathbf{x}_{\tau}) - s_{\text{cond}}^{\text{fake}}(\mathbf{x}_{\tau}) \right)}_{\Delta^{\text{real-fake}} \text{ (Distribution Matching)}} + (\alpha - 1) \underbrace{\left( s_{\text{cond}}^{\text{real}}(\mathbf{x}_{\tau}) - s_{\text{uncond}}^{\text{real}}(\mathbf{x}_{\tau}) \right)}_{\Delta^{\text{real}} \text{ (CFG Augmentation)}} \right) \frac{\partial G_{\theta}(z_{t})}{\partial \theta} \right]. \tag{6}$$
> 
> This decomposition reframes the DMD objective as a sum of two terms:
> 
> - **1. Distribution Matching (DM,**  $\Delta^{\text{real-fake}}$ ): The first term,  $s_{\text{cond}}^{\text{real}} s_{\text{cond}}^{\text{fake}}$ , strictly aligns with theoretical deduction of matching two distributions (Eq. 1 and 2).
> - **2. CFG Augmentation (CA,**  $\Delta_{\text{cfg}}^{\text{real}}$ ): The second term,  $(\alpha 1)(s_{\text{cond}}^{\text{real}} s_{\text{uncond}}^{\text{real}})$ , directly applies a scaled CFG signal as a gradient to the student's output. This component was typically overlooked.
> - This separation motivates an ablation study to isolate the true contribution of each component. We investigate three training configurations: (1) the full DMD objective ( $\Delta^{\text{real-fake}} + \Delta^{\text{real}}_{\text{cfg}}$ ), (2) CFG Augmentation only ( $\Delta^{\text{real}}_{\text{cfg}}$ ), and (3) Distribution Matching only ( $\Delta^{\text{real-fake}}$ ).
![image.png](meeting-25%2012%2011%EF%BC%88DiT%E3%80%81Diffusion%E3%80%81GPU%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%208.png)

> **[图片提取文字 (image.png)]:**
> ```
> Algorithm 1 Original&Decoupled DMD Training Procedure
> Require: Pre-trained teacher model s_{\text{real}}, CFG scale \alpha, number of steps N, proxy loss weight \lambda
> Ensure: Trained few-step generator G_{\theta}
>   1: ⊳ Initialize student generator and fake model from the teacher
>   2: G_{\theta} \leftarrow s_{\text{real}}
>   3: s_{\text{fake}} \leftarrow s_{\text{real}}
>   4: while not converged do
>   5:
>              > — Generator Update Step —
>              Sample a generation step t from the few-step schedule \{t_1, \ldots, t_N\}
>   6:
>   7:
>              Prepare generator input z_t (e.g., via backward simulation for t > t_1)
>   8:
>              Generate an image: x_{\text{gen}} \leftarrow G_{\theta}(z_t)
>   9:
>              if 'decoupled_schedule' then
> 10:
>                     > — Decoupled DMD behavior —
>                     Sample CFG augmentation noise level \tau_{\text{CA}} \sim \mathcal{U}(t, 1)
> 11:
>                     Sample Distribution Matching noise level \tau_{DM} \sim \mathcal{U}(0,1)
> 12:
>                     Re-noise the generated image for both schedules:
> 13:
>                     x_{\tau_{\text{CA}}} \leftarrow \text{renoise}(x_{\text{gen}}, \tau_{\text{CA}})
> 14:
>                     x_{\tau_{\mathrm{DM}}} \leftarrow \mathrm{renoise}(x_{\mathrm{gen}}, \tau_{\mathrm{DM}})
> 15:
> 16:
>                     Withwith torch.no_grad():
>                           ⊳ Calculate scores for the Distribution Matching (DM) term
> 17:
>                           s_{\text{cond, DM}}^{\text{real}} \leftarrow s_{\text{real}}(x_{\tau_{\text{DM}}}, \tau_{\text{DM}}, \text{text})
> 18:
>                           s_{\text{cond, DM}}^{\text{fake}} \leftarrow s_{\text{fake}}(x_{\tau_{\text{DM}}}, \tau_{\text{DM}}, \text{text})
> 19:
>                           ▷ Calculate scores for the CFG Augmentation (CA) term
> 20:
>                           s_{\text{cond, CA}}^{\text{real}} \leftarrow s_{\text{real}}(x_{\tau_{\text{CA}}}, \tau_{\text{CA}}, \text{text})
> 21:
>                           s_{\text{uncond, CA}}^{\text{real}} \leftarrow s_{\text{real}}(x_{\tau_{\text{CA}}}, \tau_{\text{CA}}, ")
> 22:
> 23:
>                     EndWith
> 24:
>                     ⊳ Compute the two components of the update direction
>                     \Delta_{\rm DM} \leftarrow s_{\rm cond, \, DM}^{\rm real} - s_{\rm cond, \, DM}^{\rm fake}
> 25:
>                     \Delta_{\text{CA}} \leftarrow (\alpha - 1) \left( s_{\text{cond, CA}}^{\text{real}} - s_{\text{uncond, CA}}^{\text{real}} \right)
> 26:
> 27:
>                     \Delta_{\text{total}} \leftarrow \Delta_{\text{DM}} + \Delta_{\text{CA}}
> 28:
>               else
> 29:
>                     > — Original DMD behavior —
>                     Sample a single noise level \tau \sim \mathcal{U}(0,1)
> 30:
>                     Re-noise the generated image: x_{\tau} \leftarrow \text{renoise}(x_{\text{gen}}, \tau)
> 31:
> 32:
>                     Withwith torch.no_grad():
>                           s_{\text{cond}}^{\text{real}} \leftarrow s_{\text{real}}(x_{\tau}, \tau, \text{text})
> 33:
>                           s_{\text{uncond}}^{\text{real}} \leftarrow s_{\text{real}}(x_{\tau}, \tau, ")
> 34:
>                           s_{\text{cond}}^{\text{fake}} \leftarrow s_{\text{fake}}(x_{\tau}, \tau, \text{text})
> 35:
>                           s_{\text{cfg}}^{\text{real}} \leftarrow s_{\text{uncond}}^{\text{real}} + \alpha (s_{\text{cond}}^{\text{real}} - s_{\text{uncond}}^{\text{real}})
> 36:
> 37:
>                     EndWith
> 38:
>                     > Compute the combined update direction
>                     \Delta_{\text{total}} \leftarrow s_{\text{cfg}}^{\text{real}} - s_{\text{cond}}^{\text{fake}}
> 39:
> 40:
>               end if
> 41:
>               Update generator by minimizing the proxy loss Update generator by minimizing the proxy loss
> 42:
>               \mathcal{L}_{\text{proxy}} \leftarrow ||G_{\theta}(z_t) - \text{stop\_grad}(G_{\theta}(z_t) + \lambda \Delta_{\text{total}})||^2
> 43:
>               Update G_{\theta} by minimizing \mathcal{L}_{\text{proxy}}
> 44:
> 45:
>              ⊳ — Fake Model Update Step —
> 46:
>               ⊳ This step can be run multiple times per generator update (TTUR)
>               Sample a new noise level \tau' \sim \mathcal{U}(0,1)
> 47:
> 48:
>               Generate a new image with detached gradient: x'_{gen} \leftarrow \text{stop\_grad}(G_{\theta}(z_t))
>               Re-noise the new image: x'_{\tau'} \leftarrow \text{renoise}(x'_{\text{gen}}, \tau')
> 49:
>               \mathcal{L}_{\text{denoise}} \leftarrow ||s_{\text{fake}}(x'_{\tau'}, \tau') - x'_{\text{gen}}||^2
> 50:
> 51:
>               Update s_{\text{fake}} using \nabla \mathcal{L}_{\text{denoise}}
> 52: end while
> ```
![image.png](meeting-25%2012%2011%EF%BC%88DiT%E3%80%81Diffusion%E3%80%81GPU%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%209.png)

CFG engine和DM Regularizer的**训练对比**

> **[图片提取文字 (image.png)]:**
> As illustrated in Fig. 2, our experiments reveal a clear division of labor between the two components. Training with CA alone is remarkably effective at converting the multi-step model into a few-step generator. Besides, the generated results also demonstrate high similarity in content to the full DMD objective, indicating the dominant role of the CA term in DMD loss. In contrast, even though it is improper to conclude that the DM term is totally incapable of doing the multi-step to few-step conversion (since in the 4-step experiment it indeed makes relatively reasonable images), a significant performance gap exists towards the CA setting, as indicated by both image visualizations and numerical indicators (Image Reward (Xu et al., 2023) and HPS v2.1 (Wu et al., 2023)).
> 
> However, we also observe that training with CA alone is unsustainable. While initially effective, the generated images progressively suffer from artifacts such as over-saturation and high-frequency noise, eventually leading to training collapse. The introduction of the Distribution Matching term eliminates these issues, enabling stable training over extended periods and yielding higher-quality final results. These empirical findings lead to two fundamental conclusions:
> 
> - 1. CFG Augmentation is the engine for few-step conversion. The ability of the distilled generator to produce high-quality samples in a few steps is almost entirely attributable to the  $\Delta^{\text{cfg}}$  term.
> - 2. Distribution Matching is a regularizer for training stability. The  $\Delta^{\text{real-fake}}$  term, while not the primary driver of distillation, plays a crucial role as a regularizer that prevents the training process from diverging and ensures the quality of the final output.
> 
> This insight fundamentally challenges the prevailing understanding of DMD-like methods: the conversion to a few-step generator is not primarily an act of matching distributions but rather a direct consequence of "baking" the CFG pattern into the student generator's predictions (we elaborate on this point in Sec. A), which is irrelevant to the fake model.
![image.png](meeting-25%2012%2011%EF%BC%88DiT%E3%80%81Diffusion%E3%80%81GPU%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2010.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 2: Ablation study on the roles of CFG Augmentation (CA) and Distribution Matching (DM). Numerical indicators are evaluated on 1k sampled prompts from COCO-10k (Lin et al., 2014).
![image.png](meeting-25%2012%2011%EF%BC%88DiT%E3%80%81Diffusion%E3%80%81GPU%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2011.png)

DMD分解成CFG和DM**两种latent**后的不同**schedule**方式

> **[图片提取文字 (image.png)]:**
> To facilitate this investigation, we first generalize the DMD gradient (Eq.6) to a  $\tau$ -decoupled form. This allows us to assign independent re-noising schedules,  $\tau_{CA}$  and  $\tau_{DM}$ , to the CA and DM components, respectively. The resulting "decoupled DMD" (d-DMD) gradient is formulated as:
> 
> $$\nabla_{\theta} \mathcal{L}_{\text{d-DMD}} = \mathbb{E}\left[-\left(\left(s_{\text{cond}}^{\text{real}}(\mathbf{x}_{\tau_{\text{DM}}}) - s_{\text{cond}}^{\text{fake}}(\mathbf{x}_{\tau_{\text{DM}}})\right) + (\alpha - 1)\left(s_{\text{cond}}^{\text{real}}(\mathbf{x}_{\tau_{\text{CA}}}) - s_{\text{uncond}}^{\text{real}}(\mathbf{x}_{\tau_{\text{CA}}})\right)\right) \frac{\partial G_{\theta}(z_t)}{\partial \theta}\right], \quad (8)$$
> 
> where d-DMD is short for decoupled DMD. This modification allows us to decouple the renoising schedule of DM and CA, allowing principled experimental analysis. With this formulation, we design an ablation study to evaluate four distinct schedule configurations for a 4-step generator:
> 
> ① Coupled-Shared: The original DMD approach where  $\tau_{CA} = \tau_{DM}$ , sampled from [0, 1].
> 
> - ② **Decoupled-Full:** Both schedules are independent but cover the full range,  $\tau_{CA}$ ,  $\tau_{DM} \in [0, 1]$ .
> - ③ **Decoupled-Constrained:** Both schedules are independent and constrained,  $\tau_{\text{CA}}$ ,  $\tau_{\text{DM}} > t$ . ④ **Decoupled-Hybrid:** The engine is constrained while the regularizer is not,  $\tau_{\text{CA}} > t$ ,  $\tau_{\text{DM}} \in [0, 1]$ .
> - The results, presented in Tab. 1 for the Lumina-Image-2.0 model (Qin et al., 2025), provide strong evidence for our hypothesis. First, we confirm that merely decoupling the schedules while keeping them global ② yields negligible impact compared to the baseline ①, demonstrating that the benefit comes from the schedule's range, not just its independence. More importantly, both configurations with constrained schedules (③ and ④) significantly outperform the baselines across multiple bench-
> 
> marks (Hu et al., 2024; Wu et al., 2023; Ma et al., 2025). Crucially, our proposed Decoupled-Hybrid
> 
> setting 4 consistently achieves the best overall scores, validating our core proposal.
> 
> The qualitative results in Fig. 5 offer further visual confirmation. Compared to the global schedule (②, top row), constraining the CA engine (③, middle row) introduces richer, finer details, confirming the benefit of a focused engine. However, this configuration still suffers from color oversaturation, a low-frequency artifact that its constrained DM regularizer fails to correct. In stark contrast, our Decoupled-Hybrid setting (④, bottom row) retains these enhanced details while effectively mitigating
> 
> the saturation artifacts, yielding the most visually appealing and natural-looking images. These observations are decisively corroborated by a comprehensive **user study** (Sec. C), where model @ achieved a unanimous 100% preference rate in model-level comparisons. 15 annotators consistently justified their choice by its ability to generate richer details, a more realistic and less "greasy" appearance, and fewer structural deformities. Furthermore, in a three-way image-level ranking, model @ was ranked first in 59.8% of cases, significantly outperforming the next-best model (③ at 33.8%).
![image.png](meeting-25%2012%2011%EF%BC%88DiT%E3%80%81Diffusion%E3%80%81GPU%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2012.png)

**推理**：text2img的**文生图**，img2img的图片自动编辑；

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](meeting-25%2012%2011%EF%BC%88DiT%E3%80%81Diffusion%E3%80%81GPU%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2013.png)

**核心算子**：

pipe→ZimagePipeline→encoder-Transformer_denoise-decoder→conv-UNetMidBlk→ResNet；

> **[图片提取文字 (image.png)]:**
> ```
> pipe: ZImagePipeline = ZImagePipeline.from_pretrained(
>         args.model path
>         if args.model path is not None
>         else os.environ.get(
>             "ZIMAGE DIR",
>             "Tongyi-MAI/Z-Image-Turbo",
>     ),
>     torch_dtype=torch.bfloat16,
> )
> if args.cache or args.parallel_type is not None:
>     if args.cache:
>         # Only warmup 4 steps (total 9 steps) for distilled models
>         args.max warmup steps = min(4, args.max warmup steps)
>     cachify(args, pipe)
> pipe.to(device)
> assert isinstance(pipe.transformer, ZImageTransformer2DModel)
> pipe.set progress bar config(disable=rank != 0)
> # Set default prompt
> prompt = (
>     "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, "
>     "red floral forehead pattern. Elaborate high bun, golden phoenix headdress, "
>     "red flowers, beads. Holds round folding fan with lady, trees, bird. Neon "
>     "lightning-bolt lamp (\phi), bright yellow glow, above extended left palm. "
>     "Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), "
>     "blurred colorful distant lights."
> )
> if args.prompt is not None:
>     prompt = args.prompt
> def run pipe(warmup: bool = False):
>     image = pipe(
>         prompt=prompt,
>         height=1024 if args.height is None else args.height,
>         width=1024 if args.width is None else args.width,
>         num inference steps=2 if warmup else (9 if args.steps is None else args.steps),
>         guidance scale=0.0, # Guidance should be 0 for the Turbo models
>         generator=torch.Generator("cpu").manual seed(0),
>     ).images[0]
>     return image
> ```
![image.png](meeting-25%2012%2011%EF%BC%88DiT%E3%80%81Diffusion%E3%80%81GPU%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2014.png)

> **[图片提取文字 (image.png)]:**
> ```
> class ZImagePipeline(DiffusionPipeline, FromSingleFileMixin):
> def __call_ (
>                 noise_pred = torch.stack(noise_pred, dim=0)
>             else:
>                 noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)
>             noise_pred = noise_pred.squeeze(2)
>             noise pred = -noise pred
>             # compute the previous noisy sample x_t -> x_t-1
>             latents = self.scheduler.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]
>             assert latents.dtype == torch.float32
>             if callback_on_step_end is not None:
>                 callback_kwargs = {}
>                 for k in callback_on_step_end_tensor_inputs:
>                     callback kwargs[k] = locals()[k]
>                 callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
>                 latents = callback_outputs.pop("latents", latents)
>                 prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
>                 negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
>             # call the callback, if provided
>             if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
>                 progress_bar.update()
>     if output_type == "latent":
>         image = latents
>     else:
>         latents = latents.to(self.vae.dtype)
>         latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
>         image = self.vae.decode(latents, return_dict=False)[0]
>         image = self.image_processor.postprocess(image, output_type=output_type)
>     # Offload all models
>     self.maybe_free_model_hooks()
>     if not return_dict:
>         return (image,)
>     return ZImagePipelineOutput(images=image)
> ```
![image.png](meeting-25%2012%2011%EF%BC%88DiT%E3%80%81Diffusion%E3%80%81GPU%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2015.png)

> **[图片提取文字 (image.png)]:**
> ```
> class Decoder(nn.Module):
>     act_fn ('str', *optional*, defaults to '"silu"'):
>         The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
>     norm_type (`str`, *optional*, defaults to `"group"`):
>         The normalization type to use. Can be either `"group"` or `"spatial"`.
> def init (
>     self,
>     in_channels: int = 3,
>     out_channels: int = 3,
>     up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
>     block out channels: Tuple[int, ...] = (64,),
>     layers_per_block: int = 2,
>     norm_num_groups: int = 32,
>     act_fn: str = "silu",
>     norm_type: str = "group", # group, spatial
>     mid_block_add_attention=True,
> ):
>     super().__init__()
>     self.layers_per_block = layers_per_block
>     self.conv_in = nn.Conv2d(
>         in_channels,
>         block out channels[-1],
>         kernel size=3,
>         stride=1,
>         padding=1,
>     )
>     self.up blocks = nn.ModuleList([])
>     temb channels = in channels if norm type == "spatial" else None
>     # mid
>     self.mid_block = UNetMidBlock2D(
>         in channels=block out channels[-1],
>         resnet_eps=1e-6,
>         resnet_act_fn=act_fn,
>         output_scale_factor=1,
>         resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
>         attention_head_dim=block_out_channels[-1],
>         resnet groups=norm num groups,
>         temb_channels=temb_channels,
>         add_attention=mid_block_add_attention,
>     )
>     # up
> ```
![image.png](meeting-25%2012%2011%EF%BC%88DiT%E3%80%81Diffusion%E3%80%81GPU%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2016.png)

> **[图片提取文字 (image.png)]:**
> ```
> class ZImagePipeline(DiffusionPipeline, FromSingleFileMixin):
> def __call__(
>      # 6. Denoising loop
>      with self.progress_bar(total=num_inference_steps) as progress_bar:
>          for i, t in enumerate(timesteps):
>              if self.interrupt:
>                  continue
>              # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
>              timestep = t.expand(latents.shape[0])
>              timestep = (1000 - timestep) / 1000
>              # Normalized time for time-aware config (0 at start, 1 at end)
>              t norm = timestep[0].item()
>              # Handle cfg truncation
>              current_guidance_scale = self.guidance_scale
>              if (
>                  self.do_classifier_free_guidance
>                  and self._cfg_truncation is not None
>                  and float(self. cfg truncation) <= 1
>              ):
>                  if t_norm > self._cfg_truncation:
>                      current guidance scale = 0.0
>              # Run CFG only if configured AND scale is non-zero
>              apply_cfg = self.do_classifier_free_guidance and current_guidance_scale > 0
>              if apply cfg:
>                  latents_typed = latents.to(self.transformer.dtype)
>                  latent model input = latents typed.repeat(2, 1, 1, 1)
>                  prompt_embeds_model_input = prompt_embeds + negative_prompt_embeds
>                  timestep_model_input = timestep.repeat(2)
>              else:
>                  latent_model_input = latents.to(self.transformer.dtype)
>                  prompt embeds model input = prompt embeds
>                  timestep_model_input = timestep
>              latent model input = latent model input.unsqueeze(2)
>              latent_model_input_list = list(latent_model_input.unbind(dim=0))
>              model_out_list = self.transformer(
>                  latent_model_input_list,
>                  timestep_model_input,
>                  prompt_embeds_model_input,
>              )[0]
>              if apply_cfg:
> ```
![image.png](meeting-25%2012%2011%EF%BC%88DiT%E3%80%81Diffusion%E3%80%81GPU%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2017.png)

> **[图片提取文字 (image.png)]:**
> ```
> class AutoencoderKL(ModelMixin, AutoencoderMixin, ConfigMixin, FromOriginalModelMixin, PeftAdapterMixin):
> def __init__(
>      norm_num_groups: int = 32,
>      sample_size: int = 32,
>      scaling_factor: float = 0.18215,
>      shift factor: Optional[float] = None,
>      latents mean: Optional[Tuple[float]] = None,
>      latents_std: Optional[Tuple[float]] = None,
>      force_upcast: bool = True,
>      use_quant_conv: bool = True,
>      use_post_quant_conv: bool = True,
>      mid_block_add_attention: bool = True,
>  ):
>      super().__init__()
>      # pass init params to Encoder
>      self.encoder = Encoder(
>          in_channels=in_channels,
>          out_channels=latent_channels,
>          down_block_types=down_block_types,
>          block_out_channels=block_out_channels,
>          layers_per_block=layers_per_block,
>          act_fn=act_fn,
>          norm_num_groups=norm_num_groups,
>          double_z=True,
>          mid_block_add_attention=mid_block_add_attention,
>      # pass init params to Decoder
>      self.decoder = Decoder(
>          in channels=latent channels,
>          out_channels=out_channels,
>          up_block_types=up_block_types,
>          block_out_channels=block_out_channels,
>          layers_per_block=layers_per_block,
>          norm_num_groups=norm_num_groups,
>          act_fn=act_fn,
>          mid block add attention=mid block add attention,
>      )
>      self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1) if use_quant_conv else None
>      self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1) if use_post_quant_conv else None
>      self.use_slicing = False
>      self.use tiling = False
>      # only relevant if vae tiling is enabled
>      self tile sample min size = self config sample size
> ```
![image.png](meeting-25%2012%2011%EF%BC%88DiT%E3%80%81Diffusion%E3%80%81GPU%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2018.png)

> **[图片提取文字 (image.png)]:**
> ```
> class UNetMidBlock2D(nn.Module):
>  def __init__(
>     output_scale_factor: float = 1.0,
> ):
>     super().__init__()
>     resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
>     self.add_attention = add_attention
>     if attn_groups is None:
>          attn_groups = resnet_groups if resnet_time_scale_shift == "default" else None
>     # there is always at least one resnet
>     if resnet_time_scale_shift == "spatial":
>          resnets = [
>             ResnetBlockCondNorm2D(
>                  in_channels=in_channels,
>                  out_channels=in_channels,
>                  temb_channels=temb_channels,
>                  eps=resnet_eps,
>                  groups=resnet_groups,
>                  dropout=dropout,
>                  time_embedding_norm="spatial",
>                  non_linearity=resnet_act_fn,
>                  output_scale_factor=output_scale_factor,
>              )
>          1
>     else:
>          resnets = [
>              ResnetBlock2D(
>                  in_channels=in_channels,
>                  out_channels=in_channels,
>                  temb_channels=temb_channels,
>                  eps=resnet_eps,
>                  groups=resnet groups,
>                  dropout=dropout,
>                  time_embedding_norm=resnet_time_scale_shift,
>                  non_linearity=resnet_act_fn,
>                  output_scale_factor=output_scale_factor,
>                  pre_norm=resnet_pre_norm,
>              )
>          ]
>     attentions = []
>     if attention_head_dim is None:
>          logger.warning(
>              f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {in_channels}."
>          )
> ```
![image.png](meeting-25%2012%2011%EF%BC%88DiT%E3%80%81Diffusion%E3%80%81GPU%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2019.png)

## Cerebras WSE vs. NV GPU

A COMPARISON OF THE CEREBRAS WAFER-SCALE INTEGRATION TECHNOLOGY WITH NVIDIA GPU-BASED SYSTEMS FOR ARTIFICIAL INTELLIGENCE

## Groq LPU

[https://medium.com/%40cognidownunder/groqs-lpu-the-ai-accelerator-that-s-leaving-gpus-in-the-dust-bb6fff67a877](https://medium.com/%40cognidownunder/groqs-lpu-the-ai-accelerator-that-s-leaving-gpus-in-the-dust-bb6fff67a877)

[https://groq.com/blog/the-groq-lpu-explained?utm_source=chatgpt.com](https://groq.com/blog/the-groq-lpu-explained?utm_source=chatgpt.com)

## Sambanova RDU

[https://sambanova.ai/blog/9-predictions-for-ai-in-2025?utm_source=chatgpt.com](https://sambanova.ai/blog/9-predictions-for-ai-in-2025?utm_source=chatgpt.com)

[https://sambanova.ai/blog/sn40l-chip-best-inference-solution](https://sambanova.ai/blog/sn40l-chip-best-inference-solution)

[https://sambanova.ai/blog/open-source-deep-research-agents](https://sambanova.ai/blog/open-source-deep-research-agents)

[https://sambanova.ai/blog/from-insight-to-action-with-sambanova-agents?utm_source=chatgpt.com](https://sambanova.ai/blog/from-insight-to-action-with-sambanova-agents?utm_source=chatgpt.com)

SambaNova SN40L Reconfigurable Dataflow Unit

## VLSI2023：Model & HW Co-optimize for sparse CNN+SA

Sense: Model-Hardware Co-design for Accelerating Sparse CNNs on Systolic Array

SA架构作sparse CNN推理：

channel cluster，完成IFM和weight在PE的负载均衡；

负载均衡的weight prune，保证kernel的稀疏比和模型精度；

自适应数据流配置计算策略，根据IFM和weight的存储需求；

> **[图片提取文字 (image.png)]:**
> neural network(CNN) and worth exploiting for CNN accelerators, but extra processing comes with hardware overhead, causing many architectures suffering from only minor profit. Meanwhile, systolic array has been increasingly competitive on CNNs acceleration for its high spatiotemporal locality and low hardware overhead. However, the irregularity of sparsity induces imbalanced workload under the rigid systolic dataflow, causing performance degradation. Thus, this paper proposed a systolicarray-based architecture, called Sense, for sparse CNN acceleration by model-hardware co-design, achieving large performance improvement. To balance input feature map(IFM) and weight loads across Processing Element(PE) array, we applied channel clustering to gather IFMs with approximate sparsity for array computation, and co-designed a load-balancing weight pruning method to keep the sparsity ratio of each kernel at a certain value with little accuracy loss, improving PE utilization and overall
> 
> Abstract—Sparsity is an intrinsic property of convolutional
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20108.png)

> **[图片提取文字 (image.png)]:**
> applied to determine the computing strategy based on the storage ratio of IFMs and weights, lowering  $1.17 \times 1.8 \times DRAM$  access compared with Swallow and further reducing system energy consumption. The whole design is implemented on ZyngZCU102 with 200MHz and performs at 471-, 34-, 53- and 191-image/s for AlexNet, VGG-16, ResNet-50 and GoogleNet respectively. Compared against sparse systolic-array-based accelerators, Swallow,
> 
> performance. Additionally, Adaptive Dataflow Configuration is
> 
> FESA and SPOTS, Sense achieves  $1 \times 2.25 \times$ ,  $1.95 \times 2.5 \times$  and  $1.17 \times 2.37 \times$  performance improvement on these CNNs respectively with reasonable overhead.
> 
> Index Terms—systolic array, hardware accelerator, sparsity, weight pruning, convolutional neural network.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20109.png)

加速器设计目标：增大（有效）计算并行度来降低延迟，增大数据复用来提高能效；

相关工作对模型作硬件友好的稀疏prune，基于稀疏方式设计加速器，挑战是：

因为稀疏的随机性导致**PE负载不均衡**，无法**同时利用weights和IFM**的稀疏性，IFM和weights的随机稀疏导致的**内存低效访问**，稀疏格式的处理需要消耗特定硬件资源而**性能受限**（FPGA Acc的LUT、BRAM）；

> **[图片提取文字 (image.png)]:**
> CNNs [10], [11], [26], [27] for higher energy efficiency and reource efficiency compared with other architectures as shown in Tab. Thus, researchers try to process sparsity with systolic architecture to improve the overall benefits. Swallow [28] overcomes the inability to exploit the sparsity of weights and IFMs, CONV layers and FC layers of CNNs with limited resource in a systolic array, and introduce a sparse-aware dataflow to boost PE utilization, achieving higher bandwidth saving and energy efficiency compared with previous sparse accelerators. However, the structured systolic dataflow essentially contradicts with the irregularity of sparsity, causing imbalanced PE loads. Considering that, FESA [29] pruned the kernels to 2~7 formalized zero distribution patterns and left IFM
> 
> Since systolic array [25] is widely applied to accelerate
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20110.png)

> **[图片提取文字 (image.png)]:**
> achieving lower sparsity processing overhead. But this pruning method is only implemented on Cifar-10 and Cifar-100 [30] currently. Thus, to balance workload with higher versatility in systolic array, SPOTS [31] designed a group-wise pruning method to divide weights into groups and prune some elements of the same position in each group, which achieves similar versatility with shape-wise pruning method [32] and improves compatibility with systolic array. Accordingly, SPOTS applied Image to Column (Im2Col) transformation of IFMs coupled with general matrix-matrix multiplication (GEMM) to better fit its pruning scheme into systolic array by skipping the weight rows and IFM columns with all zeros. However, since its pruning method is too fine-grained, the sparsity of weights after pruning is bounded by accuracy. Besides, SPOTS fails to exploit the sparsity in those rows and columns with some zeros, causing inefficient acceleration.
> 
> unprocessed to regularize dataflow as the dense systolic tempo,
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20111.png)

> **[图片提取文字 (image.png)]:**
> of system energy consumption, making it critical to further reduce memory access through dataflow. But sparse IFMs and weights can be irregular and fragmented, which leads to lower memory access efficiency. SCNN [22] employed a novel dataflow to eliminate unnecessary data transfers, but access contentions occurred when routing the products to accumulator buffer due to irregular sparse patterns. Lu et al [24] proposed a weight layout to enable efficient memory access without conflicts, but huge LUT consumption blocked the performance. Swallow harnesses a sparsity-aware dataflow with matrix multiplication tiling to promote data reuse within each channel, reducing DRAM access with little overhead. However, Swallow always preferentially reuse IFMs, while DRAM access can be variable if we choose different reuse
> 
> Additionally, memory access occupies a huge proportion
> 
> IFMs and weights in each layer.
> 
> These previous sparse systolic accelerators suffered from imbalanced workload, lacking versatility or low sparsity of weight pruning. Besides, the dataflow is inflexible for the variable ratio of IFM and weight in each layer. Thus, this paper aims to balance workload to fit with sparse systolic array, while maintaining the sparsity and versatility of weight pruning with reasonable overhead, and further
> 
> optimize DRAM access. A model-hardware co-design of
> 
> sparse CNN accelerator based on systolic array is proposed
> 
> to improve system performance and energy efficiency. Our
> 
> main contributions are as follows:
> 
> strategies. Thus, there is still room to further lower DRAM
> 
> access by choosing dataflow according to the storage ratio of
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20112.png)

## TACO22：SPOTS：Sparse CNN by SA+GEMM（Sparse、IM2COL）

**动机**：im2col耗时间且访问耗能，软件完成不能利用im2col和GEMM的pipeline；

> **[图片提取文字 (image.png)]:**
> volutional layer as a large, single **General Matrix-Matrix Multiplication (GEMM)** using a data reorganization transformation called Image-to-Column (IM2CoL). Unsurprisingly, many mainstream frameworks use this approach since highly optimized GEMM primitives are available (e.g., BLAS [4] or CuBLAS [30]). One method to accelerate the convolution computation is to offload the GEMM operation to a hardware accelerator. However, the Im2CoL operation accounts for a sizable fraction of the execution time (29% of the total time). Further, IM2Col performs many redundant memory accesses, which contributes to the overall energy consumption. Further, offloading only the GEMM operation to a hardware accelerator and doing the IM2CoL operation in software pre-
> 
> vents fine-grained pipelining of the IM2CoL transformation and the matrix multiplication opera-
> 
> tion. Thus, performing the IM2CoL operation in hardware avoids significant data transfer between
> 
> the CPU and the hardware accelerator.
> 
> Convolution as matrix multiplication. One approach to implement CNNs is to realize a con-
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20113.png)

> **[图片提取文字 (image.png)]:**
> creates a set of linearized patches. The IM2CoL unit consists of PUs where each PU is responsible for constructing a linear patch. As values are streamed in, the PU constructing the patch will forward overlapped elements to neighboring PUs. Once the PU collects all the values in a patch, it forwards in-order partial patches to the GEMM unit. This approach allows the IM2Col unit to read in values from the input feature map only once and reuse the values avoiding redundant memory accesses. We design a dynamically reconfigurable GEMM unit with a systolic-array-based design. It can be configured as a tall array to balance the work between IM2CoL and GEMM computation. To maintain a high PE utilization with CNN layers with varying shapes, the GEMM units can be configured as small GEMM units (Section 3.4). This dynamic reconfigurability enables our hardware to adapt to CNN layers with varying dimensions and shapes. Further, it also helps with sparsity
> 
> We propose a hardware unit for the IM2Col transformation that is synergistic and pipelined
> 
> with the hardware unit for GEMM. The IM2CoL unit reads the input feature map, a 3-D array, and
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20114.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> (c) The steps to generate the patches with two PUs
> 
> Fig. 4. Illustration of patch generation using the PUs in the IM2CoL unit. We show an IM2CoL unit with two PUs for exposition. (a) The input feature map with one channel. We show the sliding windows used to generate patches with a stride of 1. (b) The two PUs are interconnected by a ring network. (c) There are two rounds. Round 1 corresponds to patches belonging to the first row of sliding windows over the input feature
> 
> map. Similarly, round 2 corresponds to patches belonging to the second row of sliding windows.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20115.png)

im2col模块，和我的设计原理类似，但我的存储Buffer粒度更粗，不需要分布式buffer；

im2col过程只读取一次SRAM的特性不同，SPOTS作了H和W两个方向overlap的reserve，我只作了W方向的reserve，多任务场景下我作一个**row-wise的切换**（正好需要重复读取）？；

SPOTS中im2col模块的每个PU需要一个控制器来**分布式**控制3个buffer行为，而我是一个整体控制器控制top buffer的行为；

每个PU中的reserve buffer就是我的slab buffer，但我能**统一每个tile的overlap的大小**（不多消耗buffer），因此控制和设计更简单，pipeline粒度更细；

im2col+GEMM的pipeline相同；

SPOTS中im2col模块的PU分布式控制的对上接口的粒度是layer，即PUs动态协同控制完成layer输入的im2col，我的im2col设计的控制和模块很简单，更适合**多任务场景**？

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> (a) Overall architecture of SPOTS
> 
> (b) Overall IM2CoL architecture and patch unit
> 
> Fig. 3. (a) The overall architecture of our accelerator with the IM2CoL unit and a systolic-array-based GEMM unit. (b) The overall IM2CoL architecture and patch unit internals.
> 
> awareness by enabling our design to detect and skip zeros in the input feature map (Section 3.3). Figure 3(a) shows the overall architecture of our accelerator. The two main components are the unit for the IM2Col transformation and the GEMM unit. They are connected by two buffers that allow effective pipelining of the operations between the IM2Col unit and the GEMM unit. The compress unit detects and skips the zero blocks in the feature map and weights before they are sent to the GEMM unit. Next, we describe the details of each component.
> 
> The IM2CoL transformation creates a 2-D matrix from the 3-D input feature map, which reduces
> 
> ## 3.1 The IM2Col Unit
> 
> output controller.
> 
> convolution to matrix multiplication (Section 2.2). The IM2Col transformation is challenging because it inherits a part of the complexity of convolution, has complex memory access patterns, and results in redundant accesses. We propose a distributed hardware structure consisting of a series of PUs to both accelerate IM2Col and minimize the number of accesses to the elements of the input feature map. The key insight in our IM2Col unit is to exploit the localities resulting from the overlap between the patches as we slide the filters across the input feature map both vertically and horizontally. Each PU is responsible for building one patch at a time. One of our design goals is to read the input feature map only once from SRAM. To accomplish this goal, each patch unit has small local buffers that store some values that will be useful for building future patches. The PUs are also connected using a ring network, which allows the PUs to communicate elements locally and avoid redundant accesses to the input feature map in SRAM. Figure 3(b) shows the overall architecture of our IM2Col unit that consists of three main components: input controller, PUs, and
> 
> The input controller reads the input feature map from SRAM and forwards them to the appropriate PUs. Apart from sending values from the input feature map to the respective PUs, the input controller maintains extra metadata for every scheduled patch. This metadata carries information about the position of the current patch. For some convolution layers, the stride size is the same as the kernel size. In those cases, there is no overlap between the patches. For those scenarios, the input control forwards its output directly to the output controller by skipping the PUs.
> 
> Our IM2Col unit has multiple PUs within it. The PUs are the main components of the IM2Col unit for generating patches. Figure 3(b) shows the internals of the PU. Each PU has three buffers: the new buffer, the neighbor buffer, and the reserved buffer. The new buffer (N) maintains the newly
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20116.png)

> **[图片提取文字 (image.png)]:**
> fetched element received from the input controller. The neighbor buffer (G) stores the elements received from the neighboring PU. The reserved buffer (R) stores some of the elements previously received at that PU in the previous rounds. We store the row and column indices (i.e., coordinates) along with the value for each element. The control unit within each PU manages the buffer and generates patches. It decides whether an element needs to be forwarded to the neighboring PU and whether it should be maintained in the reserve buffer for future use.
> 
> A unique identifier identifies each patch (i.e., row and column index of top-left element). The control unit in a PU uses the patch identifier, the filter size, and the stride size to determine which elements need to be (1) fetched from the input feature map, (2) forwarded to the neighboring PUs, and (3) stored in the reserve buffer for future rounds. For example, all elements need to be fetched from the input feature map when a PU processes the first patch in the first round.
> 
> All elements that are necessary for adjacent patches in a given round are provided by the neighboring PUs. A PU typically receives  $K^2 - K \times S$  elements from the neighboring patches as long as it is not the first patch in a given round, where K is the size of the kernel and S is the stride size. We assign all patches that belong to the same column (i.e., column index of the top-left element) in different rounds to the same PU. Hence, the PUs also store some elements that may be useful to build patches in subsequent rounds in the reserved buffer. This procedure is repeated for all C channels in the feature map.
> 
> The total number of elements that are overlapped between the vertical patches for a given filter
> 
> size is  $C \times W \times (K-S)$ , where W is the width of the input feature map. This is the maximum data reuse that can be attained with the reserve buffer. Further, the width and the channel size are inversely proportional to each other. For example, the first few layers of a CNN often have a small number of channels that are wider. In contrast, the later layers of the CNN have larger channels of smaller width. Thus, a small reserve buffer can provide significant data reuse even for larger layers. When the number of overlapped elements between the vertical patches is larger than the size of the reserved buffer, the input controller skips the reserved buffer and fetches the element again from SRAM. In such cases, data reuse is restricted to horizontally adjacent patches. Finally, the output controller organizes patches formed by each PU and manages communications with the GEMM unit. It coordinates double buffering that enables the overlapped execution of the IM2CoL unit and the GEMM unit.
> 
> example, PU1 receives four elements (A1, A6, A2, A7) from the input controller and stores them in the new buffer in step 1. Similarly, PU2 receives two new elements (A3, A8). PU2 will receive other elements from the PU1 in subsequent steps (i.e., step 2).
> 
> In summary, our hardware IM2Col unit provides two benefits: energy efficiency and perfor-
> 
> mance. Accessing the smaller SRAM and performing integer operations (for computing on row and column indices) consumes significantly less energy than accessing DRAM and large SRAMs. Hence, our design provides significant energy benefits. Further, our distributed collection of PUs unlocks extra parallelism beyond parallelism among the channels, allowing multiple patches to be built simultaneously by different PUs in the IM2Col unit that boosts performance.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20117.png)

GEMM提供**两种dataflow**（tall tile、sub-tall-tiles），利用稀疏CNN提供的特性；

tall-shape GEMM的设计是为了降低IM2COL吞吐的需求，从而减少IM2COL的资源，意味着论文PU的**资源消耗较大**；

> **[图片提取文字 (image.png)]:**
> one of the inputs of the GEMM unit comes from the IM2Col unit. Using a tall-shaped array reduces the memory bandwidth requirement for the input arriving from the IM2Col unit. Thus, we can attain high PE utilization in the GEMM unit with less throughput from the Iм2Col unit. This helps us to build an IM2Col unit with fewer resources and memory bandwidth requirements. Second, the tall array helps our design to exploit sparsity in the output of the IM2Col unit to skip zeros and increase performance. As the width of the tall array is smaller than its height, fewer columns from the IM2Col transformation enter the systolic array at any instant of time, which increases the opportunity for detecting and skipping entire rows of inputs with zeros before entering the
> 
> There are two main benefits in using a tall systolic-array-based architecture for GEMM. First,
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20118.png)

output stationary的特点：128 rows，4 cols；

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> (c) Cycle by cycle execution of GEMM with an output-stationary dataflow
> 
> Fig. 5. Illustration of our GEMM unit. (a) Inputs to the GEMM unit. (b) A tall array for the GEMM unit. (c) Illustration of GEMM computation at various steps. We show the current inputs and the partial results computed till a step for each PE. We demonstrate the output-stationary attribute of our design.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20119.png)

> **[图片提取文字 (image.png)]:**
> sult by accumulating the partial products for a particular output element. This output-stationary dataflow ensures maximum reuse of the output data. Besides, with a tall array, SPOTS can attain high data reuse for the result of the IM2Col transformation (i.e., feature map input). More importantly, with output-stationary dataflow, there is no need for separate multiplication and accumulation units. This eliminates multiple levels of multiplication and addition and the routing logics between the two units (Section 2.3). Figure 5(a) shows the weight matrix from the filter and the output of the IM2Col transformation that forms the input to the GEMM unit. The values of
> 
> the filter matrix enter the GEMM unit's systolic array from left to right, while the result of the
> 
> IM2Col unit enters the systolic array from top to bottom. Figure 5(c) shows the various steps and
> 
> Our GEMM unit uses an output-stationary dataflow, where a given PE computes the final re-
> 
> partial results computed in the GEMM unit. Our design is parameterizable with *M* rows and *N* columns in the systolic array. In our design, each row handles multiple rows of the filter matrix. Our specific prototype used 128 rows of PEs and 4 columns. These numbers are chosen based on the characteristic of common CNN layers. Further, each row of the systolic array can be assigned multiple rows of the filter matrix depending on the scheduling mode. The majority of layers in state-of-the-art CNNs have fewer than 512 rows of the filter matrix in each convolution layer.
> 
> Each PE has a single **multiply-accumulate** (MAC) unit that uses two 16-bit fixed-point inputs and accumulates the result in a 24-bit register. To handle multiple rows of the filter matrix, each PE has K registers to compute the final result (e.g., in our design, we use K = 4). Each PE has three FIFOs. Two FIFOs are for each arriving input. The other FIFO works as the work queue for the MAC unit. In GEMM, the coordinates of the elements of the two input matrices should
> 
> for the MAC unit. In GEMM, the coordinates of the elements of the two input matrices should match before multiplying the inputs. In the fetch unit, we ensure that the inputs are sent to the PEs in the proper order; thus, we do **not** need additional logic to perform index matching inside a PE. Additionally, our output-stationary dataflow ensures all the partial products produced in a PE belong to the same output element. Next, we describe how to support sparsities in both inputs without requiring any index matching units inside the PEs.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20120.png)

**Sparse handling**

以block为粒度的结构化稀疏，column-block的稀疏Map，M1跳过列和行的0计算，M2用于gate行或列部分0的PE计算；

Sparse Weight——Buffer——Compress（skip）——GEMM；

Dense Input——Buffer——IM2COL——Compress（bitmap、skip）——GEMM；

> **[图片提取文字 (image.png)]:**
> unit in our accelerator (Figure 3(a)) identifies a block of zeros in the result of the Iм2Col transformation. It creates a bitmap for every block coming out of the IM2Col unit. If all elements in a block in the output of the IM2Col unit are zeros, the bit is set to zero for that block; otherwise, the bit is set to one. Subsequently, the input controller of the GEMM unit uses this bitmap and M1 level bitmaps for the weights (Figure 7(a)) to skip blocks of the input feature map and weights on the fly when they are all zeros. One unique feature of our approach is that we skip MAC operations involving zeros outside
> 
> the PEs and in the input controller. These have two advantages. First, we avoid the unnecessary
> 
> **Skipping zeros in the feature map and weights.** The *compress* component before the GEMM
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20121.png)

> **[图片提取文字 (image.png)]:**
> Second, detecting and skipping zeros centrally (inside the input controller) relieves the PEs from storing and processing any metadata, which reduces area and power consumption. Besides, our approach does not require any costly hardware units inside every PE to detect and match the nonzero pairs, unlike some prior work (Section 2.3). Figure 7(b) illustrates how the zero columns in the weight matrix and the zero rows in the output of the IM2Col unit are skipped. In addition to the zero blocks that we skip in the control unit, some PEs may still receive zero blocks (the gray blocks in C1, C2, and C4 columns in Figure 7(b)). This happens when a column of the weight matrix is partially zero. For those cases, the input controller sends one bit to the PE to indicate a zero block. The PEs will then ignore the blocks with all zeros, and the MAC units are gated to reduce energy consumption.
> 
> data traffic to stream the rows of feature maps and columns of filters to PEs when they are zeros.
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20122.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> (a) Our custom sparse format to store filters
> 
> (b) Skip rows and columns with all zeros
> 
> Fig. 7. (a) Our custom sparse format to store filters. (b) Illustration of how our design skips rows and columns with all zeros. (1) Weight matrix with the metadata about columns with all zeros. (2) The IM2Col result with the metadata about rows with all zeros. (3) If a row or a column is all zeros, all such rows and columns can be skipped (i.e., *and* operation of the row and column metadata). (4) GEMM computation when rows and columns are skipped. For example, the first element of column C4 will be fetched by the first PE in cycle 2 (skipping columns C2 and C3).
![image.png](meeting-25%2011%2019%EF%BC%88LLM%E8%B0%83%E5%BA%A6%E3%80%81FSD%E3%80%81CV%20Model%E3%80%81Acc%E3%80%81VAE%EF%BC%89/image%20123.png)

**GEMM reconfiguration**

两个数据流的NoC细节：IM2COL的多对多网络，SA是PE之间传递数据。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 8. (a) Enhancements to reorganize the tall systolic array (SA) as multiple GEMM units. (b) Illustration of how inputs are distributed in the configuration with multiple GEMM units.
> 
> enhancement allows our design to be more **adaptive** to different layer shapes and thus maintains high PE utilization under different conditions. Figure 8(a) demonstrates how a tall array can be used as two smaller arrays using the multiplexers. Hence, the PEs now either can receive the input from the PEs above (i.e., it forms a tall array) or can get the input from a different IM2Col unit. These multiplexers can be configured based on the mode register dynamically depending on the structure of a layer. The weight matrix is broadcast to all small systolic arrays when the GEMM unit is configured as smaller systolic arrays. Each small GEMM unit receives the feature map input from their assigned IM2Col units. The two GEMM units compute two independent groups of columns of the final result matrix (i.e., GEMM 1 computes result columns from 0 to N/2, GEMM computes the columns from N/2+1 to N). In our prototype, we have four IM2Col units. There is one main IM2Col and three smaller IM2Col units to support the two configurations. The main IM2Col unit is used for the tall array configuration. For the other configuration, all four IM2Col units are being used. This dynamic reorganization of the GEMM unit's systolic array coupled with the multiple IM2Col units enables our hardware to maintain high PE utilization for various CNN layers with different shapes.
![image.png](meeting-25%2012%2011%EF%BC%88DiT%E3%80%81Diffusion%E3%80%81GPU%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2020.png)

**Load Balance**

部分PE计算0而空闲，部分PE则需要计算负载，不同PE的计算负载不同，降低利用率；

SPOTS在PE前分别去除weights和IFM都是0的block（O(n+m )），从而跳过无效计算，但其中之一为0的block时会出现负载不均衡，SPOTS没有处理；

> **[图片提取文字 (image.png)]:**
> stationary or an output-stationary dataflow. Subsequently, an input-stationary dataflow can be weight stationary or feature map stationary. In input-stationary dataflow, one of the inputs is held stationary in the PEs while the other input is broadcast to each PE to ensure data reuse. When there is an uneven distribution of non-zeros in the inputs, some PEs may receive fewer inputs, forcing them to remain idle until the other PEs process their inputs before they all can receive new inputs.
> 
> SPOTS adopts an output-stationary dataflow with a tall systolic array (Section 3.2). In a tall sys-
> 
> source of the load imbalance in an accelerator. Generally, accelerators adopt either an input-
> 
> tolic array, the feature map values are passed through as many PEs as possible to ensure maximum data reuse. As described in Section 3.3, we skip the zeros in the feature map input inside the input controller before entering the systolic array. Thus, the non-zeros are skipped for *all* PEs (not just for an individual PE) in the systolic array. SPOTS's early zero detection approach avoids the potential load imbalance caused by the uneven distribution of non-zeros in the feature map. Similarly, SPOTS detects and skips the zeros in the weights outside the PE when the zeros span the whole
> 
> For partially zero columns in the weight matrix (i.e., some blocks are zeros, some non-zeros), some PEs may receive a zero block while others receive a non-zero block. This can introduce a work imbalance between the PEs. The load imbalance among the PEs can be quantified using the metric proposed by [11] where the load imbalance is quantified as follows:
> 
> filters (i.e., an entire column of the weight matrix).
> 
> $$imbalance\_percentage = \frac{maximum\_work - average\_work}{maximum\_work} \times \frac{n}{n-1}. \tag{1}$$
>  The imbalance percentage corresponds to the percentage of time the PEs with less work are not
> 
> engaged in useful work and are waiting for the PE with the maximum work. A perfectly balanced work distribution results in zero imbalance percentage. Thus, lower imbalance percentage implies fewer idle cycles for the PEs.
> 
> One way to improve the load balance in the PEs is to rearrange (shuffled) the non-zero blocks in
> 
> the weights offline to make the distribution of the non-zero blocks more balanced. However, this reshuffling can change the position of the output channels and thus requires an additional step to reorder the output before the next layer uses them [15, 23]. In Section 5 we present the average
> 
> imbalance percentage for all four CNN architectures with SPOTS. Since the PEs in SPOTS did not suffer from load imbalance, we did not use any additional load balancing steps to avoid introducing extra complexity to the design.
![image.png](meeting-25%2012%2011%EF%BC%88DiT%E3%80%81Diffusion%E3%80%81GPU%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2021.png)

**实验**

> **[图片提取文字 (image.png)]:**
> ## 4 EXPERIMENTAL METHODOLOGY
> 
> quency. FreePDK 45 does not include SRAM cells. Thus, we separately model the area and power of SRAM/DRAM using Cacti 7.0 [3]. Table 1 provides the parameters of the SPOTS prototype and the area breakdown for different components. We perform cycle-accurate simulation of the RTL model of SPOTS in Verilog using Verilator. We used the traces from the RTL simulation and estimated the power consumption of our design with Synopsys's PowerPrime tool. During our simulation, we executed each layer at a time. The pruned weights are preprocessed and are provided in our proposed sparse format. For the input feature map, we extracted each layer's data from the models in Caffe. We also developed additional infrastructure to perform fast design space exploration and to collect statistics.
> 
> We built a prototype of our design in Verilog and synthesized it using the Synopsys Design Com-
> 
> piler with the FreePDK 45nm technology [39]. Our design achieves a maximum of 500 MHz fre-
![image.png](meeting-25%2012%2011%EF%BC%88DiT%E3%80%81Diffusion%E3%80%81GPU%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2022.png)

## ITSC24：Inter-Operator Schedule for CNN on GPU

## ICS24：HW-aware(quantization) NAS for ViT

## ISCA23：Eagar Correlation Prediction based FFN-Attention Co-optimized Transformer Acc

## DAC22：length-adaptive sparse transformer Acc

## MICRO16：fused CNN Acc