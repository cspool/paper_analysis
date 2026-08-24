# 3 Method

In this section, we provide an in-depth discussion of our method. Section [3.1](#page-2-0) introduces a simple yet effective approach that enables a single tuning process to generate models with CoT with different lengths. This stage also serves as an initial step for subsequent refinements. Next, in Section [3.2,](#page-3-0) we explore multiple scenarios in which we can apply CoT-Valve to construct the dataset MixChain. In Section [3.3,](#page-4-0) we propose several advanced methods that take advantage of long-to-short datasets to improve precision and control over the generated reasoning paths in compressible fine-tuning.

## <span id="page-2-0"></span>3.1 Length-Compressible CoT Tuning

Our primary objective is to achieve a new way to control the length of reasoning paths after training a reasoning model. Existing approaches, such as prompt-based control, explicitly define sequence length in the prompt [\(Han et al.,](#page-8-18) [2024\)](#page-8-18) or utilize summary tokens [\(Ding et al.,](#page-8-19) [2024\)](#page-8-19) for guidance. However, these methods offer only limited control

over the length of CoT generated. For instance, requesting a sequence of less than 20 tokens may result in the model generating over 350 tokens (see Table [12](#page-12-0) in the Appendix), and these methods struggle to produce answers with very short lengths. To address these limitations, we introduce CoT-Valve for training one model but can adjust the length of reasoning paths.

Consider a reasoning model defined by the parameter θ. For a given question q in the dataset D, the probability of generating an answer a and its reasoning thoughts {ti} n <sup>i</sup>=1 given the question q can be described by:

$$p(a \mid t_1, \dots, t_n, q; \theta) \prod_{i=1}^{n} p(t_i \mid t_{< i}, q; \theta) \quad (1)$$

where {ti} n <sup>i</sup>=1 might include errors or unnecessary details. With short synthesized or human-annotated explanations {ti} m <sup>i</sup>=1 with m < n, the training objective is to adjust the parameter in such a way that the chain is shortened while still yielding the correct answer:

$$\max_{\Delta \theta} \mathbb{E}_{(q,a) \sim \mathcal{D}} p\left(a \mid t_1, \dots, t_m, q; \theta + \Delta \theta\right)$$

$$\prod_{i=1}^{m} p\left(t_i \mid t_{< i}, q; \theta + \Delta \theta\right) \quad (2)$$

and ∆θ denotes the change in the parameter space that steers the model towards generating a more concise chain.

Since the model, with and without ∆θ, outputs the same final answer, ∆θ can be interpreted as a task vector [\(Ilharco et al.,](#page-8-20) [2022\)](#page-8-20). The task here is to control the length of the CoT, provided that the only difference in the training set lies in intermediate reasoning steps {ti} n <sup>i</sup>=1. Those reasoning paths are different in length but ultimately lead to the same final answer. Thus, we can control the task vector to achieve the goal of adjusting the length of CoT. ∆θ is designed within a parameter-efficient space, functioning as an external branch for inference that incurs minimal overhead. Controlling this external branch enables the manipulation of the length of the reasoning path.

Task Arithmetic: Interpolation and Extrapolation of ∆θ. To manipulate this update within the parameter space, we can control the magnitude of a ∆θ as an arithmetic operation. We use two primary operations on ∆θ here: interpolation and extrapolation. Let α denote the magnitude of ∆θ for LoRA.

<span id="page-3-1"></span>> **[图片提取文字 (无描述)]:**
> Stage 2: Generate Long-to-Short Stage 3.a (CoT-Valve++): **Stage 1:** Find  $\Delta\theta_1$ ,  $\Delta\hat{\theta}_1$  or  $\Delta\hat{\theta}_2$ Reasoning Dataset Large-scale Post-trained: BSAA' Long CoT Finetuned/Distilled: Long CoT Interpolate: Long Path  $\alpha \Delta \hat{\theta}_2$  $\Delta \hat{\theta}_2 = \hat{\theta}_2 - \theta_2$  $\{\beta_L, CoT_L\}$  $\Delta\theta_1$  $\Delta\theta'$  satisfies Finetuned:  $\Delta \hat{\theta}_1$ Short CoT Medium Path  $\Delta \hat{\theta}_2$ Stage 3.b (CoT-Valve+P): Extrapolate: Short Path  $\gamma \Delta \hat{\theta}_2$ OOO Synthesized Reasoning Path System 1: Short CoT ··· Gradient Update
![](_page_3_Figure_0.jpeg)

Figure 2: Illustration of CoT-Valve. In Stage 1, we first determine  $\Delta\theta$  from distilling or post-training. Then, the trained  $\Delta\theta$  is utilized to construct the MixChain dataset. Using this dataset, we can then apply two enhanced training methods to achieve more precise control over reasoning paths, or to shorten the reasoning paths as needed.

When  $\alpha$  falls within the range of (0,1), the model smoothly transitions between longer and shorter reasoning paths, similar to weight interpolation between two models (Frankle et al., 2020; Team et al., 2025). When  $\alpha>1$ , extrapolation is introduced, further shortening the reasoning path beyond what was observed during training. This enables an exploration of the minimal reasoning length required to arrive at a given answer. Thus, by adjusting  $\alpha$  at inference, we can modulate the model's behavior, with each value of  $\alpha$  corresponding to different CoT lengths.

**Application** Unlike prompt-based approaches that can only regulate the overall length of the reasoning process using prompt words,  $\Delta\theta$  provides finer granularity control.  $\Delta\theta$  is served in the external parameter space. This allows for greater flexibility in adjusting the reasoning trajectory. Specifically, it facilitates the selective retention of long-chain reasoning in certain thoughts while applying stronger compression to simpler reasoning segments. As a result, reductions in chain length can be localized to specific portions of the inference process rather than being uniformly applied across the entire reasoning path. We remain the design of this segment selection in future work.

## <span id="page-3-0"></span>3.2 Construct the MixChain Dataset

A crucial thing for the above process is the construction of the training dataset, especially the reasoning chain  $\{t_i\}_{i=1}^n$ . To have reasoning chains with different lengths, previous approaches rely on multiple rounds of sampling, selecting reasoning paths under different random seeds, or using some hand-crafted way to remove parts of the answer (Chen

et al., 2024).

We introduce MixChain, a dataset inherently generated by our method that contains reasoning paths of varying lengths. This dataset is structured such that each question is associated with multiple reasoning paths, with lengths progressively decreasing from long to short. By simply adjusting the parameter  $\alpha$ , our approach avoids the need for repeated sampling and achieves this diverse set of reasoning paths. In contrast to multi-sampling techniques, MixChain enables a more reliable and consistent generation of shorter reasoning paths while simultaneously capturing a spectrum of reasoning lengths. To construct MixChain, we consider two possible scenarios:

- If a well-annotated dataset with humanlabeled solutions is available, such as GSM8K (Cobbe et al., 2021b) or PRM800k (Lightman et al., 2024), it can be leveraged to fine-tune the model for generating shorter reasoning chains as a cold start ( $\theta_1 \rightarrow \tilde{\theta}_1$  and  $\theta_2 \rightarrow \tilde{\theta}_2$  in Figure 2).
- In the absence of a dataset containing explicit reasoning paths, or when only final answers are available without full explanations, training solely on final answers is unlikely to enable the model to generate reasoning steps. To address this limitation, we propose an alternative method for constructing Mix-Chain. Specifically, we leverage an existing base LLM (e.g., LLaMA-3.1-8B or Qwen-32B-Instruct) as θ<sub>1</sub> and use its corresponding reasoning model (e.g., DeepSeek-R1-Distill-Llama-8B or QwQ-Preview) to derive Δθ.

The parameter update between these models serves as a form of linear interpolation, enabling the transition from θ<sup>1</sup> to θ2. This transition is then used to construct the dataset, as illustrated in Figure [2,](#page-3-1) where the parameter shift is represented by θ<sup>1</sup> → θ2.

