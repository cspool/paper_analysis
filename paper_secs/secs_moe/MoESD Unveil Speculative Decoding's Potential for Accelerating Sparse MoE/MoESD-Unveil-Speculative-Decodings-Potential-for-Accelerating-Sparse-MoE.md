# MoESD: Unveil Speculative Decoding's Potential for Accelerating Sparse MoE

Zongle Huang<sup>1</sup> Lei Zhu<sup>2</sup> Zongyuan Zhan<sup>2</sup> Ting Hu<sup>2</sup> Weikai Mao<sup>2</sup> Xianzhi Yu<sup>2</sup> Yongpan Liu1,3† Tianyu Zhang2†<sup>∗</sup>

<sup>1</sup>Tsinghua University <sup>2</sup>Huawei Noah's Ark Lab <sup>3</sup>BNRist {huangzl23}@mails.tsinghua.edu.cn {ypliu}@tsinghua.edu.cn {zhulei168,zhanzongyuan,huting35,maoweikai,yuxianzhi,zhangtianyu59}@huawei.com

### Abstract

Large Language Models (LLMs) have achieved remarkable success across many applications, with Mixture of Experts (MoE) models demonstrating great potential. Compared to traditional dense models, MoEs achieve better performance with less computation. Speculative decoding (SD) is a widely used technique to accelerate LLM inference without accuracy loss, but it has been considered efficient only for dense models. In this work, we first demonstrate that, under medium batch sizes, MoE surprisingly benefits more from SD than dense models. Furthermore, as MoE becomes sparser – the prevailing trend in MoE designs – the batch size range where SD acceleration is expected to be effective becomes broader. To quantitatively understand tradeoffs involved in SD, we develop a reliable modeling based on theoretical analyses. While current SD research primarily focuses on improving acceptance rates of algorithms, changes in workload and model architecture can still lead to degraded SD acceleration even with high acceptance rates. To address this limitation, we introduce a new metric *target efficiency* that characterizes these effects, thus helping researchers identify system bottlenecks and understand SD acceleration more comprehensively. For scenarios like private serving, this work unveils a new perspective to speed up MoE inference, where existing solutions struggle. Experiments on different GPUs show up to 2.29x speedup for Qwen2- 57B-A14B at medium batch sizes and validate our theoretical predictions.

### 1 Introduction

Recent years have witnessed remarkable success in Large Language Models (LLMs), with Mixture of Experts (MoE) architectures showing tremendous potential. Unlike dense models use a single feed-forward network (FFN) to process all inputs, MoE models replace the FFN with multiple specialized "expert" networks plus a router that selectively activates only a few experts for each input token. Such sparsity in structure enables MoEs with more parameters to achieve higher computational efficiency, and multiple state-of-the-art LLMs, such as DeepseekV3 [\[1\]](#page-10-0) and Qwen2.5-Max [\[2\]](#page-10-1), are all MoEs. MoE model architectures are evolving toward larger scales with increased sparsity [\[3,](#page-10-2) [4,](#page-10-3) [1\]](#page-10-0) and more balanced workload distribution among experts [\[5,](#page-10-4) [6\]](#page-10-5).

Speculative decoding (SD) is a lossless technique to accelerate LLM inference, but conventional wisdom suggests that its efficacy diminishes when applied to MoEs. In SD, a smaller draft model is introduced to rapidly generate multiple candidate tokens, while the larger target model verifies these predictions in parallel, preserving only correctly speculated tokens. For dense models' inference, the time taken to generate a single token and verify multiple ones is roughly the same, as both tasks require the full set of parameters to be loaded once. Therefore, SD gains acceleration through fewer forward

<sup>∗</sup> † Corresponding Author.

rounds of the target model and shorter decoding time of the draft model. However, this acceleration has been demonstrated to diminish in MoEs [\[7,](#page-10-6) [8\]](#page-10-7), as the multiple draft tokens in verification activate more experts than a single token, leading to larger memory access and significantly longer verification time compared to a standard decoding step.

In this work, we challenge the conventional belief and demonstrate that, under a moderate batch size, SD can be more effective for MoEs than for dense models. Our key insight is that when the batch size is moderate such that all experts are already activated in a single decoding step, verifying multiple draft tokens will not incur additional expert parameter loading costs. Furthermore, as the MoE becomes sparser, each expert processes fewer tokens per parameter loading, leading to lower utilization of arithmetic units and thereby creating greater acceleration opportunities for SD.

The insight above is supported by comprehensive theoretical analyses, through which we identify a new metric *target efficiency* to quantify how systemic factors (such as workload and target model architecture) affect SD speedup. In contrast to existing SD works that use acceptance rate [\[9,](#page-10-8) [10,](#page-10-9) [7,](#page-10-6) [11\]](#page-10-10), an algorithmic metric to evaluate how accurately the draft model speculates the target model, our proposed target efficiency isolates extrinsic factors like algorithm selection and focuses on intrinsic system bottlenecks caused by the target model's computational and memory access requirements. As demonstrated in the following sections, even with similar acceptance rates, systemic factors can greatly impact SD effectiveness, making our metric target efficiency necessary for a comprehensive understanding of SD acceleration.

As a further step, we build a quantitative modeling of SD speedup for MoE based on these theoretical analyses. The consistent matching between our modeling and experiment results confirms the reliability of our analyses. Additionally, the modeling itself provides an approach for analyzing the execution time of different components, making the end-to-end SD acceleration results more transparent and explainable.

Our work offers a new perspective for lossless MoE acceleration, particularly well-suited for private serving scenarios [\[12,](#page-10-11) [13,](#page-10-12) [14\]](#page-10-13). Private serving has gained popularity among enterprises seeking to safeguard data and model security, with typical applications such as in-house chatbots. These environments typically process moderate batches containing tens of requests. Additionally, our findings can be applied to latency-critical scenarios where large batch sizes are infeasible, or memoryrestricted environments where MoEs exceed GPU capacity.

In Summary, the main contributions of our work are:

- We refine the conventional belief that speculative decoding cannot effectively accelerate MoEs, demonstrating that under moderate batch sizes, SD is actually more effective in a wider range of batch sizes for sparser MoEs than dense models.
- Based on theoretical analysis, we developed a reliable modeling for SD speedup, thus making the acceleration process transparent and explainable. Existing metrics only assess algorithmic optimization efficiency and cannot fully explain SD speedup, so we introduce a new systemic metric *target efficiency* that reveals speedup opportunities inherent in the target model.
- Our findings can be applied to accelerate scenarios like private serving. Experiments on various GPUs with the Qwen2-57B-14A-Instruct model demonstrate that SD achieves the highest speedup at the moderate batch size, reaching 2.29x. These experiments also validate our theoretical prediction that SD is more favorable for sparser MoEs.

### 2 Related Work

MoE acceleration. MoE has emerged as a promising LLM architecture, and many techniques optimize its inference. Model compression methods, including pruning [\[15,](#page-11-0) [16\]](#page-11-1), quantization [\[17,](#page-11-2) [18\]](#page-11-3), distillation [\[19,](#page-11-4) [20\]](#page-11-5), and decomposition [\[21,](#page-11-6) [22\]](#page-11-7), have been applied to MoEs and achieved great acceleration. They sacrifice model quality for speedup, as in dense models. When MoEs are too large to fit in GPU memories and offloading becomes a necessity, several system-level approaches have emerged to optimize inference latency through improved scheduling techniques. Expert prefetching [\[23,](#page-11-8) [24\]](#page-11-9) predicts and pre-loads experts for upcoming layers based on previous activation patterns, thus overlapping expert loading with current layer computation. Expert caching [\[25,](#page-11-10) [26\]](#page-11-11) caches most frequently activated experts in GPU memory, leveraging expert locality to reduce expensive

offloading. Compared to them, our work unveils a new perspective for MoE acceleration that is lossless and doesn't depend on expert imbalance.

**Speculative Decoding.** Speculative decoding (SD), initially proposed by [10] and [9], has emerged as a widely adopted technique for accelerating LLM inference without sacrificing generation quality. Basic SD employs a smaller model to rapidly generate draft tokens, which are then verified in parallel by the target model that needs to be accelerated. Afterwards, more algorithms are developed to lift the acceptance rate of draft tokens. [27, 11, 28, 29, 7, 30, 31] adopt tree-structured generation patterns rather than chains to explore a broader range of potential completions. [1, 11, 7, 30, 31] propose to replace draft models with specifically trained speculative heads integrated in the target model.

Despite advances in SD algorithms, it has long been considered ineffective for *large batches* [32, 33, 27] or *MoE* [7, 8], since the verification time in these cases significantly increases. Until recently, MagicDec [34] first challenged that in long-sequence regimes, SD can effectively accelerate *large batches*, primarily due to the significantly increased KV cache altering the computation-to-memory access ratio of the model. However, SD research for *MoE* remains unexplored. In response, our work fills this gap, unveiling that under certain conditions, SD can effectively accelerate MoE models.

### 3 Theoretical Analysis

In this section, we present the theoretical analyses supporting our conclusion that SD can be more effective for MoE than dense models at moderate batch sizes. We begin by formalizing general SD speedup and introducing our new metric *target efficiency* (Sec. 3.1). Then, we focus on MoEs, analyzing how workload and MoE sparsity collectively affect the number of activated experts and SD speedup (Sec. 3.2). Based on these analyses, we develop a performance model that aligns with GPU results (Sec. 3.3). We further discuss the practical value of our theoretical findings (Sec. 3.4).

**Preliminaries.** LLM inference time is collaboratively determined by computation and memory access. When an operator is processed on a GPU, memory access and computation operations are pipelined and overlapped, causing the more time-consuming operation to become the bottleneck and determine the overall processing time, as depicted by the roofline model [35, 36]. The roofline model ridge point (RP) of hardware and the arithmetic intensity (AI) of software are defined as Eq. 1. When AI < RP, the system is *memory-bound*, and adding more computation will not significantly increase processing time. When AI > RP, the system is *compute-bound*, and increases in computation will directly reflect in processing time. In this paper, when we describe a system as "more memory-bound", we mean  $\frac{AI}{RP}$  is smaller.

<span id="page-2-1"></span>
$$\mathbf{RP} = \frac{\text{peak computation power (unit: Flops)}}{\text{peak memory bandwidth (unit: bytes/second)}} \quad \mathbf{AI} = \frac{\text{computation operation (unit: times)}}{\text{memory access volume (unit: bytes)}} \quad (1)$$

### <span id="page-2-0"></span>3.1 Formulation of Speculative Decoding Speedup and Target Efficiency

We first formalize the processing time of speculative decoding, denoted as  $T_{SD}$ . To generate a sequence of length S, SD goes through R rounds, each containing three stages: ① the draft model proposes  $\gamma$  tokens as specified by the speculation strategy; ② the target model verifies these tokens; ③ rejection sampling [9] discards incorrectly predicted tokens based on logits from target and draft models. We use  $T_T(b,s)$  and  $T_D(b,s)$  to represent the time for once forwarding of the target and draft model, respectively, where b and s are the formal arguments for batch size and the number of tokens to process. Therefore, the time for processing a batch containing B requests is given by:

$$T_{SD} = R \times (T_{propose} + T_{verify} + T_{reject}) = R \times \left(\gamma \cdot T_D(B, 1) + T_T(B, \gamma) + T_{reject}\right)$$
(2)

<span id="page-2-2"></span><sup>&</sup>lt;sup>2</sup>Since we work with typical sequence lengths and moderate batch sizes, the impact of KV-cache on performance is limited, allowing us to omit the already generated sequence length from our analysis. For cases where KV-cache becomes the dominant factor, see [34].

Then the speedup of SD to normal auto-regression decoding  $T_{AR}$  is given by:

$$Speedup = \frac{T_{AR}}{T_{SD}} = \frac{S \cdot T_T(B, 1)}{R \cdot \left(\gamma \cdot T_D(B, 1) + T_T(B, \gamma) + T_{reject}\right)}$$

$$= \frac{S}{R} \cdot \frac{1}{\gamma \cdot \frac{T_D(B, 1)}{T_T(B, 1)} + \frac{T_T(B, \gamma)}{T_T(B, 1)} + \frac{T_{reject}}{T_T(B, 1)}}$$
(4)

<span id="page-3-2"></span>
$$= \frac{S}{R} \cdot \frac{1}{\gamma \cdot \frac{T_D(B,1)}{T_T(B,1)} + \frac{T_T(B,\gamma)}{T_T(B,1)} + \frac{T_{reject}}{T_T(B,1)}} \tag{4}$$

 $\frac{S}{R}$  represents the average length of accepted tokens per SD round, which can be further expressed as  $\sigma \times (\gamma + 1)$ . Here,  $\sigma$  is the ratio of actually generated tokens to the theoretical maximum if all draft tokens were accepted. We note that  $\sigma$  differs from the acceptance rate  $\alpha$  commonly referenced in previous works [10, 9, 7], which represents the probability of the target model accepting a new draft token given the prefix.  $\sigma$  can be computed from  $\alpha$  as shown in Eq. 5. The numerator follows from [10], and the denominator accounts for all  $\gamma$  draft tokens accepted, plus a bonus token generated during the forward verification pass.

<span id="page-3-1"></span>
$$\sigma = \frac{expected \ generated \ tokens}{maximal \ possible \ accepted \ tokens} = \frac{\frac{1 - \alpha^{\gamma + 1}}{1 - \alpha}}{\gamma + 1}$$
 (5)

The denominator of Eq. 4 consists of three terms.  $\frac{T_D(B,1)}{T_T(B,1)}$  is the ratio of draft-model forward time over target-model forward time, reflecting the relative volume of draft and target models. It is also kept small (usually less than 1/10 [27, 7, 10]) to ensure the speculation is efficient.  $\frac{T_{reject}}{T_T(B,1)}$  is even smaller, since  $T_{reject}$  only involves sampling rather than model inference.  $\frac{T_T(B,\gamma)}{T_T(B,1)}$ , which is the ratio of multi-token forward time over single-token forward time, has the biggest value among these three items and significantly affects the final speedup. As indicated by Eq. 4, its increase causes speedup reduction. Two different factors drive its increase, explaining SD's ineffectiveness under (1) large batches for both dense models and MoE, and (2) MoE with small batches, respectively:

- (1) The compute-boundness. The model's  $\frac{T_T(B,\gamma)}{T_T(B,1)}$  approaches 1 when more memory-bound (smaller batch size B) but increases to  $\gamma$  when more compute-bound (larger batch size B).
- (2) The extra memory loads. For small Bs,  $T_T(B, \gamma)$  is notably greater than  $T_T(B, 1)$  as more experts are activated and need to be loaded. Since the system is still memory-bound at small Bs, memory load profoundly determines the processing time.

Therefore, we define *target efficiency* as  $\frac{T_T(B,1)}{T_T(B,\gamma)}$ , which helps understand the systemic causes of SD acceleration degradation as listed above. Our experiments shown in Fig. 2 demonstrate that target efficiency consistently reflects the trend of SD speedup variations. Despite the importance of this value, previous works have rarely noticed it, primarily due to differences in research focus. Previous SD research mainly addresses the question by lifting the acceptance rate:

Given the target model, which draft model or algorithm achieve better speedups?

In contrast, our work focuses on the following question by examining target efficiency:

Under the same level of algorithmic optimization, which types of target models or workloads are more favorable for SD?

We believe target efficiency help researchers understand SD more comprehensively. Even when target-draft pairs have the same acceptance rate  $\alpha$ , changes in the target model's architecture and the workload can significantly affect overall speedup. By introducing target efficiency, we can decouple algorithmic optimization from systemic optimization, thus helping to identify the systemic acceleration bottlenecks and assess potential speedup.

#### <span id="page-3-0"></span>Moderate Batch Size Enables Speculative Decoding Speedup for MoE

Although SD is ineffective for MoE under small batches, we demonstrate in this subsection that at moderate batch sizes—an overlooked regime in previous studies—SD speedup increases and benefits more from MoE with higher sparsity. Essentially, when the batch size falls within ranges where

<span id="page-4-3"></span>![](_page_4_Figure_0.jpeg)

Figure 1: Activation status and workload of experts. (a) and (b): Comparison between theoretical and actual number of activated experts N(t) on different datasets. (a) is for Deepseek-V2-Lite-Chat  $(\rho = 6/62)$  and (b) is for Qwen1.5-MoE-Chat  $(\rho = 4/60)$ . (c): Normalized number of tokens to process per expert  $(\overline{T_{exp}})$  versus MoE sparsity  $(\rho)$  for given input token count T.

all experts are activated but remain far from being assigned adequate workloads, FFNs become memory-bound, presenting an opportunity to leverage computational power almost for free through SD. To demonstrate this, we first formalize the expected number of activated experts, and then show MoE FFNs become more memory-bound as the model becomes sparser.

We use the Bernoulli random variable X to indicate the activation for experts:  $X_i = 1$  for expert i being activated, 0 otherwise. For simplicity, we assume Xs are i.i.d. Then, the expected number of activated experts N can be expressed as Eq. 6, where E denotes the total expert count and  $Pr(X_i)$  represents the probability that the  $i^{th}$  expert is activated.

<span id="page-4-0"></span>
$$N = \sum_{i} \mathbb{E}[X_i] = \sum_{i} Pr(X_i) = E \cdot Pr(X)$$
 (6)

Given t tokens passed through the MoE gate, then Pr(X) is expressed as Eq. 7. K denotes the number of activated experts per token, which is an architectural hyperparameter for MoE:

<span id="page-4-1"></span>
$$Pr(X) = 1 - Pr($$
 None of the  $t$  tokens activates the expert  $) = 1 - (\frac{E - K}{E})^t$  (7)

Therefore, the overall expression of N(t) is given by Eq. 8. Our derivation assumes uniformly activated experts, which is reasonable for well-trained MoE models. Load imbalance among experts can lead to routing collapse [5] and decrease computational efficiency in expert-parallel deployment [6], so state-of-the-art MoE models are typically trained with methods like incorporating auxiliary loss [37, 6] to ensure that experts have balanced loads. The experiment results also verified our theoretical derivation of N(t), as shown in Fig. 1a and 1b.

<span id="page-4-2"></span>
$$N(t) = E \cdot \left(1 - \left(\frac{E - K}{E}\right)^t\right) \tag{8}$$

We then solve how many tokens can lead to full activation. Since N(t) asymptotically approaches E when t tends to infinity, and in practice N(t) should be a finite integer, we deem  $N(t) > \tau E$  as almost full activation, where  $\tau$  is usually a large ratio such as 0.95. We further express  $K = \rho E$ , where  $\rho$  is the sparsity of MoE, then the token threshold  $T_{thres}$  can be solve by:

$$N(T_{thres}) = E \cdot \left(1 - (1 - \rho)^{T_{thres}}\right) \ge \tau E \quad \Rightarrow \quad T_{thres} = \lceil \log_{(1-\rho)} (1 - \tau) \rceil \tag{9}$$

Therefore, when B exceeds  $T_{thres}$ , the number of activated experts saturates, causing the  $B\gamma$  tokens in verification to incur only marginally larger memory access. Having addressed the second factor (namely, extra memory loads) for  $\frac{T_T(B,\gamma)}{T_T(B,1)}$ 's increase analyzed in Sec. 3.1, we now turn to the potential limitations caused by the first factor of compute-boundness. If such Bs make the system compute-bound, SD would also fail to accelerate MoE effectively. Our answer to this concern is: Sparser MoEs *delay* the transition from memory-bound to compute-bound when input tokens count increases.

We have obtained that given t tokens, N(t) experts are activated. Since each token activates K experts, the number of tokens each expert needs to process on average  $\overline{T_{exp}}$  can be computed as:

$$\overline{T_{exp}}(t;\rho) = \frac{t \cdot K}{N} = \frac{t \cdot (\rho E)}{E \cdot \left(1 - (1 - \rho)^t\right)} = \frac{\rho t}{1 - (1 - \rho)^t} \tag{10}$$

#### <span id="page-5-1"></span>**Algorithm 1** The Modeling of SD Speedup and Corresponding Fitting Method

```
1: Measurement Input: A total of m measurements denoted as M. Each M_i, i = 1, 2, ..., m contains the
  attributes including batch size B, draft length \gamma, number of activated experts per token K, total number of
  experts E, the ratio of accepted token counts to the maximal possible accepted tokens \sigma, Speedup for the
  actual speedup achieved.
```

```
2: Output: The optimal fitting parameter params*.
```

```
3: def ComputeSpeedup(params, B, \gamma, K, E, \sigma):

             bias, k_1, k_2, k_3, draft_bias, draft_k, reject_bias, reject_k, \lambda, s = params N_{ar} = E \cdot (1 - ((E - K)/E)^B), T_{ar} = B \cdot K/N_{ar} ar\_time = bias + k_1 \cdot G(B; \lambda RP, s) + k_2 \cdot N_{ar} + k_3 \cdot G(T_{ar}; \lambda RP, s)
                                                                                                                                           ▶ Unpack parameters
 5:
                                                                                                                                 ▶ Compute AR forward time
 6:
              N_{sd} = E \cdot (1 - ((E - K)/E)^{B\gamma}), T_{sd} = B \cdot \gamma \cdot K/N_{sd}
 7:

              verify\_time = bias + k_1 \cdot G(B\gamma; \lambda RP, s) + k_2 \cdot N_{sd} + k_3 \cdot G(T_{sd}; \lambda RP, s)
 8:
 9:
              draft\_time = draft\_bias + draft\_k \cdot G(B; \lambda RP, s)

             reject\_ume = reject\_bias + reject\_k \cdot B \rightharpoonup Compute rejection sampling time Speedup = <math>\sigma \cdot (\gamma + 1) \cdot \frac{a_{r\_time}}{draft\_time + ar\_time + verify\_time + reject\_time} \rightharpoonup Compute the speedup as formalized in Eq. 4
              reject\_time = reject\_bias + reject\_k \cdot B
                                                                                                                     ▶ Compute rejection sampling time
11:
12:
              return Speedup
13: params^* = \underset{params}{\operatorname{argmin}} \frac{1}{2} \sum_{i=1}^{m} \left( \textit{ComputeSpeedup}(params, \mathbf{M}_i.B, \mathbf{M}_i.\gamma, \mathbf{M}_i.K, \mathbf{M}_i.E, \mathbf{M}_i.\sigma) - \mathbf{M}_i.Speedup \right)
      ▶ Decide the optimal params* by fitting the model to the measured inputs using the least squares criterion.
```

As proven in Appendix and shown in Figure 1c, given t = T > 1,  $\overline{T_{exp}}(T; \rho)$  decreases with  $\rho$ , indicating that as MoE becomes sparser, each expert processes fewer tokens per parameter loading. Consequently, the system running sparser MoEs is more *memory-bound*, leading to lower utilization of arithmetic units. The verification stage can therefore leverage these spare resources without notably increasing processing time. In contrast, dense models are extreme cases with  $\rho = 1$ , where the FFN

consistently approaches the maximal possible arithmetic intensity of T, and the system transitions

rapidly to the compute-bound regime as T increases. We should note that our conclusion is based on a relatively large MoE FFN portion in the whole model, which holds true for current MoE models whose most parameters are experts. In a hypothetical extreme case where Attention dominates and the MoE FFN is negligible, MoE's sparsity would have

only a limited impact on overall system performance as indicated by Amdahl's Law.

### <span id="page-5-0"></span>A Modeling Method for Speculative Decoding Speedup

Given the numerous factors affecting final speedup, quantitatively understanding each factor's impact is challenging. Therefore, we developed a modeling method that makes SD speedup results more explainable and transparent. As demonstrated by Eq. 4, the core of modeling SD speedup lies in characterizing the model's forward pass time. Based on theoretical analysis in previous sections, we identified three primary factors affecting forward execution: (1) the roofline model effect, (2) the number of active experts, and (3) expert load. Since GPU execution is dynamic in practice, and not all operators are optimized to their theoretical limits, we introduced several parameters for relaxation. The values of these parameters are then automatically determined by fitting GPU measurements. These factors and their impacts on execution time are examined as follows.

(1) The roofline model effect. It manifests as execution time increases with token counts t, with a growth rate that starts slow, then accelerates, and finally stabilizes. The underlying reasons are as follows. When t is small, parameter loading time exceeds computation time, creating a memory bottleneck. Therefore, given the parameter volume, the memory access time is stable (memorybound regime). As t increases, computation time exceeds parameter loading time and becomes the bottleneck. With fixed arithmetic units in the hardware, computation time scales linearly with computational load (compute-bound regime). To characterize this trend, we design  $G(t; \lambda RP, s)$  as shown in Eq. 11, where  $\lambda RP$  represents the transition point between regimes, and s controls the increasing rate of execution time. Here, RP follows Eq. 1, and  $\lambda$  is a constant less than 1 that accounts for practical limitations in memory bandwidth utilization. G(t) exhibits a gradually increasing slope before the transition point, then shifts to a linear function afterwards, maintaining first-order gradient

continuity at the transition.

<span id="page-6-1"></span>
$$G(t; \lambda RP, s) = \begin{cases} s^t, & t \leq \lambda RP \\ s^{\lambda RP} + \left(\frac{\mathbf{d}(s^t)}{\mathbf{d}t}\big|_{t=\lambda RP}\right)(t - \lambda RP) = s^{\lambda RP} \left(1 + \ln(s) \cdot (t - \lambda RP)\right), & t > \lambda RP \end{cases}$$

$$\tag{11}$$

- (2) The number of activated experts. When it increases, the memory access volume increases, thus adding to the final processing time. We use the derived Eq. [8](#page-4-2) of N to characterize how workload and model architecture affects the number of activated experts.
- (3) Expert load. This refers to the fact that after token distribution through the MoE gating, each expert processes only a subset of tokens Texp(t; ρ) rather than the entire input token count t. Therefore, we should use G(Texp) rather than G(t) when applying the roofline model to MoE experts. This corroborates our theoretical conclusion that sparser MoEs *delay* the transition from memory-bound to compute-bound when input tokens count increases.

For the MoE target model, factors (1), (2), and (3) are all involved. We combine them in a first-order style and introduce parameters *bias*, k1, k2, and k<sup>3</sup> to adjust for non-ideal factors in actual GPU execution, with the full expression shown in lines 6 and 8 of Alg. [1.](#page-5-1) These parameters have clear practical meanings: *bias* represents the time required to load fixed parameters; k<sup>2</sup> · N represents the time needed to load N activated experts; k<sup>1</sup> · G(t) and k<sup>3</sup> · G(Texp) describe the *incremental* trend in execution time as the number of tokens increases. For the draft model, only factor (1) is involved since it is usually dense, with the modeling form shown in line 9 of Alg. [1.](#page-5-1)

With the expression of SD speedup determined, we fit the measurement inputs to automatically determine the relaxation parameter values, with the optimization criterion being the minimization of Mean Squared Error (MSE) between the model outputs and the ground truth, as shown in line 13 of Alg. [1.](#page-5-1) By applying these optimized parameters in our model (i.e., the *ComputeSpeedup* function in line 3), we obtain the complete modeling for SD speedup. An illustrative diagram of this process and more fitting details are provided in Appendix [C.](#page-16-0)

Since our theoretical analyses capture the primary tradeoffs and provide a solid foundation for the modeling, the fitting is very efficient. The fitting results with 21 measurements are displayed in Fig. [4,](#page-9-0) which show consistent trends with GPU results under various cases. These results validate the reliability of our modeling, thereby establishing it as an effective tool for analyzing the components of the model's forward pass and quantitatively understanding the tradeoffs between different factors. As shown in Sec. [4.2,](#page-8-0) we explain some unexpected results with the help of the model.

#### <span id="page-6-0"></span>3.4 Practical Values of Theoretical Findings

While previous sections focuses on theoretical analysis, this section demonstrates how these findings translate to practical speedups. Our theoretical analysis has already revealed that SD speedup for MoE is most effective at *moderate* batch sizes, with its trend initially increasing and then decreasing. We discuss their practical values considering both basic deployment and extended configurations.

Basic deployment. (1) Moderate batch sizes are common in private serving, which are increasingly adopted for data security, with representative applications like enterprise in-house chatbots. (2) When latency requirements are strict, large batch sizes are often not feasible. LLM serving must satisfy multiple service level objectives (SLOs) [\[38\]](#page-12-6), including time-to-first-token (TTFT) and time-peroutput-token (TPOT). Large batches reduce per-request computational resources, causing latency violations. In such cases, moderate batch sizes are common. (3) Our work actually reveals that SD on MoE relaxes the traditional *latency-throughput trade-off*. Specifically, MoE models exhibit a regime where SD speedup increases (lower latency) alongside larger batch sizes (higher throughput).

From the model's perspective, moderate batch sizes represent an "*efficiency gap*" in MoEs. At this scale, all parameters must be loaded (unlike small batches with selective expert activation), yet GPU FLOPs are not fully utilized (unlike large batches). Our findings provide a novel perspective to address this efficiency challenge without compromising model quality.

Extended configurations. We consider typical system optimizations on MoE like *offloading* and *expert parallelism (EP)*. When MoE models exceed GPU memory capacity, FFN parameters are offloaded to CPU memory [\[39\]](#page-12-7). This degrades parameter loading bandwidth from GPU memory bandwidth to much lower PCIe bandwidth, making the system more memory-bound. Consequently,

<span id="page-7-0"></span>![](_page_7_Figure_0.jpeg)

Figure 2: SD speedup (left y-axis) as a function of batch size and corresponding target efficiency values (right y-axis). Across different hardware platforms and MoE models, SD speedup first increases and then decreases, verifying our theoretical predictions. The target efficiency shows consistent trends with final speedup, validating its effectiveness.

additional computation does not significantly increase processing time, creating favorable conditions for SD. Notably, existing optimizations like expert prefetching [23, 24] and caching [25, 26] lose efficiency under moderate batch sizes since nearly all experts are activated.

Our findings are also compatible with EP. In EP, experts are distributed across multiple GPUs, which affects neither N(t) nor  $\overline{T_{exp}}$ , making our previous analyses remain valid. Since components besides MoE FFN are also parallelized, MoE FFN continues to constitute a significant portion of processing time, allowing memory-boundness effects to remain observable in end-to-end performance. Notably, under extensive EP configurations, the inefficiency of SD for MoE at a small batch size may vanish, considering the additional memory bandwidth offered by large amounts of EP GPUs.

### <span id="page-7-1"></span>4 Experiments

**Models and datasets.** We conducted experiments on two pairs of MoE target models and draft models: Qwen2-57B-A14B-Instruct with Qwen2-0.5B-Instruct [40], and Mixtral-8x7B-Instruct-v0.1 [41] with Eagle speculation head [7]. They represent two prevalent SD approaches: Qwen2 employs a standalone small model from the same model family as the draft model, while Mixtral uses a specifically trained speculation head. When we need to examine MoEs with different sparsity, we modify the num\_experts\_per\_token in the model's config.json file. For comparison with dense models, we use Opt-30b and Opt-350m [42] as the target and draft models. Models are evaluated on HumanEval [43] and MT-bench [44] datasets for code generation and conversation tasks, following previous works [7, 45, 11]. The tokenized prompt lengths range from 38 to 391 tokens for HumanEval and 5 to 356 tokens for MT-bench.

**Frameworks and hardware.** We used the existing vllm [46] framework for our experiments to verify theoretical predictions. Vllm supports batched speculative decoding, cudagraph optimization, and reports comprehensive data such as  $T_D$ ,  $T_T$ ,  $T_{reject}$  and  $\sigma$ , thus being suitable for our experiments. To prevent unstable performance at the beginning, all data were obtained by averaging the results from the last five of the total ten runs. We conducted experiments on different hardware platforms including 2xGPU-A, 2xGPU-B, 4xGPU-A, 4xGPU-C.

#### <span id="page-7-2"></span>4.1 Speedup Trend of Speculative Decoding for MoE

Figure 2 plots the end-to-end SD speedup (left y-axis) for MoE across various settings, validating our theoretical prediction about acceleration behavior. As batch size grows, speedup initially increases due to expert loading saturation, and then decreases due to compute-boundness. We denote the maximal speedup across batch sizes as  $\mathbf{x}$  and summarize the results in Table 1. For both models, SD achieves higher acceleration with longer  $\gamma$  for tasks with more predictable patterns (e.g., code generation) or less randomness (e.g., lower temperature), aligning with conclusions from previous research. Figure 5 in Appendix A.1 further presents SD speedup trends under more settings, including individual runs and their mean to show the statistical significance of our findings.

<span id="page-8-1"></span>Table 1: The peak speedup (x) of SD across different datasets, temperatures,  $\gamma$ s and models on 2xGPU-A

| Device  | Dataset   | Temp | $\gamma = 2$ |          |          |      |          | $\gamma =$ | 3        |      | $\gamma = 4$ |          |          |      |
|---------|-----------|------|--------------|----------|----------|------|----------|------------|----------|------|--------------|----------|----------|------|
|         |           |      | $T_{AR}$     | $T_{SD}$ | $\sigma$ | x    | $T_{AR}$ | $T_{SD}$   | $\sigma$ | x    | $T_{AR}$     | $T_{SD}$ | $\sigma$ | x    |
| Qwen2   | humaneval | 0.0  | 18.89        | 11.61    | 0.94     | 1.63 | 15.93    | 8.11       | 0.93     | 1.96 | 15.93        | 7.31     | 0.91     | 2.18 |
|         | humaneval | 1.0  | 19.13        | 12.93    | 0.83     | 1.48 | 21.20    | 14.09      | 0.73     | 1.50 | 19.13        | 11.14    | 0.67     | 1.72 |
|         | mtbench   | 0.0  | 20.92        | 16.70    | 0.71     | 1.25 | 16.00    | 12.43      | 0.62     | 1.29 | 20.92        | 17.53    | 0.55     | 1.19 |
|         | mtbench   | 1.0  | 21.15        | 17.33    | 0.68     | 1.22 | 19.09    | 14.83      | 0.57     | 1.29 | 19.09        | 15.93    | 0.48     | 1.20 |
| Mixtral | humaneval | 0.0  | 20.86        | 12.47    | 0.78     | 1.67 | 21.00    | 12.46      | 0.66     | 1.69 | 20.86        | 11.69    | 0.58     | 1.79 |
|         | humaneval | 1.0  | 21.52        | 15.58    | 0.61     | 1.38 | 21.39    | 16.03      | 0.46     | 1.33 | 21.48        | 16.23    | 0.39     | 1.32 |
|         | mtbench   | 0.0  | 21.61        | 16.10    | 0.61     | 1.34 | 21.61    | 16.43      | 0.46     | 1.32 | 21.36        | 16.89    | 0.39     | 1.26 |
|         | mtbench   | 1.0  | 21.33        | 17.70    | 0.53     | 1.21 | 21.33    | 17.84      | 0.43     | 1.20 | 21.33        | 18.05    | 0.35     | 1.18 |

<span id="page-8-2"></span>Table 2: The peak speedup (x) of SD across different datasets, temperatures,  $\gamma$ s and hardware on Qwen2

| Device   | Dataset   | Temp | $\gamma = 2$ |          |      |      | $\gamma = 3$ |          |      |      | $\gamma = 4$ |          |      |      |
|----------|-----------|------|--------------|----------|------|------|--------------|----------|------|------|--------------|----------|------|------|
|          |           |      | $T_{AR}$     | $T_{SD}$ | σ    | x    | $T_{AR}$     | $T_{SD}$ | σ    | x    | $T_{AR}$     | $T_{SD}$ | σ    | x    |
|          | humaneval | 0.0  | 15.96        | 9.34     | 0.95 | 1.71 | 15.96        | 7.95     | 0.93 | 2.01 | 15.96        | 6.96     | 0.90 | 2.29 |
| 2xGPU-B  | humaneval | 1.0  | 17.39        | 12.82    | 0.82 | 1.36 | 13.20        | 8.98     | 0.74 | 1.47 | 13.20        | 7.17     | 0.75 | 1.84 |
| ZXGF U-B | mtbench   | 0.0  | 24.42        | 16.74    | 0.71 | 1.46 | 24.42        | 16.84    | 0.62 | 1.45 | 24.42        | 17.05    | 0.54 | 1.43 |
|          | mtbench   | 1.0  | 18.24        | 14.38    | 0.67 | 1.27 | 16.25        | 13.28    | 0.56 | 1.22 | 16.25        | 13.76    | 0.48 | 1.18 |
|          | humaneval | 0.0  | 11.20        | 6.77     | 0.95 | 1.65 | 11.20        | 5.89     | 0.93 | 1.90 | 11.20        | 5.38     | 0.90 | 2.08 |
| 4xGPU-A  | humaneval | 1.0  | 11.72        | 8.51     | 0.81 | 1.38 | 12.05        | 8.30     | 0.73 | 1.45 | 11.23        | 7.70     | 0.67 | 1.46 |
| 4XGPU-A  | mtbench   | 0.0  | 11.26        | 8.92     | 0.72 | 1.26 | 11.26        | 9.10     | 0.61 | 1.24 | 11.26        | 9.82     | 0.52 | 1.15 |
|          | mtbench   | 1.0  | 11.78        | 10.32    | 0.67 | 1.14 | 11.30        | 9.42     | 0.58 | 1.20 | 11.78        | 11.25    | 0.47 | 1.05 |
|          | humaneval | 0.0  | 17.84        | 10.00    | 0.95 | 1.79 | 17.84        | 8.33     | 0.93 | 2.14 | 17.84        | 7.94     | 0.90 | 2.25 |
| 4xGPU-C  | humaneval | 1.0  | 17.89        | 12.27    | 0.80 | 1.46 | 17.89        | 11.07    | 0.74 | 1.62 | 17.89        | 10.91    | 0.65 | 1.64 |
| 4xGPU-C  | mtbench   | 0.0  | 20.40        | 15.87    | 0.71 | 1.29 | 20.40        | 16.22    | 0.62 | 1.26 | 20.40        | 16.33    | 0.54 | 1.25 |
|          | mtbench   | 1.0  | 20.58        | 16.02    | 0.68 | 1.28 | 18.11        | 14.75    | 0.54 | 1.23 | 18.11        | 15.54    | 0.48 | 1.17 |

We further evaluate Qwen2-57B-A14B-Instruct on multiple hardware platforms (Table 2) to verify the generality of our conclusions. Combined with results of Qwen2 in Table 1, two observations can be made: (1) GPUs with higher ridge points yield larger SD speedups (e.g.,  $2\times$ GPU-A vs.  $2\times$ GPU-B,  $4\times$ GPU-A vs.  $4\times$ GPU-C), since they provide more arithmetic units for verification. (2) Scaling from  $2\times$ GPU-A to  $4\times$ GPU-A reduces absolute runtimes ( $T_{AR}$  and  $T_{SD}$ ), but the SD speedups slightly degrade. This is because the large model benefits from inter-GPU parallelization, whereas the small draft model remains single-GPU, making its relative forward cost higher.

<span id="page-8-3"></span>![](_page_8_Figure_5.jpeg)

Figure 3: Comparison of target efficiency: MoE vs dense model.

Figure 2 also highlights the effectiveness of our metric target efficiency. It is computed as  $\frac{T_T(B,1)}{T_T(B,\gamma)}$  as explained in Sec. 3.1, where both  $T_T(B,1)$  and  $T_T(B,\gamma)$  are obtained from vllm runtime logs. Target efficiency values are annotated on the right y-axis, showing a consistent trend with the end-to-end speedup. In contrast, the acceptance rate across batch sizes merely *fluctuates within a small range*, unable to effectively explain the dramatic changes in speedup.

We further compare the behaviors of MoE and dense models in SD. Since the effectiveness of target efficiency has been established in analyses above, and to avoid interference from algorithmic factors such as acceptance rate, we compare their target efficiency. As shown in Figure 3, the target efficiency for MoE first increases and then decreases, while that for the dense model decreases continuously. Consequently, although SD for

MoE is less effective with small batches, it exhibits stronger potential across a wider range of larger batch sizes. Regarding end-to-end performance, SD speedups become more pronounced for MoE when the batch size exceeds 16, as supplemented in Figure 6 in Appendix A.2.

### <span id="page-8-0"></span>4.2 Impact of MoE Sparsity and Validation of Modeling Method

To evaluate MoE sparsity's impact on SD acceleration, we varied the number of activated experts per token (K) of Qwen2-57B-A14B-Instruct. Directly changing K without training affects the target model's performance and speculation accuracy, so we adjust the speedup by multiplying the raw speedup with  $\frac{\sigma_{K=8}}{\sigma_K}$ , whose rationale is exhibited by Eq. 4. Fig. 4 shows the adjusted speedup

<span id="page-9-0"></span>![](_page_9_Figure_0.jpeg)

Figure 4: Comparison between GPU results and our modeling for Qwen2-57B-A14B-Instruct with varying sparsity ρ and draft length γ.

alongside our modeling results for comparison. The parameters used in the modeling are decided using 21 GPU measurements, as explained in Section [3.3.](#page-5-0) The impact of measurement selection for parameter fitting on the modeling's reliability is supplemented in Appendix [C.](#page-16-0)

There are three key observations. First, the modeling consistently aligns with experiment results across varying sparsity (ρ) and draft length (γ), validating our modeling's reliability.

Second, while the SD speedup in most MoEs exhibits an initial increase followed by a decrease, very sparse MoEs (K = 1, 2) show continuously decreasing speedup. This appears to conflict with the theoretical analysis, but after examining the components of our modeling, we identified the reason as follows. These very sparse MoEs have a disproportionately low ratio of FFN, thus making the memory-boundness of MoE FFN hard to manifest systematically as indicated by Amdahl's Law. The Qwen2-57B-A14B model is designed based on K = 8, but by reducing K to 1 or 2, we actually artificially *synthesize* a model where Attention dominates. In practice, however, sparser MoEs typically incorporate more FFN parameters to maintain a balanced ratio between FFN and Attention components, resulting in acceleration patterns more similar to K = 8 cases.

Finally, as MoE models become sparser, the system's transition from memory-bound to computebound is delayed. This is evidenced by two phenomena in Fig. [4:](#page-9-0) With smaller ρ, (1) the batch size for the maximal speedup (x) becomes larger; (2) the range of batch sizes that maintain speedup above a certain decay threshold (annotated by the brown dashed line in Fig. [4](#page-9-0) for x/ √ 2) is wider. These validate our theoretical analysis and indicate that SD has broader applicability in sparser MoEs.

### 5 Conclusion and Limitation

In this work, we challenge the conventional wisdom that speculative decoding cannot effectively accelerate MoE models, and point out that with moderate batch sizes, sparser MoEs actually gain greater benefits from SD due to the more memory-bound FFN. We support this conclusion with both theoretical analysis and experimental verification. Considering the complex interplay of multiple factors affecting the final speedup, we develop a reliable modeling for SD, which enables us to comprehend the acceleration process in a transparent and explainable manner. We also introduce target efficiency to help researchers comprehensively understand how SD acceleration is affected by the target model architecture and workload. Our work offers a new perspective for MoE acceleration, particularly effective for private serving with moderate batch sizes or memory-constrained scenarios. Most of our analysis assumes the volume of KV-cache is relatively smaller than that of parameters, while the behavior of SD when KV-cache dominates has already been analyzed by MagicDec [\[34\]](#page-12-2). These two works can be combined to offer a more comprehensive view of SD at varying batch sizes.

### Acknowledgments and Disclosure of Funding

This work is supported by the National Natural Science Foundation of China (Grant Nos. 92267203) and Deng Feng Fund.

### References

- <span id="page-10-0"></span>[1] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. *arXiv preprint arXiv:2412.19437*, 2024.
- <span id="page-10-1"></span>[2] Qwen Team. Qwen2.5 technical report. *arXiv preprint arXiv:2412.15115*, 2024.
- <span id="page-10-2"></span>[3] Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Yu Wu, et al. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models. *arXiv preprint arXiv:2401.06066*, 2024.
- <span id="page-10-3"></span>[4] Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, et al. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. *arXiv preprint arXiv:2405.04434*, 2024.
- <span id="page-10-4"></span>[5] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *arXiv preprint arXiv:1701.06538*, 2017.
- <span id="page-10-5"></span>[6] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. *arXiv preprint arXiv:2006.16668*, 2020.
- <span id="page-10-6"></span>[7] Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle: Speculative sampling requires rethinking feature uncertainty. *arXiv preprint arXiv:2401.15077*, 2024.
- <span id="page-10-7"></span>[8] Anish Saxena, Po-An Tsai, Hritvik Taneja, Aamer Jaleel, and Moinuddin Qureshi. Utility-driven speculative decoding for mixture-of-experts, 2025.
- <span id="page-10-8"></span>[9] Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. Accelerating large language model decoding with speculative sampling. *arXiv preprint arXiv:2302.01318*, 2023.
- <span id="page-10-9"></span>[10] Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In *International Conference on Machine Learning*, pages 19274–19286. PMLR, 2023.
- <span id="page-10-10"></span>[11] Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D Lee, Deming Chen, and Tri Dao. Medusa: Simple llm inference acceleration framework with multiple decoding heads, 2024. *URL https://arxiv. org/abs/2401.10774*, 2024.
- <span id="page-10-11"></span>[12] Rahul. What are private llms? running large language models privately - privategpt and beyond, 2024. <https://zilliz.com/learn/what-are-private-llms>.
- <span id="page-10-12"></span>[13] Esther Julie. What is a private llm and why should you build one?, 2024. [https://www.inoru.](https://www.inoru.com/blog/what-is-a-private-llm-and-why-should-you-build-one/) [com/blog/what-is-a-private-llm-and-why-should-you-build-one/](https://www.inoru.com/blog/what-is-a-private-llm-and-why-should-you-build-one/).
- <span id="page-10-13"></span>[14] Hanbo Huang, Yihan Li, Bowen Jiang, Lin Liu, Bo Jiang, Ruoyu Sun, Zhuotao Liu, and Shiyu Liang. Position: On-premises llm deployment demands a middle path: Preserving privacy without sacrificing model confidentiality, 2025.

- <span id="page-11-0"></span>[15] Yanyue Xie, Zhi Zhang, Ding Zhou, Cong Xie, Ziang Song, Xin Liu, Yanzhi Wang, Xue Lin, and An Xu. Moe-pruner: Pruning mixture-of-experts large language model using the hints from its router. *arXiv preprint arXiv:2410.12013*, 2024.
- <span id="page-11-1"></span>[16] Jaeseong Lee, Aurick Qiao, Daniel F Campos, Zhewei Yao, Yuxiong He, et al. Stun: Structuredthen-unstructured pruning for scalable moe pruning. *arXiv preprint arXiv:2409.06211*, 2024.
- <span id="page-11-2"></span>[17] Elias Frantar and Dan Alistarh. Qmoe: Practical sub-1-bit compression of trillion-parameter models. *arXiv preprint arXiv:2310.16795*, 2023.
- <span id="page-11-3"></span>[18] HamidReza Imani, Abdolah Amirany, and Tarek El-Ghazawi. Mixture of experts with mixture of precisions for tuning quality of service. *arXiv preprint arXiv:2407.14417*, 2024.
- <span id="page-11-4"></span>[19] Felipe Cruz Salinas, Kenichi Kumatani, Robert Gmyr, Linquan Liu, and Yu Shi. Knowledge distillation for mixture of experts models in speech recognition. Technical report, Technical Report MSR-TR-2022-6, Microsoft Research, May 2022. https://www . . . , 2022.
- <span id="page-11-5"></span>[20] Fangxun Shu, Yue Liao, Le Zhuo, Chenning Xu, Lei Zhang, Guanghao Zhang, Haonan Shi, Long Chen, Tao Zhong, Wanggui He, et al. Llava-mod: Making llava tiny via moe knowledge distillation. *arXiv preprint arXiv:2408.15881*, 2024.
- <span id="page-11-6"></span>[21] Cheng Yang, Yang Sui, Jinqi Xiao, Lingyi Huang, Yu Gong, Yuanlin Duan, Wenqi Jia, Miao Yin, Yu Cheng, and Bo Yuan. Moe-i<sup>2</sup> : Compressing mixture of experts models through inter-expert pruning and intra-expert low-rank decomposition. *arXiv preprint arXiv:2411.01016*, 2024.
- <span id="page-11-7"></span>[22] Pingzhi Li, Zhenyu Zhang, Prateek Yadav, Yi-Lin Sung, Yu Cheng, Mohit Bansal, and Tianlong Chen. Merge, then compress: Demystify efficient smoe with hints from its routing policy. *arXiv preprint arXiv:2310.01334*, 2023.
- <span id="page-11-8"></span>[23] Leyang Xue, Yao Fu, Zhan Lu, Luo Mai, and Mahesh Marina. Moe-infinity: Activation-aware expert offloading for efficient moe serving. *arXiv e-prints*, pages arXiv–2401, 2024.
- <span id="page-11-9"></span>[24] Shuzhang Zhong, Ling Liang, Yuan Wang, Runsheng Wang, Ru Huang, and Meng Li. Adapmoe: Adaptive sensitivity-based expert gating and management for efficient moe inference. In *Proceedings of the 43rd IEEE/ACM International Conference on Computer-Aided Design*, pages 1–9, 2024.
- <span id="page-11-10"></span>[25] Xin He, Shunkang Zhang, Yuxin Wang, Haiyan Yin, Zihao Zeng, Shaohuai Shi, Zhenheng Tang, Xiaowen Chu, Ivor Tsang, and Ong Yew Soon. Expertflow: Optimized expert activation and token allocation for efficient mixture-of-experts inference. *arXiv preprint arXiv:2410.17954*, 2024.
- <span id="page-11-11"></span>[26] Peng Tang, Jiacheng Liu, Xiaofeng Hou, Yifei Pu, Jing Wang, Pheng-Ann Heng, Chao Li, and Minyi Guo. Hobbit: A mixed precision expert offloading system for fast moe inference. *arXiv preprint arXiv:2411.01433*, 2024.
- <span id="page-11-12"></span>[27] Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Zhengxin Zhang, Rae Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, et al. Specinfer: Accelerating large language model serving with tree-based speculative inference and verification. In *Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3*, pages 932–949, 2024.
- <span id="page-11-13"></span>[28] Zhenyu He, Zexuan Zhong, Tianle Cai, Jason D Lee, and Di He. Rest: Retrieval-based speculative decoding. *arXiv preprint arXiv:2311.08252*, 2023.
- <span id="page-11-14"></span>[29] Ruslan Svirschevski, Avner May, Zhuoming Chen, Beidi Chen, Zhihao Jia, and Max Ryabinin. Specexec: Massively parallel speculative decoding for interactive llm inference on consumer devices. *Advances in Neural Information Processing Systems*, 37:16342–16368, 2024.
- <span id="page-11-15"></span>[30] Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle-2: Faster inference of language models with dynamic draft trees. *arXiv preprint arXiv:2406.16858*, 2024.
- <span id="page-11-16"></span>[31] Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle-3: Scaling up inference acceleration of large language models via training-time test. *arXiv preprint arXiv:2503.01840*, 2025.

- <span id="page-12-0"></span>[32] Xiaoxuan Liu, Jongseok Park, Langxiang Hu, Woosuk Kwon, Zhuohan Li, Chen Zhang, Kuntai Du, Xiangxi Mo, Kaichao You, Alvin Cheung, Zhijie Deng, Ion Stoica, and Hao Zhang. Turbospec: Closed-loop speculation control system for optimizing llm serving goodput, 2025.
- <span id="page-12-1"></span>[33] Qidong Su, Christina Giannoula, and Gennady Pekhimenko. The synergy of speculative decoding and batching in serving large language models, 2023.
- <span id="page-12-2"></span>[34] Ranajoy Sadhukhan, Jian Chen, Zhuoming Chen, Vashisth Tiwari, Ruihang Lai, Jinyuan Shi, Ian En-Hsu Yen, Avner May, Tianqi Chen, and Beidi Chen. Magicdec: Breaking the latencythroughput tradeoff for long context generation with speculative decoding. *arXiv preprint arXiv:2408.11049*, 2024.
- <span id="page-12-3"></span>[35] Zhihang Yuan, Yuzhang Shang, Yang Zhou, Zhen Dong, Zhe Zhou, Chenhao Xue, Bingzhe Wu, Zhikai Li, Qingyi Gu, Yong Jae Lee, Yan Yan, Beidi Chen, Guangyu Sun, and Kurt Keutzer. Llm inference unveiled: Survey and roofline model insights, 2024.
- <span id="page-12-4"></span>[36] Georg Ofenbeck, Ruedi Steinmann, Victoria Caparros, Daniele G Spampinato, and Markus Püschel. Applying the roofline model. In *2014 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)*, pages 76–85. IEEE, 2014.
- <span id="page-12-5"></span>[37] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity, 2022.
- <span id="page-12-6"></span>[38] Zhibin Wang, Shipeng Li, Yuhang Zhou, Xue Li, Rong Gu, Nguyen Cam-Tu, Chen Tian, and Sheng Zhong. Revisiting slo and goodput metrics in llm serving, 2024.
- <span id="page-12-7"></span>[39] kvcache ai. Ktransfromers: A flexible framework for experiencing cutting-edge llm inference optimizations, 2025. <https://github.com/kvcache-ai/ktransformers/tree/main>.
- <span id="page-12-8"></span>[40] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, et al. Qwen2 technical report, 2024.
- <span id="page-12-9"></span>[41] Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mixtral of experts, 2024.
- <span id="page-12-10"></span>[42] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. *arXiv preprint arXiv:2205.01068*, 2022.
- <span id="page-12-11"></span>[43] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, et al. Evaluating large language models trained on code, 2021.
- <span id="page-12-12"></span>[44] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena, 2023.
- <span id="page-12-13"></span>[45] Lefan Zhang, Xiaodan Wang, Yanhua Huang, and Ruiwen Xu. Learning harmonized representations for speculative sampling, 2025.
- <span id="page-12-14"></span>[46] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In *Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles*, 2023.
- <span id="page-12-15"></span>[47] NVIDIA Docs Hub. Matrix multiplication background user's guide. [https://docs.nvidia.](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) [com/deeplearning/performance/dl-performance-matrix-multiplication/index.](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) [html](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html).

- <span id="page-13-0"></span>[48] Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. Flashattention-3: Fast and accurate attention with asynchrony and low-precision, 2024.
- <span id="page-13-1"></span>[49] Kan Zhu, Yufei Gao, Yilong Zhao, Liangyu Zhao, Gefei Zuo, Yile Gu, Dedong Xie, Tian Tang, Qinyu Xu, Zihao Ye, Keisuke Kamahori, Chien-Yu Lin, Ziren Wang, Stephanie Wang, Arvind Krishnamurthy, and Baris Kasikci. Nanoflow: Towards optimal large language model serving throughput, 2025.

<span id="page-14-0"></span>![](_page_14_Figure_0.jpeg)

Figure 5: SD speedup trends across more settings with individual runs and averages shown.

### A Supplementary Experimental Results

This section presents additional experimental results referenced in Section [4,](#page-7-1) which are included here due to space limitations.

### <span id="page-14-1"></span>A.1 Trends of SD speedup under more configurations

Figure [5](#page-14-0) presents additional trends of SD speedup across different datasets, temperatures, and model types, serving as a supplement to Figure [2.](#page-7-0) The results demonstrate that SD speedup exhibits a consistent first-increase-then-decrease pattern, which aligns well with our theoretical analysis.

To confirm the statistical significance of our findings, we also present the five individual runs that constitute the averages in Figure [5.](#page-14-0) The variance across different runs is minimal, which is expected since the random seed is fixed across all runs to ensure identical workloads.

While the overall trend follows the first-increase-then-decrease pattern, local fluctuations are observable in the curves. For instance, Figure [5\(](#page-14-0)c) exhibits a sawtooth-like decreasing trend. This phenomenon can be attributed to GPU *quantization effects*, as documented in NVIDIA's documentation [\[47\]](#page-12-15). When dimensions are not evenly divisible by the GPU's native tile sizes, computational performance degrades. AR decoding is more sensitive to this effect than SD, making the time ratio of

<span id="page-15-0"></span>![](_page_15_Figure_0.jpeg)

Figure 6: End-to-end speedup comparison of MoE and dense models under various settings.

AR to SD (namely, SD speedup) fluctuate. Despite these local variations, the overall speedup trend follows our theoretical predictions, confirming the validity of our conclusions.

#### <span id="page-15-1"></span>A.2 End-to-end speedup comparison between MoE and dense models

To isolate the effects of acceptance rate variations and enable a clearer focus on system bottlenecks, we have compared MoE and dense models using target efficiency in Section 4.1. In this section, we further compare their end-to-end speedup across various settings in Figure 6 as a supplement.

Two key observations emerge from the experiment results. First, while SD speedups for MoE models initially increase before declining, SD speedups for dense models continue to decrease. Consequently, SD achieves more substantial end-to-end speedups for MoE models at moderate batch sizes, which aligns with the trend in Figure 3 in Section 4.1. Second, the extent to which SD favors MoE over dense models varies across different configurations. For instance, at temperature = 1 (second row), SD demonstrates greater relative benefits for MoE compared to temperature = 0 (first row). This variation stems from diverse acceptance rates under different settings, which can obscure the observation of systemic bottlenecks. In summary, target efficiency serves as a reliable comparison metric while controlling for the confounding effects of algorithmic optimizations.

## **B** Proof of $\overline{T_{exp}}(T; \rho)$ 's Trend with Varying $\rho$

In Section 3.2, Fig. 1(c) demonstrates that: Given input token count t=T>1, the number of tokens each expert processes on average  $\overline{T_{\rm exp}}(T;\rho)=\frac{\rho T}{1-(1-\rho)^T}$  decreases as  $\rho$  decreases. We prove this by showing  $\frac{{\rm d}(\overline{T_{\rm exp}}(T;\rho))}{{\rm d}\rho}>0$  when T>1.

$$\frac{d(\overline{T_{\exp}}(T;\rho))}{d\rho} = \frac{d(\frac{\rho^T}{1-(1-\rho)^T})}{d\rho} = \frac{T(-\rho T(1-\rho)^{T-1} - (1-\rho)^T + 1)}{(1-(1-\rho)^T)^2}$$
(12)

Since  $\rho$  represents MoE sparsity  $\in (0,1)$ , the original proposition is equivalent to proving:

$$\mathbf{F}(\rho;T) = (1-\rho)^{T-1}(\rho T + 1 - \rho) < 1 \tag{13}$$

Note that  $\mathbf{F}(\rho;T) \to 1$  as  $\rho \to 0$ . Therefore, if we can prove that  $\mathbf{F}(\rho;T)$  decreases as  $\rho$  increases from 0 to 1, then the original proposition is proven. We demonstrate this by computing  $\frac{d(\mathbf{F}(\rho;T))}{d\rho}$ :

$$\frac{d(\mathbf{F}(\rho;T))}{d\rho} = \frac{d((1-\rho)^{T-1}(\rho T + 1 - \rho))}{d\rho} = -\rho T(T-1)(1-\rho)^{T-2}$$
(14)

<span id="page-16-2"></span>![](_page_16_Figure_0.jpeg)

Figure 7: The overall diagram of the modeling method.

When T>1,  $\frac{d(\mathbf{F}(\rho;T))}{d\rho}<0$ . This confirms that  $\mathbf{F}(\rho;T)$  decreases as  $\rho$  increases, which proves our original proposition: when T>1,  $\overline{T_{\rm exp}}(T;\rho)$  decreases as  $\rho$  decreases.

### <span id="page-16-0"></span>C More details about the Modeling Method

The main design considerations and expressions of our modeling method have been presented in Section 3.3. In this section, we provide additional content on the following topics to give a more comprehensive view of the modeling method:

- Description and an illustrative diagram of the modeling process. (Appendix C.1)
- Fitting Details of the modeling shown in Figure 4 in Section 4.1. (Appendix C.2)
- How the modeling is affected by alternative measurement selection. (Appendix C.3)

The value of our modeling is twofold. On one hand, it achieves alignment with real measurements with only a small number of simple parameters, thus validating the correctness of our theoretical analyses. On the other hand, it provides the decomposition of various factors in the end-to-end results, making the entire SD acceleration process explainable and transparent.

#### <span id="page-16-1"></span>C.1 Description and Overview of the Modeling Process

Figure 7 presents the overall diagram of our modeling method. Building upon the theoretical analyses in Sections 3.1 and 3.2, we derive an expression for SD speedup as a function of workloads. This expression contains several relaxation parameters to be determined for complete modeling. We determine these parameters through empirical profiling. We first collect a small set of real measurements comprising various workloads and their corresponding SD speedups. We then perform parameter fitting using these measurements under the least squares criterion to obtain optimal parameter values. Since our theoretically-derived SD speedup expression already captures the fundamental performance tradeoffs, the fitting process is computationally lightweight and robust, as will be demonstrated in Appendix C.2 and C.3. Once the optimal parameters are obtained, the resulting expression can predict SD speedups for arbitrary workloads.

We now explain why profiling is necessary and why we cannot derive the complete SD speedup expression purely through theoretical analysis, examining both software and hardware considerations.

**Software considerations:** Actual execution times can deviate significantly from theoretical predictions for complex operators with diverse implementations. On GPUs, GEMM operations are indeed predictable due to their regular structure and highly optimized implementations. However, prediction becomes challenging for operators such as Attention, which involve customized kernel optimizations (e.g., FlashAttention1/2, eager attention, SDPA attention) and operator fusion strategies (incorporating various nonlinear layers or positional encodings such as RoPE and its variants). To illustrate this complexity, we examine profiling results from Qwen2-57B-A14B (hidden size 3584) and Mixtral-8x7B (hidden size 4096). For FFN, Qwen takes a shorter time than Mixtral (143us vs 226us), aligning with their relative hidden sizes. However, for Attention, Qwen takes a longer time than Mixtral (271us vs 115us), contradicting the theoretical expectation based on hidden size scaling.

**Hardware considerations:** GPU microarchitectures vary across different series, which can greatly impact execution times. For instance, attention efficiency depends heavily on hardware-aware

programming optimizations, while different GPUs vary in cache configurations and thread-memory interaction patterns across memory hierarchy levels. By taking advantage of new capabilities in modern hardware, FlashAttention-3 successfully increases GPU utilization from 35% to 75% on H100 GPUs [48]. Moreover, many critical hardware details remain undisclosed by GPU vendors, making theoretical predictions impractical.

Therefore, modeling speedup trends with pure analytical methods requires *case-by-case* analysis for different operator implementations and GPU microarchitectures. In contrast, the hyperparameter approach offers a more generalizable paradigm and is easy to use: all parameters possess clear physical interpretations, only minimal profiling data are required, and the computational overhead is low. Our method achieves a balance between effectiveness and practicality: on one hand, it captures the primary performance drivers (i.e. the number of activated experts and roofline trends); on the other hand, it avoids getting entangled in low-level implementation complexities. This approach is also used by other system optimization frameworks such as NanoFlow [49], which similarly adopt a two-stage strategy of profiling followed by runtime execution.

#### <span id="page-17-0"></span>C.2 Fitting Details of the Modeling shown in Figure 4

We first explain how we select the 21 measurements. Due to GPU resource and time constraints, we obtained a total of 228 GPU measurements across varying experimental settings, including 6 different numbers of activated experts per token (K), 2 draft lengths  $(\gamma)$ , and 19 batch sizes (B). These measurements are sorted first by K, then by  $\gamma$  within each K group, and finally by B within each  $(K,\gamma)$  combination, forming the total dataframe (df). We then uniformly sampled measurements from this sorted dataset with a fixed stride, namely M = df[begin:end:11]. This sampling strategy enables our selected measurements to contain different settings, making the modeling more robust.

The SD speedup function (namely, *ComputeSpeedup* defined in line 3 of Algorithm 1) is nonlinear. To optimize its MSE, we employed the scipy.optimize.least\_squares function with the Trust Region Reflective (TRR) algorithm. TRR is an optimization method for bound-constrained nonlinear least squares problems that combines trust region methods with reflection techniques. It constructs quadratic models within trust regions and uses reflection strategies near boundaries to maintain feasibility while ensuring convergence. The fitting process for these 21 data points is efficient, completing in approximately 0.114 seconds. Our modeling incorporates 10 parameters requiring relaxation, with their search boundaries specified as follows:

- bias: It represents the time required to load the dense parameters of the target model. We denote the model's non-FFN parameter count as  $V_{dense}$ . Consequently, the theoretical minimum loading time can be calculated as  $bias_{min} = \frac{V_{dense} \times bitwidth}{peak memory bandwidth}$ . For the upper bound of the relaxation range, we set  $bias_{max} = 5 \times bias_{min}$ .
- k1: It adjusts the intensity of the roofline effect of dense components. It should be larger than 0 to ensure the execution time increases as the token count increases. We don't set a definite upper limit for k1, as its value is affected by other parameters. Given the hardware with fixed arithmetic units, the execution time grows linearly with the token count in the compute-bound regime. As shown in line 6 of Algorithm 1, k1 appears as a coefficient in the term  $k1 \cdot G(t; \lambda, s)$ , whose gradient in the compute-bound regime is  $k1 \cdot ln(s) \cdot s^{\lambda RP}$ . As s approaches 1, k1 needs to continuously increase to counterbalance ln(s) that approaches 0.
- k2: It represents the time required to load one expert. Given a target model, we denote the parameter count per expert as  $V_{exp}$ . Consequently, the theoretical minimum loading time can be calculated as  $k2_{min} = \frac{V_{exp} \times bitwidth}{peak\ memory\ bandwidth}$ . For the upper bound of the relaxation range, we set  $k2_{max} = 5 \times k_{2min}$ .
- k3: It adjusts the intensity of the roofline effect of sparse components. Similar to k1, we set  $k3_{min} = 0$  and  $k3_{max} = inf$ .
- $draft\_bias$ : It represents the time required to load the dense draft model. We denote the draft model's parameter count as  $V_{draft}$ . Consequently, the theoretical minimum loading time can be calculated as  $draft\_bias_{min} = \frac{V_{draft} \times bitwidth}{peak\ memory\ bandwidth}$ . For the upper bound of the relaxation range, we set  $draft\_bias_{max} = 5 \times draft\_bias_{min}$ .
- $draft\_k$ : It adjusts the intensity of the roofline effect of the dense draft model. Similar to kl, we set  $draft\_k_{min} = 0$  and  $draft\_k_{max} = inf$ .

- *reject\_bias*: It represents the fixed overhead when performing rejection sampling. Vllm reports its elapsed time during SD, and we denote the maximum across measurements as Trej . We then set *reject\_bias*min = 0 and *reject\_bias*max = Trej .
- *reject\_k*: It represents the incremental processing time in rejection sampling as the input token count increases. For simplicity, we set *reject\_k*min = 0 and *reject\_k*max = Trej just like *reject\_bias*.
- λ: It represents the ratio of the empirical ridge point to the theoretical ridge point. Since memory bandwidth is typically less utilized than arithmetic units, we set λmin = 0.2 and λmax = 1.
- s: It adjusts the growing rate of execution time as input token count increases. Since s serves as the base of G(t), it must exceed 1 to ensure monotonic growth. However, s should not be too large, as it would result in an excessively steep growth rate. In experiments, we set smin = 1 and smax = 2.

### <span id="page-18-0"></span>C.3 Exploration of Alternative Measurement Selection

In this section, we demonstrate the impact of varying the number (m) of measurements used for fitting on the modeling results. Given that our model incorporates 10 parameters, a minimum of 10 profiling data points (m ≥ 10) are required to determine all parameters. We present the modeling fitting with m ranging from 10 to 228. The data selection method follows the stride-based approach described in the previous section, specifically M = df[begin:end:stride]. Measurement count m and stride satisfy the following relation: m = ⌈228/stride⌉.

We present the MSE values of different ms and their corresponding fitting figures in Table [3.](#page-19-0) We also list the distinct batch sizes involved in the selected measurements, which helps explain why some configurations show inferior model fit. Due to integer division constraints, ms are not continuous at larger magnitudes. Generally, the modeling fits well with the real measurements, except for m = 10, 12, 13. The reasons are as follows. When m = 10, the number of measurements equals the parameter count, resulting in insufficient data for robust fitting. When m = 12 and m = 13, the distribution of the measurement data is biased. With stride-based selection, measurements at m = 12 and m = 13 demonstrate notable gaps in batch size coverage (specifically, m = 12 does not include batch sizes greater than 40, while m = 13 does not include batch sizes within 1∼24). Their MSE values are larger than that of m = 11, despite the latter containing fewer data points for fitting. Based on this analysis, we recommend prioritizing uniform data distribution when selecting measurements, as this approach enables the development of more reliable models even with smaller datasets.

Table 3

<span id="page-19-0"></span>

| $\overline{m}$ | stride | MSE   | Figure | Batch Size Involved                                                       |
|----------------|--------|-------|--------|---------------------------------------------------------------------------|
| 10             | 25     | 2.216 | 8      | [1, 12, 16, 20, 36, 40, 44, 60, 80, 100]                                  |
| 11             | 22     | 1.764 | 9      | [1, 4, 8, 16, 20, 28, 32, 40, 44, 56, 100]                                |
| 12             | 20     | 4.288 | 10     | [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]                              |
| 13             | 18     | 2.681 | 11     | [1, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 80, 100]                      |
| 14             | 17     | 2.041 | 12     | [1, 2, 8, 16, 24, 32, 40, 44, 48, 52, 56, 60, 80, 100]                    |
| 15             | 16     | 1.668 | 13     | [1, 2, 4, 12, 16, 24, 28, 36, 40, 48, 52, 56, 60, 80, 100]                |
| 16             | 15     | 1.508 | 14     | [1, 2, 4, 8, 16, 20, 24, 32, 36, 40, 48, 52, 56, 60, 80, 100]             |
| 17             | 14     | 1.563 | 15     | [1, 2, 4, 8, 12, 20, 24, 28, 32, 40, 44, 48, 52, 56, 60, 80, 100]         |
| 18             | 13     | 1.525 | 16     | [1, 2, 4, 8, 12, 16, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 80, 100]     |
| 19             | 12     | 2.080 | 17     | [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 80, 100] |
| 21             | 11     | 1.679 | 18     | [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 80, 100] |
| 23             | 10     | 1.800 | 19     | [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 80, 100] |
| 26             | 9      | 1.716 | 20     | [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 80, 100] |
| 29             | 8      | 1.524 | 21     | [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 80, 100] |
| 33             | 7      | 1.526 | 22     | [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 80, 100] |
| 38             | 6      | 1.715 | 23     | [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 80, 100] |
| 46             | 5      | 1.644 | 24     | [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 80, 100] |
| 57             | 4      | 1.509 | 25     | [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 80, 100] |
| 76             | 3      | 1.553 | 26     | [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 80, 100] |
| 114            | 2      | 1.485 | 27     | [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 80, 100] |
| 228            | 1      | 1.523 | 28     | [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 80, 100] |

<span id="page-19-1"></span>![](_page_19_Figure_2.jpeg)

Figure 8: Comparison between GPU results and modeling with 10 measurements.

<span id="page-20-0"></span>![](_page_20_Figure_0.jpeg)

Figure 9: Comparison between GPU results and modeling with 11 measurements.

<span id="page-20-1"></span>![](_page_20_Figure_2.jpeg)

Figure 10: Comparison between GPU results and modeling with 12 measurements.

<span id="page-21-0"></span>![](_page_21_Figure_0.jpeg)

Figure 11: Comparison between GPU results and modeling with 13 measurements.

<span id="page-21-1"></span>![](_page_21_Figure_2.jpeg)

Figure 12: Comparison between GPU results and modeling with 14 measurements.

<span id="page-22-0"></span>![](_page_22_Figure_0.jpeg)

Figure 13: Comparison between GPU results and modeling with 15 measurements.

<span id="page-22-1"></span>![](_page_22_Figure_2.jpeg)

Figure 14: Comparison between GPU results and modeling with 16 measurements.

<span id="page-23-0"></span>![](_page_23_Figure_0.jpeg)

Figure 15: Comparison between GPU results and modeling with 17 measurements.

<span id="page-23-1"></span>![](_page_23_Figure_2.jpeg)

Figure 16: Comparison between GPU results and modeling with 18 measurements.

<span id="page-24-0"></span>![](_page_24_Figure_0.jpeg)

Figure 17: Comparison between GPU results and modeling with 19 measurements.

<span id="page-24-1"></span>![](_page_24_Figure_2.jpeg)

Figure 18: Comparison between GPU results and modeling with 21 measurements.

<span id="page-25-0"></span>![](_page_25_Figure_0.jpeg)

Figure 19: Comparison between GPU results and modeling with 23 measurements.

<span id="page-25-1"></span>![](_page_25_Figure_2.jpeg)

Figure 20: Comparison between GPU results and modeling with 26 measurements.

<span id="page-26-0"></span>![](_page_26_Figure_0.jpeg)

Figure 21: Comparison between GPU results and modeling with 29 measurements.

<span id="page-26-1"></span>![](_page_26_Figure_2.jpeg)

Figure 22: Comparison between GPU results and modeling with 33 measurements.

<span id="page-27-0"></span>![](_page_27_Figure_0.jpeg)

Figure 23: Comparison between GPU results and modeling with 38 measurements.

<span id="page-27-1"></span>![](_page_27_Figure_2.jpeg)

Figure 24: Comparison between GPU results and modeling with 46 measurements.

<span id="page-28-0"></span>![](_page_28_Figure_0.jpeg)

Figure 25: Comparison between GPU results and modeling with 57 measurements.

<span id="page-28-1"></span>![](_page_28_Figure_2.jpeg)

Figure 26: Comparison between GPU results and modeling with 76 measurements.

<span id="page-29-0"></span>![](_page_29_Figure_0.jpeg)

Figure 27: Comparison between GPU results and modeling with 114 measurements.

<span id="page-29-1"></span>![](_page_29_Figure_2.jpeg)

Figure 28: Comparison between GPU results and modeling with 228 measurements.