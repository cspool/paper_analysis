# Long-tailed Distribution-aware Router For Mixture-of-Experts in Large Vision-Language Models

Chaoxiang Cai, Longrong Yang, Minghe Weng, Xuewei Li, Zequn Qin<sup>∗</sup> , Xi Li<sup>∗</sup>

*Abstract*—The mixture-of-experts (MoE) architecture, which replaces dense networks with sparse ones, has attracted significant attention in large vision-language models (LVLMs) for achieving comparable performance while activating far fewer parameters. Existing MoE architectures for LVLMs primarily focus on token-to-expert routing (TER), encouraging different experts to specialize in processing specific tokens. However, these methods typically rely on the load balancing mechanism, neglecting the inherent distributional differences between vision and language modalities. To address this limitation, we propose the Long-Tailed Distribution-aware Router (LTDR) for visionlanguage TER, which tackles two key challenges: (1) Modalityspecific distribution-aware routing. We observe that language TER generally follows a relatively uniform distribution, whereas vision TER exhibits a long-tailed distribution. This modality discrepancy motivates the design of specialized routing strategies for each modality. (2) Vision-specific dynamic expert activation. Recognizing the importance of high-information vision tail tokens, we introduce a data-augmentation-inspired strategy that increases the number of activated experts, ensuring sufficient learning for these rare but informative tokens. On vision-language and vision benchmarks, our approach achieves consistent improvements, boosting performance by 1.2% / 2.1% on vision-language and 1.6% on vision benchmarks.

*Index Terms*—Long-tailed distribution, Modality-specific routing, Large Vision-Language Models, Mixture-of-Experts.

### I. INTRODUCTION

Recent advances in large vision-language models (LVLMs) [1], [2], which bridge vision and language, have demonstrated impressive instruction-following and generalization capabilities. However, real-world necessitate models that can handle diverse tasks. Traditional approaches that train separate models for each task incur significant redundancy and resource consumption. Despite efforts to expand datasets or models [3], [4], [5], the demand for substantial resources remains. The mixture-of-experts (MoE) [6] architecture enables scalable parameter growth without a proportional increase in inference costs. Its effectiveness in model scaling has been demonstrated in various recent works [7], [8], [9], [10], [11]. MoE-LLaVA [12] achieves performance comparable to LLaVA-7B and LLaVA-13B [13] while activating only 3B parameters.

The core of MoE lies in its token-to-expert routing (TER). While most implementations [14] utilize trainable routers

Chaoxiang Cai and Zequn Qin are with the School of Software Technology, Zhejiang University, Ningbo 315100, China. Longrong Yang, Minghe Weng and Xi Li are with the College of Computer Science and Technology, Zhejiang University, Hangzhou 310027, China. Xuewei Li is with the School of Electronic and Information, Shanghai Dianji University, Shanghai 201306, China. E-mail: {cxcai, longrongyang, wengminghe, xilizju}@zju.edu.cn, zequnqin@gmail.com, xueweili@sdju.edu.cn.

(Corresponding authors: Zequn Qin and Xi Li)

to predict routing probabilities, they typically enforce load balancing constraints on TER to prevent expert overload or underload, thereby promoting a uniform distribution of tokens across experts. However, this uniform-load strategy is suboptimal for multi-modal tasks, where vision tokens follow a long-tailed distribution [15], [16] in contrast to the more uniform distribution of language tokens [17], [18]. Indiscriminate load balancing hinders the effective learning of vision tokens. As illustrated in Fig. 1 (a), vision tokens comprise a majority of low-information background tokens (the head) and a minority of high-information foreground tokens (the tail). Enforcing load balancing scatters these critical, but sparse, foreground tokens across different experts, which impedes the consolidation of similar information and prevents the selection of specialized experts for these important tokens.

1

We perform an analysis of this question. Fig. 1 (b) plots the TER probability variance (x-axis) against the token count (y-axis). The distribution of vision token counts across probability variances is long-tailed. The majority of tokens, which possess low probability variances, typically correspond to lowinformation background tokens, as the router struggles to assign specialized experts for them. In contrast, high-information foreground tokens generally exhibit high variances. As shown in Fig. 1 (b), the application of load balancing reduces the count of tokens with high variances, suggesting that the load balancing hinders the expert specialization for foreground tokens. While language token counts are uniformly distributed across probability variances, indicating that language tokens are compatible with load balancing. Fig. 1 (c) shows that removing load balancing for vision tokens leads to improved performance. Based on these, our goal is to ensure that sparse yet high-information vision tail tokens are sufficiently learned within specialized experts as shown in Fig. 1 (a).

To ensure sufficient learning of vision tail tokens, we propose the Long-Tailed Distribution-aware Router (LTDR). Our approach solves two key challenges: (1) *Modality-specific distribution-aware routing.* Increasing the variance of routing probabilities allows vision tokens, especially vision tail tokens, to be learned within specialized experts. Consequently, we remove load balancing for vision TER to align with its longtailed distribution, thus improving the variance of routing probabilities. For language TER, we retain load balancing as it aligns well with its uniform distribution. (2) *Vision-specific dynamic expert activation.* Recognizing the importance of highinformation vision tail tokens, we employ a data-augmentation strategy to increase the number of activated experts for them, thereby improving fault tolerance during specialized expert selection. Our method achieves improvements of 1.2% / 2.1% on vision-language benchmarks and 1.6% on vision benchmarks.

![](_page_1_Figure_1.jpeg)

Fig. 1. (a) Our goal is to ensure that vision tail tokens are sufficiently learned within specialized experts. (b) Distribution of TER probability variance. Language TER with load balancing is uniform, vision TER without load balancing exhibits a long-tailed distribution and vision TER with load balancing shows a biased long-tailed characteristic. (c) GMoE with and without load balancing. Removing load balancing from vision tokens improves performance.

Our contributions can be summarized as follows:

- We identify the detrimental effect of traditional load balancing on vision tokens. This effect stems from a distribution mismatch, where the uniform routing preference of load balancing leads to insufficient learning of important, yet long-tailed, vision tail tokens.
- We propose the long-tailed distribution-aware router (LTDR), which contains two key modules: (1) Modality-specific distribution-aware router for vision-language tasks with diverse distributions; and (2) Vision-specific dynamic expert activation for vision tail tokens that necessitate sufficient representation learning.
- Extensive experiments demonstrate the effectiveness of our approach, achieving average improvements of 1.2% / 2.1% on vision-language and 1.6% on vision benchmarks.

#### II. RELATED WORKS

#### A. Mixture-of-Experts in Large Vision-Language Models

The success of large language models (LLMs) has driven rapid progress in LVLMs [12]. LVLMs [1], [13] encode visual inputs into LLM-compatible representations, enabling effective vision-language integration. Current efforts focus on dataset scaling [19] and parameter-efficient adaptation [20], [21], [22], [23] to improve efficiency. However, LVLMs still suffer from limited generalization and high inference costs, motivating the adoption of MoE architectures.

The effectiveness of MoE models largely depends on their token-to-expert routing mechanisms. Traditional MoE employs a trainable linear layer to predict routing probabilities [24], often with load-balancing constraints to ensure uniform expert utilization. Building on this, various advanced routing

strategies have been proposed. Task-based routing [25], [26], [27] assigns tokens to predefined experts, while cluster-based routing [7], [8], [28] groups feature-similar tokens to enhance specialization. Dynamic routing [29], [30] further adapts the number of activated experts, reducing reliance on fixed hyper-parameters. Recent work also explores richer routing dynamics, including modeling task relationships [31], mitigating gradient conflicts [32], and sequential routing [33], highlighting the critical role of routing in improving MoE efficiency.

Nevertheless, existing methods largely overlook routing under long-tailed distributions, which are common in real-world scenarios. Prior works [34], [35] mainly address long-tailed imbalance at the sample level and rely on conventional routing strategies, without introducing dedicated routing designs. In contrast, our LTDR targets long-tailed distributions at the token level and further enables modality-specific routing, providing a more comprehensive and effective solution.

#### B. Long-Tailed Distribution

Real-world data typically exhibits a long-tailed distribution, causing models to perform well on head classes while overfitting tail classes [36], [37], [38], [39], [40], [41], [42]. Long-tailed visual classification addresses this challenge under severe class imbalance. Traditional methods can be grouped into re-sampling, re-weighting, and ensemble methods.

(1) Re-sampling methods [43], [44] balance data via over-sampling tail classes or under-sampling head classes. Approaches such as BBN [45] adopt dual-branch designs to jointly learn original and re-balanced distributions, while other methods use collaborative branches with loss adjustments to improve both head and tail performance. (2) Re-weighting

methods [46], [47] assign larger weights to tail samples and down-weight head classes, typically based on inverse class frequency. Advanced designs, such as LDAM [48] and EQL [49], further improve performance by introducing margin-based objectives or selectively ignoring gradients for rare categories. (3) Ensemble methods [50], [51] leverage multiple experts to address long-tailed learning. Methods like RIDE [34] reduce variance through independent expert training and dynamic routing, while other approaches employ multi-stage or multiexpert fusion strategies to balance head and tail performance and identify hard samples more effectively.

### III. METHODOLOGY

### *A. Preliminaries*

Large Vision-Language Models. LVLMs integrate the capabilities of LLMs with visual processing technologies, enabling vision-language understanding and generation. As shown in Fig. 1 (a), the text input is first transformed through a word embedding, which projects the text t into a continuous vector space, resulting in the language token sequence T = [t1, t2, · · · , t<sup>N</sup> ] ∈ R <sup>N</sup>×<sup>D</sup>. N means the sequence length of language tokens, and D denotes the hidden layer size of the LLM.

Similarly, the RGB image input v ∈ R H×W×3 , where H, W, and 3 denote the height, width, and channels of the image at its original resolution, the visual encoder processes the image to extract a sequence of vision tokens Z = [z1, z2, · · · , zM] ∈ RM×<sup>C</sup> . M is the sequence length of vision tokens, and C is the hidden layer size of the visual encoder. To align vision tokens with language tokens in the same vector space, a visual projection layer is employed to map Z ∈ RM×<sup>C</sup> to V = [v1, v2, · · · , vM] ∈ RM×<sup>D</sup>, where D matches the hidden layer size of the LLM. This alignment ensures that both vision and language can be processed jointly by the subsequent layers of the model.

Subsequently, vision and language tokens are concatenated and fed into the LLM. The LLM consists of layers of multihead self-attention (MSA) and feed-forward network (FFN), with layer normalization (LN) and residual connection applied to stabilize training and enhance performance. As shown in Eq. 1 ∼ Eq. 4, where L is the layer number of LLM, the LVLM achieve a deep understanding of the relationships between vision and language, enabling effective performance on visionlanguage tasks.

$$\mathbf{x}_0 = [v_1, v_2, \cdots, v_M, \cdots, t_1, t_2, \cdots, t_N]$$
 (1)

$$\mathbf{x}'_{\ell} = \text{MSA}(\text{LN}(\mathbf{x}_{\ell-1})) + \mathbf{x}_{\ell-1}, \ell \in \{1, \dots, L\}$$
 (2)

$$\mathbf{x}_{\ell} = \text{FFN}(\text{LN}(\mathbf{x'}_{\ell})) + \mathbf{x'}_{\ell}, \ell \in \{1, \dots, L\}$$
 (3)

$$\mathcal{Y} = LN(\mathbf{x}_L) \tag{4}$$

The output of the LVLM is optimized through a generative loss in an auto-regressive manner. Given an image and its corresponding instruction text, the LVLM aims to generate the output text sequence Y = [y1, y2, · · · , yO] ∈ R <sup>O</sup>×<sup>D</sup> by progressively prediction, where O is the sequence length of the text output. The loss function is defined in Eq. 5, Y<i

indicates the output sequence before token y<sup>i</sup> , θ denotes the trainable parameters of the model. We only calculate the loss for the generated text.

$$\mathcal{L}_{\text{regressive}} = -\sum_{i=1}^{O} \log p(y_i \mid \mathcal{V}, \mathcal{T}, \mathcal{Y}_{< i}, \theta)$$
 (5)

Mixture-of-Experts. We replace FFN layers as MoE layers, following MoE-LLaVA [12]. A MoE layer typically contains multiple FFNs, denoted as an experts ensemble E = [e1, e2, · · · , eK], K is the number of total experts. The router implements a linear layer to predict the routing probability of assigning tokens to experts. As shown in Eq. 6, the router produces weight logits f(x) = W · x, which are normalized by the softmax function. The matrix W ∈ R <sup>D</sup>×<sup>K</sup> denotes the lightweight trainable parameters for routing, and P(x)<sup>i</sup> is the routing score of the input x for the i-th expert. The final output in Eq. 7 is computed as a weighted sum of the outputs from the Top-k experts with the highest softmax probabilities. E(x)<sup>i</sup> is the output of the i-th expert, and the weight for each expert is determined by its routing score.

$$\mathcal{P}(\mathbf{x})_i = \frac{e^{f(\mathbf{x})_i}}{\sum_{j=1}^K e^{f(\mathbf{x})_j}}$$
(6)

$$MoE(\mathbf{x}) = \sum_{i=1}^{k} \mathcal{P}(\mathbf{x})_i \cdot \mathcal{E}(\mathbf{x})_i$$
 (7)

Due to the presence of multiple experts, it is necessary to impose the expert load balancing constraint on MoE layers. Traditional methods [12], [32] incorporate differentiable load balancing loss [52] into each MoE layer to encourage experts to handle tokens in a balanced manner. As shown in Eq. 8, F<sup>i</sup> is the fraction of tokens processed by expert E<sup>i</sup> , and G<sup>i</sup> is the average routing probability of expert E<sup>i</sup> .

$$\mathcal{L}_{\text{balancing}} = K \cdot \sum_{i=1}^{K} \mathcal{F}_i \cdot \mathcal{G}_i$$
 (8)

Our Method LTDR. Since language tokens follow a uniform distribution, while vision tokens exhibit a long-tailed distribution, we focus on optimizing vision-language TER to make experts handle different distributional modality tokens effectively. We find that the load balancing mechanism leads to the scattered vision tail tokens in experts, impeding the learning of specialized experts. Therefore, as illustrated in Fig. 2, our method consists of two modules: (1) Modalityspecific Distribution-aware Router (MsDaR). We retain load balancing for language TER as it aligns with the uniform distribution of language tokens, while abandon load balancing for vision TER to adaptively align with the long-tailed distribution of visual tokens. Without load balancing, vision tokens, especially vision tail tokens, exhibit higher routing probability variance, enabling expert specialization. (2) Visionspecific Dynamic Expert Activation (VsDEA). Given the high importance of vision tail tokens, we define the head and tail tokens of vision, and increase the number of activated experts for vision tail tokens, achieving a data-augmentation strategy to improve fault tolerance and learning effectiveness.

![](_page_3_Figure_1.jpeg)

Fig. 2. (a) Modality-specific distribution-aware router allows vision and language to be routed with different expert load to adapt to their respective modality distributions. (b) Vision-specific dynamic expert activation enables a data-augmentation strategy to make experts process important vision tail tokens sufficiently.

#### B. Modality-specific Distribution-aware Router

Existing MoE architectures on modality differences fall into modality-aware [53], [54], [55] and distribution-aware [34], [35]. [53], [55] center on modalities by using a hierarchical MoE and modality-specific expert groups, while [34], [35] focus on long-tailed distribution by enhancing expert diversity and reducing dynamic routing. These methods are constrained by load balancing (Eq. 8), without considering modality token distribution differences.

Recent works [15], [16], [17], [18] have shown that language follows a uniform distribution, whereas vision exhibits a long-tailed distribution. This divergence stems from the characteristics of vision, which contains a few foreground patches and a large number of background patches. [56], [57] also emphasize that the structural and semantic differences between vision and language necessitate specific processing. To this end, we optimize the TER distribution (Eq. 6) for vision and language.

Load balancing conflicts with the long-tailed vision distribution, an intuitive way is to release vision tokens from load balancing to increase their routing probability variance (RPV). As shown in Eq. 9, for a vision token  $v_i \in \mathcal{V}$ , its routing probabilities  $\mathcal{P}(v_i) \in \mathbb{R}^K$ , and  $\operatorname{RPV}(v_i)$  is its variance of  $\mathcal{P}(v_i)$ . The number of vision head tokens is sufficient to generalize across various experts, resulting in uniform routing probability (low RPV). If the limited number of vision tail tokens are distributed uniformly across experts, it would lead to poor generalization. Therefore, a modality-specific distribution-aware router is essential to benefit the sufficient learning of experts for vision tokens. We modify the traditional  $\mathcal{L}_{\text{balancing}}$  in Eq. 10, where  $\mathcal{T}$  means the language token sequence.

$$RPV(v_i) = Variance(\mathcal{P}(v_i))$$
 (9)

$$\mathcal{L}_{\text{balancing}} = \sum_{i=1}^{K} \mathcal{F}_i(\mathcal{T}) \cdot \mathcal{G}_i(\mathcal{T})$$
 (10)

In this architecture, language tokens maintain their original load balancing behavior, while vision tokens are released from the constraint of load balancing. This enables vision tokens, especially vision tail tokens, which represent information-rich content, to undergo specialized expert processing. As shown in Fig. 2 (a), by enhancing the RPV of vision tokens, these tokens can be allocated to specialized experts instead of being uniformly distributed, facilitating more specialized and efficient learning.

#### C. Vision-specific Dynamic Expert Activation

Given the high importance of vision tail tokens, we enhance their processing for sufficient expert learning. However, the complexity and training instability introduced by long-tailed tokens necessitate a simple yet effective solution. Since the input and output token sequence lengths fix before and after each FFN layer, traditional sampling methods are impractical for adjusting head or tail token counts. We thus employ a data-augmentation strategy to enhance the expert learning and processing for vision tail tokens. As illustrated in Fig. 2 (b), our innovation lies in defining and identifying vision tail tokens and activating more experts to learn and process vision tail tokens. First, we define whether a vision token  $v_i \in \mathcal{V}$  is a vision head token or vision tail token by the following Eq. 11.

$$Type(v_i) = tail, if RPV(v_i) > Mean(RPV(V))$$
 (11)

 $\operatorname{Mean}(\operatorname{RPV}(\mathcal{V}))$  is the mean value of  $\operatorname{RPV}(\mathcal{V}) \in \mathbb{R}^M$ . We serve vision tokens with larger RPV than the  $\operatorname{Mean}(\operatorname{RPV}(\mathcal{V}))$  of total vision tokens as vision tail tokens, and the remainder as heads. Vision tail tokens generally exhibit higher RPV than head tokens, as they are processed by specialized experts. We use mean RPV as the dynamic threshold for vision tail token recognition, as it filters out most of vision head tokens. Subsequently, we route identified vision tail tokens to more experts than originally assigned, resulting in a modified version of Eq. 7 tailored for vision tail tokens in Eq. 12.

TABLE I

COMPARISONS OF DIFFERENT LVLMS AND OUR METHOD. MOE-LLAVA USES 4TOP2 WHILE OUR METHOD WITH THE VSDEA MODULE USES TOP4.

WE CALCULATE THE AVERAGE PERFORMANCE "AVG" ACROSS ALL DATASETS EXCEPT FOR MME.

| Method               | LLM                   | GQA  | ScienceQA-IMG | TextVQA | POPE | MME    | MMBench | MM-Vet | Avg  |
|----------------------|-----------------------|------|---------------|---------|------|--------|---------|--------|------|
|                      |                       |      | Dense Model   |         |      |        |         |        |      |
| IDEFICS [58]         | IDEFICS-65B [58]      | 45.2 | -             | 30.9    | -    | -      | 54.5    | -      | -    |
| LLaVA-1.5 [13]       | LLaMA-13B [59]        | 63.3 | 71.6          | 61.3    | 85.9 | 1531.3 | 67.7    | 35.4   | 64.2 |
| LLaVA-1.5 [13]       | Vicuna-7B [60]        | 62.0 | 66.8          | 58.2    | 85.9 | 1510.7 | 64.3    | 30.5   | 61.2 |
| Qwen-VL [4]          | Qwen-7B [4]           | 59.3 | 67.1          | 63.8    | -    | -      | 38.2    | -      | -    |
| TinyGPT-V [61]       | Phi2-2.7B [62]        | 33.6 | -             | -       | -    | -      | -       | -      | -    |
| MobileVLM [63]       | MobileLLaMA-2.7B [63] | 59.0 | 61.0          | 47.5    | 84.9 | 1288.9 | 59.6    | -      | -    |
| LLaVA-Phi [64]       | Phi2-2.7B [62]        | -    | 68.4          | 48.6    | 85.0 | 1335.1 | 59.8    | 28.9   | -    |
|                      |                       |      | Sparse Model  |         |      |        |         |        |      |
| MoE-LLaVA-4Top2 [12] | StableLM-1.6B [65]    | 60.3 | 62.6          | 50.1    | 85.7 | 1318.2 | 60.2    | 26.9   | 57.6 |
| Our Method           | StableLM-1.6B [65]    | 61.1 | 63.4          | 51.1    | 86.6 | 1363.5 | 60.6    | 29.9   | 58.8 |
| MoE-LLaVA-4Top2 [12] | Phi2-2.7B [62]        | 61.4 | 68.5          | 51.4    | 86.3 | 1423.0 | 65.2    | 34.3   | 61.1 |
| Our Method           | Phi2-2.7B [62]        | 62.2 | 69.3          | 52.9    | 87.5 | 1446.5 | 66.8    | 34.9   | 62.3 |

TABLE II

COMPARISONS OF OUR METHOD WITH MOLMO AND GMOE. MOLMO ADOPTS A 64TOP8 CONFIGURATION, WHEREAS OUR METHOD, ENHANCED BY THE VSDEA MODULE, USES A MORE EFFICIENT 64TOP12. SIMILARLY, GMOE OPERATES ON 4TOP2/6TOP2, WHILE OUR APPROACH ACHIEVES COMPARABLE RESULTS WITH ONLY TOP4. WE CALCULATE THE AVERAGE PERFORMANCE "AVG" ACROSS ALL DATASETS.

|                          | Vision-Language Evaluation         |                     |                     |                     |                     |                     |                     |                     |  |  |  |
|--------------------------|------------------------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|--|--|--|
| Method                   | MoE Topk                           | ChartQA             | DocVQA              | AI2D                | VQA                 | AndroidControl      | CountBenchQA        | Avg                 |  |  |  |
| Molmo [66]<br>Our Method | Top8 Top12 (VsDEA) / Top8 (others) | 65.7<br><b>68.1</b> | 79.8<br><b>81.3</b> | 85.2<br><b>87.4</b> | 82.6<br><b>83.3</b> | 81.8<br><b>83.2</b> | 74.0<br><b>77.6</b> | 78.1<br><b>80.2</b> |  |  |  |
|                          | Vision Evaluation                  |                     |                     |                     |                     |                     |                     |                     |  |  |  |
| Method                   | MoE Topk                           | PACS                |                     | VI                  | CS                  | Office              | -Home               | Avg                 |  |  |  |
| 1,1001100                |                                    | 4Top2               | 6Top2               | 4Top2               | 6Top2               | 4Top2               | 6Top2               |                     |  |  |  |
| GMoE [67]<br>Our Method  | Top2 Top4 (VsDEA) / Top2 (others)  | 88.9<br><b>90.1</b> | 87.6<br><b>92.1</b> | 92.5<br><b>95.7</b> | 96.5<br><b>96.9</b> | 70.5<br><b>70.6</b> | 71.0<br><b>71.4</b> | 84.5<br><b>86.1</b> |  |  |  |

$$\operatorname{MoE}(x) = \begin{cases} \sum_{i=1}^{a} \frac{e^{\mathcal{P}(x)_i}}{\sum_{j=1}^{a} e^{\mathcal{P}(x)_j}} \cdot \mathcal{E}(x)_i, & \text{vision tail token} \\ \sum_{i=1}^{k} \frac{e^{\mathcal{P}(x)_i}}{\sum_{j=1}^{k} e^{\mathcal{P}(x)_j}} \cdot \mathcal{E}(x)_i, & \text{others} \end{cases}$$

$$(12)$$

The weights of the selected experts will be renormalized. Vision tail tokens can be handled by more experts  $(k < a \le K)$ . Reducing the possibility of expert incorrect routing.

#### IV. EXPERIMENTS

#### A. Experimental Setups

**Benchmarks.** We aim to study the vision TER within the MoE architecture. To ensure a robust evaluation, we conduct experiments on a suite of both vision-language and vision tasks. A detailed description of the benchmarks and tuning datasets can be found in the supplementary material. This comprehensive evaluation framework allows for a rigorous assessment of our approach across a diverse capabilities.

**Baselines.** We employ MoE-LLaVA [12] and Molmo [66] for vision-language tasks, and GMoE [67] for vision tasks. MoE-LLaVA selects the two most relevant experts from a total of four experts, while Molmo selects 8 experts from 64 experts. GMoE is designed for visual generalization and enhances the ability of cross-domain data generalization.

**Configurations.** For vision-language tasks, we build upon two architectures: MoE-LLaVA (with StableLM-1.6B and Phi-2-2.7B backbones) and Molmo (with OLMoE-1B-7B backbone). For vision tasks, we adopt the pre-trained ViT-S/16 backbone following GMoE. Besides, our routing strategy is applied at each training batch. Further implementation details are provided in the supplementary material.

### B. Comprehensive Evaluations

**Vision-Language.** As shown in Tab. I  $\sim$  Tab. II, our method demonstrates robust image-text understanding, achieving superior performance compared to the MoE-LLaVA and Molmo. Specifically, our method achieves average improvements of 1.2% over StableLM-1.6B, 1.2% over Phi2-2.7B, and 2.1% over OLMoE-1B-7B. The improvements across different models also highlight the scalability of our method.

**Vision.** Tab. II assesses the generalization capability of our method on vision tasks. Our method enhances the performance of GMoE, yielding an average improvement of 1.6%.

### C. Ablation Studies

**Accuracy.** Tab. III show ablation studies on MoE-LLaVA and Molmo. The results confirm the importance of modality-specific distribution-aware router (MsDaR) and vision-specific dynamic expert activation (VsDEA) modules.

 $\label{eq:table_initial} \textbf{TABLE III} \\ \textbf{Modular ablation studies on MoE-LLaVA and Molmo}.$ 

| Method                          | GQA     | ScienceQA-IMG | TextVQA | POPE | MMBench        | MM-Vet       | Avg  |
|---------------------------------|---------|---------------|---------|------|----------------|--------------|------|
| MoE-LLaVA-4Top2 (StableLM-1.6B) | 60.3    | 62.6          | 50.1    | 85.7 | 60.2           | 26.9         | 57.6 |
| + MsDaR                         | 61.1    | 62.3          | 51.2    | 86.6 | 59.9           | 27.9         | 58.2 |
| + MsDaR&VsDEA (LTDR)            | 61.1    | 63.4          | 51.1    | 86.6 | 60.6           | 29.9         | 58.8 |
| MoE-LLaVA-4Top2 (Phi2-2.7B)     | 61.4    | 68.5          | 51.4    | 86.3 | 65.2           | 34.3         | 61.1 |
| + MsDaR                         | 61.8    | 68.6          | 51.8    | 86.7 | 66.2           | 34.5         | 61.6 |
| + MsDaR&VsDEA (LTDR)            | 62.2    | 69.3          | 52.9    | 87.5 | 66.8           | 34.9         | 62.3 |
| Method                          | ChartQA | DocVQA        | AI2D    | VQA  | AndroidControl | CountBenchQA | Avg  |
| Molmo-64Top8 (OLMoE-1B-7B)      | 65.7    | 79.8          | 85.2    | 82.6 | 81.8           | 74.0         | 78.1 |
| + MsDaR                         | 67.8    | 80.7          | 86.4    | 83.0 | 81.7           | 76.9         | 79.4 |
| + MsDaR&VsDEA (LTDR)            | 68.1    | 81.3          | 87.4    | 83.3 | 83.2           | 77.6         | 80.2 |

TABLE IV
RUNNING TIME COMPARISON ON MOE-LLAVA WITH STABLELM-1.6B AND MOLMO WITH OLMOE-1B-7B.

| Method          | GPU      | GQA     | ScienceQA-IMG | TextVQA | POPE | MMBench        | MM-Vet       | Avg (s) |
|-----------------|----------|---------|---------------|---------|------|----------------|--------------|---------|
| MoE-LLaVA-4Top2 | V100-30G | 2284    | 331           | 1277    | 1552 | 835            | 366          | 1108    |
| Our Method      | V100-30G | 2252    | 368           | 1259    | 1530 | 825            | 363          | 1100    |
| MoE-LLaVA-4Top2 | A800-80G | 1771    | 285           | 1265    | 1269 | 603            | 310          | 917     |
| Our method      | A800-80G | 1698    | 301           | 1057    | 1116 | 595            | 310          | 846     |
| Method          | GPU      | ChartQA | DocVQA        | AI2D    | VQA  | AndroidControl | CountBenchQA | Avg (s) |
| Molmo-64Top8    | A800-80G | 184     | 200           | 190     | 196  | 89             | 705          | 261     |
| Our Method      | A800-80G | 187     | 203           | 194     | 200  | 90             | 716          | 265     |

![](_page_5_Figure_5.jpeg)

Fig. 3. Expert load of MoE-LLaVA with StableLM-1.6B and LTDR on MME. LTDR does not significantly increase the load on the slowest experts.

**Running Time.** As shown in Tab. IV. While our method activates more experts for vision tail tokens, its inference time does not increase significantly. This stems from the all-to-all communication waiting principle: our method enhances expert activation without overburdening the slowest expert. Details of expert load is provided in Fig. 3.

**Expert Load.** Although our method activates more experts, it does not increase inference time. We attribute this to the all-to-all communication, in which the inference speed is determined by the slowest expert. As shown in Fig. 3, our method enhances expert activation without substantially increasing the slowest expert load, thereby preserving inference efficiency.

Memory and Utilization. We quantify VsDEA's Memory

usage (G) and GPU Utilization (%) using a V100-30G GPU. The analyzed results are shown in Tab. V. It is worth noting that our method and the baseline show almost identical metrics, which means that our method does not introduce additional computational overhead.

#### D. Comparison Studies

**Routing Strategies.** We compare different routing strategies in Tab. VI, including task, cluster, instruction, dynamic, conflict mitigation, modality and distribution. Please refer to the supplementary material for detailed implementations. Our method achieves state-of-the-art performance, demonstrating its superior performance.

 $TABLE\ V \\ Memory\ used\ (G)\ and\ GPU\ Utilization\ (\%)\ comparison\ on\ MoE-LLaVA-4Top2\ with\ Stable LM-1.6B.$ 

| Model                 | П   | GQ.          | A        | ScienceQ     | A-IMG    | TextV          | 'QA      | POP          | E        | MM           | Œ        | MMBe           | ench     | MM-          | Vet      | Avg          | 3              |
|-----------------------|-----|--------------|----------|--------------|----------|----------------|----------|--------------|----------|--------------|----------|----------------|----------|--------------|----------|--------------|----------------|
|                       | - [ | Memo (G)     | Util (%) | Memo (G)     | Util (%) | Memo (G)       | Util (%) | Memo (G)     | Util (%) | Memo (G)     | Util (%) | Memo (G)       | Util (%) | Memo (G)     | Util (%) | Memo (G)     | Util (%)       |
| Vanilla<br>Our Method | d   | 9.30<br>9.32 | 65<br>67 | 9.08<br>9.03 | 61<br>57 | 10.43<br>10.43 | 63<br>59 | 8.63<br>8.63 | 67<br>67 | 9.11<br>9.11 | 66<br>67 | 10.47<br>10.47 | 64<br>65 | 9.08<br>9.08 | 31<br>33 | 9.44<br>9.44 | 59.57<br>59.29 |

TABLE VI
COMPARISON OF ROUTING STRATEGIES ON MOE-LLAVA-4TOP2 WITH
STABLELM-1.6B. THE BEST IS MARKED WITH BOLDFACE.

| Method                      | GQA  | TextVQA | POPE | MM-Vet | Avg  |
|-----------------------------|------|---------|------|--------|------|
| Vanilla                     | 60.3 | 50.1    | 85.7 | 26.9   | 55.7 |
| + Task [25], [26], [27]     | 58.2 | 49.2    | 81.5 | 25.2   | 53.5 |
| + Cluster [7], [8]          | 57.0 | 50.3    | 86.1 | 27.3   | 55.1 |
| + Instruct [68]             | 58.1 | 50.0    | 85.9 | 26.6   | 55.1 |
| + Dynamic [29], [30]        | 61.0 | 49.2    | 85.7 | 28.2   | 56.0 |
| + STGC [32]                 | 60.9 | 50.7    | 85.9 | 28.2   | 56.4 |
| + Modality [54], [55], [53] | 60.4 | 49.2    | 85.9 | 28.4   | 55.9 |
| + Distribution [35]         | 60.1 | 50.5    | 86.5 | 27.8   | 56.2 |
| + LTDR (Ours)               | 61.1 | 51.1    | 86.6 | 29.9   | 57.2 |

TABLE VII

COMPARISON OF SHARPENING VISION TOKEN ROUTING STRATEGIES ON MOE-LLAVA-4TOP2 WITH STABLELM-1.6B.

| Mehtod                                    | ScienceQA-IMG        | TextVQA              | MME                        | MM-Vet               |
|-------------------------------------------|----------------------|----------------------|----------------------------|----------------------|
| Vanilla                                   | 62.6                 | 50.1                 | 1318.2                     | 26.9                 |
| + Temperature<br>+ Entropy<br>+ Heuristic | 62.8<br>62.4<br>62.5 | 50.2<br>50.8<br>50.3 | 1302.5<br>1330.7<br>1299.6 | 26.8<br>27.2<br>26.6 |
| + LTDR (Ours)                             | 63.4                 | 51.1                 | 1363.5                     | 29.9                 |

Sharpening Vision Token Routing Strategies. We assess the effect of sharpening vision token routing by comparing temperature scaling, entropy regularization, and heuristic approach in Tab. VII. Temperature: Gumbel-Softmax with temperature 0.7. Entropy: entropy regularization to promote confident routing. Heuristic: variance constraints on vision token routing distributions. Simply sharpening vision token routing does not lead to notable gains. We hypothesize this stems from differing behaviors of head and tail tokens: Tail tokens, being fewer, benefit from sharper routing to assign them to suitable experts more effectively. Head tokens, which are abundant and already learn adequately from multiple experts, may adversely affect the learning of tail tokens.

Vision Token Selection in VsDEA. We choose vision tail tokens (VTTs), whose RPV exceeds the mean RPV of all vision tokens. Two strategies are compared in the upper of Tab. VIII: Vision head tokens (VHTs), whose RPV are below the mean RPV of all vision tokens. It selects a large proportion of tokens ( $\approx$ 87%) compared to VTTs ( $\approx$ 13%). Instruction-aware Tokens (IATs). The attention scores between the instruction and vision tokens are used to identify the top 15% of vision tokens for VsDEA. As shown in Tab. VIII, enhancing expert activations for vision head tokens also improve the model's ability to learn visual information, although the benefits are less significant than those achieved with VTTs. Moreover, selecting a large proportion of tokens

#### TABLE VIII

COMPARISON OF VISION TOKEN SELECTION STRATEGIES IN VSDEA ON MOE-LLAVA-4TOP2 WITH STABLELM-1.6B. INCLUDING THE SELECTION OF VISION HEAD AND TAIL TOKENS, AS WELL AS THE SELECTION OF INSTRUCTION-AWARE VISION TAIL TOKENS. FOR VISION TAIL TOKENS, IT INCLUDES CASES UNDER DIFFERENT FIXED RATIOS.

| Method                | ScienceQA-IMG | TextVQA | MM-Vet   Avg |
|-----------------------|---------------|---------|--------------|
| Vanilla               | 62.6          | 50.1    | 26.9   57.6  |
| VHTs                  | 62.0          | 50.3    | 27.8   58.0  |
| IATs                  | 61.0          | 50.8    | 26.9 57.7    |
|                       | VTTs          |         |              |
| 10% (fixed)           | 62.2          | 50.9    | 25.5   57.9  |
| 15% (fixed)           | 62.2          | 50.8    | 25.3 57.9    |
| 20% (fixed)           | 62.8          | 50.1    | 25.1 57.4    |
| Ours (≈13%, adaptive) | 63.4          | 51.1    | 29.9   58.8  |

#### TABLE IX

Comparison with the "Share + Routing" strategy of DeepSeekMoE [11] on MoE-LLaVA-4Top2 with StableLM-1.6B.  $S_s+R_r$  denotes S share expert(s) of size s combined with R routing experts of size r. The best is marked with **Boldface**.

| Method                                | ScienceQA-IMG | POPE | MMBench | MM-Vet A  | vg  |
|---------------------------------------|---------------|------|---------|-----------|-----|
| Vanilla                               | 62.5          | 85.7 | 60.2    | 26.9   58 | 8.8 |
| + 1 <sub>1.0</sub> + 3 <sub>1.0</sub> | 62.5          | 86.2 | 57.6    | 27.9   58 | 8.5 |
| $+1_{1.0} + 12_{0.25}$                | 62.7          | 86.2 | 57.6    | 27.9 58   | 8.6 |
| $+1_{1.0}+16_{0.25}$                  | 62.8          | 86.2 | 58.7    | 28.1 58   | 8.9 |
| + LTDR (4 <sub>1.0</sub> )            | 63.4          | 86.6 | 60.6    | 29.9 60   | 0.1 |

TABLE X
EVALUATION ON HIGH VISION-TOKEN SCENARIO (NLVR2) USING
MOE-LLAVA-4TOP2 WITH STABLELM-1.6B

| Method     | Accuracy (%) | Inference Time (s) |
|------------|--------------|--------------------|
| Vanilla    | 52.9         | 1121               |
| Our Method | 54.4         | 1124               |

increases the inference time cost. The influence of IATs is minimal, potentially due to noise from both visual and textual information, which may hinder its ability to reliably identify the important vision tokens. Finally, we conduct quantitative comparisons against three fixed thresholds in the lower of Tab. VIII to further demonstrate the consistent performance advantage of our method.

**DeepSeekMoE.** We also compare with DeepSeekMoE [11] in Tab. IX, which partitions K experts into mK experts and activates mk of them. Experts are subdivided as  $k_s$  share experts and  $k_r$  routing experts. Evaluations under  $1_{1.0} + 3_{1.0}$ ,  $1_{1.0} + 12_{0.25}$  and  $1_{1.0} + 16_{0.25}$  reveal that DeepSeekMoE under performs our method, despite its increase specialization. This supports our hypothesis that vision and language TER differ, highlighting the need for tailored modality-aware routing.

TABLE XI VALIDATION ON LARGE-SCALE INSTRUCTION-TUNING DATASETS USING MOE-LLAVA-4TOP2 WITH STABLELM-1.6B.

| Method<br>Data<br>GQA                                   | ScienceQA-IMG | TextVQA              | POPE         | MME              | MMBench      | MM-Vet<br>Avg                |
|---------------------------------------------------------|---------------|----------------------|--------------|------------------|--------------|------------------------------|
|                                                         |               | ShareGPT [13]        |              |                  |              |                              |
| Vanilla<br>665K<br>60.3<br>Our Method<br>665K<br>61.1   | 62.6<br>63.4  | 50.1<br>51.1         | 85.7<br>86.6 | 1318.2<br>1363.5 | 60.2<br>60.6 | 26.9<br>57.6<br>29.9<br>58.8 |
|                                                         |               | Open-LLaVA-NeXT [70] |              |                  |              |                              |
| Vanilla<br>1021K<br>61.0<br>Our Method<br>1021K<br>61.4 | 63.0<br>65.4  | 51.2<br>52.5         | 86.7<br>87.3 | 1360.3<br>1409.2 | 61.0<br>62.4 | 29.2<br>58.6<br>34.8<br>60.6 |

TABLE XII CONFIDENCE INTERVALS OF THE MOE-LLAVA-4TOP2 WITH STABLELM-1.6B AND PHI-2.7B ACROSS SEEDS.

| Method                      | GQA                    | ScienceQA-IMG          | TextVQA                | POPE                   | MME                          | MMBench                | MM-Vet<br>Avg                                    |
|-----------------------------|------------------------|------------------------|------------------------|------------------------|------------------------------|------------------------|--------------------------------------------------|
| StableLM-1.6B<br>Our Method | 60.3±0.20<br>61.1±0.15 | 62.4±0.20<br>63.4±0.15 | 50.0±0.15<br>51.0±0.10 | 85.5±0.21<br>86.6±0.15 | 1295.1±26.93<br>1342.9±24.66 | 60.0±0.15<br>60.7±0.12 | 26.9±0.40<br>57.5±0.12<br>29.6±0.25<br>58.8±0.06 |
| Phi2-2.7B<br>Our Method     | 61.2±0.20<br>62.2±0.15 | 68.4±0.21<br>68.5±0.20 | 51.3±0.23<br>52.1±0.10 | 86.2±0.21<br>86.7±0.00 | 1391.9±29.79<br>1414.5±26.45 | 65.3±0.15<br>66.7±0.20 | 34.1±0.15<br>61.1±0.10<br>34.2±0.20<br>61.7±0.06 |

### *E. Generalization Studies*

Higher Vision-Token Scenario. We evaluate inference performance on a multi-image visual question answering task, comparing our method with the baseline in terms of both accuracy and inference speed. Specifically, we use the NLVR2 dataset [69], in which each sample consists of two images and a textual statement, and the model must determine whether the statement correctly describes both images. Experimental results, presented in Tab. X, show that our method outperforms the baseline by 1.5% in scenarios involving a higher number of vision tokens, while maintaining comparable inference time and introducing no significant latency overhead.

Larger Training Dataset. We conduct experiments using an expanded training dataset to provide a more comprehensive comparison. Specifically, we incorporate the Open-LLaVA-NeXT training set, which adds 350K samples to the original 665K samples. Results summarized in Tab. XI demonstrate that our method achieves greater performance improvements (1.2%->2.0%) compared to the baseline after training on larger datasets. Furthermore, our method on small datasets outperforms the baseline on large datasets.

### *F. Confidence intervals of the models across different seeds*

We assess StableLM-1.6B and Phi2-2.7B using three random seeds to evaluate the consistency of our method in inference. As shown in Tab. XII, our approach consistently outperforms all baselines across these evaluations.

### *G. Visualization Examples*

We analyze the distribution of expert load on MoE-LLaVA-4Top2 with StableLM-1.6B. The expert loads of total tokens and image-text tokens are shown in Fig. 4 ∼ Fig. 5.

Fig. 4 indicates that our method does not significantly amplify the expert load imbalance compared to the baseline, which aligns with the analyses in Fig. 3. Fig. 5 depicts that language tokens follow a relatively uniform distribution. In contrast, our method eliminates the vision load balancing; the resulting imbalanced distribution of vision tokens shows that more vision tokens select specialized experts. This supports our hypothesis that the TER for language tokens adheres to a uniform distribution, whereas the TER for vision tokens follows a long-tailed distribution.

Our method dynamically adjusts token-to-expert paths to improve the language expert load balancing and vision expert specialization, better handling modality-specific distributions. Fig. 6 shows token activation maps. Our method significantly changes the original token activation path.

### V. CONCLUSION

We reveal the distinct token-to-expert routing (TER) distributions in vision-language tasks: language TER follows a uniform distribution, while vision TER exhibits a long-tailed distribution. This challenges the traditional load balancing mechanism in MoE: experts should receive an equal count of tokens to avoid a small number of experts gaining a disproportionately large share of preferences by the router. To address this, we propose the Long-Tailed Distributionaware Router (LTDR) for vision-language TER, addressing two key challenges: (1) Modality-specific distribution-aware routing. We retain the load balancing mechanism for language TER but abandon it for vision TER, enabling important vision tail tokens to be routed to specialized experts. (2) Visionspecific expert activation. Recognizing the importance of vision tail tokens, we employ a data-augmentation strategy, which increases the number of activated experts to ensure their thorough processing. To verify the effectiveness of our LTDR, we conduct extensive experiments on both visionlanguage and vision benchmarks. Experimental results verify the effectiveness of our approach.

![](_page_8_Figure_1.jpeg)

Fig. 4. Expert token load across layers. Bar heights indicate token proportions assigned to experts. LTDR yields a more balanced expert utilization.

![](_page_8_Figure_3.jpeg)

Fig. 5. Expert token load cross modal. Bar heights indicate token proportions assigned to experts. LTDR yields more balanced cross-modal expert utilization.

![](_page_8_Figure_5.jpeg)

Fig. 6. Top-10 pathways for text and image, with Top-2 in color and others in gray. LTDR significantly modifies the original token activation maps.

TABLE XIII INSTRUCTION-TUNING DATASETS AND BENCHMARKS ON BOTH VISION-LANGUAGE AND VISION TASKS.

|                                                                                                                                                                                                            | Instruction-tuning Datasets                                                                                                                                                                                                                                                                       | Benchmarks                                 |                                                                                                    |                                                                                                      |                                            |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|--------------------------------------------|
| MoE-LLaVA                                                                                                                                                                                                  | Molmo                                                                                                                                                                                                                                                                                             | GMoE                                       | MoE-LLaVA                                                                                          | Molmo                                                                                                | GMoE                                       |
| LLaVA (158K) [13]<br>ShareGPT (40K) [75]<br>VQAv2 (83K) [71]<br>GQA (72K) [73]<br>OKVQA (9K) [80]<br>OCRVQA (80K) [86]<br>A-OKVQA (66K) [90]<br>TextCaps (22K) [92]<br>RefCOCO (48K) [93]<br>VG (86K) [94] | VQAv2 (440K) [71]<br>TextVQA (35K) [76]<br>OKVQA (9K) [80]<br>ChartQA (28K) [74]<br>DocVQA (39K) [79]<br>InfographicVQA (24K) [87]<br>AI2D (15K) [82]<br>A-OKVQA (17K) [90]<br>AndroidControl (300K) [85]<br>ScienceQA (6K) [78]<br>TabWMP (23K) [95]<br>ST-VQA (25K) [96]<br>TallyQA (250K) [97] | PACS [72]<br>VLCS [77]<br>Office-Home [81] | GQA [73]<br>ScienceQA [78]<br>TextVQA [76]<br>POPE [83]<br>MME [84]<br>MMBench [88]<br>MM-Vet [91] | ChartQA [74]<br>DocVQA [79]<br>AI2D [82]<br>VQAv2.0 [71]<br>AndroidControl [85]<br>CountBenchQA [89] | PACS [72]<br>VLCS [77]<br>Office-Home [81] |

TABLE XIV DETAILED TRAINING HYPER-PARAMETERS.

| Epoch           | Learning Rate      | Learning Rate Schedule         | Weight Decay | Load Balancing Loss Coefficient |
|-----------------|--------------------|--------------------------------|--------------|---------------------------------|
| 1               | 2e-5               | Cosine                         | 0.0          | 0.01                            |
| Text Max Length | Batch Size per GPU | Train Step                     | Precision    | The Number of a in VsDEA        |
| 2048            | 16                 | original(others) / 2000(Molmo) | Fp16         | 4(others) / 12(Molmo)           |

### VI. IMPLEMENTATION DETAILS

We utilize the instruction tuning datasets and benchmarks of MoE-LLaVA [12], Molmo [66] and GMoE [67]. The composition of these datasets are detailed in Tab. XIII. Our training configurations are based on MoE-LLaVA [12], Molmo [66], and GMoE [67]. The detailed hyper-parameters and implementation specifics are provided in Tab. XIV.

### VII. ADDITIONAL COMPARISON STUDIES

### *A. Detail Implementations of Routing Strategies*

Task. Similar to MoLA [27], aims to enhance similar routing for data from the same task while ensure distinct routing for data from different tasks. We conduct experiments using the LLaVA-mix-665k dataset. Empirically, we categorize the data into four task types: Caption, VQA, OCR and Regionaware. Each task type is assigned an expert label: 0 for *Caption*, 1 for *VQA*, 2 for *OCR*, and 3 for *Region-aware*. The dataset distributions are: *Caption* accounts for 3.5%, *VQA* for 61.6%, *OCR* for 12.8%, and *Region-aware* for 22.1%.

Cluster. Following MoCLE [8], we encode instructions from different datasets using the all-MiniLM-L6-v2 <sup>1</sup> and cluster their embeddings using the k-means clustering algorithm. After clustering, in line with MoCLE's approach, we initialize K learnable embeddings, where each embedding corresponds to a cluster center. When a sample belongs to the k-th cluster center, the k-th learnable embedding is extracted and passed to the router to predict routing scores. We set k-th=128, consistent with MoCLE's practice, and do not incorporate the load balancing loss.

Instruct. Following the approach of LoRA-MoE [68], we compute the average of instruction token representations for each instance and use this as input to predict its routing scores across experts. Based on these routing scores, the Topk experts are selected for each sample to generate the final prediction. In alignment with LoRA-MoE's methodology, we do not include the load balancing loss in our implementation.

Dynamic. Drawing inspiration from DYNMOE [30], we highlight its gating mechanism, which allows tokens to dynamically determine the number of experts to activate. Additionally, an adaptive process automatically adjusts the number of experts during training. We reference its experimental results on the MoE-LLaVA-4Top2 with StableLM-1.6B model, where an average of 1.25 experts out of 4 are activated per token.

STGC. STGC [32] employs token-level gradients to identify conflicting tokens within experts. Additionally, it introduces a regularization loss designed to encourage conflicting tokens to route away from their current experts to alternative ones, thereby minimizing interference among tokens within the same expert. We reference their experimental results on the MoE-LLaVA-4Top2 using the StableLM-1.6B model.

Modality. Following MoMa [55], we partition experts into two groups dedicated to vision and language, respectively. To maintain the total number of experts K (set to 4) and the number of activated experts k (set to 2), we designate 2 experts as vision experts and the remaining 2 as language experts, activating 1 expert from each modality group.

Distribution. Following RIDE [34], we cluster tokens into six categories via k-means (matching the 4Top2 expert paths, C 2 4 . Each category is mapped to a specific expert pair (e.g., category 1 → experts 1 & 2). Tokens of each category are directed to their assigned expert pair.

<sup>1</sup>https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2.

![](_page_10_Figure_1.jpeg)

Fig. 7. The modality-aware MoE architecture which divides experts for vision and language.

### *B. Modality-aware MoE Architecture*

To compare with modality-aware MoE architectures [54], [55], [53] as shown in Fig. 7, we adopt the following configurations:

MoE-LLaVA-v2Top1-t2Top1. We partition experts into two groups dedicated to vision and language. As illustrated in Fig. 7, to maintain the total number of experts K (set to 4) and the number of activated experts k (set to 2), we designate 2 experts as vision experts and the remaining 2 as language experts, activating 1 expert from each modality group.

MoE-LLaVA-v4Top2-t4Top2. We expand the 4 experts into 8 by splitting FFN intermediate hidden dimension [11], assigning 4 as vision experts and the remaining 4 as language experts, and activating 2 experts from each modality group.

MoE-LLaVA-v2Top1-t2Top1-MsDaR. Similar to *MoE-LLaVA-v2Top1-t2Top1*, but it removes the expert load balancing constraint for the vision expert group (similar to the distribution-aware router in our proposed approach, LTDR).

MoE-LLaVA-v2Top1-t2Top1-MsDaR-shared. Akin to the former, but includes a shared expert for world knowledge [11].

The results in Tab. XV indicate that the modality-aware MoE does not enhance the performance of MoE-LLaVA. Moreover, increasing the number of experts results in a performance decline, suggesting that a larger number of experts exacerbates expert load balancing issues, which negatively impacts vision TER. When the expert load balancing constraint is removed for the vision expert group, *MoE-LLaVA-v2Top1 t2Top1-MsDaR* shows improved performance compared to MoE-LLaVA-4Top2, validating the effectiveness of our Ms-DaR module. Finally, the addition of an extra expert in *MoE-LLaVA-v2Top1-t2Top1-MsDaR-shared* does not yield benefits, highlighting the distinctions between vision-language MoE and language MoE [11].

### VIII. ADDITIONAL CORROBORATING STUDIES

### *A. Strategy-swap Ablation on Vision and Language*

We perform a strategy-swap ablation on the language-side while keeping the vision-side fixed. As shown in Tab. XVI, removing the load balancing loss on the language side (language+MsDaR) results in fluctuating performance, yielding a little improvement over the vanilla model. Moreover, introducing text-specific dynamic expert activation (TsDEA) does not lead to consistent performance gains. In contrast, removing the load balancing loss on the vision side (vision+MsDaR) yields stable improvements, and further incorporating vision-specific dynamic expert activation (VsDEA) provides additional gains, achieving an average improvement of 1.2%.

### *B. Broader Cross-Router Performance*

In addition to comparing the different backbones in Section.IV-B, we also report the cross-router results, as summarized in Tab. XVII. In the Top-Dynamic setting, expert activation is determined by token-score ranking and expert capacity, where the expert capacity is defined as numtoken/numexperts × 3. LTDR with Top-1 achieves a 0.9% improvement over the vanilla model. Although dynamic routers provide additional gains, their performance remains lower than that of LTDR combined with Top-a.

### *C. Impact of Reducing Load Balancing*

We analyze the effect of reducing load balancing, with the results presented in Tab. XVIII. Specifically, we decrease the vision-side load balancing coefficient from 0.01 to 0.001. The results indicate that this adjustment is less effective than removing load balancing entirely.

### *D. Generalizability across datasets/models*

We compare the mean RPV using thresholds of 10%, 15%, and 20%, as the observed mean RPV (13%) falls within this range. This allows us to visually assess how small variations in the threshold affect performance. We evaluate these fixedproportion thresholds on the MoE-LLaVA with StableLM-1.6B model across multiple test datasets, as detailed in Appendix VII. Here, we further present generalizability experiments on the MoE-LLaVA with Phi-2-2.7B model, as shown in Tab. XIX. The results show that mean-RPV consistently outperforms the comparison thresholds. We also compare mean-RPV with the learnable strategy in Tab. XX; however, the tuning method yields performance that lies between that of the original model and the LTDR-based model.

### *E. The link between RPV and token informativeness*

We evaluate the link between RPV and token informativeness from two aspects: 1) Performance-based analysis (interpret ability). Higher model performance suggests that the corresponding tokens carry richer information. As described in Section.IV-D, we sort tokens by RPV and classify those above the mean RPV as vision tail tokens (13%) and the remaining ones as vision head tokens (87%). Although vision head tokens

TABLE XV COMPARISON WITH MODALITY-AWARE MOE [54], [55], [53] ON MOE-LLAVA-4TOP2 WITH STABLELM-1.6B.

| Method                               | GQA  | ScienceQA-IMG | TextVQA | POPE | MME    | MMBench | MM-Vet | Avg  |
|--------------------------------------|------|---------------|---------|------|--------|---------|--------|------|
| MoE-LLaVA-4Top2                      | 60.3 | 62.6          | 50.1    | 85.7 | 1318.2 | 60.2    | 26.9   | 57.6 |
| MoE-LLaVA-v2Top1-t2Top1              | 60.4 | 61.6          | 49.2    | 85.9 | 1293.3 | 61.1    | 28.4   | 57.7 |
| MoE-LLaVA-v4Top2-t4Top2              | 60.3 | 58.6          | 46.8    | 85.7 | 1296.6 | 55.4    | 26.4   | 55.5 |
| MoE-LLaVA-v2Top1-t2Top1-MsDaR        | 60.9 | 61.3          | 51.0    | 86.5 | 1324.5 | 61.1    | 28.4   | 58.2 |
| MoE-LLaVA-v2Top1-t2Top1-MsDaR-shared | 61.0 | 62.5          | 51.3    | 86.5 | 1333.8 | 60.0    | 28.0   | 58.2 |
| Our Method                           | 61.1 | 63.4          | 51.1    | 86.6 | 1363.5 | 60.6    | 29.9   | 58.8 |

TABLE XVI STRATEGY-SWAP ABLATION STUDIES ON VISION AND LANGUAGE MODALITIES.

| MoE-LLaVA_StableLM | GQA  | ScienceQA-IMG | TextVQA  | POPE | MME    | MMBench | MM-Vet<br>Avg |
|--------------------|------|---------------|----------|------|--------|---------|---------------|
| Vanilla            | 60.3 | 62.6          | 50.1     | 85.7 | 1318.2 | 60.2    | 26.9<br>57.6  |
|                    |      |               | Language |      |        |         |               |
| + MsDaR            | 60.8 | 62.0          | 50.2     | 85.9 | 1254.1 | 60.0    | 28.0<br>57.8  |
| + MsDaR&TsDEA      | 60.1 | 61.2          | 50.4     | 86.2 | 1282.4 | 60.1    | 28.6<br>57.8  |
|                    |      |               | Vision   |      |        |         |               |
| + MsDaR            | 61.1 | 62.3          | 51.2     | 86.6 | 1324.3 | 59.9    | 27.9<br>58.2  |
| + MsDaR&VsDEA      | 61.1 | 63.4          | 51.1     | 86.6 | 1363.5 | 60.6    | 29.9<br>58.8  |

TABLE XVII BROADER CROSS-ROUTER PERFORMANCE ACROSS DIFFERENT EXPERT ACTIVATION NUMBERS.

| MoE-LLaVA_StableLM      | GQA  | ScienceQA-IMG | TextVQA | POPE | MME    | MMBench | MM-Vet<br>Avg |
|-------------------------|------|---------------|---------|------|--------|---------|---------------|
| Vanilla+Top-1 (k=1)     | 58.6 | 55.8          | 45.0    | 85.2 | 1245.3 | 56.2    | 27.2<br>54.7  |
| Vanilla+Top-2 (k=2)     | 60.3 | 62.6          | 50.1    | 85.7 | 1318.2 | 60.2    | 26.9<br>57.6  |
| LTDR+Top-1 (k=1)        | 59.7 | 58.2          | 45.6    | 85.8 | 1302.9 | 57.1    | 27.0<br>55.6  |
| LTDR+Dynamic (avg k=3)  | 60.8 | 62.1          | 51.1    | 86.8 | 1332.8 | 60.2    | 28.9<br>58.3  |
| LTDR+Top-a (avg k=2.26) | 61.1 | 63.4          | 51.1    | 86.6 | 1363.5 | 60.6    | 29.9<br>58.8  |

are far more numerous than tail tokens, applying the VsDEA strategy to head tokens yields substantially lower performance than applying it to tail tokens. This performance gap indicates that high-RPV vision tail tokens encode more informative content. 2) Statistics-based analysis (statistical). A token group with a higher mean L2 norm of its vector representations reflects richer underlying information. We compare the mean L2 norm of the top 13% vision token vectors (vision tail tokens) with those of tokens ranked between the top 13%–26% and 26%–39% by RPV. As shown in Tab. XXI, top 13% exhibits a higher mean L2 norm than the latter two, providing additional evidence that high-RPV vision tail tokens carry richer information.

### IX. VISUALIZATION EXAMPLES

### *A. Routing Probability Variance*

We also compare the token routing probability variance (RPV) between vision head tokens and vision tail tokens across GQA [73], MMBench [88] and TextVQA [76]. As shown in Fig. 8, for images that yield 576 tokens from CLIP, Mean(RPV(V)) denotes the mean RPV of all vision tokens V, Vhead and Vtail denote vision head tokens and vision tail tokens. The bars in the figure is the mean token count with RPV ranging from left to right for images (*e.g.*, in the upper left figure, the count of tokens with RPV ranging from 0.00 to 0.01 is 442). Our method significantly increases the mean RPV of vision tail tokens. Given that RPV reflects the TER probability distribution, these results demonstrate that vision tail tokens gain the ability to select their specialized experts. Meanwhile, the mean RPV of vision head tokens remains unchanged, indicating that vision head tokens are not affected. Moreover, we add three sets of RPV distribution visualization comparisons: training stages RPV (Fig. 9), and cross-router RPV (Fig. 10). The RPV distributions at 1,500, 3,000, and 4,500 training steps on TextVQA indicate that, although both the baseline and our method improve mean RPV,

TABLE XVIII IMPACT OF REDUCING LOAD BALANCING.

| MoE-LLaVA_StableLM | GQA  | ScienceQA-IMG | TextVQA | POPE | MME    | MMBench | MM-Vet<br>Avg |
|--------------------|------|---------------|---------|------|--------|---------|---------------|
| Vanilla            | 60.3 | 62.6          | 50.1    | 85.7 | 1318.2 | 60.2    | 26.9<br>57.6  |
| Reducing-based     | 60.2 | 62.4          | 50.9    | 86.3 | 1315.7 | 59.1    | 28.4<br>57.8  |
| LTDR               | 61.1 | 63.4          | 51.1    | 86.6 | 1363.5 | 60.6    | 29.9<br>58.8  |

TABLE XIX GENERALIZABILITY ACROSS DIFFERENT MODELS.

| MoE-LLaVA_StableLM   | GQA  | ScienceQA-IMG | TextVQA | POPE | MME    | MMBench | MM-Vet<br>Avg |
|----------------------|------|---------------|---------|------|--------|---------|---------------|
| Phi-2B with 10%      | 62.1 | 68.3          | 51.6    | 86.5 | 1406.3 | 65.6    | 33.7<br>61.3  |
| Phi-2B with 15%      | 62.0 | 68.4          | 51.8    | 86.6 | 1423.1 | 66.3    | 34.0<br>61.5  |
| Phi-2B with 20%      | 61.3 | 67.8          | 51.3    | 86.2 | 1414.5 | 66.5    | 34.1<br>61.2  |
| Phi-2B with mean-RPV | 62.2 | 69.3          | 52.9    | 87.5 | 1446.5 | 66.8    | 34.9<br>62.3  |

TABLE XX COMPARISON WITH SYSTEMATIC TUNING POLICY.

| MoE-LLaVA_StableLM | GQA  | ScienceQA-IMG | TextVQA | POPE | MME    | MMBench | MM-Vet<br>Avg |
|--------------------|------|---------------|---------|------|--------|---------|---------------|
| Vanilla            | 60.3 | 62.6          | 50.1    | 85.7 | 1318.2 | 60.2    | 26.9<br>57.6  |
| Tuning-based       | 61.0 | 62.1          | 50.6    | 86.3 | 1322.3 | 59.9    | 28.9<br>58.1  |
| LTDR               | 61.1 | 63.4          | 51.1    | 86.6 | 1363.5 | 60.6    | 29.9<br>58.8  |

TABLE XXI STATISTICS-BASED ANALYSIS.

| MoE-LLaVA_StableLM | mean L2 norm |
|--------------------|--------------|
| Top-13%            | 0.3158       |
| Top-13% - Top-26%  | 0.2124       |
| Top-26% - Top-39%  | 0.1475       |

### our approach achieves substantially greater enhancement. This suggests that the baseline has limited capacity to effectively capture critical visual information. Cross-Router: In the Top-Dynamic setting, expert activation is determined by token score ranking and expert capacity, where expert capacity is defined as numtoken/numexperts × 3. The results indicate that the mean RPV of the vision tail under Top-1 is significantly lower than in other configurations, suggesting that increasing the number of parameters can enhance vision tail token learning. However, despite Top-Dynamic having the largest number of parameters, it does not outperform Top-a in terms

of vision tail RPV, implying that simply increasing parameter

count does not guarantee performance improvement.

### *B. Visualization Cases.*

Our visualizations in Fig. 11 reveal that vision tail tokens focus on critical instruction-aware image patches, capturing question-relevant visual information which enhances answer accuracy.

![](_page_13_Figure_1.jpeg)

Fig. 8. The Routing probability variance distribution of vision head tokens and vision tail tokens.

![](_page_13_Figure_3.jpeg)

Fig. 9. The Routing probability variance distribution at three training steps on TextVQA.

![](_page_13_Figure_5.jpeg)

Fig. 10. The Routing probability variance distribution cross routers.

## **Image-text Visualization Cases and Model Outputs**

![](_page_14_Picture_2.jpeg)

*Q1: Is there a bicycle in the image?*

MoE-LLaVA: No

MoE-LLaVA+LTDR: Yes

*Q2: Is there a person in the image?*

MoE-LLaVA: No

MoE-LLaVA+LTDR: Yes

![](_page_14_Picture_9.jpeg)

*Q1:Which kind of furniture is wooden?*

MoE-LLaVA: Couch

MoE-LLaVA+LTDR: Bookcase

*Q2:What is the brown item of furniture?*

MoE-LLaVA: Mirror

MoE-LLaVA+LTDR: Couch

| x=5                 |    |         |      |      |
|---------------------|----|---------|------|------|
| x=5<br>if x>10:     |    |         |      |      |
| print('x            | is | larger  | than | 10') |
| else:               |    |         |      |      |
| <pre>print('x</pre> | is | smaller | thar | 10') |

*Q1: The image shows a python code. Is the output of the code 'x is larger than 10'?*

MoE-LLaVA: Yes

MoE-LLaVA+LTDR: No

*Q2: The image shows a python code. Is the output of the code 'x is smaller than 10'?*

MoE-LLaVA: No

MoE-LLaVA+LTDR: Yes

| Country     | Sales Volume | Revenue      | Profit       | Profit Margin |
|-------------|--------------|--------------|--------------|---------------|
| USA         | 40.080       | \$15.971.880 | \$3.086.421  | 19,3%         |
| China       | 35.070       | \$15.866.670 | \$3.032.162  | 19,1%         |
| Australia   | 27.054       | \$14.812.566 | \$2.868.636  | 19,4%         |
| India       | 23.046       | \$10.608.174 | \$1.853.710  | 17,5%         |
| South Korea | 16.032       | \$10.494.948 | \$1.975.844  | 18,8%         |
| Total / Avg | 141.282      | \$67.754.238 | \$12.816.772 | 18,8%         |

*Q1: Which country has a below-average profit margin?*

MoE-LLaVA: South Korea MoE-LLaVA+LTDR: India

*Q2: Which country has the highest profit margin?*

MoE-LLaVA: USA

MoE-LLaVA+LTDR: Australia

![](_page_14_Picture_30.jpeg)

*Q: Which of these organisms contains matter that was once part of the phytoplankton?* 

MoE-LLaVA: Sea Otter

MoE-LLaVA+LTDR: Black Rockfish

![](_page_14_Picture_34.jpeg)

*Q1: What is the name of this chapter?*

MoE-LLaVA: Harry Potte

MoE-LLaVA+LTDR: King's Cross

### REFERENCES

- [1] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat *et al.*, "Gpt-4 technical report," *arXiv preprint arXiv:2303.08774*, 2023.
- [2] A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Yang, A. Fan *et al.*, "The llama 3 herd of models," *arXiv e-prints*, pp. arXiv–2407, 2024.
- [3] Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, S. Xing, M. Zhong, Q. Zhang, X. Zhu, L. Lu *et al.*, "Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks," in *Proc. CVPR*, 2024, pp. 24 185–24 198.
- [4] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou, and J. Zhou, "Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond," *arXiv preprint arXiv:2308.12966*, 2023.
- [5] H. Lu, W. Liu, B. Zhang, B. Wang, K. Dong, B. Liu, J. Sun, T. Ren, Z. Li, H. Yang *et al.*, "Deepseek-vl: towards real-world vision-language understanding," *arXiv preprint arXiv:2403.05525*, 2024.
- [6] R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton, "Adaptive mixtures of local experts," *Neural Comput.*, vol. 3, no. 1, pp. 79–87, 1991.
- [7] S. Dou, E. Zhou, Y. Liu, S. Gao, W. Shen, L. Xiong, Y. Zhou, X. Wang, Z. Xi, X. Fan *et al.*, "Loramoe: Alleviating world knowledge forgetting in large language models via moe-style plugin," in *Proc. ACL*, 2024, pp. 1932–1945.
- [8] Y. Gou, Z. Liu, K. Chen, L. Hong, H. Xu, A. Li, D.-Y. Yeung, J. T. Kwok, and Y. Zhang, "Mixture of cluster-conditional lora experts for vision-language instruction tuning," *arXiv preprint arXiv:2312.12379*, 2023.
- [9] J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang *et al.*, "Qwen technical report," *arXiv preprint arXiv:2309.16609*, 2023.
- [10] S. Chen, Z. Jie, and L. Ma, "Llava-mole: Sparse mixture of lora experts for mitigating data conflicts in instruction finetuning mllms," *arXiv preprint arXiv:2401.16160*, 2024.
- [11] D. Dai, C. Deng, C. Zhao, R. Xu, H. Gao, D. Chen, J. Li, W. Zeng, X. Yu, Y. Wu *et al.*, "Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models," in *Proc. ACL*, 2024, pp. 1280–1297.
- [12] B. Lin, Z. Tang, Y. Ye, J. Huang, J. Zhang, Y. Pang, P. Jin, M. Ning, J. Luo, and L. Yuan, "Moe-llava: Mixture of experts for large visionlanguage models," *IEEE Trans. Multimedia.*, 2026.
- [13] H. Liu, C. Li, Y. Li, and Y. J. Lee, "Improved baselines with visual instruction tuning," in *Proc. CVPR*, 2024, pp. 26 296–26 306.
- [14] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, "Outrageously large neural networks: The sparsely-gated mixture-of-experts layer," *arXiv preprint arXiv:1701.06538*, 2017.
- [15] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. CVPR*, 2016, pp. 770–778.
- [16] A. Krizhevsky, I. Sutskever, and G. E. Hinton, "Imagenet classification with deep convolutional neural networks," *Proc. NeurIPS*, vol. 25, 2012.
- [17] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," *Proc. NeurIPS*, vol. 30, 2017.
- [18] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "Bert: Pre-training of deep bidirectional transformers for language understanding," in *Proc. NAACL*, 2019, pp. 4171–4186.
- [19] Y. Zhang, R. Zhang, J. Gu, Y. Zhou, N. Lipka, D. Yang, and T. Sun, "Llavar: Enhanced visual instruction tuning for text-rich image understanding," *arXiv preprint arXiv:2306.17107*, 2023.
- [20] N. Houlsby, A. Giurgiu, S. Jastrzebski, B. Morrone, Q. De Laroussilhe, A. Gesmundo, M. Attariyan, and S. Gelly, "Parameter-efficient transfer learning for nlp," in *Proc. ICML*. PMLR, 2019, pp. 2790–2799.
- [21] B. Lester, R. Al-Rfou, and N. Constant, "The power of scale for parameter-efficient prompt tuning," in *Proc. EMNLP*, 2021.
- [22] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen *et al.*, "Lora: Low-rank adaptation of large language models." *Proc. ICLR*, vol. 1, no. 2, p. 3, 2022.
- [23] Q. Ye, H. Xu, G. Xu, J. Ye, M. Yan, Y. Zhou, J. Wang, A. Hu, P. Shi, Y. Shi *et al.*, "mplug-owl: Modularization empowers large language models with multimodality," *arXiv preprint arXiv:2304.14178*, 2023.
- [24] D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun, N. Shazeer, and Z. Chen, "Gshard: Scaling giant models with conditional computation and automatic sharding," *arXiv preprint arXiv:2006.16668*, 2020.

- [25] S. Gururangan, M. Lewis, A. Holtzman, N. A. Smith, and L. Zettlemoyer, "Demix layers: Disentangling domains for modular language modeling," in *Proc. NAACL*, 2022.
- [26] Y. Jain, H. Behl, Z. Kira, and V. Vineet, "Damex: Dataset-aware mixture-of-experts for visual understanding of mixture-of-datasets," *Proc. NeurIPS*, vol. 36, 2024.
- [27] Y. Zhou, Z. Zhao, S. Du, H. Li, J. Yao, Y. Zhang, and Y. Wang, "Exploring training on heterogeneous data with mixture of low-rank adapters," in *Proc. ICML*, 2024, pp. 62 294–62 306.
- [28] F. Xu, D. Chen, T. Jia, S. Deng, and H. Wang, "Cbdmoe: Consistentbut-diverse mixture of experts for domain generalization," *IEEE Trans. Multimedia.*, vol. 26, pp. 9814–9824, 2024.
- [29] Q. Huang, Z. An, N. Zhuang, M. Tao, C. Zhang, Y. Jin, K. Xu, L. Chen, S. Huang, and Y. Feng, "Harder task needs more experts: Dynamic routing in moe models," in *Proc. ACL*, 2024, pp. 12 883–12 895.
- [30] Y. Guo, Z. Cheng, X. Tang, Z. Tu, and T. Lin, "Dynamic mixture of experts: An auto-tuning approach for efficient transformer models," in *Proc. ICLR*, 2025.
- [31] J. Ma, Z. Zhao, X. Yi, J. Chen, L. Hong, and E. H. Chi, "Modeling task relationships in multi-task learning with multi-gate mixture-of-experts," in *Proc. KDD*, 2018, pp. 1930–1939.
- [32] L. Yang, D. Shen, C. Cai, F. Yang, T. Gao, D. ZHANG, and X. Li, "Solving token gradient conflict in mixture-of-experts for large visionlanguage model," in *Proc. ICLR*, 2025.
- [33] Z. Zhong, M. Xia, D. Chen, and M. Lewis, "Lory: Fully differentiable mixture-of-experts for autoregressive language model pre-training," *arXiv preprint arXiv:2405.03133*, 2024.
- [34] X. Wang, L. Lian, Z. Miao, Z. Liu, and S. Yu, "Long-tailed recognition by routing diverse distribution-aware experts," in *Proc. ICLR*, 2021.
- [35] Y. Jin, M. Li, Y. Lu, Y.-m. Cheung, and H. Wang, "Long-tailed visual recognition via self-heterogeneous integration with knowledge excavation," in *Proc. CVPR*, 2023, pp. 23 695–23 704.
- [36] Z. Liu, Z. Miao, X. Zhan, J. Wang, B. Gong, and S. X. Yu, "Largescale long-tailed recognition in an open world," in *Proc. CVPR*, 2019, pp. 2537–2546.
- [37] Y. Cui, M. Jia, T.-Y. Lin, Y. Song, and S. Belongie, "Class-balanced loss based on effective number of samples," in *Proc. CVPR*, 2019, pp. 9268–9277.
- [38] B. Kang, Y. Li, S. Xie, Z. Yuan, and J. Feng, "Exploring balanced feature spaces for representation learning," in *Proc. ICLR*, 2020.
- [39] A. K. Menon, S. Jayasumana, A. S. Rawat, H. Jain, A. Veit, and S. Kumar, "Long-tail learning via logit adjustment," in *Proc. ICLR*, 2021.
- [40] M. Song, X. Qu, J. Zhou, and Y. Cheng, "From head to tail: Towards balanced representation in large vision-language models through adaptive data calibration," in *Proc. CVPR*, 2025, pp. 9434–9444.
- [41] L. Jin, Z. Lu, Z. Li, Y. Pan, L. Dai, J. Tang, and R. Jain, "Causal inference hashing for long-tailed image retrieval," *IEEE Trans. Image Process.*, 2025.
- [42] W. Zhao, W. Li, Y. Li, L. Yang, Z. Liang, E. Hu, W. Zhang, and H. Yang, "Constructing balanced training samples: A new perspective on longtailed classification," *IEEE Trans. Multimedia.*, vol. 27, pp. 5130–5143, 2025.
- [43] H. Han, W.-Y. Wang, and B.-H. Mao, "Borderline-smote: a new oversampling method in imbalanced data sets learning," in *Proc. ICIC*. Springer, 2005, pp. 878–887.
- [44] M. Buda, A. Maki, and M. A. Mazurowski, "A systematic study of the class imbalance problem in convolutional neural networks," *Neural Netw.*, vol. 106, pp. 249–259, 2018.
- [45] B. Zhou, Q. Cui, X.-S. Wei, and Z.-M. Chen, "Bbn: Bilateral-branch network with cumulative learning for long-tailed visual recognition," in *Proc. CVPR*, 2020, pp. 9719–9728.
- [46] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," in *Proc. ICCV*, 2017, pp. 2980–2988.
- [47] Y.-X. Wang, D. Ramanan, and M. Hebert, "Learning to model the tail," *Proc. NeurIPS*, vol. 30, 2017.
- [48] K. Cao, C. Wei, A. Gaidon, N. Arechiga, and T. Ma, "Learning imbalanced datasets with label-distribution-aware margin loss," *Proc. NeurIPS*, vol. 32, 2019.
- [49] J. Tan, C. Wang, B. Li, Q. Li, W. Ouyang, C. Yin, and J. Yan, "Equalization loss for long-tailed object recognition," in *Proc. CVPR*, 2020, pp. 11 662–11 671.
- [50] J. Ren, C. Yu, X. Ma, H. Zhao, S. Yi *et al.*, "Balanced meta-softmax for long-tailed visual recognition," *Proc. NeurIPS*, vol. 33, pp. 4175–4186, 2020.
- [51] Y. Hong, S. Han, K. Choi, S. Seo, B. Kim, and B. Chang, "Disentangling label distribution for long-tailed visual recognition," in *Proc. CVPR*, 2021, pp. 6626–6636.

- [52] W. Fedus, B. Zoph, and N. Shazeer, "Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity," *J. Mach. Learn. Res.*, vol. 23, no. 120, pp. 1–39, 2022.
- [53] H. Nguyen, X. Han, C. W. Harris, S. Saria, and N. Ho, "On expert estimation in hierarchical mixture of experts: Beyond softmax gating functions," *arXiv preprint arXiv:2410.02935*, 2024.
- [54] J. Chen, L. Guo, J. Sun, S. Shao, Z. Yuan, L. Lin, and D. Zhang, "Eve: efficient vision-language pre-training with masked prediction and modality-aware moe," in *Proc. AAAI*, vol. 38, 2024, pp. 1110–1119.
- [55] X. V. Lin, A. Shrivastava, L. Luo, S. Iyer, M. Lewis, G. Ghosh, L. Zettlemoyer, and A. Aghajanyan, "Moma: Efficient early-fusion pre-training with mixture of modality-aware experts," *arXiv preprint arXiv:2407.21770*, 2024.
- [56] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark *et al.*, "Learning transferable visual models from natural language supervision," in *Proc. ICML*. PmLR, 2021, pp. 8748–8763.
- [57] W. Kim, B. Son, and I. Kim, "Vilt: Vision-and-language transformer without convolution or region supervision," in *Proc. ICML*. PMLR, 2021, pp. 5583–5594.
- [58] H. Laurençon, L. Saulnier, L. Tronchon, S. Bekman, A. Singh, A. Lozhkov, T. Wang, S. Karamcheti, A. Rush, D. Kiela *et al.*, "Obelics: An open web-scale filtered dataset of interleaved image-text documents," *Proc. NeurIPS*, vol. 36, pp. 71 683–71 702, 2023.
- [59] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar *et al.*, "Llama: Open and efficient foundation language models," *arXiv preprint arXiv:2302.13971*, 2023.
- [60] W.-L. Chiang, Z. Li, Z. Lin, Y. Sheng, Z. Wu, H. Zhang, L. Zheng, S. Zhuang, Y. Zhuang, J. E. Gonzalez *et al.*, "Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality," *See https://vicuna. lmsys. org (accessed 14 April 2023)*, vol. 2, no. 3, p. 6, 2023.
- [61] Z. Yuan, Z. Li, W. Huang, Y. Ye, and L. Sun, "Tinygpt-v: Efficient multimodal large language model via small backbones," *arXiv preprint arXiv:2312.16862*, 2023.
- [62] M. Javaheripi, S. Bubeck, M. Abdin, J. Aneja, S. Bubeck, C. C. T. Mendes, W. Chen, A. Del Giorno, R. Eldan, S. Gopi *et al.*, "Phi-2: The surprising power of small language models," *Microsoft Research Blog*, vol. 1, no. 3, p. 3, 2023.
- [63] X. Chu, L. Qiao, X. Lin, S. Xu, Y. Yang, Y. Hu, F. Wei, X. Zhang, B. Zhang, X. Wei *et al.*, "Mobilevlm: A fast, reproducible and strong vision language assistant for mobile devices," *arXiv preprint arXiv:2312.16886*, vol. 1, no. 2, p. 3, 2023.
- [64] Y. Zhu, M. Zhu, N. Liu, Z. Xu, and Y. Peng, "Llava-phi: Efficient multimodal assistant with small language model," in *Proc. EMCLR*, 2024, pp. 18–22.
- [65] M. Bellagente, J. Tow, D. Mahan, D. Phung, M. Zhuravinskyi, R. Adithyan, J. Baicoianu, B. Brooks, N. Cooper, A. Datta *et al.*, "Stable lm 2 1.6 b technical report," *arXiv preprint arXiv:2402.17834*, 2024.
- [66] M. Deitke, C. Clark, S. Lee, R. Tripathi, Y. Yang, J. S. Park, M. Salehi, N. Muennighoff, K. Lo, L. Soldaini *et al.*, "Molmo and pixmo: Open weights and open data for state-of-the-art vision-language models," in *Proc. CVPR*, 2025, pp. 91–104.
- [67] B. Li, Y. Shen, J. Yang, Y. Wang, J. Ren, T. Che, J. Zhang, and Z. Liu, "Sparse mixture-of-experts are domain generalizable learners," in *Proc. ICLR*, 2023.
- [68] Z. Chen, Z. Wang, Z. Wang, H. Liu, Z. Yin, S. Liu, L. Sheng, W. Ouyang, and J. Shao, "Octavius: Mitigating task interference in mllms via lora-moe," in *Proc. ICLR*, 2024.
- [69] A. Suhr, S. Zhou, A. Zhang, I. Zhang, H. Bai, and Y. Artzi, "A corpus for reasoning about natural language grounded in photographs," in *Proc. ACL*, 2019, pp. 6418–6428.
- [70] L. Chen and L. Xing, "Open-llava-next: An open-source implementation of llava-next series for facilitating the large multi-modal model community," *GitHub-xiaoachen98/Open-LLaVA-NeXT: AnopensourceimplementationfortrainingLLaVA-NeXT*, 2024.
- [71] Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, and D. Parikh, "Making the v in vqa matter: Elevating the role of image understanding in visual question answering," in *Proc. CVPR*, 2017, pp. 6904–6913.
- [72] D. Li, Y. Yang, Y.-Z. Song, and T. M. Hospedales, "Deeper, broader and artier domain generalization," in *Proc. ICCV*, 2017, pp. 5542–5550.
- [73] D. A. Hudson and C. D. Manning, "Gqa: A new dataset for real-world visual reasoning and compositional question answering," in *Proc. CVPR*, 2019, pp. 6700–6709.

- [74] A. Masry, X. L. Do, J. Q. Tan, S. Joty, and E. Hoque, "Chartqa: A benchmark for question answering about charts with visual and logical reasoning," in *Findings of the association for computational linguistics: ACL 2022*, 2022, pp. 2263–2279.
- [75] ShareAI Lab, "Sharegpt-chinese-english-90k: A bilingual chineseenglish human-machine dialogue dataset," 2023. [Online]. Available: https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k
- [76] A. Singh, V. Natarajan, M. Shah, Y. Jiang, X. Chen, D. Batra, D. Parikh, and M. Rohrbach, "Towards vqa models that can read," in *Proc. CVPR*, 2019, pp. 8317–8326.
- [77] I. Albuquerque, J. Monteiro, M. Darvishi, T. H. Falk, and I. Mitliagkas, "Generalizing to unseen domains via distribution matching," *arXiv preprint arXiv:1911.00804*, 2019.
- [78] P. Lu, S. Mishra, T. Xia, L. Qiu, K.-W. Chang, S.-C. Zhu, O. Tafjord, P. Clark, and A. Kalyan, "Learn to explain: Multimodal reasoning via thought chains for science question answering," *Proc. NeurIPS*, vol. 35, pp. 2507–2521, 2022.
- [79] M. Mathew, D. Karatzas, and C. Jawahar, "Docvqa: A dataset for vqa on document images," in *Proc. WACV*, 2021, pp. 2200–2209.
- [80] K. Marino, M. Rastegari, A. Farhadi, and R. Mottaghi, "Ok-vqa: A visual question answering benchmark requiring external knowledge," in *Proc. CVPR*, 2019, pp. 3195–3204.
- [81] H. Venkateswara, J. Eusebio, S. Chakraborty, and S. Panchanathan, "Deep hashing network for unsupervised domain adaptation," in *Proc. CVPR*, 2017, pp. 5018–5027.
- [82] A. Kembhavi, M. Salvato, E. Kolve, M. Seo, H. Hajishirzi, and A. Farhadi, "A diagram is worth a dozen images," in *Proc. ECCV*. Springer, 2016, pp. 235–251.
- [83] Y. Li, Y. Du, K. Zhou, J. Wang, W. X. Zhao, and J.-R. Wen, "Evaluating object hallucination in large vision-language models," in *Proc. EMNLP*, 2023, pp. 292–305.
- [84] C. Fu, P. Chen, Y. Shen, Y. Qin, M. Zhang, X. Lin, J. Yang, X. Zheng, K. Li, X. Sun *et al.*, "Mme: A comprehensive evaluation benchmark for multimodal large language models," in *Proc. NeurIPS*, 2025.
- [85] W. Li, W. Bishop, A. Li, C. Rawles, F. Campbell-Ajala, D. Tyamagundlu, and O. Riva, "On the effects of data scale on ui control agents," *Proc. NeurIPS*, vol. 37, pp. 92 130–92 154, 2024.
- [86] A. Mishra, S. Shekhar, A. K. Singh, and A. Chakraborty, "Ocr-vqa: Visual question answering by reading text in images," in *Proc. ICDAR*. IEEE, 2019, pp. 947–952.
- [87] M. Mathew, V. Bagal, R. Tito, D. Karatzas, E. Valveny, and C. Jawahar, "Infographicvqa," in *Proc. WACV*, 2022, pp. 1697–1706.
- [88] Y. Liu, H. Duan, Y. Zhang, B. Li, S. Zhang, W. Zhao, Y. Yuan, J. Wang, C. He, Z. Liu *et al.*, "Mmbench: Is your multi-modal model an all-around player?" in *Proc. ECCV*. Springer, 2024, pp. 216–233.
- [89] L. Beyer, A. Steiner, A. S. Pinto, A. Kolesnikov, X. Wang, D. Salz, M. Neumann, I. Alabdulmohsin, M. Tschannen, E. Bugliarello *et al.*, "Paligemma: A versatile 3b vlm for transfer," *arXiv preprint arXiv:2407.07726*, 2024.
- [90] D. Schwenk, A. Khandelwal, C. Clark, K. Marino, and R. Mottaghi, "A-okvqa: A benchmark for visual question answering using world knowledge," in *Proc. ECCV*. Springer, 2022, pp. 146–162.
- [91] W. Yu, Z. Yang, L. Li, J. Wang, K. Lin, Z. Liu, X. Wang, and L. Wang, "Mm-vet: evaluating large multimodal models for integrated capabilities," in *Proc. ICML*, 2024, pp. 57 730–57 754.
- [92] O. Sidorov, R. Hu, M. Rohrbach, and A. Singh, "Textcaps: a dataset for image captioning with reading comprehension," in *Proc. ECCV*, 2020, pp. 742–758.
- [93] S. Kazemzadeh, V. Ordonez, M. Matten, and T. Berg, "Referitgame: Referring to objects in photographs of natural scenes," in *Proc. EMNLP*, 2014, pp. 787–798.
- [94] R. Krishna, Y. Zhu, O. Groth, J. Johnson, K. Hata, J. Kravitz, S. Chen, Y. Kalantidis, L.-J. Li, D. A. Shamma *et al.*, "Visual genome: Connecting language and vision using crowdsourced dense image annotations," *Int. J. Comput. Vis.*, vol. 123, no. 1, pp. 32–73, 2017.
- [95] P. Lu, L. Qiu, K.-W. Chang, Y. N. Wu, S.-C. Zhu, T. Rajpurohit, P. Clark, and A. Kalyan, "Dynamic prompt learning via policy gradient for semistructured mathematical reasoning," in *Proc. ICLR*, 2023.
- [96] A. F. Biten, R. Tito, A. Mafla, L. Gomez, M. Rusinol, E. Valveny, C. Jawahar, and D. Karatzas, "Scene text visual question answering," in *Proc. ICCV*, 2019, pp. 4291–4301.
- [97] M. Acharya, K. Kafle, and C. Kanan, "Tallyqa: Answering complex counting questions," in *Proc. AAAI*, vol. 33, no. 01, 2019, pp. 8076– 8084.