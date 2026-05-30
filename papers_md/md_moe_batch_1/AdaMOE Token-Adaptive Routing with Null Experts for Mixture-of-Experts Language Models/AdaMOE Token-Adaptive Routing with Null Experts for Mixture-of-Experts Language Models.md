# <span id="page-0-0"></span>A*da*MOE: Token-Adaptive Routing with Null Experts for Mixture-of-Experts Language Models

Zihao Zeng<sup>1</sup>[\\*](#page-0-0), Yibo Miao<sup>1</sup>\*, Hongcheng Gao<sup>2</sup> , Hao Zhang<sup>3</sup> , Zhijie Deng<sup>1</sup>[†](#page-0-0) <sup>1</sup>Qing Yuan Research Institute, SEIEE, Shanghai Jiao Tong University <sup>2</sup>University of Chinese Academy of Sciences <sup>3</sup>University of California, San Diego {zengzihao, miaoyibo, zhijied}@sjtu.edu.cn gaohongcheng23@mails.ucas.ac.cn, haozhang@ucsd.edu

# Abstract

Mixture of experts (MoE) has become the standard for constructing production-level large language models (LLMs) due to its promise to boost model capacity without causing significant overheads. Nevertheless, existing MoE methods usually enforce a constant top-k routing for all tokens, which is arguably restrictive because various tokens (e.g., "<EOS>" vs. "apple") may require various numbers of experts for feature abstraction. Lifting such a constraint can help make the most of limited resources and unleash the potential of the model for downstream tasks. In this sense, we introduce A*da*MO<sup>E</sup> to realize token-adaptive routing for MoE, where different tokens are permitted to select a various number of experts. A*da*MO<sup>E</sup> makes minimal modifications to the vanilla MoE with top-k routing—it simply introduces a fixed number of *null experts*, which do not consume any FLOPs, to the expert set and increases the value of k. A*da*MO<sup>E</sup> does not force each token to occupy a fixed number of null experts but ensures the average usage of the null experts with a load-balancing loss, leading to an adaptive number of null/true experts used by each token. A*da*MO<sup>E</sup> exhibits a strong resemblance to MoEs with expert choice routing while allowing for trivial auto-regressive modeling. A*da*MO<sup>E</sup> is easy to implement and can be effectively applied to pre-trained (MoE- )LLMs. Extensive studies show that A*da*MO<sup>E</sup> can reduce average expert load (FLOPs) while achieving superior performance. For example, on the ARC-C dataset, applying our method to fine-tuning Mixtral-8x7B can reduce FLOPs by 14.5% while increasing accuracy by 1.69%. Code is available at [this link.](https://github.com/CengZihao/AdaMoE)

# 1 Introduction

Large language models (LLMs) have exhibited exceptional performance across diverse tasks and do-

mains [\(Touvron et al.,](#page-9-0) [2023;](#page-9-0) [Chiang et al.,](#page-8-0) [2023;](#page-8-0) [Chowdhery et al.,](#page-8-1) [2023;](#page-8-1) [Zhang et al.,](#page-9-1) [2022\)](#page-9-1). Nevertheless, LLMs' efficacy is heavily impacted by the substantial number of parameters they possess, with some high-performing LLMs containing up to 540B parameters [\(Chowdhery et al.,](#page-8-1) [2023\)](#page-8-1). The mixture of experts (MoE) mechanism [\(Shazeer](#page-9-2) [et al.,](#page-9-2) [2017\)](#page-9-2) offers a compelling way to enhance model capability without a corresponding increase in computational overhead. Recent research further underscores the merits of MoE, vividly demonstrating its potential to support production-level applications [\(Jiang et al.,](#page-8-2) [2024;](#page-8-2) [Qwen,](#page-9-3) [2024\)](#page-9-3).

MoE operates on the core assumption that a (small) subset of experts is sufficient to handle a single token effectively. MoE-LLMs, with Mixtral [\(Jiang et al.,](#page-8-2) [2024\)](#page-8-2) and DeepSeekMoE [\(Dai](#page-8-3) [et al.,](#page-8-3) [2024\)](#page-8-3) as popular examples, often replace the feed-forward network (FFN) in the model with a set of FFN experts. A token-level router is introduced to sparsely activate the experts for various tokens, so the computational cost is constrained to a low level. We can also build experts with parameterefficient fine-tuning (PEFT) modules [\(Hu et al.,](#page-8-4) [2021;](#page-8-4) [Liu et al.,](#page-9-4) [2022\)](#page-9-4) like LoRA, giving rise to Mo-LoRA approaches [\(Zadouri et al.,](#page-9-5) [2023\)](#page-9-5).

MoE routinely routes each token to a fixed amount of experts, typically the k ones with top routing probabilities. However, not all tokens require the same number of experts for feature abstraction. Intuitively, the semantic tokens deserve a higher concentration of experts, while others with less significant meaning can be processed more swiftly. Lifting the top-k routing constraint can help make the most of limited resources and unleash the potential of the model. To achieve this, MoE with expert choice routing [\(Zhou et al.,](#page-9-6) [2022\)](#page-9-6) performs expert-level routing, where each expert chooses a fixed number of tokens for processing and different tokens could be processed by different numbers of experts. Yet, an unacceptable drawback

<sup>\*</sup>Equal contribution.

<sup>†</sup>Corresponding authors.

<span id="page-1-0"></span><rs> Tracy didn't go home that evening and resisted Riley's attacks.\nQuestion: What does Tracy need to do before this?\nA. make a new plan\nB. Go home and see Riley\nC. Find somewhere to go\nAnswer:

![](_page_1_Figure_1.jpeg)

Figure 1: The number of selected experts for various tokens in an  $\mathcal{A}da$ MoE variant of Mixtral-8x7b. As shown, after applying  $\mathcal{A}da$ MoE, the model possesses the ability to perform token-adaptive routing. Also note that some tokens only require 1 expert for feature abstraction, which offers the opportunity for inference acceleration.

is that it is not suited to casual language modeling due to the reliance on future tokens for the top-*k* token selection (Zhou et al., 2022).

This work introduces  $\mathcal{A}da$ MoE, a novel method designed to achieve token-level adaptive routing in MoE, allowing different tokens to select varying numbers of experts. An illustrative example is presented in Figure 1. AdaMoE requires minimal changes to the vanilla MoE with top-k routing by incorporating a fixed number of null experts into the expert set. These null experts do not consume any computational resources. By increasing the value of k, more experts can be activated. To encourage the average usage rate of the null experts, AdaMoE minimizes a load balancing loss. This leads to an adaptive number of null experts and true experts being employed by each token. Notably, AdaMoE shares similarities with existing MoE with expert choice routing while also enabling straightforward causal language modeling.

AdaMoE is easy to implement and can be applied to both pre-trained regular LLMs and MoE-LLMs for supervised fine-tuning. For the former, we experiment on Llama2-7B (Touvron et al., 2023) by introducing LoRA experts and corresponding routers to the model. For the latter, we experiment on Mixtral-8x7B (Jiang et al., 2024) by augmenting the original router with extra weights for the null experts. The results underscore the effectiveness of AdaMoE's token-adaptive mechanism in enhancing both computational efficiency and model performance. For example, when fine-tuning Llama2-7B, AdaMoE achieves much higher accuracy across almost all evaluated datasets. Moreover, when fine-tuning Mixtral-8x7B with

AdaMoE on ARC-Challenge (Clark et al., 2018), we observed a 14.5% reduction in total FLOPs, accompanied by a 1.69% increase in accuracy.

#### <span id="page-1-1"></span>2 Related Works

#### 2.1 Mixture of Experts

Mixture of Experts (MoE) (Jacobs et al., 1991; Shazeer et al., 2017) is an efficient scaling technique that allows for larger model sizes with less computation, resulting in enhanced performance. MoE models can be trained and used for inference more efficiently compared to dense models, requiring substantially fewer computational resources. Due to these advantages, pioneering works (Jiang et al., 2024; Dai et al., 2024) have applied MoE to transformer-based language models and demonstrated their superiority. Typically, they replace the feed-forward network (FFN) in each layer of the model with a routing function and multiple FFNs, referred to as experts, with only a subset of these experts being activated at any time. We refer to these models, which combine MoE and large language models (LLMs), as MoE-LLMs.

In addition to MoE-LLMs, fine-tuning techniques have also seen significant advancements. Pre-trained LLMs are often fine-tuned for downstream tasks. However, as models increase in size, full fine-tuning becomes increasingly computationally expensive (Brown et al., 2020; Chang et al., 2024). LoRA (Hu et al., 2021) addresses this challenge by providing an effective fine-tuning methodology for scenarios with constrained computational resources. LoRA freezes model weights and injects trainable rank decomposition matrices, thereby modifying the behavior of dense linear layers with-

<span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

Figure 2: Comparison of Routing Mechanisms: vanilla MoE v.s.  $\mathcal{A}da$ MoE. Left: In vanilla MoE, each token selects the top 2 experts based on the routing probabilities. Right:  $\mathcal{A}da$ MoE introduces an additional set of *null experts* and makes each token select the top 4 experts, which can include both the true and null experts. For example, token 1 selects three true experts, while token 2 selects only one true expert. Despite this variation, the average number of true experts selected per token remains two, maintaining parity with the vanilla method.

out substantially changing the original model parameters (Lester et al., 2021; An et al., 2022). Recent studies (Zadouri et al., 2023; Liu et al., 2023; Dou et al., 2023) convincingly show that integrating LoRA with MoE offers a promising approach for achieving high performance with minimal parameter updates. Methods like MixLoRA (Li et al., 2024), MoLE (Wu et al., 2023), and Lo-RAMoE (Dou et al., 2023) combine MoE with LoRA by learning multiple pairs of low-rank matrices, known as LoRA experts, and use a router to compute the probabilities of each expert for the inputs. MoLA (Gao et al., 2024) explores the relationship between the number of LoRA experts and the depth of model layers. For consistency and convenience, we will refer to these methods collectively as Mo-LoRA in the following text.

#### 2.2 Routing Strategies

The early MoE architecture utilized gate units as the router to select experts for each token (Shazeer et al., 2017; Lepikhin et al., 2020). Following the success of the Switch Transformer (Fedus et al., 2022) in large-scale pre-training, MoE received increased attention, leading to the development of more advanced routing algorithms. For example, BASE Layers (Lewis et al., 2021) use a linear assignment problem to maximize token-expert affinities while ensuring each expert receives an equal number of tokens. Hash layers (Roller et al., 2021) employ hashing techniques on input tokens to allocate different sets of weights. A different approach, Expert-Choice Routing (Zhou et al., 2022), allows experts to select their preferred tokens, achiev-

ing a more balanced expert load and better costeffectiveness. Furthermore, DeepMind's Mixtureof-Depths (MoD) (Raposo et al., 2024) introduces a router to determine the necessity of computation for each input token at each layer.

#### 3 Method

In this section, we introduce  $\mathcal{A}da$ MoE, which incorporates null experts to allow for more flexible and efficient expert selection for various tokens. An illustrative comparison between vanilla MoE and  $\mathcal{A}da$ MoE is presented in Figure 2.

#### 3.1 Preliminary on MoE

MoE has been widely applied in two scenarios for large language models: MoE-LLMs (Jiang et al., 2024; Dai et al., 2024) and Mo-LoRA (Dou et al., 2023; Gao et al., 2024; Li et al., 2024) , as briefly introduced in Section 2. The core component of both is the MoE layer, which consists of n specialized experts  $E_i: \mathbb{R}^{d_{in}} \to \mathbb{R}^{d_{out}}, i=1,\ldots,n$  and a router  $G: \mathbb{R}^{d_{in}} \to \mathbb{R}^n$ . The experts often have the same parameterization, such as feed-forward neural networks (FFNs) in MoE-LLMs or LoRA modules in Mo-LoRA. The router usually activates the k (k < n) experts with the highest routing probabilities (i.e., the top-k experts) and distributes input tokens to corresponding experts.

Given an input token  $x \in \mathbb{R}^{d_{in}}$ , the routing process works as:

$$G(x) \in \mathbb{R}^n := \operatorname{Softmax} \left( \operatorname{TopK}(x \cdot W_g, k) \right) , (1)$$

where  $W_g \in \mathbb{R}^{d_{in} \times n}$  is the parameter matrix of the router, and  $\text{TopK}(\cdot, k)$  retains only the top-k ele-

ments, setting the rest to  $-\infty$  (so that after Softmax, the corresponding routing probabilities are zero). The output of the MoE layer is then computed as:

$$y \in \mathbb{R}^{d_{out}} := \sum_{i=1}^{n} G(x)_i \cdot E_i(x) . \tag{2}$$

Additionally, an auxiliary loss is applied during the training stage to encourage a balanced load across experts within the same MoE layer (Fedus et al., 2022). Given a batch  $\mathcal{B}$  of tokens, this load balancing loss for a MoE layer is defined as:

<span id="page-3-1"></span>
$$\ell_{load} := \alpha \cdot n \cdot \sum_{i=1}^{n} f_i \cdot P_i , \qquad (3)$$

where  $\alpha$  is a hyperparameter, and  $f_i$  represents the fraction of tokens dispatched to expert  $E_i$ ,

$$f_i = \frac{1}{|\mathcal{B}|} \sum_{x \in \mathcal{B}} \mathbb{1}\{G(x)_i \neq 0\} ,$$
 (4)

and  $P_i$  denotes the average fraction of the router probability allocated for expert  $E_i$ , i.e.,

$$P_i = \frac{1}{|\mathcal{B}|} \sum_{x \in \mathcal{B}} \text{Softmax} (x \cdot W_g)_i .$$
 (5)

#### 3.2 Drawback of Top-k Router

Almost all traditional MoE methods adopt a top-k routing strategy for expert selection (Fedus et al., 2022; Lepikhin et al., 2020; Jiang et al., 2024). Therefore, each token passes through exactly k experts and occupies the same amount of computation. We first question the rationality of the fixed top-k routing with the following studies.

Concretely, the SocialIQA dataset (Sap et al., 2019) is fed into Mixtral-8x7B (Jiang et al., 2024), which employs a top-2 routing strategy for expert selection. We record the routing distribution for all tokens in each MoE layer of the model. To evaluate the sharpness of the routing distribution, we count the number of top experts whose cumulative routing probabilities exceed 50% and according to this, all tokens can be divided into four categories. The proportions of the tokens are displayed in Figure 3.

As shown, the proportions of tokens within different counts show substantial variation. Namely, the sharpness of the routing distribution varies significantly. A considerable number of tokens have highly uneven routing distributions. Some tokens tend towards a single expert, while a significant proportion of tokens distribute attention to more than

<span id="page-3-0"></span>![](_page_3_Figure_12.jpeg)

Figure 3: Proportions of the number of top experts with cumulative routing probabilities exceeding 50% for tokens in the SocialIQA dataset. Each bar represents the proportion of different counts of tokens at the corresponding MoE layer in Mixtral-8x7B.

2 experts. These observations imply that the traditional fixed top-k routing strategy, which selects the same number of experts for each token, may not be optimal. This is also implied by the argument in MoD (Raposo et al., 2024) that some tokens may not need to pass through all MoE layers.

### 3.3 Null Experts for Token-Adaptive Router

 $\mathcal{A}da$ MoE achieves token-adaptive expert selection by incorporating *null experts*, which are defined as an empty operation requiring zero FLOPs to process the token feature. In the context of LLMs, common operations satisfying this requirement include a constant zero mapping and an identity mapping (we take the zero mappings null expert as the default choice in the following just for simplicity). Consequently, an  $\mathcal{A}da$ MoE layer includes n+m experts, where  $\{E_i\}_{i=1}^n$  are true experts and  $\{E_i\}_{i=n+1}^{n+m}$  are null experts, and a top-k router  $G: \mathbb{R}^d \to \mathbb{R}^{n+m}$ , which functions the same as the vanilla MoE router except for its output dimension.

**Token-level adaptive routing.** The router still performs fixed top-k selection but with k larger than in vanilla MoE. When null experts are chosen, no additional computation occurs due to their definition. Consequently, the number of true experts selected varies for different tokens.

**Prespecified expert load.** We can adjust the number of null experts according to the compute

<span id="page-4-1"></span>![](_page_4_Picture_0.jpeg)

![](_page_4_Picture_1.jpeg)

Figure 4: Left: Adding null experts to Mo-LoRA. Right: Adding null experts to the MoE layer of MoE-LLMs.

budget, and then reinforce the usage of null experts with a load balancing loss (see Section [3.4\)](#page-4-0). This way, the load of true experts (or the overall FLOPs) can be easily adjusted to an appropriate degree.

Autoregressive task suitability. Expert-choice routing [\(Zhou et al.,](#page-9-6) [2022\)](#page-9-6) also allows varying numbers of experts for different tokens but struggles with autoregressive text generation since it requires considering both past and future tokens. In contrast, our token-choice method avoids this issue.

Bypassing MoE layers. MoD [\(Raposo et al.,](#page-9-14) [2024\)](#page-9-14) uses expert-choice routing to let tokens bypass some FFN layers, speeding up inference. Similarly, in A*da*MOE, if all selected experts for a token are null experts, the token effectively bypasses the A*da*MO<sup>E</sup> layer, achieving a similar effect.

### <span id="page-4-0"></span>3.4 More Details

Load balancing loss with null experts. Including null experts in the load balancing loss is necessary to prevent tokens from disproportionately selecting true experts. However, since all null experts are identical in nature, it is unnecessary to balance the load among them. Treating null experts as distinct entities for load balancing can significantly hinder performance, as shown in Table [3.](#page-7-0)

To address this, we modify the load balancing loss in Equation [\(3\)](#page-3-1) as

<span id="page-4-3"></span>
$$\ell_{null} = \alpha \cdot (n+m) \cdot \sum_{i=1}^{n+m} \tilde{f}_i \cdot P_i , \qquad (6)$$

where

$$\tilde{f}_i = \begin{cases} f_i & \text{if } i \leq n \\ \frac{1}{m} \sum\limits_{j=n+1}^{n+m} f_j & \text{if } i > n \end{cases}.$$

By using an average load among the null experts, we make no distinction between them, which can avoid unnecessary constraints on the router.

Load balancing constraints: from tight to loose. In practice, we anneal the weight α of our load balancing loss to chase a better balanceefficiency trade-off. In particular, we first set a larger α to enforce strict load balancing, ensuring tokens do not disproportionately select true experts, leading to a more even load distribution among all experts. In the latter, we use a smaller α to give tokens greater freedom in choosing experts. The empirical efficacy of doing so is verified in Table [5.](#page-7-1)

Normalization of routing probabilities. In vanilla MoE, TopK(x · Wg, k) is normalized using the Softmax activation function. With null experts, we have two options: 1) normalizing over all selected top-k experts, or 2) normalizing over only the true experts within the top-k ones. We choose the latter to ensure that the weighted average output by the A*da*MO<sup>E</sup> layer remains consistent with the scale of that from the vanilla MoE layer.

### <span id="page-4-2"></span>3.5 Compatibility with Vanilla (MoE-)LLMs

A*da*MO<sup>E</sup> is designed to be plug-and-play, able to be seamlessly integrated with pre-trained LLMs and MoE-LLMs, as illustrated in Figure [4.](#page-4-1) Due to resource constraints, we mainly focus on finetuning such models. For fine-tuning regular LLMs with the Mo-LoRA architecture, we need to add a randomly initialized router and multiple LoRA experts to the corresponding module. When applying A*da*MO<sup>E</sup> to MoE-LLMs, the router's output dimensions are expanded to provide corresponding probabilities for null experts. The parameters for the new dimensions can be derived from the original parameter values. This ensures the expanded router balances the load across all experts, including both true and null experts, at the beginning of the finetuning process. For more specific implementation details, see Section [4.2.1.](#page-6-0) To fine-tune A*da*MOE, we need to adjust the router and experts to meet our token-adaptive routing strategy and follow the

<span id="page-5-1"></span>![](_page_5_Figure_0.jpeg)

Figure 5: Performance comparison across five datasets: RTE, COLA, SQA, CQA, and OQA. The baseline is fine-tuned Llama2-7B using the vanilla Mo-LoRA method with top-1/top-2 routing. Acc. represents accuracy, and Load represents the average number of experts used per Mo-LoRA module or  $\mathcal{A}da$ MoE layer.  $\mathcal{A}da$ MoE use different configurations: m5k2 (5 null experts, top-2 selection), m9k4, m7k4 and m5k4. As shown,  $\mathcal{A}da$ MoE achieves higher accuracy across almost all datasets compared to the baseline. The exact accuracy values can be found in Table 7.

detailed modifications outlined in Section 3.4 to achieve adaptive routing.

### 4 Experiments

In this section, we demonstrate the superior performance of  $\mathcal{A}da$ MoE across various benchmarks, particularly in reducing expert load and enhancing task performance through its token-adaptive routing strategy. We first apply our method to regular LLMs with the Mo-LoRA architecture (Section 4.1). Then apply it to traditional MoE-LLMs (Section 4.2). Additionally, We also include extensive ablation studies to provide further insights into our approach's effectiveness.

#### <span id="page-5-0"></span>4.1 Application to Regular LLMs

#### 4.1.1 Experiments Setup

Model and datasets. We select Llama2-7B (Touvron et al., 2023) as our base model due to its strong performance and popularity within the AI community. To validate the effectiveness of our method, we evaluate it on two distinct task types using five widely recognized datasets. The first task focuses on semantic understanding, for which we use two datasets from the renowned GLUE Benchmark (Wang et al., 2018): Recognizing Textual Entailment (RTE) and the Corpus of Linguistic Acceptability (COLA). The second task involves commonsense reasoning and includes the following datasets: ScienceQA (SQA) (Lu et al., 2022), CommonsenseQA (CQA) (Talmor et al., 2018), and OpenBookQA (OQA) (Mihaylov et al., 2018).

Baseline and implementation details. To high-

light our method's significance, we use the typical Mo-LoRA method as the baseline for comparison. For each MoE/AdaMoE layer, we set n = 4 (4 true experts). For the baseline, we set k = 1, 2 for the top-k routing strategy, which are the most common choices. For our AdaMoE, we selected various configurations for k (the number of top-k experts) and m (the number of null experts). We use AdamW (Loshchilov and Hutter, 2017) as the optimizer with a learning rate of 3e-4. The rank of each LoRA expert is set to 8, and the initialization of the LoRA modules follows the original LoRA implementation (Hu et al., 2021). For each LLM layer, we applied LoRA to  $(W_a, W_k, W_v, W_o)$  in the selfattention modules and  $(W_{aate}, W_{down}, W_{up})$  in the MLP modules. We trained on each dataset for 2 epochs, using 3 random seeds, and averaged the results to obtain the final performance metrics.

#### 4.1.2 Experiments Results

The results are shown in Figure 5. We use accuracy as the main metric to evaluate the model's performance  $^1$ . It is evident that  $\mathcal{A}da$ MoE achieves higher accuracy across almost all datasets compared to the traditional baseline. For instance, on the RTE and OQA datasets, all configurations of  $\mathcal{A}da$ MoE surpass the baseline in accuracy. This trend continues across the other datasets, demonstrating the robustness and effectiveness of  $\mathcal{A}da$ MoE in achieving better performance with more adaptive expert utilization.

<span id="page-5-2"></span><sup>&</sup>lt;sup>1</sup>LoRA expert load has minimal impact on total FLOPs; therefore, it is not considered a primary evaluation metric.

<span id="page-6-3"></span>

|                         | Metric  | WINO  | HELLA | PIQA  | SIQA  | OQA   | ARC-C | Avg.  |
|-------------------------|---------|-------|-------|-------|-------|-------|-------|-------|
| Original Mixtral-8x7B   | Acc.    | 55.96 | 53.62 | 68.06 | 64.59 | 65.40 | 83.73 | 65.23 |
| Fine-tuned Mixtral-8x7B | Acc.    | 80.43 | 84.10 | 90.48 | 76.36 | 89.00 | 87.46 | 84.64 |
|                         | Acc.    | 81.93 | 85.50 | 90.32 | 76.97 | 88.20 | 89.15 | 85.35 |
| $\mathcal{A}da$ MoE     | %FLOPs↓ | 14.99 | 14.10 | 18.07 | 16.31 | 13.22 | 14.55 | 15.21 |
|                         | Load    | 1.66  | 1.68  | 1.59  | 1.63  | 1.70  | 1.67  | 1.66  |

Table 1: Comparison of performance and computational efficiency across six datasets: WINO, HELLA, PIQA, SIQA, OQA and ARC-C. Metrics include Acc. (accuracy), %FLOPs↓ (percentage of FLOPs reduction by AdaMoE compared to the baselines), and Load (the average number of experts used per MoE/AdaMoE layer). The baselines are original/fine-tuned Mixtral-8x7B, both using the top-2 routing strategy (Load = 2.00). AdaMoE not only reduces FLOPs but also achieves better accuracy across most datasets compared to the fine-tuned Mixtral-8x7B with LoRA.

#### <span id="page-6-1"></span>4.2 Application to MoE-LLMs

#### <span id="page-6-0"></span>4.2.1 Experiments Setup

Model and datasets. We use Mixtral-8x7B (Jiang et al., 2024) as the base model, where each MoE layer has 8 FFN experts and a top-2 router. We selected six well-known datasets from different categories for our experiments: WinoGrande (WINO) (Sakaguchi et al., 2021) for coreference resolution, Hellaswag (HELLA) (Zellers et al., 2019), PIQA (Bisk et al., 2020), and SIQA (Sap et al., 2019) for commonsense reasoning, Open-BookQA (OQA) (Mihaylov et al., 2018) for reading comprehension, and ARC-Challenge (ARC-C) (Clark et al., 2018) for science examination.

**Baseline and implementation details.** Due to the substantial resources required for pre-training, we focus on fine-tuning. To save memory, we use 4bit quantization and the QLoRA method (Dettmers et al., 2024). The LoRA target modules for the baseline are gate, w1, w2, and w3. For our AdaMoE, we modify this architecture as described in Section 3.5. Specifically, we add null experts to each MoE layer, and the router expands its output dimension to assign probabilities to all experts. To simplify the modification, we define an additional module, gate2, whose parameters can be derived from gate. <sup>2</sup> Together, gate and gate2 form the router that assigns weights to all experts. Thus, the LoRA target modules for our method are gate, gate2, w1, w2, and w3. The rank of the LoRA module is set to 8, and the learning rate is 5e-5. Due to the tendency of MoE-LLMs to overfit during fine-tuning, we use 1000 samples for training on each dataset and train for 2 epochs. In the 2

epochs, we set different values of  $\alpha$  in Equation (6) to  $\alpha_1 = 0.02$ ,  $\alpha_2 = 0.0001$ , as described in Section 3.4. All evaluations are conducted using Open-Compass (Contributors, 2023) to assess accuracy.

### 4.2.2 Experiment Results

In this section, we present the results for the configuration with m=8 and k=3 (i.e., 8 null experts and top-3 expert selection), as shown in Table 1. Additional results are in Section 4.3 and Table 8.

**Accuracy.** AdaMoE outperforms the baseline on WinoGrande, HellaSwag, SIQA, and ARC-Challenge. Although the baseline slightly surpasses AdaMoE on PIQA and OpenBookQA, AdaMoE achieves a higher average accuracy.

**FLOPs.** FFNs account for the majority of the FLOPs during inference. This issue is exacerbated in Mixtral-8x7B, which replaces the FFN with a set of 8 FFNs and selects the top-2 during each inference step. This greatly increases the computational load. *Ada*MoE significantly reduces FLOPs across all datasets, achieving an average reduction of 15.21% compared to the baseline. This demonstrates that *Ada*MoE is more computationally efficient while maintaining competitive performance.

**Load.** The Load metric indicates the average number of experts used per MoE/AdaMoE layer. The baseline method has a Load of 2. In contrast, AdaMoE achieves a lower average Load of 1.66, indicating more efficient utilization of experts.

Overall, the results confirm the effectiveness of the token-adaptive mechanism in improving both computational efficiency and model performance.

#### <span id="page-6-4"></span>4.3 Ablation

In this section, we provide results for various m and k, beyond the single configuration shown in Section 4.2. We also present ablation studies for

<span id="page-6-2"></span><sup>&</sup>lt;sup>2</sup>For instance, if gate2 has an output dimension of 16, meaning there are 16 null experts, the parameters of gate2 can be copied from gate in two segments.

<span id="page-7-2"></span>

|      | Baseline |       | AdaMOE |       |       |       |       |       |       |       |  |
|------|----------|-------|--------|-------|-------|-------|-------|-------|-------|-------|--|
| m, k | 0, 2     | 8, 3  | 16, 4  | 32, 4 | 32, 5 | 32, 6 | 40, 6 | 40, 7 | 40, 8 | 48, 8 |  |
| Acc. | 76.36    | 76.97 | 76.92  | 66.27 | 72.93 | 76.46 | 69.86 | 76.05 | 77.23 | 74.67 |  |
| Load | 2.00     | 1.63  | 1.66   | 0.77  | 1.05  | 1.54  | 1.01  | 1.49  | 1.64  | 1.48  |  |

Table 2: Performance of different m and k combinations on the SIQA dataset. The Baseline represents fine-tuned Mixtral-8x7B using LoRA method, with a Load of 2. Bold values indicate accuracy higher than the baseline.

<span id="page-7-0"></span>

|       | RTE   |      | COLA  |      | SQA   |      | OQA   |      |        | SIQA  |      |
|-------|-------|------|-------|------|-------|------|-------|------|--------|-------|------|
|       | Acc.  | Load | Acc.  | Load | Acc.  | Load | Acc.  | Load | Option | Acc.  | Load |
| ℓbal  | 56.68 | 1.77 | 83.68 | 1.77 | 65.65 | 1.78 | 69.80 | 1.76 | 1)     | 80.19 | 1.50 |
| ℓnull | 67.51 | 1.77 | 85.01 | 1.77 | 66.64 | 1.80 | 71.40 | 1.77 | 2)     | 81.27 | 1.54 |

Table 3: Comparison of accuracy and load on four datasets using load balancing loss with and without balancing among null experts. ℓbal represents the loss with load balancing constraints among null experts, and ℓnull represents the loss without these constraints. Bold values indicate higher accuracy.

Table 4: 1) Normalizing all selected top-k experts, and 2) normalizing only the true experts within the top-k.

A*da*MOE, corresponding to Section [3.4.](#page-4-0) Additional ablation experiments can be found in Appendix [A.](#page-10-0)

More results for Section [4.2.](#page-6-1) We tested different combinations of m and k on the SIQA dataset, with results shown in Table [2.](#page-7-2) Compared to the m = 8, k = 3 configuration in Section [4.2,](#page-6-1) A*da*MO<sup>E</sup> can further reduce the expert load (FLOPs) while maintaining competitive performance. For example, with m = 32, k = 6, the expert load is 1.54 (79.57% of baseline FLOPs), yet accuracy remains higher than the baseline. There are also accuracy differences among configurations with similar loads. For instance, m = 40, k = 7 and m = 48, k = 8 have nearly identical loads but differ in accuracy. This discrepancy highlights areas for further exploration in future research.

Load balancing loss with null experts. To verify the effectiveness of the modified load balancing loss introduced in Equation [\(6\)](#page-4-3), we selected two datasets from each of the semantic understanding and commonsense reasoning tasks. The results, illustrated in Table [3,](#page-7-0) show that lifting the load balancing constraints among null experts significantly improves the performance of the fine-tuned model on the RTE, COLA, SQA, and OQA datasets.

Load balancing constraints: from tight to loose. The effectiveness of the annealing training process described in Section [3.4](#page-4-0) is validated in Table [5.](#page-7-1) The tight load balancing constraints in the first epoch effectively control the expert load in A*da*MOE, meeting our expectations. The loose con-

<span id="page-7-1"></span>

|    |          | WINO  |      | SIQA  |      |
|----|----------|-------|------|-------|------|
|    |          | Acc.  | Load | Acc.  | Load |
| α1 | Baseline | 78.14 | 2.00 | 75.38 | 2.00 |
|    | AdaMOE   | 76.24 | 1.65 | 75.90 | 1.62 |
| α2 | Baseline | 80.43 | 2.00 | 76.36 | 2.00 |
|    | AdaMOE   | 81.93 | 1.66 | 76.97 | 1.63 |

Table 5: Performance for finetuning Mixtral-8x7B with A*da*MO<sup>E</sup> on the WINO and SIQA datasets for two epochs with α<sup>1</sup> = 0.02 and α<sup>2</sup> = 0.0001.

straints in the second epoch allow tokens greater freedom in selecting experts, thereby enhancing performance with almost no increase in expert load. For example, on the WINO dataset, the accuracy increased by 5.69% compared to the result after epoch 1, with almost no increase in expert load.

Normalization of routing probabilities. We tried the two options mentioned in Section [3.4](#page-4-0) on the SIQA dataset, and the results are shown in Table [4.](#page-7-0) As we can see, option 2) is a superior choice, showing a significant improvement in accuracy , with only a minor change in expert load.

# 5 Conclusion

MoE has been a promising method for training powerful models with fewer parameters. In this paper, we introduced A*da*MOE, which uses null experts to enable token-level adaptive expert allocation and overcome the drawbacks of fixed expert

allocation. Extensive experiments validate its effectiveness. The A*da*MO<sup>E</sup> approach significantly enhances efficiency and adaptability, paving the way for more capable large language models.

# Limitations

One potential drawback of this work is that we did not pre-train a MoE-LLM using our A*da*MO<sup>E</sup> method. Pre-training an MoE-LLM would have allowed us to thoroughly evaluate the full capabilities and performance improvements of our method, but the significant resources required made it impractical for our current study. Additionally, we did not explore the scenario of null experts as identity mappings, where null experts would also need zero FLOPs to process input tokens. We hypothesize that this approach might accelerate training convergence because null experts as identity mappings would potentially update their corresponding router parameters more frequently.

We acknowledge these limitations and leave these aspects for future work. Addressing these issues could provide a more comprehensive evaluation of the A*da*MO<sup>E</sup> method and potentially uncover additional benefits or areas for improvement.

# References

- <span id="page-8-9"></span>Shengnan An, Yifei Li, Zeqi Lin, Qian Liu, Bei Chen, Qiang Fu, Weizhu Chen, Nanning Zheng, and Jian-Guang Lou. 2022. Input-tuning: Adapting unfamiliar inputs to frozen pretrained models. *arXiv preprint arXiv:2203.03131*.
- <span id="page-8-13"></span>Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. 2020. Piqa: Reasoning about physical commonsense in natural language. In *Thirty-Fourth AAAI Conference on Artificial Intelligence*.
- <span id="page-8-7"></span>Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. *Advances in neural information processing systems*, 33:1877–1901.
- <span id="page-8-8"></span>Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, et al. 2024. A survey on evaluation of large language models. *ACM Transactions on Intelligent Systems and Technology*, 15(3):1–45.
- <span id="page-8-0"></span>Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion

- Stoica, and Eric P. Xing. 2023. [Vicuna: An open](https://lmsys.org/blog/2023-03-30-vicuna/)[source chatbot impressing gpt-4 with 90%\\* chatgpt](https://lmsys.org/blog/2023-03-30-vicuna/) [quality.](https://lmsys.org/blog/2023-03-30-vicuna/)
- <span id="page-8-1"></span>Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2023. Palm: Scaling language modeling with pathways. *Journal of Machine Learning Research*, 24(240):1–113.
- <span id="page-8-5"></span>Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. *arXiv:1803.05457v1*.
- <span id="page-8-15"></span>OpenCompass Contributors. 2023. Opencompass: A universal evaluation platform for foundation models. [https://github.com/open-compass/](https://github.com/open-compass/opencompass) [opencompass](https://github.com/open-compass/opencompass).
- <span id="page-8-3"></span>Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y Wu, et al. 2024. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models. *arXiv preprint arXiv:2401.06066*.
- <span id="page-8-14"></span>Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2024. Qlora: Efficient finetuning of quantized llms. *Advances in Neural Information Processing Systems*, 36.
- <span id="page-8-10"></span>Shihan Dou, Enyu Zhou, Yan Liu, Songyang Gao, Jun Zhao, Wei Shen, Yuhao Zhou, Zhiheng Xi, Xiao Wang, Xiaoran Fan, et al. 2023. Loramoe: Revolutionizing mixture of experts for maintaining world knowledge in language model alignment. *arXiv preprint arXiv:2312.09979*.
- <span id="page-8-12"></span>William Fedus, Barret Zoph, and Noam Shazeer. 2022. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research*, 23(120):1–39.
- <span id="page-8-11"></span>Chongyang Gao, Kezhen Chen, Jinmeng Rao, Baochen Sun, Ruibo Liu, Daiyi Peng, Yawen Zhang, Xiaoyuan Guo, Jie Yang, and VS Subrahmanian. 2024. Higher layers need more lora experts. *arXiv preprint arXiv:2402.08562*.
- <span id="page-8-4"></span>Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. *arXiv preprint arXiv:2106.09685*.
- <span id="page-8-6"></span>Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. 1991. Adaptive mixtures of local experts. *Neural computation*, 3(1):79–87.
- <span id="page-8-2"></span>Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. 2024. Mixtral of experts. *arXiv preprint arXiv:2401.04088*.

- <span id="page-9-11"></span>Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. 2020. Gshard: Scaling giant models with conditional computation and automatic sharding. *arXiv preprint arXiv:2006.16668*.
- <span id="page-9-7"></span>Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. The power of scale for parameter-efficient prompt tuning. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 3045–3059.
- <span id="page-9-12"></span>Mike Lewis, Shruti Bhosale, Tim Dettmers, Naman Goyal, and Luke Zettlemoyer. 2021. Base layers: Simplifying training of large, sparse models. In *International Conference on Machine Learning*, pages 6265–6274. PMLR.
- <span id="page-9-9"></span>Dengchun Li, Yingzi Ma, Naizheng Wang, Zhiyuan Cheng, Lei Duan, Jie Zuo, Cal Yang, and Mingjie Tang. 2024. Mixlora: Enhancing large language models fine-tuning with lora based mixture of experts. *arXiv preprint arXiv:2404.15159*.
- <span id="page-9-4"></span>Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, and Colin A Raffel. 2022. Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning. *Advances in Neural Information Processing Systems*, 35:1950–1965.
- <span id="page-9-8"></span>Qidong Liu, Xian Wu, Xiangyu Zhao, Yuanshao Zhu, Derong Xu, Feng Tian, and Yefeng Zheng. 2023. Moelora: An moe-based parameter efficient finetuning method for multi-task medical applications. *arXiv preprint arXiv:2310.18339*.
- <span id="page-9-20"></span>Ilya Loshchilov and Frank Hutter. 2017. [De](https://arxiv.org/abs/1711.05101)[coupled weight decay regularization.](https://arxiv.org/abs/1711.05101) *Preprint*, arXiv:1711.05101.
- <span id="page-9-17"></span>Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. *Advances in Neural Information Processing Systems*, 35:2507–2521.
- <span id="page-9-19"></span>Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018. Can a suit of armor conduct electricity? a new dataset for open book question answering. *arXiv preprint arXiv:1809.02789*.
- <span id="page-9-3"></span>Qwen. 2024. Qwen1.5-moe: Matching 7b model performance with 1/3 activated parameters. [https:](https://qwenlm.github.io/blog/qwen-moe) [//qwenlm.github.io/blog/qwen-moe](https://qwenlm.github.io/blog/qwen-moe).
- <span id="page-9-14"></span>David Raposo, Sam Ritter, Blake Richards, Timothy Lillicrap, Peter Conway Humphreys, and Adam Santoro. 2024. Mixture-of-depths: Dynamically allocating compute in transformer-based language models. *arXiv preprint arXiv:2404.02258*.

- <span id="page-9-13"></span>Stephen Roller, Sainbayar Sukhbaatar, Jason Weston, et al. 2021. Hash layers for large sparse models. *Advances in Neural Information Processing Systems*, 34:17555–17566.
- <span id="page-9-21"></span>Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2021. Winogrande: An adversarial winograd schema challenge at scale. *Communications of the ACM*, 64(9):99–106.
- <span id="page-9-15"></span>Maarten Sap, Hannah Rashkin, Derek Chen, Ronan LeBras, and Yejin Choi. 2019. Socialiqa: Commonsense reasoning about social interactions. *arXiv preprint arXiv:1904.09728*.
- <span id="page-9-2"></span>Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. 2017. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *arXiv preprint arXiv:1701.06538*.
- <span id="page-9-18"></span>Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. 2018. Commonsenseqa: A question answering challenge targeting commonsense knowledge. *arXiv preprint arXiv:1811.00937*.
- <span id="page-9-0"></span>Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*.
- <span id="page-9-16"></span>Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. 2018. Glue: A multi-task benchmark and analysis platform for natural language understanding. *arXiv preprint arXiv:1804.07461*.
- <span id="page-9-10"></span>Xun Wu, Shaohan Huang, and Furu Wei. 2023. Mole: Mixture of lora experts. In *The Twelfth International Conference on Learning Representations*.
- <span id="page-9-5"></span>Ted Zadouri, Ahmet Üstün, Arash Ahmadian, Beyza Ermi¸s, Acyr Locatelli, and Sara Hooker. 2023. Pushing mixture of experts to the limit: Extremely parameter efficient moe for instruction tuning. *arXiv preprint arXiv:2309.05444*.
- <span id="page-9-22"></span>Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. Hellaswag: Can a machine really finish your sentence? *arXiv preprint arXiv:1905.07830*.
- <span id="page-9-1"></span>Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022. Opt: Open pre-trained transformer language models. *arXiv preprint arXiv:2205.01068*.
- <span id="page-9-6"></span>Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Zhao, Andrew M Dai, Quoc V Le, James Laudon, et al. 2022. Mixture-of-experts with expert choice routing. *Advances in Neural Information Processing Systems*, 35:7103–7114.

# <span id="page-10-0"></span>A Additional Ablation

Robustness. We primarily considered the impact of two hyperparameters, the number of epochs and the rank of LoRA module. We evaluated their impact on the SQA dataset, as shown in Table [6.](#page-11-1) Regardless of the number of training epochs and rank of the LoRA, our method outperforms the baseline consistently. Therefore, we can conclude that our method demonstrates strong robustness across different epochs and ranks of LoRAs.

# B Additional Results

Exact values for results in Figure [5.](#page-5-1) Table [7](#page-11-0) presents the exact values corresponding to Figure [5,](#page-5-1) averaged from results with three random seeds. The performance metrics evaluated include accuracy and average number of experts used per Mo-LoRA module/A*da*MO<sup>E</sup> layer across five datasets: RTE, COLA, SQA, CQA, and OQA.

For the baseline model, two configurations (k1 and k2) are tested. The k1 configuration, with a fixed top-1 routing strategy, achieves an accuracy of 65.06 on RTE, 85.39 on COLA, 63.35 on SQA, 76.09 on CQA, and 65.81 on OQA. The k2 configuration, with a fixed top-2 routing strategy, results in slightly lower accuracy values of 63.66 on RTE, 84.69 on COLA, 58.41 on SQA, 76.77 on CQA, and 65.57 on OQA.

The A*da*MO<sup>E</sup> model is evaluated with four configurations (m5k4, m7k4, m9k4, and m5k2). The m5k4 configuration achieves an accuracy of 66.64 on RTE, 85.01 on COLA, 66.64 on SQA, 76.96 on CQA, and 66.48 on OQA, with a load ranging from 1.77 to 1.80. The m7k4 configuration shows improved accuracy, reaching 67.19 on RTE, 85.83 on COLA, 68.17 on SQA, 77.64 on CQA, and 66.79 on OQA, with a load between 1.44 and 1.47. The m9k4 configuration presents an accuracy of 68.38 on RTE, 84.95 on COLA, 63.62 on SQA, 76.47 on CQA, and 67.98 on OQA, with a load consistently around 1.22. Lastly, the m5k2 configuration records an accuracy of 67.01 on RTE, 84.61 on COLA, 66.82 on SQA, 76.52 on CQA, and 67.89 on OQA, with a significantly lower load of 0.44.

Additional results for Section [4.2.](#page-6-1) Table [8](#page-12-0) presents the performance of various m and k combinations for the A*da*MO<sup>E</sup> model across different datasets, including ARC-C, HELLA, OQA, PIQA, and WINO. These results supplement the experimental findings discussed in Section [4.2.](#page-6-1)

For the ARC-C dataset, the baseline model with m = 0 and k = 2 achieved an accuracy of 87.46

and a load of 2.00. The A*da*MO<sup>E</sup> configurations demonstrated varying performance, with the highest accuracy of 89.15 observed for m = 8 and k = 3, accompanied by a load of 1.67. As the values of m and k increased, the load generally decreased, with the lowest load of 1.34 recorded for m = 40 and k = 8. On the HELLA dataset, the baseline model achieved an accuracy of 84.10 and a load of 2.00. The A*da*MO<sup>E</sup> model's best performance was observed with m = 8 and k = 3, achieving an accuracy of 85.50 and a load of 1.68. Similar to the ARC-C results, higher values of m and k led to reduced loads, with a minimum load of 1.37 for m = 40 and k = 7. For the OQA dataset, the baseline model achieved an accuracy of 89.94 and a load of 2.00. The A*da*MO<sup>E</sup> configurations showed varying results, with the highest accuracy of 89.2 observed for m = 16 and k = 4, and the load varying between 1.49 and 1.71. The lowest load of 1.50 was recorded for m = 40 and k = 8. In the case of the PIQA dataset, the baseline model reached an accuracy of 90.48 and a load of 2.00. The best accuracy among the A*da*MO<sup>E</sup> configurations was 90.32 for m = 8 and k = 3, with a load of 1.59. The load decreased as m and k values increased, reaching a minimum of 1.32 for m = 40 and k = 7. Finally, on the WINO dataset, the baseline model achieved an accuracy of 80.43 and a load of 2.00. The highest accuracy of 81.93 was observed for m = 8 and k = 3, with a load of 1.66. The load showed a decreasing trend with increasing values of m and k, with the lowest load of 1.45 recorded for m = 40 and k = 8.

# C Additional Discussion

The top-p router can also implement token-adaptive expert selection. It selects experts based on the sum of routing probabilities exceeding a threshold p. This allows for a variable number of experts to be chosen for different tokens. However, compared to our A*da*MOE, this approach has the following drawbacks:

- 1. The value of p cannot be predefined according to the compute budget, and finding an appropriate p often requires multiple attempts.
- 2. It cannot enable tokens to bypass some layers.

Moreover, our method is actually compatible with the top-p approach. We can incorporate null experts and simultaneously use top-p. This compatibility opens up avenues for further exploration in the future.

<span id="page-11-1"></span>

|          |      |       | Epoch |       | Rank of LoRAs |
|----------|------|-------|-------|-------|---------------|
|          |      | 1     | 10    | 8     | 32            |
| Baseline | Acc. | 45.95 | 87.19 | 45.95 | 46.72         |
|          | Load | 2.00  | 2.00  | 2.00  | 2.00          |
| AdaMOE   | Acc. | 48.88 | 88.54 | 48.88 | 49.01         |
|          | Load | 1.92  | 1.88  | 1.92  | 1.89          |

Table 6: Robustness of our method under different epochs and ranks of LoRAs.

<span id="page-11-0"></span>

|          |      | Metric       | RTE           | COLA          | SQA           | CQA           | OQA           |
|----------|------|--------------|---------------|---------------|---------------|---------------|---------------|
| Baseline | k1   | Acc.<br>Load | 65.06<br>1.00 | 85.39<br>1.00 | 63.35<br>1.00 | 76.09<br>1.00 | 65.81<br>1.00 |
|          | k2   | Acc.<br>Load | 63.66<br>2.00 | 84.69<br>2.00 | 58.41<br>2.00 | 76.77<br>2.00 | 65.57<br>2.00 |
|          | m5k4 | Acc.<br>Load | 66.64<br>1.78 | 85.01<br>1.77 | 66.64<br>1.80 | 76.96<br>1.77 | 66.48<br>1.78 |
|          | m7k4 | Acc.<br>Load | 67.19<br>1.47 | 85.83<br>1.45 | 68.17<br>1.47 | 77.64<br>1.44 | 66.79<br>1.47 |
| AdaMOE   | m9k4 | Acc.<br>Load | 68.38<br>1.22 | 84.95<br>1.23 | 63.62<br>1.22 | 76.47<br>1.22 | 67.98<br>1.22 |
|          | m5k2 | Acc.<br>Load | 67.01<br>0.44 | 84.61<br>0.44 | 66.82<br>0.44 | 76.52<br>0.44 | 67.89<br>0.44 |

Table 7: Exact values for Figure [5,](#page-5-1) averaged from results with 3 random seeds.

<span id="page-12-0"></span>

| ARC-C | Baseline |       |       | AdaMOE |       |       |       |
|-------|----------|-------|-------|--------|-------|-------|-------|
| m, k  | 0,2      | 8,3   | 16,4  | 24,5   | 32,6  | 40,7  | 40,8  |
| Acc.  | 87.46    | 89.15 | 87.12 | 86.10  | 85.08 | 86.10 | 85.76 |
| Load  | 2.00     | 1.67  | 1.70  | 1.56   | 1.49  | 1.59  | 1.34  |
| HELLA | Baseline |       |       | AdaMOE |       |       |       |
| m, k  | 0,2      | 8,3   | 16,4  | 24,5   | 32,6  | 40,7  | 40,8  |
| Acc.  | 84.10    | 85.50 | 83.10 | 81.30  | 80.40 | 82.50 | 79.20 |
| Load  | 2.00     | 1.68  | 1.64  | 1.45   | 1.39  | 1.37  | 1.44  |
| OQA   | Baseline |       |       | AdaMOE |       |       |       |
| m, k  | 0,2      | 8,3   | 16,4  | 24,5   | 32,6  | 40,7  | 40,8  |
| Acc.  | 89 94    | 88.2  | 89.2  | 86.6   | 86.8  | 85    | 82.6  |
| Load  | 2.00     | 1.70  | 1.71  | 1.49   | 1.54  | 1.56  | 1.50  |
| PIQA  | Baseline |       |       | AdaMOE |       |       |       |
| m, k  | 0,2      | 8,3   | 16,4  | 24,5   | 32,6  | 40,7  | 40,8  |
| Acc.  | 90.48    | 90.32 | 89.99 | 88.30  | 86.67 | 86.78 | 85.42 |
| Load  | 2.00     | 1.59  | 1.53  | 1.46   | 1.39  | 1.32  | 1.33  |
| WINO  | Baseline |       |       | AdaMOE |       |       |       |
| m, k  | 0,2      | 8,3   | 16,4  | 24,5   | 32,6  | 40,7  | 40,8  |
| Acc.  | 80.43    | 81.93 | 79.32 | 78.17  | 77.66 | 71.43 | 79.16 |
| Load  | 2.00     | 1.66  | 1.72  | 1.71   | 1.73  | 1.59  | 1.45  |

Table 8: Performance of more m and k combinations on various datasets. As a supplement to the experimental results in Section [4.2.](#page-6-1)