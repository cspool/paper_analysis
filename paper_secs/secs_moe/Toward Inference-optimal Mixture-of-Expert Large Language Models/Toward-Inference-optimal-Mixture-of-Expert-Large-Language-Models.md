# Toward Inference-optimal Mixture-of-Expert Large Language Models

Longfei Yun<sup>1\*</sup> Yonghao Zhuang<sup>2\*</sup> Yao Fu<sup>3</sup> Eric P Xing<sup>2</sup> Hao Zhang<sup>1</sup>

#### Abstract

Mixture-of-Expert (MoE) based large language models (LLMs), such as the recent Mixtral and DeepSeek-MoE, have shown great promise in scaling model size without suffering from the quadratic growth of training cost of dense transformers. Like dense models, training MoEs requires answering the same question: given a training budget, what is the optimal allocation on the model size and number of tokens? We study the scaling law of MoE-based LLMs regarding the relations between the model performance, model size, dataset size, and the expert degree. Echoing previous research studying MoE in different contexts, we observe the diminishing return of increasing the number of experts, but this seems to suggest we should scale the number of experts until saturation, as the training cost would remain constant, which is problematic during inference time. We propose to amend the scaling law of MoE by introducing inference efficiency as another metric besides the validation loss. We find that MoEs with a few (4/8) experts are the most serving efficient solution under the same performance, but costs 2.5-3.5x more in training. On the other hand, training a (16/32) expert MoE much smaller (70-85%) than the loss-optimal solution, but with a larger training dataset is a promising setup under a training budget.

## 1 Introduction

Recent developments, such as Mixtral (Jiang et al., 2024), DeepSeek-MoE (Dai et al., 2024), spotlight Mixture-of-Experts (MoE) models as a superior alternative to Dense Transformers. An MoE layer works by routing each input token to a selected group of experts for processing. Remarkably, increasing the number of experts in an MoE model (almost) does not raise the computational cost, enabling the model to incorporate more knowledge through extra parameters without inflating pre-training expenses. This approach seemingly presents a "free lunch" that we could just infinitely scale the number of experts – yet raises a critical question: is scaling up the number of experts in MoE models always as beneficial as it seems? In this paper, we answer this question and investigate the optimal number of experts for MoEs by examining two key factors: the scaling behavior and inference efficiency.

To understand how performance improves when scaling MoE models, we first study its scaling behavior. Previous works on Transformer model (Kaplan et al., 2020; Hoffmann et al., 2022) have established a power-law relationship linking the model's validation loss L to both the number of parameters N and training tokens D, which is referred to as the scaling law. Together with an estimation of training cost C(N,D), there is an optimal (N,D) within a training budget  $C_0$  (i.e., arg min L(N,D), s.t.  $C(N,D) \leq C_0$ ). We name this configuration a loss-optimal budget allocation.

Our *first contribution* is to enhance the existing scaling law to incorporate the number of experts E. Existing works either do not study the scaling behavior against E, or ignore the influence of the number of training tokens D – both are crucial to optimize the training budget allocation for MoEs. Akin to the Transformer scaling law, we observe that the number of expert, model size, and training dataset size all conform to a power-law relationship with validation loss. Consistent with previous work (Clark et al., 2022), our MoE scaling law

<sup>\*</sup>Equal contribution. <sup>1</sup>UC San Diego <sup>2</sup>Carnegie Mellon University <sup>3</sup>The University of Edinburgh

also reveals a diminishing return for increasing the number of experts, which saturates at a threshold *E*max.

Although our findings suggest a loss-optimal configuration with *E*max experts, such a setup is not practical for actual deployment. The main reason is that an excessive number of experts makes the model impractical for inference. In contrast to pretraining, LLM inference is notably memory-intensive, as it requires storing intermediate states (KV-cache) of all tokens. With more experts, the available memory for storing KV caches is squeezed. As a result, the batch size – hence throughput – decreases, leading to increased cost per query. This observation suggests scaling MoE must be subject to inference cost. Our *second contribution* is to incorporate inference cost, characterized by a new metric – cost per token – as a novel constraint for budget allocation for MoE models, in addition to the validation loss in existing works [\\*](#page-1-0). This dual-metric approach allows for a more comprehensive evaluation balancing model quality with practical resource constraints.

By jointly considering the scaling behavior under inference efficiency constraints, we first study loss-optimal models with different numbers of experts. We found that MoE models with 4 or 8 experts exhibit more efficient inference and higher performance compared to MoE models with more experts. However, they necessitate 2.4x-4.3x more training budgets to reach the same performance with models with more experts, making them impractical from the training side.

We further notice that for MoE with more experts, given a training budget, when the model shrinks a lot from the loss-optimal size, the performance only experience a marginal change. On the other hand, the inference cost grows linearly with the model size, and benefits a lot from a smaller model. This observation motivates us to train a model much smaller than the loss-optimal configuration. Such a model, though suffers from a marginal drop in quality, has a significantly lower inference cost. Because the budget saved from using a smaller model can be utilized to train on more tokens, we refer to this as an over-trained configuration. To evaluate the potential of over-trained models with more experts, we compare them with loss-optimal models with fewer experts under the same training budget. Under the same quality of a loss-optimal 4-expert MoE, an over-trained 8- or 16-expert MoE only needs 47.0% to 52.0% inference cost. With the same inference cost, an over-trained 16-expert MoE can save up to 68.4% training budget.

Our main contributions can be summarized as follows:

- We study the scaling law of MoE LLMs, revealing the relation between the validation loss and all 3 critical factors: model size, dataset size, and number of experts;
- We introduce a novel perspective to analyze the optimal training budget allocation for MoE models, which considers inference cost as a key component;
- We demonstrate that a smaller, over-trained MoE model with additional experts can surpass larger, fewer expert models in both quality and inference efficiency.

## **2 Background**

#### <span id="page-1-1"></span>2.1 Mixture of Expert Model

Many works on sparse models [\(Jacobs et al., 1991;](#page-9-5) [Jordan & Jacobs, 1994;](#page-9-6) [Shazeer et al.,](#page-10-0) [2017;](#page-10-0) [Lepikhin et al., 2020;](#page-10-1) [Fedus et al., 2022\)](#page-9-7) have been introduced to continue scaling the sizes of large language models with a marginal increase on compute, among which, Mixture-of-Expert (MoE) is perhaps the most succesful example. An MoE layer consists of a router and a set of experts. Every input token is routed to a subset of *K* experts, and the outputs of these experts are combined to produce the final output [\(Figure 7\)](#page-11-0). It is common to replace the Feed-forward layer (FFN) in a Transformer model with MoE layers. The architecture of each expert is identical to the replaced FFN.

<span id="page-1-0"></span><sup>\*</sup>In dense models, we cannot scale the number of parameters without increasing the training cost, hence the inference cost is predetermined and need not be separately considered in its scaling law.

A critical factor in MoEs is the number of parameters activated to process a single token. We introduce a notion *Corresponding Dense Model*, which refers to a dense Transformer model with an identical number of layers and hidden dimension size as the MoE model. If the Corresponding Dense Model of an MoE has a size of *N*, its total activated number of parameters for a token is roughly (*Ka* + (1 − *a*))*N*. Here *a* is the proportion of the size of MLP layers relative to the size of the dense model. Since all components of the model scale simultaneously, *a* is a constant for a given architecture.

## 2.2 Scaling Law

Recent research [\(Kaplan et al., 2020;](#page-9-2) [Hoffmann et al., 2022;](#page-9-3) [Brown et al., 2020\)](#page-9-8) indicate that scaling the number of parameters in a dense Transformer model or the size of the training dataset yields a predictable outcome on the model's final perplexity. Such correlation typically follows a power law of the parameters (N) and training tokens (D):

$$L(N,D) = L_0 + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} \tag{1}$$

where *L*0, *A*, *B*, *α*, *β* are constants whose values depend solely on the model architecture and the training data corpus, i.e. the quality of the dataset.

A common practice to determine the most effective allocation of the training budget is to utilize scaling laws:

<span id="page-2-0"></span>
$$\underset{N,D}{\operatorname{argmin}}L(N,D) \text{ s.t. } \operatorname{FLOPs}(N,D) = C \tag{2}$$

We refer to this choice of (*N*, *D*) under the constraint as the *loss-optimal* configuration.

They also propose to calculate the loss-optimal configuration as follows [\(Appendix A\)](#page-11-1):

<span id="page-2-1"></span>
$$N_{\text{opt}}(C) = G\left(\frac{C}{6}\right)^{a}, \quad D_{\text{opt}}(C) = G^{-1}\left(\frac{C}{6}\right)^{b},$$
where  $G = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha + \beta}}, a = \frac{\beta}{\alpha + \beta}, b = \frac{\alpha}{\alpha + \beta}$ 
(3)

Because *α* ≈ *β*, it is concluded that *N* and *D* should be scaled proportionally in computeoptimal training.

[Clark et al.](#page-9-4) [\(2022\)](#page-9-4) explores the scaling behavior of MoE models. They introduce a multiplicative factor to capture the interaction between *N* and *E*. Furthermore, they incorporate a saturation threshold (*E*max) to account for the diminishing returns observed when the number of experts (*E*) becomes excessively large. However, this study does not consider the impact of the dataset size (*D*) on the model's performance. As a result, it fails to provide recommendations on the loss-optimal configuration for a given training budget.

#### 2.3 LLM Inference

At inference, LLMs generate tokens following an auto-regressive paradigm. At the first iteration (prompt stage), the model generates the hidden states for all prompt tokens. In subsequent iterations (decoding stage), the model generates the hidden state for the most recently generated token and uses the accumulated hidden states to predict the next token. These hidden states, known as KV cache, are retrained in memory for compute efficiency. During the decoding phase, each iteration merely computes the hidden state of one token per request, resulting in low compute intensity on accelerators. To minimize the cost per query, we want to batch many requests to boost the serving throughput. Consequently, the size of the cumulative KV cache across all requests, even with optimizations like Multiquery attention (MQA), is very large and becomes significantly memory-bound. Hence, the available memory to store KV caches dictates the batch size – hence throughput and cost per query.

## <span id="page-3-2"></span>**3 Method: Scaling law of MoE model**

Though the scaling law for dense Transformer is already well developed, it still lacks exploration in the context of MoE models. In this section, we develop the MoE's scaling law from some previous exploration [\(Clark et al., 2022\)](#page-9-4).

#### 3.1 Experiment setup

To study the scaling behavior, we train a sweep of models with a dense model size ranging from 100 million to 730 million parameters. The detail of each model's hyper-parameters is in [Table 1.](#page-3-0) For every dense model, we trained with 4, 8, 16, and 32 experts, with a dense Transformer as the baseline. We construct the training dataset by uniformly sampling from SlimPajama [\(Soboleva et al., 2023\)](#page-10-2), with a size ranging from 2.5B to 20B. More training details can be found in [Appendix B.](#page-11-2)

<span id="page-3-0"></span>

| Name | dmodel | nlayers | nheads | Actual # Params (w/o embedding) |
|------|--------|---------|--------|---------------------------------|
| 100M | 768    | 12      | 8      | 81, 395, 712                    |
| 200M | 896    | 14      | 8      | 184, 64, 768                    |
| 320M | 1024   | 16      | 12     | 289, 406, 976                   |
| 730M | 1536   | 16      | 16     | 679, 477, 248                   |

Table 1: Model Configurations

## 3.2 Formulate the scaling law for MoE

Observations from [Figure 1](#page-4-0) show parallel lines for various dense model sizes, indicating a consistent slope across all sizes. This uniformity in slope is also apparent under different numbers of training tokens, with the lines differing primarily in their intercepts. On top of that, provided other factors do not become limiting, increasing the number of experts leads to a proportional decrease in validation loss. This trend holds true regardless of the dense model size and the number of training tokens used.

Based on the sweep of experimental runs, we observe a similar finding to the existing work [\(Clark et al., 2022\)](#page-9-4), that not all models across *N* benefit equally from *E*, though *E* roughly follows a power-law to a certain extent. As a result, we inherit the interaction term of *N* and *E* from the existing work.

However, the relation between *D* and *E* has not yet been explored. From [Figure 1,](#page-4-0) we observe that the benefit between two distinct number of experts *E* remains constant across different numbers of tokens *D*, indicating that an interaction term between *D* and *E* is unnecessary. It is also reasonable to conjecture that, when the router's error rate is roughly the same, a fixed number of tokens are dispatched to the correct expert to be learned, regardless of the number of experts. We also find the same saturating trend: as *E* increases, the benefit decreases, which is evident from scaling *E* from 16 to 32.

As a result, building upon the existing research [\(Kaplan et al., 2020;](#page-9-2) [Clark et al., 2022\)](#page-9-4) and experimental heuristics, we introduce a new scaling law that extends their theories to the MoE architecture:

<span id="page-3-1"></span>
$$\log L(N, D, E) \triangleq \log(\frac{A}{N^{\alpha}} + \frac{B}{\hat{E}^{\beta}} + \frac{C}{D^{\gamma}} + F) + d \log N \log \hat{E}$$
where  $\frac{1}{\hat{E}} \triangleq \frac{1}{E - 1 + \left(\frac{1}{E_{\text{start}}} - \frac{1}{E_{\text{max}}}\right)^{-1}} + \frac{1}{E_{\text{max}}}$ 
(4)

The first term represents the ideal performance achievable in a hypothetical space. However, the routing mechanism constrains the actual performance, leading to the introduction of the second term. *Estart* and *Emax* are two terms fitted to model the saturation, ensuring that the scaling behavior is bounded on both sides. *E*ˆ signifies that for *E* ≫ *Estart* and *E* ≪ *Emax*, performance varies near-linearly. The peak performance is equivalent to the performance obtained with *Emax* experts without saturation.

<span id="page-4-0"></span>![](_page_4_Figure_0.jpeg)

Figure 1: **Validation losses for different** *D***.** Scattered dots show the actual losses, and dotted lines correspond to values fitted by Equation 4.

#### 3.3 Fit Result

Figure 1 displays the predicted outcomes derived from our scaling law. More details of the fitting procedure is in Appendix D. The goodness-of-fit is evident, demonstrating that the fitted validation loss closely mirrors the actual validation loss. Therefore, Equation 4 serves as an ideal model to represent the relationship between validation loss, model size, number of tokens, and number of experts.

## 4 Method: Estimating inference cost for MoE

Although scaling the number of experts in MoE models can procure higher performance without increasing the training budget, it incurs a significantly higher inference cost. Therefore, when determining the "optimal" number of experts, it is essential to consider the inference cost. In this section, we model the inference costs and analyze the MoE model's inference cost as the number of the experts increases.

#### 4.1 Inference cost estimation

As highlighted in prior research (Narayanan et al., 2023), there is a linear relationship between the time to generate output and the number of output tokens. In other words, the latency in generating each token remains consistent. Thus, the throughput of a model m is in the form of  $T_m(N_m) = \frac{b_m(G)}{Lat_m(G,b)}$ , where G is the number of GPUs to serve the model, b is the maximal batch size, and Lat is the latency of a single iteration to generate a token. We derive the maximal batch size, latency and throughput in Appendix C.

**Inference cost** We define the inference cost in terms of dollars per token:

<span id="page-4-1"></span>
$$C_{\text{Model},G} = \frac{GC_0}{T_{\text{Model}}(G)} \tag{5}$$

Here C represents the cost per token, while G denotes the number of GPUs utilized.  $C_0$  is defined as the cost of operating a single GPU per second, which is usually considered a constant. Since the vendor has the flexibility to use any number of GPUs, we define C(m) of model m based on the most cost-effective GPU utilization, i.e.,  $C(m) = \min_G(C_{m,G})$ , meaning we select the minimum cost across different GPU numbers for the most economical option of model m.

#### <span id="page-4-2"></span>4.2 MoE inference cost

As discussed in Appendix C, the size of an MoE model has  $N_{MoE} = (1 + (E - 1) \cdot 1/3)N$ . We take this term into Equation 5 and Equation 6 to estimate the inference cost of MoE.

In this paper, we profile the inference cost on 8x40 GB A100 GPU with NVLink connected, and use the state-of-the-art serving system vLLM (Kwon et al., 2023) to launch our model.

Figure 2 (left) shows the inference cost under different model sizes. Conversely, Figure 2 (right) plots the maximum model size for different inference budget. The relationship

<span id="page-5-0"></span>![](_page_5_Figure_0.jpeg)

Figure 2: **MoE inference cost.** Cost increases proportionally with model size.

between inference cost and model size is mostly smooth and monotonic, and all exceptions occur because the minimum number of GPUs required to serve the model increases, which results in a gap in the inference cost.

## <span id="page-5-3"></span>**5 Results: Budget Allocation with Inference Efficiency**

<span id="page-5-2"></span>![](_page_5_Figure_4.jpeg)

Figure 3: **Trade-off between inference cost, model performance, and training cost.** Inference cost of and model performance for MoE models under different training budgets (left); Model performance with different training FLOPs (middle); Inference cost of different training FLOPs (right). Under the same budget, more experts means a better quality but higher inference cost. Fewer experts can reach a lower inference cost with the same quality, but needs much more training FLOPs

Previous analysis already reveals a trade-off between inference cost and performance for MoE with different number of experts: on one hand, the scaling law (Section [3\)](#page-3-2) shows that more experts (larger E) result in a higher performance; on the other hand, more experts result in a larger inference cost (Section [4.2\)](#page-4-2). In this section, we first reveals another trade-off between training budget and inference cost (Section [5.1\)](#page-5-1), then propose a budget allocation considering all these trade-offs (Section [5.2\)](#page-6-0). The key idea of our purposed budget allocation is to relax the loss-optimal constraint during training, allowing a model with a sub-optimal performance, but a much lower inference cost.

#### <span id="page-5-1"></span>5.1 Trade-off between training and inference

For MoE model with different experts, there exists a trilemma among training budget, inference cost, and model quality. As shown in [Figure 3\(](#page-5-2)middle), for any fixed training budget, MoE with more experts have a higher performance (i.e., a lower loss). However, it suffers from a higher inference cost, as shown in [Figure 3\(](#page-5-2)right).

Since model training only runs for once, while model inference may serve unlimited number of requests, we also studies the correlation between the two inference metrics: model quality and inference cost. [Figure 3\(](#page-5-2)left) plots the model quality and inference cost under different training budgets, but guarantees that the model is loss-optimal. MoE with 4 or 8 experts shows the best quality (lowest validation loss) under a certain inference cost.

An explanation is that, the inference cost is approximately linear to the total number of parameters (Figure 2). Under a fixed inference cost, if the number of experts is halved, the number of equivalent dense model's parameters is approximately doubled. For a loss-optimal configuration, the training dataset is scaled with the dense model's parameters, thus it is also doubled. In most cases, the gain of doubling both training dataset and the dense model's parameters outperforms the loss of halving the number of experts, and thus using fewer experts is more suggested.

However, since both the dense model and training dataset needs to be scaled up, MoE with fewer experts demands a much higher training budget to reach the same performance. By revisiting Figure 3 (middle), there is a consistent trend that, to achieve the same loss with MoE models of fewer experts, it requires an increasing percentage in FLOPs. A 16-expert MoE only needs 23.7% to 42.8% of the FLOPs to reach the same model performance of a 4-expert MoE. When the total FLOPs increases, such a gap grows even larger.

This observation underscores that though MoE models with fewer experts (such as 4 or 8) consistently improve performance than more experts across both metrics on the inference side. However, this advantage comes at the cost of a much larger training cost.

## <span id="page-6-0"></span>5.2 Over-training smaller MoE with more data

<span id="page-6-1"></span>![](_page_6_Figure_4.jpeg)

Figure 4: **loss-cost curve for a given training budget.** The over-trained 16-expert model achieves both better performance and lower inference cost than loss-optimal 4/8 expert model.

Though MoE of fewer experts has a lower inference cost, it needs an innegligible extra training budget. However, Figure 4 demonstrates that, for a given training budget, the model with 8 or 16 experts outperforms the optimal 4-expert models in a specific region, achieving both improved performance and reduced cost. This motivates us to consider such a case: what if we shift from the loss-optimal configuration to a model with fewer parameters, which leads to a much smaller inference cost? Since we can reuse the budget saved from model size to train more tokens, the model's quality only experiences a marginal drop within a range. We call this an *over-trained* budget allocation. In this part, we study the potential of such over-trained model with more experts, and compare them with loss-optimal models with fewer experts under different scenarios.

More specifically, given a training budget B, we first find the loss-optimal budget allocation  $(N_E, D_E)$  under a fixed number of experts E. The validation loss and inference cost for this model is correspondingly  $L_E^{opt}$  and  $I_E$ . Then for MoE with a larger number of expert E' > E, we study its over-trained configuration, where its quality is anchored by  $L_E$ , say  $L_{E'} \leq L_E^{opt}$ . We compute the lowest inference cost  $I_{E'}^{\min}$  for E' experts under the quality constraint above, and compare  $I_{E'}^{\min}$  with  $I_E$ . On the other hand, we also consider the lowest validation loss  $L_{E'}^{\min}$ , under a bound that  $I_{E'} \leq I_E$ , and compare  $L_{E'}^{\min}$  with  $L_E^{opt}$ .

The practical meaning of the two is that, if an over-trained model reaches the quality of a loss-optimal model, can it have a lower inference cost? Or on the other hand, if the two model has the same inference cost, which one has a higher quality.

**Optimal inference cost for a bounded loss.** Based on the scaling law, the loss L is monotonic to model size N before the loss-optimal point. Besides, the inference cost I is also

monotonic to N. As a result, to minimize inference cost I, the model size N should be as low as possible, meaning the loss is as large as possible. As a result,  $I_{E'}^{\min}$  corresponds to the case when the loss is exactly  $L_{E}^{opt}$ .

Based on the above analysis, we do dichotomy search for equation  $L_{E'}(N,B) = L_E^{opt}$  to find the solution  $N_{E'}$ , and use it to compute  $I_{E'}^{min}$  (the detail is in Algorithm 1). Figure 5 (left) shows the result for E=1 (dense Transformer) and 4 (4-expert MoE). To reach the model performance (validation loss) similar to that of the dense model, over-training an 8-expert MoE with the same training budget has the lowest inference cost, which is 31.6%-38.1% as large as that of the dense model when B ranges from 5.15e21 to 8.18e21. When using 4-expert loss-optimal MoE's quality as a standard, 8-expert over-trained MoE saves 49.0%-52.3% inference cost per token, and 16-expert over-trained MoE saves 48%-53% inference cost. MoE with more experts has a higher cost than 8- or 16-expert. We reason this as that 4-expert MoE's optimal loss is too far away from 32-or-more expert MoE's, making the over-training no longer appealing as it already leaves the "flat area" in the size-loss curve.

<span id="page-7-0"></span>![](_page_7_Figure_2.jpeg)

Figure 5: **Optimal inference cost for a bounded loss.** Minimum achievable inference cost with a bounded loss (left). Ratio of model size to the base model (right).

Figure 5 (right) further shows how much smaller than the loss-optimal model is trained. When using the loss-optimal dense Transformer as baseline, with a training budget ranging from 2.12e21 to 5.96e21 (which means the dense model has a number of parameters from 3.36B to 6.14B), an 8-expert MoE uses 23.3%  $\sim$  28.2% activated parameters of the loss-optimal dense model and 21.0%  $\sim$  25.1% of the loss-optimal 4-expert MoE. Two consistent trends emerge: first, as the number of experts increases, the ratio of activated parameters in the MoE model compared to the base model decreases. Second, a higher budget correlates with a lower dense model parameter ratio.

**Optimal loss for bounded inference cost.** Similarly, given a training budget B, we firstly compute the loss-optimal configuration for E-expert MoE, with its inference cost  $I_E$  and  $L_E$ . For MoE with E' experts, we compute the model size  $N_{E'}$  which has an inference cost of  $I_{E'}$ . The monotonicity discussed before guarantees that this is the model size with the lowest loss under the inference bound. Then we use the scaling law to estimate its loss, say  $L_{E'}^{\min} = Loss(N_{E'}, B)$ . (the detail of the algorithm is in Algorithm 2).

Figure 6 (left) shows the result when the base model is a dense Transformer or 4-experts MoE. Overtraining more experts always has a better validation loss, but the gain of scaling from 16 to 32 experts already shows a diminishing return.

Alike the bounded loss case, we also study how small is the over-trained model. Figure 6 (right) gives the ratio between the size of the over-trained E'-expert MoE and the loss-optimal E-expert MoE. When using the loss-optimal dense Transformer as the baseline, an 8-expert MoE uses 84.1% as large as the loss-optimal base model under a training budget of 5.15e21, while other number of experts also varies in a range of  $37.1\%\sim125.2\%$ . If the baseline is the loss-optimal 4-experts MoE under the same training budget, the ratio varies from  $30.7\%\sim69.5\%$ , if we continue to scale the loss-optimal 4-experts MoE model, it will need 52.1% more FLOPs in order to achieve the same loss of 8-experts MoE.

**Recommended training setup.** Over-training a smaller model with a larger dataset exhibits a great potential to reach an inference efficiency. When model quality is the most

<span id="page-8-0"></span>![](_page_8_Figure_0.jpeg)

Figure 6: **Optimal loss for a bounded inference cost.** Minimum achievable loss with a bounded inference cost (left). Ratio of model size to the base model (right).

concerning factor, training a 32-experts MoE as 30% large as the loss-optimal 32-experts MoE is preferred. If the inference cost is more important, training a 16-experts MoE as 16% large as the loss-optimal 16-experts MoE is preferred. [Figure 5](#page-7-0) (right) and [Figure 6](#page-8-0) (right) prove that such a conclusion is scalable. With the growth of the training budget, the ratio of over-trained model's size against a loss-optimal model is approximately a constant.

## **6 Related Work**

**Scaling laws** Previous works extensively study the scaling behavior on different cases, especially for Transformer. [Kaplan et al.](#page-9-2) [\(2020\)](#page-9-2) note a power-law relationship between model size, training dataset size, and the pretrained model's quality. They suggested that when the model scales 5.5x larger, the training tokens needs to grow 1.8x larger. [Hoffmann et al.](#page-9-3) [\(2022\)](#page-9-3), however, showed that the scaling of model size and training dataset should be scaled in equal proportions. [Muennighoff et al.](#page-10-4) [\(2023\)](#page-10-4) and [Frantar et al.](#page-9-10) [\(2023\)](#page-9-10) studied the scaling behavior for data-constrained training and sparse models, respectively, by introducing new terms to describe the data repetition and sparsity. [Clark et al.](#page-9-4) [\(2022\)](#page-9-4) is the only attempt of MoE scaling law. It shows that MoE shows a unified scaling trend among different gating mechanisms. However, this work does not include training dataset size into consideration. As a result, unlike the later works, it cannot show a proportion between scaling model size and training dataset.

**MoE pre-training practice** Starting from [Lepikhin et al.](#page-10-1) [\(2020\)](#page-10-1), MoE architecture has been adapted with Transformer as a more cost-efficient way to scale the number of parameters. [\(Zoph, 2022;](#page-10-5) [Fedus et al., 2022\)](#page-9-7) discussed new loss function and routing mechanisms to improve the training and fine-tuning efficiency. Recent practices [\(Dai et al., 2024;](#page-9-1) [Jiang](#page-9-0) [et al., 2024\)](#page-9-0) have scaled MoE into billions of activated parameters, with a performance even stronger than the state-of-the-art Transformer models of the same size. However, these pretrained MoEs designs the hyper-parameters in an ad hoc way, simply following the scaling law of Transformers to decide the training budget allocation.

**Budget allocation with inference cost** The closest work to this paper is [Sardana & Frankle](#page-10-6) [\(2023\)](#page-10-6), which also recognized that inference cost should be considered in the training budget allocation problem. However, this work relied on oversimplified assumptions. It estimated inference cost with a total number of requests, which is unpredictable. Besides, it simply assumed a constant Model FLOPs Utilization (MFU) at both the training and the inference stage, while our profiling shows that MFU varies 10x with different batch sizes.

## **7 Conclusion**

This paper studies the problem of how to scale the number of experts in the fast-developing MoE large language models. We first extend the scaling law, originally developed for dense transformer LLMs, to the context of MoEs, establishing a new relation between the validation loss and the number of experts, the number of training tokens, and the model size. We then discuss the need and the unique challenge to additionally consider inference efficiency when scaling MoEs. Our findings provide new insights on how to appropriately scale MoE models under compute constraints.

## **References**

- <span id="page-9-8"></span>Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. *Advances in neural information processing systems*, 33:1877–1901, 2020.
- <span id="page-9-4"></span>Aidan Clark, Diego de Las Casas, Aurelia Guy, Arthur Mensch, Michela Paganini, Jordan Hoffmann, Bogdan Damoc, Blake A. Hechtman, Trevor Cai, Sebastian Borgeaud, George van den Driessche, Eliza Rutherford, Tom Hennigan, Matthew J. Johnson, Albin Cassirer, Chris Jones, Elena Buchatskaya, David Budden, Laurent Sifre, Simon Osindero, Oriol Vinyals, Marc'Aurelio Ranzato, Jack W. Rae, Erich Elsen, Koray Kavukcuoglu, and Karen Simonyan. Unified scaling laws for routed language models. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), ´ *International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA*, volume 162 of *Proceedings of Machine Learning Research*, pp. 4057–4086. PMLR, 2022. URL <https://proceedings.mlr.press/v162/clark22a.html>.
- <span id="page-9-1"></span>Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y Wu, et al. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models. *arXiv preprint arXiv:2401.06066*, 2024.
- <span id="page-9-11"></span>Nan Du, Yanping Huang, Andrew M Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, et al. Glam: Efficient scaling of language models with mixture-of-experts. In *International Conference on Machine Learning*, pp. 5547–5569. PMLR, 2022.
- <span id="page-9-7"></span>William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *The Journal of Machine Learning Research*, 23(1):5232–5270, 2022.
- <span id="page-9-10"></span>Elias Frantar, Carlos Riquelme, Neil Houlsby, Dan Alistarh, and Utku Evci. Scaling laws for sparsely-connected foundation models. *arXiv preprint arXiv:2309.08520*, 2023.
- <span id="page-9-3"></span>Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. *arXiv preprint arXiv:2203.15556*, 2022.
- <span id="page-9-12"></span>Peter J Huber. Robust estimation of a location parameter. In *Breakthroughs in statistics: Methodology and distribution*, pp. 492–518. Springer, 1992.
- <span id="page-9-5"></span>Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. Adaptive mixtures of local experts. *Neural computation*, 3(1):79–87, 1991.
- <span id="page-9-0"></span>Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. *arXiv preprint arXiv:2401.04088*, 2024.
- <span id="page-9-6"></span>Michael I Jordan and Robert A Jacobs. Hierarchical mixtures of experts and the em algorithm. *Neural computation*, 6(2):181–214, 1994.
- <span id="page-9-2"></span>Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. *arXiv preprint arXiv:2001.08361*, 2020.
- <span id="page-9-9"></span>Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In *Proceedings of the 29th Symposium on Operating Systems Principles*, pp. 611–626, 2023.

- <span id="page-10-1"></span>Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. *arXiv preprint arXiv:2006.16668*, 2020.
- <span id="page-10-13"></span>Dong C Liu and Jorge Nocedal. On the limited memory bfgs method for large scale optimization. *Mathematical programming*, 45(1-3):503–528, 1989.
- <span id="page-10-12"></span>Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. *arXiv preprint arXiv:1711.05101*, 2017.
- <span id="page-10-4"></span>Niklas Muennighoff, Alexander M Rush, Boaz Barak, Teven Le Scao, Aleksandra Piktus, Nouamane Tazi, Sampo Pyysalo, Thomas Wolf, and Colin Raffel. Scaling data-constrained language models. *arXiv preprint arXiv:2305.16264*, 2023.
- <span id="page-10-11"></span>Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, et al. Efficient large-scale language model training on gpu clusters using megatron-lm. In *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis*, pp. 1–15, 2021.
- <span id="page-10-3"></span>Deepak Narayanan, Keshav Santhanam, Peter Henderson, Rishi Bommasani, Tony Lee, and Percy Liang. Cheaply estimating inference efficiency metrics for autoregressive transformer models. In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023.
- <span id="page-10-8"></span>Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, pp. 3505–3506, 2020.
- <span id="page-10-6"></span>Nikhil Sardana and Jonathan Frankle. Beyond chinchilla-optimal: Accounting for inference in language model scaling laws. *arXiv preprint arXiv:2401.00448*, 2023.
- <span id="page-10-0"></span>Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixtureof-experts layer. *arXiv preprint arXiv:1701.06538*, 2017.
- <span id="page-10-10"></span>Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. *arXiv preprint arXiv:1909.08053*, 2019.
- <span id="page-10-9"></span>Shaden Smith, Mostofa Patwary, Brandon Norick, Patrick LeGresley, Samyam Rajbhandari, Jared Casper, Zhun Liu, Shrimai Prabhumoye, George Zerveas, Vijay Korthikanti, et al. Using deepspeed and megatron to train megatron-turing nlg 530b, a large-scale generative language model. *arXiv preprint arXiv:2201.11990*, 2022.
- <span id="page-10-2"></span>Daria Soboleva, Faisal Al-Khateeb, Robert Myers, Jacob R Steeves, Joel Hestness, and Nolan Dey. SlimPajama: A 627B token cleaned and deduplicated version of RedPajama. [https://www.cerebras.net/blog/](https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama) [slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama](https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama), June 2023. URL <https://huggingface.co/datasets/cerebras/SlimPajama-627B>.
- <span id="page-10-7"></span>Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee Lacroix, Baptiste Rozi ´ ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. ` Llama: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*, 2023.
- <span id="page-10-5"></span>Barret Zoph. Designing effective sparse expert models. In *IEEE International Parallel and Distributed Processing Symposium, IPDPS Workshops 2022, Lyon, France, May 30 - June 3, 2022*, pp. 1044. IEEE, 2022. URL <https://doi.org/10.1109/IPDPSW55747.2022.00171>.

## <span id="page-11-1"></span>A Optimal Allocation For Dense Model

Given that the training budget can be approximated as C = 6ND (Kaplan et al., 2020) and the optimal allocation problem illustrated in Equation 2, we can solve this convex optimization problem with an equality constraint by adding a Lagrange multiplier.

$$\mathcal{L}(N,D,\lambda) = L(N,D) = L_0 + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} + \lambda(6ND - C)$$

The dual problem is  $g(\lambda) = \inf_{N,D} \mathcal{L}(N,D,\lambda)$ .

By taking the derivative with respect to *N* and *D*, we have:

$$\begin{split} \frac{\partial \mathcal{L}}{\partial N} &= \frac{\partial}{\partial N} \left( L_0 + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} + \lambda (6ND - C) \right) \\ &= \frac{\partial}{\partial N} \left( \frac{A}{N^{\alpha}} \right) + \frac{\partial}{\partial N} \left( \lambda (6ND - C) \right) \\ &= -\frac{\alpha A}{N^{\alpha + 1}} + 6\lambda D \\ \frac{\partial \mathcal{L}}{\partial D} &= \frac{\partial}{\partial D} \left( L_0 + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} + \lambda (6ND - C) \right) \\ &= \frac{\partial}{\partial D} \left( \frac{B}{D^{\beta}} \right) + \frac{\partial}{\partial D} \left( \lambda (6ND - C) \right) \\ &= -\frac{\beta B}{D^{\beta + 1}} + 6\lambda N \end{split}$$

Let both the derivatives equal 0 and also apply the constraint C = 6ND. We can calculate the loss-optimal configuration as shown in Equation 3.

## <span id="page-11-2"></span><span id="page-11-0"></span>**B** Training Details

![](_page_11_Figure_8.jpeg)

Figure 7: MoE architecture.

**Model Details** As seen in Figure 7, a Transformer's MoE layer is composed of E feedforward networks, labeled FFN<sub>1</sub> to FFN<sub>E</sub>. Given an input token  $u_t^l$  (i.e. logits of token t in the l-th layer) to this MoE layer, its output is a sum of the outputs from these experts, calculated as  $\sum_{e=1}^{E} \mathcal{G}i$ ,  $t \cdot \text{FFN} \, e \left( u_t^l \right)$ . Here,  $\mathcal{G}i$ , t is a vector determined by a gating mechanism GATE(·). It's decided that each token is routed to no more than K experts, which causes the gating

values  $G_{i,t}$  to be non-zero for those experts involved, indicating their respective contributions to the overall output of the network.

$$\mathbf{h}_{t}^{l} = \sum_{i=1}^{E} \left( \mathcal{G}_{i,t} \operatorname{FFN}_{i} \left( u_{t}^{l} \right) \right) + u_{t}^{l},$$

$$\mathcal{G}_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \operatorname{Topk} \left( \left\{ s_{j,t} \mid 1 \leqslant j \leqslant E \right\}, K \right), \\ 0, & \text{otherwise} \end{cases}$$

$$s_{i,t} = \operatorname{Softmax}_{i} \left( u_{t}^{l^{T}} \right),$$

Recent works (Du et al., 2022; Zoph, 2022; Fedus et al., 2022; Lepikhin et al., 2020) suggest to replace one of every two FFN layers in a Transformer model by MoE, and use Top-2 gating (Shazeer et al., 2017) as the routing mechanism. In this paper, we also inherit from such a context. Besides, our model architecture follows the practice of Llama (Touvron et al., 2023), which uses a gated-MLP as the feed-forward layer, and the MLP intermediate hidden dimension size is 2.6x large as the model's hidden dimension.

To train our model, we have forked Megatron-Deepspeed (Rasley et al., 2020; Smith et al., 2022) framework. Models are trained using data, tensor parallelism on up to 32 GPUs.

**Dataset** we specify our dataset choice as SlimPajama (Soboleva et al., 2023), a high-quality dataset refined through content filtering and deduplication processes. It is an open-source version of the LLaMA pretraining data blend, comprising 82% internet content (with 67% from CommonCrawl and 15% from C4), 4.5% code (sourced from Github), 4.5% from Wikipedia, 4.5% from books, 2.5% from Arxiv, and 2% from StackExchange. Given that this dataset closely resembles the one used for pretraining LLaMA models, there is less concern about adapting the findings to various datasets. From this dataset, our experiments utilize up to 20 billion tokens for training and 0.58 billion tokens for validation purposes.

Training Details All models were trained on A100 GPUs, utilizing a blend of data, tensor, and model parallelism as outlined in Shoeybi et al. (2019); Narayanan et al. (2021). The training involved a sequence length of 2048 and a batch size of 256 (i.e. 0.5M tokens per batch). All models are optimized with AdamW (Loshchilov & Hutter, 2017). Due to empirical observations, it has been determined that larger models necessitate a reduced learning rate to avoid divergence, whereas smaller models can withstand a higher learning rate. Consequently, we establish the learning rate based on previous experience (Kaplan et al., 2020):

$$LR(N) \approx 0.003239 + -0.0001395 \log(N)$$

We also employ a linear warm-up of the learning rate with the initial 3% tokens. The learning rate then decays to 10% of the maximum value through a cosine schedule.

#### <span id="page-12-0"></span>C Inference Cost Estimation

**Model size** The model size of MoE model refers to the size of the corresponding dense model, as described in Section 2.1. The total number of parameters can be approximately described as proportional to N\*(1+(E-1)c), where the factor c is influenced by the model architecture. In our setup, we replace a FFN layer by MoE for every two Transformer layers. The width of the Gated-MLP layers i is fixed at around 2.67 times the width of the model hidden state h (Touvron et al., 2023), so FFN layers take 2/3 of all parameters in the dense model. Consequently, c equals 1/3.

In a Transformer layer, the parameter count primarily stems from two components: the self-attention module and the feed-forward network.

Within the self-attention mechanism, four matrices of parameters exist:  $W_k$ ,  $W_v$ ,  $W_q$ ,  $W_o$ , each having dimensions  $h \times h$ . Additionally, the bias components contribute 4h parameters. Therefore, the self-attention mechanism altogether encompasses  $4h^2 + 4h$  parameters.

Regarding the gated MLP, there are three linear projections involved: the gate projection, which is  $h \times 2.67h$ ; the up projection, also  $h \times 2.67h$ ; and the down projection, which is  $2.67h \times h$ . Consequently, the MLP component holds a total of  $8.01h^2 + 6.34h$  parameters.

Excluding the linear term, the proportion of parameters attributed to the MLP relative to the total is approximately  $\frac{8.01}{8.01+6.34} \approx \frac{2}{3}$ .

**Maximal batch size** For every token processed, the KV-cache memory for a token is 2hl, with the hidden dimension size h and the number of layers l. Assume a model has  $N_m$  parameters, each GPU has  $M_0$  memory, the available memory for KV-cache is  $GM_0 - N_m$ . Assume that the average output length is n, and the average prompt length is p. The memory for a single request's KV-cache grows from 2phl to 2(n+p)hl, and the expectation is (2p+n)hl. Hence, the maximum number of simultaneous requests that can be served is given by  $b = \frac{GM_0 - N_m}{(2p+n)hl}$ .

**Latency** When serving with a batch size b, the decoding iteration's batch size is b. Given the average output length n, we can expect that on average, b/n requests will be completed in a decoding iteration. On the other hand, to maintain the batch size stable, it needs b/n new requests, necessitating an additional prompt iteration. Hence, the latency per iteration for model m has:

$$L_m(b,G) = L_m^P(b/n,G) + L_m^D(b,G)$$

were  $L_m^P(b,G)$ ,  $L_m^D(b,G)$  are the prompt and decoding latency with a batch size b on G GPUs. The prompt and decoding stages exhibit distinct levels of computing intensities. To assess the latency of each stage, we separately conduct a detailed profiling of various models for each stage. This data is used to estimate latency for other models through linear interpolation on batch size and model size.

**Throughput** Let k = 2p + n, the throughput  $T_m$  of model m has:

<span id="page-13-0"></span>
$$T_{m} = \frac{GM_{0} - N_{m}}{khl(L_{m}^{P}(\frac{GM_{0} - N_{m}}{k_{n}hl}, G) + L_{m}^{D}(\frac{GM_{0} - N_{m}}{khl}, G))}$$
(6)

Since p and n depend solely on the request's traffic patterns, together with k is a constant. We approximate their values with the ShareGPT dataset.

<span id="page-13-1"></span>Furthermore, there is  $N \propto h^2 l$ . To estimate the hl term in the model's throughput, we take a simple assumption that hidden state and number of layers roughly keep a linear relationship. As a result, there is  $hl = \mu N^{2/3}$ , where  $\mu$  is a constant. As Figure 8 shows, the accurate predicted hl assures that our assumption is reasonable.

![](_page_13_Figure_10.jpeg)

Figure 8: **Fitted hl with**  $N^{2/3}$ . Dots represent the actual hl value, the line indicate the fitted value.

## <span id="page-14-0"></span>D Detail of Fitting the Scaling Law

To estimate  $(\alpha, \beta, \gamma, A, B, C, d, F)$ , we effectively minimize the Huber loss (Huber, 1992):

$$\min_{A,B,C,d,F,\alpha,\beta,\gamma} \sum_{\text{Run } i} \text{Huber}_{\delta} \left( \log \hat{L} \left( N_i, D_i, \hat{E}_i \right) - \log L_i \right)$$

We use the L-BFGS (Liu & Nocedal, 1989) algorithm to find local minima of the objective above, started on a grid of initialisation given by:  $\alpha \in \{0., 0.5, ..., 2.\}, \beta \in \{0., 0.5, ..., 2.\}, \gamma \in \{0., 0.5, ..., 2.\}, a \in \{0, 5, ..., 25\}, b \in \{0, 5, ..., 25\}, c \in \{0, 5, ..., 25\}, d \in \{0, 5, ..., 25\}, f \in \{1., -.5, ..., 1.\}$ . We use  $\delta = 10^{-3}$  for the Huber loss, which is robust shown in previous work (Hoffmann et al., 2022).

We also compute RMSLE value and Huber loss value, which are 3.908e-3 and 1.033e-3, respectively, indicating that the error is extremely low.

#### **E** Bound Metrics

Here we provide the detail algorithm for section 5 about studying the optimal inference cost under a bounded loss, or optimal loss under an inference cost. In both cases, the bound is defined by the loss-optimal MoE with fewer experts.

## <span id="page-14-1"></span>Algorithm 1 Optimal Inference Cost For A Bounded Loss.

```
Input: A training budget B
```

A base model with number of experts *E* 

A MoE model with a larger number of experts E'

Total GPU number G

**Output**: Lowest inference cost  $I_{r'}^{\min}$ 

```
1: (N_E, D_E) \leftarrow \text{Optimal\_config}(B)
2: L_E \leftarrow \text{Scaling\_law}(N_E, D_E, E)
3: \mathbf{for} \ g \leftarrow 1 \ \mathbf{to} \ G \ \mathbf{do}
4: I_E \leftarrow \min(\text{Get\_cost}(N_E, E, g), I_E)
5: \mathbf{end} \ \mathbf{for}
6: N_{E'} \leftarrow \text{Dichotomy\_search}(E', L_E)
7: \mathbf{for} \ g \leftarrow 1 \ \mathbf{to} \ G \ \mathbf{do}
8: I_{E'}^{\min} \leftarrow \min(\text{Get\_cost}(N_{E'}, E', g), I_{E'}^{\min})
9: \mathbf{end} \ \mathbf{for}
10: \mathbf{return} \ I_{E'}^{\min}
```

#### <span id="page-14-2"></span>Algorithm 2 Optimal Loss For A Bounded Inference Cost.

```
Input: A training budget B
```

A base model with number of experts *E* 

A MoE model with a larger number of experts E'

Total GPU number G

**Output**: Lowest validation loss  $L_{E'}^{\min}$ 

```
1: (N_E, D_E) \leftarrow \texttt{Optimal\_config}(B)

2: \mathbf{for} \ g \leftarrow 1 \ \mathbf{to} \ G \ \mathbf{do}

3: I_E \leftarrow \min(\texttt{Get\_cost}(N_E, E, g), I_E)

4: \mathbf{end} \ \mathbf{for}

5: N_{E'} \leftarrow \texttt{Dichotomy\_search}(E', I_E)

6: D_{E'} \leftarrow \texttt{Dataset\_size}(B, N_{E'})

7: L_{E'} \leftarrow \texttt{Scaling\_law}(N_{E'}, D_{E'}, E')

8: \mathbf{return} \ L_{E'}^{\min}
```