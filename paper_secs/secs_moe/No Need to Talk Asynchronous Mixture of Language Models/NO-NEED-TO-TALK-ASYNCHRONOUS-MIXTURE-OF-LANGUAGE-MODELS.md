# NO NEED TO TALK: ASYNCHRONOUS MIXTURE OF LANGUAGE MODELS

Anastasiia Filippova † EPFL Angelos Katharopoulos Apple David Grangier Apple Ronan Collobert Apple

### ABSTRACT

We introduce SMALLTALK LM, an innovative method for training a mixture of language models in an almost asynchronous manner. Each model of the mixture specializes in distinct parts of the data distribution, without the need of highbandwidth communication between the nodes training each model. At inference, a lightweight router directs a given sequence to a single expert, according to a short prefix. This inference scheme naturally uses a fraction of the parameters from the overall mixture model. Unlike prior works on asynchronous LLM training, our routing method does not rely on full corpus clustering or access to metadata, making it more suitable for real-world applications. Our experiments on language modeling demonstrate that SMALLTALK LM achieves significantly lower perplexity than dense model baselines for the same total training FLOPs and an almost identical inference cost. Finally, in our downstream evaluations we outperform the dense baseline on 75% of the tasks.

### <span id="page-0-0"></span>1 INTRODUCTION

Recent research has demonstrated that scaling large language models (LLMs) by increasing model capacity and expanding training data consistently leads to significant performance improvements on a wide range of downstream tasks [\(Kaplan et al.,](#page-11-0) [2020;](#page-11-0) [Brown et al.,](#page-10-0) [2020;](#page-10-0) [Henighan et al.,](#page-11-1) [2020;](#page-11-1) [Hoffmann et al.,](#page-11-2) [2022;](#page-11-2) [Dubey et al.,](#page-11-3) [2024\)](#page-11-3). Scaling introduces substantial operating and engineering costs for both inference and training. In general, training is achieved on a large number of nodes via synchronous gradient descent techniques, which relies on high-bandwidth communication to scale. Inference of large models may require multiple compute nodes to distribute the model, which relies on low-latency communication. In both cases, state-of-the-art interconnect hardware is critical, and careful engineering is required to scale, and maintain a large compute cluster. While mainstream machine learning frameworks have eased the engineering work on the scaling side, access to a large number of well interconnected nodes remains a privilege in the machine learning community.

In this paper, we explore strategies to mitigate the communication cost of large language models, both at training and inference, while keeping the inference efficient. We show that efficient training and inference can be achieved without relying on fast interconnects, and without compromising model performance, both in terms of perplexity or downstream task accuracy.

In recent studies aimed at mitigating reliance on high-bandwidth interconnects, researchers have developed algorithms that reduce the need for frequent or comprehensive gradient synchronizations. Such techniques include asynchronous training [\(Douillard et al.,](#page-10-1) [2023;](#page-10-1) [Aji & Heafield,](#page-10-2) [2017;](#page-10-2) [Zhang](#page-13-0) [et al.,](#page-13-0) [2015;](#page-13-0) [Liu et al.,](#page-12-0) [2024\)](#page-12-0) and gradient compression [\(Lin et al.,](#page-12-1) [2017;](#page-12-1) [Dettmers,](#page-10-3) [2016\)](#page-10-3) which effectively decrease communication overhead. By performing updates less frequently or by communicating less data, these methods sustain high training throughput while diminishing dependence on high-speed interconnects. However, these algorithms still require some level of gradient synchronization, and resulting models often under perform (in terms of perplexity) compared to training approaches synchronizing at every step [\(Diskin et al.,](#page-10-4) [2021\)](#page-10-4).

Regarding efficient inference, a number of sparse parameter activation techniques have recently become popular [\(Shazeer et al.,](#page-13-1) [2017;](#page-13-1) [Fedus et al.,](#page-11-4) [2022;](#page-11-4) [Artetxe et al.,](#page-10-5) [2022;](#page-10-5) [Lewis et al.,](#page-12-2) [2021;](#page-12-2) [Du et al.,](#page-11-5) [2022\)](#page-11-5), in particular the Switch Transformer mixture of experts (MoE). These approaches

<sup>†</sup>Work done while interning at Apple.

are effective at reducing significantly the number of active model parameters at inference time. Yet, they require routing decisions to be made for each token, demanding rapid interconnections and requiring all parameters to be accessible in RAM.

To achieve both low-bandwidth training and efficient inference, several prior works have explored asynchronously training mixtures of models (Gross et al., 2017; Li et al., 2022; Gururangan et al., 2023). In these methods, the input space is clustered using unsupervised techniques like K-Means, with each model in the mixture trained on an assigned cluster. These approaches reduce communication overhead and have demonstrated improvements in model perplexity, while maintaining similar inference costs to baseline models (Gross et al., 2017; Gururangan et al., 2023). However, in the context of language processing, their practical applicability on standard downstream tasks with relatively short input prefixes remains a subject of debate, as they rely on full corpus clustering or access to metadata (Li et al., 2022; Gururangan et al., 2023).

In this paper, we introduce SMALLTALK LM, an asynchronous mixture of language models, which combines the advantage of asynchronous LM training methods (significantly reducing interconnect requirements), and sparse activation methods (it naturally uses only a subset of the overall architecture parameters during inference). SMALLTALK LM achieves better perplexity, and better accuracy on a majority of downstream tasks, compared to a regular dense language model trained with the same amount of FLOPs. Our main contributions are as follows:

- We introduce an algorithm that trains a mixture of independent language models, significantly reducing bandwidth requirements during training compared to regular distributed training methods. Our approach leverages a lightweight router (which accounts for less than 1.5% of the size of each expert) to efficiently route sequences. This router determines the most suitable expert for a given sequence based on a short prefix (see § 2).
- We empirically demonstrate that our method achieves significantly lower perplexity than a dense baseline for near-identical training and inference FLOPs and identical training data volume (see Fig. 2). Furthermore, we show that for a constant mixture model size, the improvement in perplexity increases with the number of experts.
- We evaluate our method on a set of downstream tasks, showing that it achieves better or similar accuracy compared to a dense baseline on 75% of the tasks (see § 3.3 and App. B).

#### <span id="page-1-0"></span>2 Method

In this section, we formalize our proposed method: SMALLTALK LM. In § 2.1, we introduce general language modeling background, and present a mixture of experts framework in this context. Subsequently, in § 2.2, we show how we can use independent language models to perform routing to different experts in the mixture. We also present the training procedure and implementation details of our method. Finally, we discuss its marginal computational overhead and minimal distributed communication costs.

#### <span id="page-1-1"></span>2.1 BACKGROUND

Language modeling Let  $\mathcal V$  denote a fixed vocabulary, and consider a sequence of tokens  $\mathbf x=(x_1,x_2,\dots,x_S)$ , where each  $x_s\in\mathcal V$  for  $s=1,2,\dots,S$ . Language modeling is about modeling the data distribution  $p(\mathbf x)$ , which in practice is cast as training a language model  $p(x_{s+1}\mid \mathbf x_{1:s};\theta)$  generating a token  $x_{s+1}$  given a context sequence  $\mathbf x_{1:s}=(x_1,x_2,\dots,x_s)$ . Typically, the model parameterized by  $\theta$  is a neural network, and parameters are obtained by minimizing the negative log-likelihood:

<span id="page-1-2"></span>
$$\mathcal{L}(\mathbf{x}; \theta) = -p(\mathbf{x}; \theta) = -\sum_{s=1}^{S-1} \log p(x_{s+1} \mid \mathbf{x}_{1:s}; \theta).$$
 (1)

**Mixture of Experts** There exist many variants of the mixture of experts model (Jacobs et al., 1991; Jordan & Jacobs, 1994; Tresp, 2000; Collobert et al., 2001; Shahbaba & Neal, 2009). Following the original formulation and applying it to language modeling, we factorize the conditional distribution for predicting a given token by summing over a set of E experts:

<span id="page-2-1"></span>
$$\mathcal{L}(\mathbf{x}; \theta) = -\sum_{s=1}^{S-1} \log \sum_{e=1}^{E} p(x_{s+1} \mid \mathbf{x}_{1:s}, e; \theta^e) \, p(e \mid \mathbf{x}_{1:s}; \theta^r), \tag{2}$$

where each expert  $p(x_{s+1} \mid \mathbf{x}_{1:s}, e; \theta^e)$  is intended to focus on different parts of the data distribution, and the router  $p(e \mid \mathbf{x}_{1:s}; \theta^r)$  assigns weights to the expert predictions. Each expert e has its own set of independent parameters  $\theta^e$ , and the router is parameterized by  $\theta^r$ .

While this type of mixture of experts partitions the input space across different models, all experts are still involved when computing the conditional likelihood in Equation (2), both at training and inference time. Hard mixtures of experts circumvent this issue by hard-assigning a sequence  $\mathbf{x}_{1:s}$  to a *single* expert  $e^*$ , minimizing an upper bound of Equation (2):

$$\mathcal{L}(\mathbf{x};\theta) \le -\sum_{s=1}^{S-1} \log p(x_{s+1} \mid \mathbf{x}_{1:s}, e^{\star}; \theta^{e^{\star}}) \, p(e^{\star} \mid \mathbf{x}_{1:s}; \theta^r), \tag{3}$$

The way  $e^{\star}$  is chosen depends of the goal of the approach. For example, in Jacobs et al. (1991),  $e^{\star}$  is sampled at training time, according to the distribution  $p(e \mid \mathbf{x}_{1:s}; \theta^r)$ . In other works, like Collobert et al. (2001), it is chosen as the most relevant expert (that is  $e^{\star} = \arg\max_{e} p(e \mid \mathbf{x}_{1:s}; \theta^r)$ ), both at training and inference. Regarding the training procedure, mixture models with both soft and hard assignments are typically trained with an Expectation-Minimization (EM) scheme, alternating between experts and router optimization.

#### <span id="page-2-0"></span>2.2 SMALLTALK LM

**Routing With Independent Language Models** SMALLTALK LM can be viewed as an instance of hard mixture of experts, where a sequence is routed to the most relevant expert  $e^*$ , according to the router posterior  $p(e \mid \mathbf{x}_{1:s}; \theta^r)$ , both at training and inference. However, instead of being a monolithic neural network model, the router is implemented as a language model per expert. More precisely, with Bayes' rule we have:

$$e^* = \arg\max_{e} p(e \mid \mathbf{x}_{1:s}; \theta^r)$$
 (4)

<span id="page-2-2"></span>
$$= \underset{e}{\operatorname{arg\,max}} \frac{p(\mathbf{x}_{1:s} \mid e ; \theta^r) \, p(e)}{\sum_{i=1}^{E} p(\mathbf{x}_{1:s} \mid i ; \theta^r) \, p(i)}$$
 (5)

$$= \arg\max_{e} p(\mathbf{x}_{1:s} \mid e; \theta^r)$$
 (6)

$$= \arg\max_{e} p(\mathbf{x}_{1:s} \mid \theta^{r,e}), \tag{7}$$

where the likelihood  $p(\mathbf{x}_{1:s} \mid \theta^{r,e})$  defines a routing language model attributed to the e-th expert with *independent* parameters  $\theta^{r,e}$ , and expert priors p(e) are supposed to be uniform. So far, in our definition, there is nothing that prevents  $\theta^{r,e} = \theta^e$ , namely to use the experts themselves to implement the router. Interestingly, in that case selecting  $e^*$  as above is intuitive because it amounts to selecting the best expert for a given sequence (in terms of log-likelihood).

**Deriving a Practical Model** In order for the above formulation to result in a model that is useful in practice, we have to solve the following main challenges. Firstly, we need to be able to route using a small prefix of the full sequence since in most real world applications we do not have access to a full sequence (e.g. in question answering we do not have the answer). Secondly, using the experts to implement the router results in a large computational overhead since we need to evaluate every single one of them, largely negating the benefits of the hard mixture of experts.

We solve both of the above problems with the following two key ideas:

1. We use a *short prefix* of length M to compute the score for each expert as follows:

$$p(e^{\star} \mid \mathbf{x}_{1:s}) \approx p(e^{\star} \mid \mathbf{x}_{1:M}). \tag{8}$$

This enables us to use SMALLTALK LM in real-world applications where only a very small prefix may be available. We show in § 3.4 that our method outperforms the dense baseline even when using as little as 32 tokens for routing.

2. We implement the router with E language models that are *orders of magnitude smaller* than the experts. This results in marginal computational overhead, as little as 1% during training and 3% during inference (§ 3.2).

Using the log-likelihood of a prefix to route a sequence is an intuitive solution since a model that is good at predicting the beginning of a sequence is likely to be good at the full sequence. Another interpretation is that the prefix can be used to perform some sort of thematic identification of the whole sequence that is then used to route to the appropriate expert.

On the other hand, the efficacy of smaller language models as routers is less intuitive. Even though one can expect a large model to be able to perform well on sequences that are beyond the capabilities of smaller models, the task of routing boils down to identifying whether a sequence is similar to the training data or not. We show that even really small models (4.4M parameters) can perform that task very well and result in no discernible difference in performance compared to larger routers (§ 3.4).

#### <span id="page-3-1"></span>Algorithm 1 SMALLTALK LM training.

```
1: Train the routers
 2: \mathbf{X} \leftarrow N new sequences from the dataset
 3: \mathbf{X}_{1:E} = \text{random\_assignments}(\mathbf{X})
 4: for i = 1 ... T do
          for e=1\dots E do
               \theta^{r,e} \approx \arg\min_{\theta^{r,e}} \mathcal{L}(\mathbf{X}_e; \theta^{r,e})
 6:
                                                               # Optimize Equation 9 with SGD for the e-th router
 7:
          \mathbf{X} \leftarrow N new sequences from the dataset
          \mathbf{X}_{1:E} = \text{balanced\_assignments}(\mathbf{X}, \theta^r)
                                                                          # Segment the data according to Equation 4
10: end for
11: Train the experts
12: X \leftarrow M new sequences from the dataset comprising the total number of training tokens
13: \mathbf{X}_{1:E} = \text{balanced\_assignments}(\mathbf{X}, \theta^r)
14: for e = 1 ... E do
          \theta^e \approx \arg\min_{\theta^e} \mathcal{L}(\mathbf{X}_e; \theta^e)
                                                              # Optimize Equation 1 with SGD for the e-th expert
16: end for
```

**Training Procedure** We train our proposed hard mixture of experts with EM. Namely we alternate between a step of optimizing equation 1 and a step of assignments according to equation 4. In our framework, assignments  $e^*$  purposefully do not depend on the experts, and the routers are independent language models. This allows us to split the training in two stages:

1. We train the routers using EM without using the experts at all, namely we minimize

<span id="page-3-0"></span>
$$\mathcal{L}(\mathbf{x}, \theta^r) = -\log p(e^* \mid \mathbf{x}_{1:M}; \theta^r) = -\sum_{s=1}^{M-1} \log p\left(x_{s+1} | \mathbf{x}_{1:s}; \theta^{r,e^*}\right), \tag{9}$$

in the likelihood maximization step and perform the assignments using equation 4 in the expectation step.

2. We train the experts as E independent language models on disjoint dataset segments selected by the trained routers.

This procedure is described in pseudocode in Algorithm 1. In the specific case where the experts and the routers are the same language models ( $\theta^e = \theta^{r,e}$ ) our training algorithm comes down to a regular EM algorithm where the sequences are routed to the best performing expert. In that light, we can think of each router as lightweight approximation of the corresponding expert. See § 3.4 for experiments where the experts themselves are used for routing.

**Balancing the Assignments to Experts** A common hurdle to overcome when training mixture of experts models is to ensure that the modeling load is shared as evenly as possible across all experts, or in other words that the model is utilizing the capacity of all experts efficiently instead of relying on a few good ones. Common ways of tackling this problem are either introducing losses that penalize

<span id="page-4-0"></span>![](_page_4_Figure_1.jpeg)

Figure 1: Balanced assignments. In this scenario we have 3 experts (columns with different colors) and 3 sequences to assign (rows) under the constraint that each expert should get 1 sequence. In (a) we assign sequentially each row, by negative log-likelihood, leading to sub-optimal assignment because the first expert is full when we try to assign the last row. On the other hand, in (b), we first sort wrt to the minimum log-likelihood which results in optimal assignments.

uneven assignments or introducing noise into the assignments namely by sometimes performing "sub-optimal" assignments to avoid local-minima. These techniques are used in all mixture models since their introduction in [Jacobs et al.](#page-11-8) [\(1991\)](#page-11-8) until more recently in [Fedus et al.](#page-11-4) [\(2022\)](#page-11-4).

In our setup, we can solve this problem by ensuring that each expert is assigned equally sized distinct portions of the dataset. However, a challenge arises when assigning data sequentially based solely on Equation [4.](#page-2-2) In such cases, an expert might already have more than its fair share of the dataset assigned yet we continue to find sequences for which it has the lowest loss among all experts, as illustrated in Figure [1a.](#page-4-0) To overcome this issue, during training we perform the assignments considering the whole set of sequences rather than one sequence at a time. In particular, we sort the sequences based on − max<sup>e</sup> log p(x1:<sup>M</sup> | e; θ r ) and perform the assignments in that order. This ensures that sequences with high likelihood will be assigned first and avoids the edge-case where a sequence with low-loss cannot be assigned to the appropriate expert because it is at capacity. In Algorithm [1](#page-3-1) we refer to this procedure as *balanced assignments* and it is demonstrated in Figure [1b.](#page-4-0)

During inference, no balancing is performed, and the expert is selected solely based on equation [4.](#page-2-2)

Computational and Communication Cost It is important to note that the computational cost of training and inference with the routers is significantly lower than that of training and inference with the experts. This difference arises due to the routers' much smaller size, smaller batch sizes, and the fewer tokens they train on (see Appendix [A.3](#page-16-0) for further details). As a result, the additional computational overhead from training and inference with the routers is negligible.

Regarding communication requirements for a distributed implementation, in Algorithm [1](#page-3-1) all operations except for the balanced assignments are completely independent and embarrassingly parallelizable without any communication. Namely, each expert or each router can be trained on its own on a different node or set of nodes. The assignments, however, require that every node has access to the scores of every router in order to perform the assignment according to equation [4.](#page-2-2) This requires a small amount of communication as each node needs to share 1 score per sequence which results in 2MB per 1M sequences if we represent the scores in 16-bit floats. A detailed analysis is provided in the Appendix [A.4.](#page-19-0)

### <span id="page-4-1"></span>3 EXPERIMENTS

In this section, we experimentally analyze the performance of our proposed method. First, in § [3.2](#page-5-0) we benchmark our approach against a dense LLM trained using regular distributed training techniques. Second, in § [3.3,](#page-6-1) we evaluate our method on downstream tasks, demonstrating that it achieves performance comparable to baseline models at the same level of perplexity, improving upon a dense model given the total training FLOPs spent. Finally, in § [3.4,](#page-7-0) we analyse the core component of our method, the routing. We show that the performance of the mixture does not depend significantly on the size of the router. Furthermore, we show that our approach outperforms less complex routing methods, such as the TF-IDF encoding with K-Means clustering proposed by [Gururangan et al.](#page-11-7) [\(2023\)](#page-11-7). Moreover, we demonstrate that the perplexity gains are not driven by a few dominant experts but result from each expert contributing positively to the overall performance.

### <span id="page-5-1"></span>3.1 EXPERIMENTAL SETUP

We use the RedPajama-V2 dataset [\(Computer,](#page-10-7) [2023\)](#page-10-7), which is a large-scale collection of text data designed for training language models. The dataset is built from the ground up based on publicly available web data, consisting of 84 crawls provided by [Common Crawl](#page-10-8) [\(2024\)](#page-10-8).

Both the experts and routers utilize a Transformer architecture [\(Vaswani,](#page-13-4) [2017\)](#page-13-4) with rotary positional encoding [\(Su et al.,](#page-13-5) [2024\)](#page-13-5). All experiments use sequences of 1, 024 tokens. We train the models using the AdamW optimizer [\(Loshchilov & Hutter,](#page-12-4) [2017\)](#page-12-4) with β<sup>1</sup> = 0.9, β<sup>2</sup> = 0.99, and a weight decay of 0.1. Gradient clipping is applied with a maximum norm of 0.1.

For the experts, we employ a learning rate schedule featuring a linear warm-up to 5 × 10−<sup>4</sup> over the first 3, 000 steps, followed by a cosine decay for the remainder of the training period. We experiment with two model sizes: a 335M-parameter model with 4, 8, 16, and 32 experts and a 1.3B-parameter model with 4, 16, and 32 experts. Tables [1](#page-15-0) and [2](#page-15-1) describe the architectures and the training parameters in more detail.

For the routers, we use models with 4.4M parameters. The routers are trained for 128, 000 steps using a constant learning rate of 1 × 10<sup>−</sup><sup>4</sup> , following a linear warm-up over the first 1, 000 steps. Scheduler choice described in details in App. [A.](#page-14-0) Routers are trained with a batch size of 32 and perform all-to-all communication of the loss on the dataset chunk approximately every 45 million training tokens (see App. [A.4\)](#page-19-0). The length of the prefix used for routing is set to 256 tokens, which is 25% of the context length.

Comparison to the Dense Model To ensure a fair comparison to the dense baseline, we designed our experiments such that the computational cost during inference and the total computational cost during training are the same. We achieve this by comparing the performance of models where each expert has the same architecture and number of parameters as the dense model, resulting in the same inference cost. In addition, we train SMALLTALK LM for the same number of total tokens which means we spend the same number of total training FLOPs.

#### <span id="page-5-0"></span>3.2 LANGUAGE MODELING RESULTS

Across all experimental configurations, our method *consistently outperforms the baseline* while utilizing the same volume of data, with only minor increases in computational costs. Specifically, for the 1.3B parameter model, our approach results in less than a 1% increase in training FLOPs and less than a 3% increase in inference FLOPs, while achieving performance gains of up to almost 18% in perplexity. Similarly, for the 335M parameter model, the cost increases are modest, with less than a 4% increase in training FLOPs and less than a 10% increase in inference FLOPs, alongside performance gains of up to nearly 14% in perplexity (see Appendix [A.3](#page-16-0) for details on FLOPs calculation). Figures [2a](#page-6-0) and [2b](#page-6-0) illustrate the perplexity on a held-out test set relative to the total training cost, measured in petaFLOPs, across different model sizes. Additionally, Figure [2c](#page-6-0) shows the test perplexity as a function of the number of training tokens for the 1.3B parameter model.

Moreover, it is interesting to compare our model with the smaller 335M parameter experts to the 1.3B dense baseline. When using 32 experts and roughly the same training budget in FLOPs, SMALLTALK LM achieves slightly better perplexity (9.07 vs 9.11) but requires three times less compute during inference.

Finally, we observed that for a constant model size, the improvement in perplexity increases with the number of experts. This suggests that our method more efficiently utilizes the training data to improve performance, likely due to the specialization of experts on different segments of the data distribution.

Communication Overhead Our approach performs minimal communication and does not require a fast interconnect. As detailed in § [2,](#page-1-0) we first train the routers which are then used to efficiently shard the dataset and each expert is trained completely independently. During their training, the routers communicate approximately 100 times with each transmission involving less than 6

<span id="page-6-0"></span>![](_page_6_Figure_1.jpeg)

- (a) FLOPs vs perplexity, 335M.
- (b) FLOPsvs perplexity, 1.3B.
- (c) Tokens vs perplexity, 1.3B.

Figure 2: Better perplexity for the same price. Test perplexity comparison between our approach and the dense baseline, as a function of training cost measured in PFLOPs. In (a), we report results for models with 335M parameters using 4, 8, 16, and 32 experts and in (b) for models with 1.3B parameters using 4, 16, and 32 experts. In addition, (c) shows the perplexity comparison between our approach and the dense baseline, plotted against the cumulative number of tokens processed throughout the training for the 1.3B parameter models. We observe that our method significantly outperforms the baseline across all experimental configurations. Notably, our 335M parameter model with 32 experts achieves a perplexity of 9.07, outperforming the 1.3B dense baseline's perplexity of 9.1. This improvement is achieved with a training budget of  $2.5 \times 10^{21}$  FLOPs, which is comparable to the baseline's  $2.2 \times 10^{21}$  FLOPs, while requiring three times less computational cost during inference  $(0.87 \times 10^{12} \text{ FLOPs})$  compared to  $2.81 \times 10^{12} \text{ FLOPs})$ . See § 3.2 and App. A for a detailed description of our experimental setup.

<span id="page-6-2"></span>![](_page_6_Figure_6.jpeg)

Figure 3: **Downstream evaluation.** Accuracy with respect to perplexity on (a) ARC Challenge, (b) ARC Easy, (c) HellaSwag and (d) MMLU, for 1.3B parameter dense baselines trained on 266B, 1T and 2T tokens (empty symbols) and mixture models with 1.3B parameter experts and 4, 16 and 32 experts respectively (filled symbols). The models that have the same symbol shape have near identical training and inference FLOPs.

megabytes per router resulting in truly minimal communication requirements. See A.4 for details about the calculation of these numbers.

#### <span id="page-6-1"></span>3.3 DOWNSTREAM EVALUATION

To demonstrate the applicability of our approach in real-world scenarios, we performed zero-shot evaluation on a set of natural language processing (NLP) tasks, including ARC Challenge and ARC Easy (Clark et al., 2018), HellaSwag (Zellers et al., 2019), SciQ (Welbl et al., 2017), and MMLU (Hendrycks et al., 2020). Figure 3 shows both accuracy and perplexity for each task (see Appendix B for details regarding perplexity and accuracy computation).

Specifically, our largest configuration, a model with 32 1.3B parameter experts, demonstrates better performance on four out of five tasks, achieving gains of 3%, 2%, 3%, and 1% on ARC Challenge (3a), ARC Easy (3b), HellaSwag (3c), and MMLU (3d), respectively, under the same inference cost.

Moreover, on the MMLU benchmark, our approach either beats the baseline or achieves the same performance on 75% of the tasks (42 out of 56), as shown in Table 4.

<span id="page-7-1"></span>![](_page_7_Figure_1.jpeg)

Figure 4: **Routing analysis.** (a) Test perplexity over training steps for different router sizes using a 335M parameter model with 4 experts. We compare routers of sizes 335M (where the model routes data for itself), 110M, 65M, and 4.4M parameters. (b) Test perplexity as a function of routing prefix length during inference for 1.3B parameter model with 4, 16 and 32 experts. We examine how reducing the prefix length  $\hat{M}$  used during inference affects performance when the data is partitioned during training using a prefix size  $M \ge \hat{M}$ . (c) Test perplexity over training steps for a 335M parameter model with 16 experts, comparing our proposed routing using TF-IDF document encoding followed by SVD projection and balanced K-Means clustering.

#### <span id="page-7-0"></span>3.4 ROUTING ANALYSIS

In this section we analyse the most critical component of SMALLTALK LM, the routing. In particular, we perform experiments to investigate the impact of the size of the router to the mixture model, the impact of the size of the prefix and finally whether the EM algorithm presented in § 2 is critical or it could be replaced by simple content based clustering.

**Impact of Router Size** One of the most critical findings of our work is the fact that *the size of the model* used for routing *does not impact* the performance of the mixture (Figure 4a). This allows us to utilize small router models which reduce dramatically the computational cost of routing during training and inference without compromising the model's performance.

In detail, we investigated the effect of router size on the performance of our method by experimenting with a 335M parameter model configured with 4 experts, using four different router sizes for data partitioning: a 335M parameter router (where the model routes data for itself, optimizing for the data it handles best), and routers with 110M, 64M, and 4.4M parameters. All models were trained with a batch size of 32 for 256, 000 steps, using a routing prefix length of 256 tokens, following the same experimental setup as in § 3.2 (see Appendix A for more details about model architectures and training parameters). Figure 4a illustrates the test perplexity wrt the number of training tokens for these configurations. We observe that all router sizes perform practically identically.

**Impact of Prefix Length** An important hyperparameter for our mixture model is the length of the context used for routing to the expert model. In our experiments, we use a prefix length of 256 which may be too large for some real-world applications like conversational AI. However, we show in Figure 4b that SMALLTALK LM outperforms the dense baseline even when routing with shorter prefixes. In particular, we show that even with prefixes of just 32 tokens ( $\frac{1}{8}$ th of the length during training) the model still outperforms the dense baseline.

Comparison With Routing Using TF-IDF An alternative approach might suggest that if a tiny LLM suffices for routing, simpler methods such as TF-IDF encoding combined with balanced K-Means clustering, as proposed by Gururangan et al. (2023), could be adequate. To test this hypothesis, we trained a 335M parameter model using the routing strategy outlined in (Gururangan et al., 2023): applying TF-IDF transformation to the text, followed by Singular Value Decomposition (SVD) projection into a low-dimensional space, and then clustering using balanced K-Means.

We then trained 16 experts on these clustered results. Our findings reveal that routing with TF-IDF encoding under performs when the prefix is short, indicating limitations of this simple routing approach. As shown in Figure 4c, our proposed routing method significantly outperforms the TF-

IDF-based routing. This demonstrates that our approach is more effective in capturing the nuances of the data distribution, even when using limited context for routing decisions.

Experts Do Specialize A potential concern is that the observed performance gains might be driven primarily by a few experts among the E, while the remaining contribute minimally or may even under perform on their assigned data. To address this concern, we conducted an analysis of each expert's performance individually. Figures 5a, 5b, and 5c compare the perplexity achieved by our method and the dense baseline on the routed dataset segments for the 1.3B parameter model, utilizing mixtures of 4, 16, and 32 experts, respectively.

Our findings demonstrate that all experts contribute positively, with consistent perplexity improvements observed across all routed dataset segments. Moreover, the performance gap between our method and the baseline widens as the number of experts increases, suggesting enhanced specialization due to finer partitioning of the data distribution.

Notably, despite the absence of capacity constraints on experts during inference, unlike the enforced capacities during training, all experts are actively utilized and receive substantial portions of the data. The consistent improvements across all routed dataset segments confirm that the experts do specialize and they all contribute to the overall improvement instead of only a few dominant experts.

<span id="page-8-0"></span>![](_page_8_Figure_5.jpeg)

Figure 5: **Experts Do specialize.** Test perplexity comparison between our method and the dense baseline on the routed dataset segments for the 1.3B parameter model, using mixtures of 4 experts in (a) (trained on 266B tokens), 16 experts in (b) (1T tokens), and 32 experts in (c) (2T tokens). Each bar represents a dataset segment, with the color intensity indicating the percentage of data assigned to that expert – darker shades correspond to a higher proportion of data. Overlapping bars depict the perplexity achieved by the dense baseline (translucent) and our proposed mixture model (opaque). The results demonstrate that all experts specialize effectively on their assigned segments of the data distribution, leading to consistent improvements over the baseline. While the data distribution among experts is not perfectly even – with some experts receiving more data than others – all experts receive a substantial portion of the data. This shows that each expert contributes meaningfully to the overall performance gains.

#### <span id="page-8-1"></span>4 RELATED WORK

Data-parallel training requires synchronization of the gradients after each backward pass, which can be slow if the network bandwidth is limited. In this section, we discuss existing methods that aim to make data parallelism more communication-efficient via reducing the frequency of the synchronization. Since our paper also explores methods for efficient inference without performance degradation, we discuss this branch of research as well.

**Distributed optimization** To reduce the frequency of synchronisation during training of large models, researchers have explored distributed optimization techniques both theoretically and empirically (Mcdonald et al., 2009; Stich, 2019; Lian et al., 2018; Wang & Joshi, 2019; McMahan et al., 2017; Lin et al., 2018; Zinkevich et al., 2010). The core principle of these methods is to allow each worker to execute several local training iterations before engaging in global synchronization – each device performs multiple optimizer updates per communication round. This technique also has been studied for LLM pre-training (Diskin et al., 2021; Douillard et al., 2023) and fine-tuning (Borzunov et al., 2022; Liu et al., 2024; Hilmkil et al., 2021; Ro et al., 2022).

While theses approaches reduces the frequency of communication, thereby decreasing communication overhead, they often result in degraded model performance compared to every step synchroniza-

tion [\(Diskin et al.,](#page-10-4) [2021\)](#page-10-4), or they lack direct performance comparisons in the context of pre-training [\(Douillard et al.,](#page-10-1) [2023\)](#page-10-1). The degradation in performance is often attributed to the staleness of gradients and the divergence between local and global models due to delayed synchronization [\(Zhang](#page-13-11) [et al.,](#page-13-11) [2016;](#page-13-11) [Yu et al.,](#page-13-12) [2019;](#page-13-12) [Mitra et al.,](#page-12-10) [2021\)](#page-12-10). Moreover, to our knowledge, all experiments have been conducted at a small scale (≤ 400M parameters) [\(Douillard et al.,](#page-10-1) [2023;](#page-10-1) [Liu et al.,](#page-12-0) [2024;](#page-12-0) [Ro et al.,](#page-12-9) [2022\)](#page-12-9). Therefore, scalable solutions that can reduce communication frequency for LLM training without compromising performance at larger scales are still needed.

Mixture of Experts Mixture of Experts (MoE) models have emerged as a prominent approach to increase model capacity without a proportional increase in inference costs [\(Jacobs et al.,](#page-11-8) [1991;](#page-11-8) [Jordan & Jacobs,](#page-11-9) [1994\)](#page-11-9). The key idea behind MoE is to partition the input space into subsets corresponding to specific subtasks, each managed by a specialized expert. By training different experts on distinct tasks, MoE models aim to enhance performance while reducing computational overhead during inference.

Early implementations of MoE employed algorithms such as Gaussian Processes [\(Tresp,](#page-13-2) [2000;](#page-13-2) [Theis](#page-13-13) [& Bethge,](#page-13-13) [2015;](#page-13-13) [Deisenroth & Ng,](#page-10-11) [2015\)](#page-10-11), Dirichlet Processes [\(Shahbaba & Neal,](#page-13-3) [2009\)](#page-13-3), and Support Vector Machines (SVMs) [\(Collobert et al.,](#page-10-6) [2001\)](#page-10-6) to model these specialized functions. Recent advancements have integrated deep neural networks as experts, providing scalable and flexible implementations suitable for large-scale models.

In natural language processing, researchers have generalized the original MoE idea to enhance largescale language models. For instance, [Shazeer et al.](#page-13-1) [\(2017\)](#page-13-1) introduced a Mixture of Experts feed forward layer in which, in contrast to prior works, routing decision is made for every token. This layer takes a token representation as an input and routes it to the best-matched top-k experts. By doing so the number of active parameters is decreased. Similar approaches have been effectively employed in models like the Switch Transformer [\(Fedus et al.,](#page-11-4) [2022\)](#page-11-4), GShard [\(Lepikhin et al.,](#page-12-11) [2021\)](#page-12-11), GLaM [\(Du et al.,](#page-11-5) [2022\)](#page-11-5) and Mixtral of Experts [\(Jiang et al.,](#page-11-12) [2024\)](#page-11-12) which leverage MoE layers to achieve state-of-the-art results with reduced number of active parameters costs. However, since the routing decision in models like Switch Transformer MoE is made on the token level: (1) all experts must be retained in RAM during both training and inference phases, necessitating substantial memory resources, and (2) a high communication overhead is required for data transfer and gradient synchronization across experts [\(Lepikhin et al.,](#page-12-11) [2021;](#page-12-11) [Fedus et al.,](#page-11-4) [2022;](#page-11-4) [Riquelme et al.,](#page-12-12) [2021\)](#page-12-12). In that respect this approach is completely orthogonal to our proposed SMALLTALK LM. In fact, each of the experts in our method could be such a token-level mixture model which would result in even smaller inference FLOPs for a specific achieved perplexity albeit with a significantly more complex implementation and higher RAM requirements during inference.

Independent Training of Mixture of Experts Models To train experts fully in parallel, recent research has also explored independent training of the mixture of experts models [\(Gross et al.,](#page-11-6) [2017;](#page-11-6) [Li et al.,](#page-12-3) [2022;](#page-12-3) [Gururangan et al.,](#page-11-7) [2023\)](#page-11-7). For instance, [Gross et al.](#page-11-6) [\(2017\)](#page-11-6) proposed to train a gater independently from the experts and use the output of the gater to hard assign the input image to experts. In language processing, [Li et al.](#page-12-3) [\(2022\)](#page-12-3) and [Gururangan et al.](#page-11-7) [\(2023\)](#page-11-7) proposed methods that train expert models separately and merge them post-training. However, these approaches rely on full document context or metadata for routing, which can be impractical in real-time applications involving shorter inputs. For instance, [Gururangan et al.](#page-11-7) [\(2023\)](#page-11-7) focus on performance in text classification tasks, which may not fully reflect real-world scenarios.

### <span id="page-9-0"></span>5 CONCLUSION

In this work, we present SMALLTALK LM – a method that significantly reduces communication overhead during LLM training. It also outperforms dense model baselines trained with a similar amount of FLOPs and the same data volume and comparable inference costs. In addition, our proposed lightweight routing with short prefixes makes SMALLTALK LM practical for real-world downstream tasks, where we show it largely outperforms the dense baselines.

These findings open up multiple directions for future research, including exploring different routing strategies for data partitioning among experts and investigating further the impact on downstream task performance. Another promising avenue is scaling the mixture to hundreds or even thousands of experts, which could further enhance model performance while benefiting from sparse inference.

### ACKNOWLEDGMENTS

We thank Jagrid Digani and Awni Hannun for their insightful discussions on the algorithm and experiments. We thank Justin Deschenaux for his contributions to the early stages of the project.

### REFERENCES

- <span id="page-10-2"></span>Alham Fikri Aji and Kenneth Heafield. Sparse communication for distributed gradient descent. In Martha Palmer, Rebecca Hwa, and Sebastian Riedel (eds.), *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing*, pp. 440–445. Association for Computational Linguistics, September 2017.
- <span id="page-10-5"></span>Mikel Artetxe, Shruti Bhosale, Naman Goyal, Todor Mihaylov, Myle Ott, Sam Shleifer, Xi Victoria Lin, Jingfei Du, Srinivasan Iyer, Ramakanth Pasunuru, Giridharan Anantharaman, Xian Li, Shuohui Chen, Halil Akin, Mandeep Baines, Louis Martin, Xing Zhou, Punit Singh Koura, Brian O'Horo, Jeffrey Wang, Luke Zettlemoyer, Mona Diab, Zornitsa Kozareva, and Veselin Stoyanov. Efficient large scale language modeling with mixtures of experts. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.), *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pp. 11699–11732. Association for Computational Linguistics, December 2022.
- <span id="page-10-10"></span>Alexander Borzunov, Dmitry Baranchuk, Tim Dettmers, Max Ryabinin, Younes Belkada, Artem Chumachenko, Pavel Samygin, and Colin Raffel. Petals: Collaborative inference and fine-tuning of large models. *Association for Computational Linguistics*, 2022.
- <span id="page-10-12"></span>James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs, 2018.
- <span id="page-10-0"></span>Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. *Advances in Neural Information Processing Systems*, 33:1877–1901, 2020.
- <span id="page-10-9"></span>Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. *arXiv preprint arXiv:1803.05457*, 2018.
- <span id="page-10-6"></span>Ronan Collobert, Samy Bengio, and Yoshua Bengio. A parallel mixture of svms for very large scale problems. *Advances in Neural Information Processing Systems*, 14, 2001.
- <span id="page-10-8"></span>Common Crawl. Common crawl: Open web data, 2024. Accessed: 2024-09-15.
- <span id="page-10-7"></span>Together Computer. Redpajama: An open source recipe to reproduce llama training dataset, 2023.
- <span id="page-10-11"></span>Marc Deisenroth and Jun Wei Ng. Distributed gaussian processes. In *International conference on machine learning*, pp. 1481–1490. PMLR, 2015.
- <span id="page-10-3"></span>Tim Dettmers. 8-bit approximations for parallelism in deep learning. *International Conference on Learning Representations*, 2016.
- <span id="page-10-4"></span>Michael Diskin, Alexey Bukhtiyarov, Max Ryabinin, Lucile Saulnier, Anton Sinitsin, Dmitry Popov, Dmitry V Pyrkin, Maxim Kashirin, Alexander Borzunov, Albert Villanova del Moral, et al. Distributed deep learning in open collaborations. *Advances in Neural Information Processing Systems*, 34:7879–7897, 2021.
- <span id="page-10-1"></span>Arthur Douillard, Qixuan Feng, Andrei A Rusu, Rachita Chhaparia, Yani Donchev, Adhiguna Kuncoro, Marc'Aurelio Ranzato, Arthur Szlam, and Jiajun Shen. Diloco: Distributed lowcommunication training of language models. *arXiv preprint arXiv:2311.08105*, 2023.

- <span id="page-11-5"></span>Nan Du, Yanping Huang, Andrew M Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, et al. Glam: Efficient scaling of language models with mixture-of-experts. In *International Conference on Machine Learning*, pp. 5547– 5569. PMLR, 2022.
- <span id="page-11-3"></span>Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. *arXiv preprint arXiv:2407.21783*, 2024.
- <span id="page-11-4"></span>William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research*, 23(120):1–39, 2022.
- <span id="page-11-14"></span>Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac'h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 07 2024.
- <span id="page-11-6"></span>Sam Gross, Marc'Aurelio Ranzato, and Arthur Szlam. Hard mixtures of experts for large scale weakly supervised vision. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pp. 6865–6873, 2017.
- <span id="page-11-7"></span>Suchin Gururangan, Margaret Li, Mike Lewis, Weijia Shi, Tim Althoff, Noah A Smith, and Luke Zettlemoyer. Scaling expert language models with unsupervised domain discovery. *arXiv preprint arXiv:2303.14177*, 2023.
- <span id="page-11-10"></span>Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. *arXiv preprint arXiv:2009.03300*, 2020.
- <span id="page-11-1"></span>Tom Henighan, Jared Kaplan, Mor Katz, Mark Chen, Christopher Hesse, Jacob Jackson, Heewoo Jun, Tom B Brown, Prafulla Dhariwal, Scott Gray, et al. Scaling laws for autoregressive generative modeling. *arXiv preprint arXiv:2010.14701*, 2020.
- <span id="page-11-11"></span>Agrin Hilmkil, Sebastian Callh, Matteo Barbieri, Leon Rene S ´ utfeld, Edvin Listo Zec, and Olof ¨ Mogren. Scaling federated learning for fine-tuning of large language models. In *International Conference on Applications of Natural Language to Information Systems*, pp. 15–23. Springer, 2021.
- <span id="page-11-2"></span>Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. *arXiv preprint arXiv:2203.15556*, 2022.
- <span id="page-11-8"></span>Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. Adaptive mixtures of local experts. *Neural computation*, 3(1):79–87, 1991.
- <span id="page-11-12"></span>Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. *arXiv preprint arXiv:2401.04088*, 2024.
- <span id="page-11-9"></span>Michael I Jordan and Robert A Jacobs. Hierarchical mixtures of experts and the em algorithm. *Neural computation*, 6(2):181–214, 1994.
- <span id="page-11-0"></span>Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. *CoRR*, 2020.
- <span id="page-11-13"></span>Taku Kudo and John Richardson. SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. In Eduardo Blanco and Wei Lu (eds.), *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, pp. 66–71, Brussels, Belgium, November 2018. Association for Computational Linguistics.

- <span id="page-12-11"></span>Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. In *International Conference on Learning Representations*, 2021.
- <span id="page-12-2"></span>Mike Lewis, Shruti Bhosale, Tim Dettmers, Naman Goyal, and Luke Zettlemoyer. Base layers: Simplifying training of large, sparse models. In *International Conference on Machine Learning*, pp. 6265–6274. PMLR, 2021.
- <span id="page-12-3"></span>Margaret Li, Suchin Gururangan, Tim Dettmers, Mike Lewis, Tim Althoff, Noah A Smith, and Luke Zettlemoyer. Branch-train-merge: Embarrassingly parallel training of expert language models. *arXiv preprint arXiv:2208.03306*, 2022.
- <span id="page-12-6"></span>Xiangru Lian, Ce Huang, Yijun Li, and Ji Liu. Asynchronous decentralized parallel stochastic gradient descent. In *Proceedings of the 35th International Conference on Machine Learning*, pp. 3043–3052. PMLR, 2018.
- <span id="page-12-1"></span>Yujun Lin, Song Han, Huizi Mao, Yu Wang, and William J Dally. Deep gradient compression: Reducing the communication bandwidth for distributed training. *arXiv preprint arXiv:1712.01887*, 2017.
- <span id="page-12-8"></span>Yujun Lin, Song Han, Huizi Mao, Yu Wang, and William J Dally. Deep gradient compression: Reducing the communication bandwidth for distributed training. In *International Conference on Learning Representations*, 2018.
- <span id="page-12-0"></span>Bo Liu, Rachita Chhaparia, Arthur Douillard, Satyen Kale, Andrei A Rusu, Jiajun Shen, Arthur Szlam, and Marc'Aurelio Ranzato. Asynchronous local-sgd training for language modeling. *arXiv preprint arXiv:2401.09135*, 2024.
- <span id="page-12-4"></span>Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In *International Conference on Learning Representations*, 2017.
- <span id="page-12-5"></span>Ryan Mcdonald, Mehryar Mohri, Nathan Silberman, Dan Walker, and Gideon Mann. Efficient large-scale distributed training of conditional maximum entropy models. *Advances in neural information processing systems*, 22, 2009.
- <span id="page-12-7"></span>H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-efficient learning of deep networks from decentralized data. In *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics*, pp. 1273–1282. PMLR, 2017.
- <span id="page-12-10"></span>Praneeth Mitra, Idan Bistritz, Ji Wang, Dan Alistarh, and Wotao Yin. Linear convergence in federated learning: Tackling client heterogeneity and asynchronous updates. In *Advances in Neural Information Processing Systems*, volume 34, pp. 14606–14619, 2021.
- <span id="page-12-13"></span>Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, highperformance deep learning library. *Advances in neural information processing systems*, 32, 2019.
- <span id="page-12-14"></span>Pitch Patarasuk and Xin Yuan. Bandwidth optimal all-reduce algorithms for clusters of workstations. *Journal of Parallel and Distributed Computing*, 69(2):117–124, 2009.
- <span id="page-12-15"></span>Ofir Press, Noah A. Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables input length extrapolation. In *The Tenth International Conference on Learning Representations*. OpenReview.net, 2022.
- <span id="page-12-12"></span>Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, Andre´ Susano Pinto, Daniel Keysers, and Neil Houlsby. Scaling vision with sparse mixture of experts. *Advances in Neural Information Processing Systems*, 34:8583–8595, 2021.
- <span id="page-12-9"></span>Jae Hun Ro, Theresa Breiner, Lara McConnaughey, Mingqing Chen, Ananda Theertha Suresh, Shankar Kumar, and Rajiv Mathews. Scaling language model size in cross-device federated learning. *arXiv preprint arXiv:2204.09715*, 2022.

- <span id="page-13-3"></span>Babak Shahbaba and Radford Neal. Nonlinear models using dirichlet process mixtures. *Journal of Machine Learning Research*, 10(8), 2009.
- <span id="page-13-1"></span>Noam Shazeer, \*Azalia Mirhoseini, \*Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In *International Conference on Learning Representations*, 2017.
- <span id="page-13-8"></span>Sebastian U Stich. Local SGD converges fast and communicates little. In *International Conference on Learning Representations*, 2019.
- <span id="page-13-5"></span>Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. *Neurocomputing*, 568:127063, 2024.
- <span id="page-13-14"></span>Rajeev Thakur, Rolf Rabenseifner, and William Gropp. Optimization of collective communication operations in mpich. *The International Journal of High Performance Computing Applications*, 19 (1):49–66, 2005.
- <span id="page-13-13"></span>Lucas Theis and Matthias Bethge. Generative image modeling using spatial lstms. *Advances in neural information processing systems*, 28, 2015.
- <span id="page-13-2"></span>Volker Tresp. Mixtures of gaussian processes. *Advances in neural information processing systems*, 13, 2000.
- <span id="page-13-4"></span>A Vaswani. Attention is all you need. *Advances in Neural Information Processing Systems*, 2017.
- <span id="page-13-9"></span>Jianyu Wang and Gauri Joshi. Adaptive communication strategies to achieve the best error-runtime trade-off in local-update sgd. *Proceedings of Machine Learning and Systems*, 1:212–229, 2019.
- <span id="page-13-7"></span>Johannes Welbl, Nelson F. Liu, and Matt Gardner. Crowdsourcing multiple choice science questions. In Leon Derczynski, Wei Xu, Alan Ritter, and Tim Baldwin (eds.), *Proceedings of the 3rd Workshop on Noisy User-generated Text*, pp. 94–106. Association for Computational Linguistics, September 2017.
- <span id="page-13-12"></span>Hao Yu, Sen Yang, and Shenghuo Zhu. Parallel restarted sgd with faster convergence and less communication: Demystifying why model averaging works for deep learning. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 33, 2019.
- <span id="page-13-6"></span>Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In *Annual Meeting of the Association for Computational Linguistics*, 2019.
- <span id="page-13-0"></span>Sixin Zhang, Anna E Choromanska, and Yann LeCun. Deep learning with elastic averaging sgd. *Advances in neural information processing systems*, 28, 2015.
- <span id="page-13-11"></span>Wei Zhang, Suyog Gupta, Xiangru Lian, and Ji Liu. Staleness-aware async-sgd for distributed deep learning. In *Proceedings of the 25th International Joint Conference on Artificial Intelligence*, pp. 2350–2356, 2016.
- <span id="page-13-10"></span>Martin Zinkevich, Markus Weimer, Lihong Li, and Alex Smola. Parallelized stochastic gradient descent. *Advances in neural information processing systems*, 23, 2010.

### CONTENTS

| 1 | Introduction                                 | 1  |  |  |  |
|---|----------------------------------------------|----|--|--|--|
| 2 | Method                                       | 2  |  |  |  |
|   | 2.1<br>Background                            | 2  |  |  |  |
|   | 2.2<br>SMALLTALK LM                          | 3  |  |  |  |
| 3 | Experiments                                  | 5  |  |  |  |
|   | 3.1<br>Experimental Setup                    | 6  |  |  |  |
|   | 3.2<br>Language Modeling Results<br>         | 6  |  |  |  |
|   | 3.3<br>Downstream evaluation                 | 7  |  |  |  |
|   | 3.4<br>Routing analysis                      | 8  |  |  |  |
| 4 | Related Work                                 |    |  |  |  |
| 5 | Conclusion                                   | 10 |  |  |  |
| A | Language modeling experiments                | 15 |  |  |  |
|   | A.1<br>Architecture and training details<br> | 15 |  |  |  |
|   | A.2<br>Notation                              | 16 |  |  |  |
|   | A.3<br>Computational Cost                    | 17 |  |  |  |
|   | A.3.1<br>Transformer                         | 17 |  |  |  |
|   | A.3.2<br>Mixture of LLMs                     | 19 |  |  |  |
|   | A.4<br>Communication overhead                | 20 |  |  |  |
| B | Downstream evaluation                        | 22 |  |  |  |
| C | Additional Results                           | 23 |  |  |  |

### <span id="page-14-0"></span>A LANGUAGE MODELING EXPERIMENTS

### <span id="page-14-1"></span>A.1 ARCHITECTURE AND TRAINING DETAILS

Our architecture is based on the transformer decoder [\(Vaswani,](#page-13-4) [2017\)](#page-13-4). We use rotary positional encoding [\(Su et al.,](#page-13-5) [2024\)](#page-13-5). We use a SentencePiece [\(Kudo & Richardson,](#page-11-13) [2018\)](#page-11-13) tokenizer with a vocabulary of 32000 tokens. The model specific parameters for the different sizes are summarized in Table [1.](#page-15-0) We implemented our EM scheme for training the routers in PyTorch [\(Paszke et al.,](#page-12-13) [2019\)](#page-12-13). After segmenting the training set, the experts were trained independently using Jax [\(Bradbury et al.,](#page-10-12) [2018\)](#page-10-12). All training was done in bfloat16. The optimizer states and operations are in float32. The training parameters for specific models are summarised in Table [2.](#page-15-1)

We employ different learning rate schedules for routers and experts due to their distinct roles. Experts are trained for accurate next-token prediction, using a cosine decay learning rate to optimize their performance. In contrast, routers are used solely for routing purposes. We only need an estimation of how well a router performs on a prefix of a sequence relative to other routers. Since we are concerned with relative performance rather than absolute accuracy, applying the same training strategy to all routers is sufficient for effective routing decisions. Moreover, using a constant learning rate for routers reduces computational overhead and eliminates the need to tune the number of training steps.

<span id="page-15-0"></span>

| Model parameters | Role   | Hidden size | FFW expansion factor | Layers | Attention heads |
|------------------|--------|-------------|----------------------|--------|-----------------|
| 335M             | expert | 1024        | 4                    | 24     | 16              |
| 1.3B             | expert | 2048        | 4                    | 24     | 16              |
| 4.4M             | router | 96          | 4                    | 12     | 12              |
| 64M              | router | 416         | 4                    | 12     | 12              |
| 110M             | router | 768         | 4                    | 12     | 12              |

Table 1: Model parameters.

<span id="page-15-1"></span>

| Model             | Steps (k) | Tokens (B) | Batch size | # GPUs |
|-------------------|-----------|------------|------------|--------|
| 335M (dense)      | 256       | 133        | 512        | 8      |
| 335M (4 experts)  | 256       | 133        | 128        | 8      |
| 335M (dense)      | 512       | 266        | 512        | 32     |
| 335M (8 experts)  | 256       | 266        | 128        | 8      |
| 335M (dense)      | 1024      | 532        | 512        | 32     |
| 335M (16 experts) | 256       | 532        | 128        | 8      |
| 1.3B (dense)      | 512       | 266        | 512        | 32     |
| 1.3B (4 experts)  | 512       | 266        | 128        | 8      |
| 1.3B (dense)      | 1024      | 1000       | 1024       | 64     |
| 1.3B (16 experts) | 512       | 1000       | 128        | 8      |
| 1.3B (dense)      | 1024      | 2000       | 2048       | 128    |
| 1.3B (32 experts) | 512       | 2000       | 128        | 8      |
| 4.4M (router)     | 128       | 4          | 32         | 1      |

Table 2: Training parameters.

### <span id="page-15-2"></span>A.2 NOTATION

For the dense baseline LLM and expert LLM:

- Number of layers (L)
- Hidden dimension size (H)
- Number of attention heads (A)
- Sequence length (S)
- Batch size (B)
- Vocabulary size (V )
- Feedforward network dimension (Dff)

#### For the router LLM:

- Prefix length for routing (M)
- Number of layers (Lr)
- Hidden dimension size (Hr)
- Number of attention heads (Ar)
- Batch size (Br)
- Feedforward network dimension (D<sup>r</sup>ff ) parameters specific to the router model

• Number of training tokens per router between communications T

We also define Nsteps, Nsteps expert, and Nsteps router as the number of training steps for the dense baseline, the expert in the mixture, and the router, respectively. The sequence length (S) and vocabulary size (V ) are consistent across the dense baseline, router, and expert models. Finally, we denote E as the number of experts (routers).

### <span id="page-16-0"></span>A.3 COMPUTATIONAL COST

#### <span id="page-16-1"></span>A.3.1 TRANSFORMER

To estimate the computational cost of Transformer model during training and inference, we calculated the total floating-point operations (FLOPs) based on the model's architectural parameters.

For training, we computed the FLOPs per training step by summing the contributions from the embedding layer, multi-head attention (MHA), feedforward layers (FFN), and the output projection layer. The calculations are as follows:

• Embedding Layer: Although embeddings are typically implemented as lookups with negligible computational cost, for completeness, we estimate the FLOPs as:

$$FLOPs_{emb} = B \times S \times H$$

- Multi-Head Attention (MHA):
  - 1. Linear Projections (Queries, Keys, Values):

$$FLOPs_{proj} = 3 \times 2 \times B \times S \times H \times H = 6 \times B \times S \times H^{2}$$

2. Scaled Dot-Product Attention:

$$FLOPs_{attn} = FLOPs_{QK} + FLOPs_{V} = 2 \times B \times S^{2} \times H + 2 \times B \times S^{2} \times H = 4 \times B \times S^{2} \times H$$

3. Output Projection:

$$FLOPs_{out\_proj} = 2 \times B \times S \times H \times H$$

Total Multi-Head Attention (MHA):

$$\begin{aligned} \text{FLOPs}_{\text{MHA}} &= \text{FLOPs}_{\text{proj}} + \text{FLOPs}_{\text{attn}} + \text{FLOPs}_{\text{out-proj}} \\ &= 6 \times B \times S \times H^2 + 4 \times B \times S^2 \times H + 2 \times B \times S \times H^2 \\ &= 8 \times B \times S \times H^2 + 4 \times B \times S^2 \times H \end{aligned}$$

• Feedforward Network (FFN):

$$FLOPs_{FFN} = 2 \times 2 \times B \times S \times H \times D_{ff} = 4 \times B \times S \times H \times D_{ff}$$

• Output projection:

$$FLOPs_{out} = FLOPs_{out_{proj}} + FLOPs_{softmax} = 2 \times B \times S \times H \times V + 3 \times B \times S \times V$$

Total FLOPs per Layer:

$$FLOPs_{layer} = (FLOPs_{MHA} + FLOPs_{FFN}) \times L$$

Total Forward Pass FLOPs per Step:

$$FLOPs_{forward} = FLOPs_{emb} + FLOPs_{layer} + FLOPs_{out}$$

#### Total Training FLOPs per Step:

The backward pass is approximately twice as computationally intensive as the forward pass. Therefore:

$$FLOPs_{train\_step} = FLOPs_{forward} + 2 \times FLOPs_{forward} = 3 \times FLOPs_{forward}$$

#### Total Training FLOPs:

Multiplying the FLOPs per training step by the total number of training steps:

Total Training FLOPs = FLOPs<sub>train\_step</sub> 
$$\times$$
  $N_{steps} = 3 \times$  FLOPs<sub>forward</sub>  $\times$   $N_{steps}$ 

### Total Training FLOPs:

Total Training FLOPs = 
$$3 \times N_{\text{steps}} \times \left[ B \times S \times H + L \times \left( 8 \times B \times S \times H^2 + 4 \times B \times S^2 \times H + 4 \times B \times S \times H \times D_{\text{ff}} \right) + 2 \times B \times S \times H \times V + 3 \times B \times S \times V \right]$$
 (10)

#### Total Inference FLOPs:

<span id="page-17-0"></span>Total Inference FLOPs = 
$$\begin{bmatrix} B \times S \times H \\ + L \times \left( 8 \times S \times H^2 + 4 \times S^2 \times H + 4 \times S \times H \times D_{\rm ff} \right) \\ + 2 \times S \times H \times V + 3 \times S \times V \end{bmatrix}$$
(11)

### <span id="page-18-0"></span>A.3.2 MIXTURE OF LLMS

The total training FLOPs for the mixture model consist of four main components: training routers, sharding data for routers, training experts, and sharding data for the experts.

For a model trained with batch size B over Nsteps steps, the total number of sequences shaded can be estimated as Nsteps × B × E, where each model receives approximately Nsteps × B sequences during training.

Thus, the FLOPs required for sharding can be calculated as the inference FLOPs with an effective batch size of Nsteps × B × E (see Equation [11\)](#page-17-0).

### Therefore Total Training FLOPs:

<span id="page-18-2"></span><span id="page-18-1"></span>Total Training FLOPs = Training FLOPs Routers + Training FLOPs Experts = 
$$(12) + 3 \times N_{\text{steps.router}} \times \left[ B_r \times S \times H_r \right. \\ + L_r \times \left( 8 \times B_r \times S \times H_r^2 + 4 \times B_r \times S^2 \times H_r + 4 \times B_r \times S \times H_r \times D_{r_{\text{fl}}} \right) \\ + 2 \times B_r \times S \times H_r \times V + 3 \times B_r \times S \times V \right] \times E \quad \text{(Eq. 13: Training Routers)}$$

$$(13) + \left( N_{\text{steps.router}} \times B_r \times E \right) \times \left[ M \times H_r \right. \\ + L_r \times \left( 8 \times M \times H_r^2 + 4 \times M^2 \times H_r + 4 \times M \times H_r \times D_{r_{\text{fl}}} \right) \\ + 2 \times M \times H_r \times V + 3 \times M \times V \right] \times E \quad \text{(Eq. 14: Sharding data for routers)}$$

$$(14)$$

$$3 \times N_{\text{steps.expert}} \times \left[ B \times S \times H \right. \\ + L \times \left( 8 \times B \times S \times H^2 + 4 \times B \times S^2 \times H + 4 \times B \times S \times H \times D_{\text{fl}} \right) \\ + 2 \times B \times S \times H \times V + 3 \times B \times S \times V \right] \times E \quad \text{(Eq. 15: Training Experts)}$$

$$(15) + \left( N_{\text{steps.expert}} \times B \times E \right) \times \left[ M \times H_r \\ + L_r \times \left( 8 \times M \times H_r^2 + 4 \times M^2 \times H_r + 4 \times M \times H_r \times D_{r_{\text{fl}}} \right) \\ + 2 \times M \times H_r \times V + 3 \times M \times V \right] \times E \quad \text{(Eq. 16: Sharding data for experts)}$$

In the above formula:

- <span id="page-18-4"></span><span id="page-18-3"></span>• Equations [15](#page-18-3) and [13](#page-18-1) correspond to the total training FLOPs for E experts and E routers respectively.
- Equations [16](#page-18-4) and [14](#page-18-2) represents the inference cost of E routers on B batches with a prefix length M used to partition the data for experts and routers respectively.

Table [3](#page-19-1) presents the training and inference costs for our proposed mixture of language models compared to the dense baseline LLMs.

The expert balancing during training introduces only a negligible computational overhead relative to both training and inference costs. Therefore, it is excluded from our FLOPs calculations.

<span id="page-19-1"></span>

| Model                             | Perplexity                                                      | Training cost                                                          | Inference cost                                                     |
|-----------------------------------|-----------------------------------------------------------------|------------------------------------------------------------------------|--------------------------------------------------------------------|
| 335M (dense)<br>335M (4 experts)  | $11.78 \\ 10.78 \downarrow 8.49\%$                              | $31.02 \\ 31.02 + 0.22 \uparrow 0.07\%$                                | $0.79 \\ 0.79 + 0.01 \uparrow 1.27\%$                              |
| 335M (dense)<br>335M (8 experts)  | $11.25 \\ 10.20 \downarrow 9.33\%$                              | $62.03 \\ 62.03 + 0.75 \uparrow 1.20\%$                                | $0.79 \\ 0.79 + 0.02 \uparrow 2.53\%$                              |
| 335M (dense)<br>335M (16 experts) | $10.80 \\ 9.64 \downarrow\ 10.74\%$                             | $124.06 \\ 124.06 + 2.71 \uparrow 2.19\%$                              | $0.79 \\ 0.79 + 0.04 \uparrow 5.06\%$                              |
| 335M (dense)<br>335M (32 experts) | $10.5 \\ 9.07 \downarrow 13.62\%$                               | $248.12 \\ 248.12 + 10.28 \uparrow 4.14\%$                             | $0.79 \\ 0.79 + 0.08 \uparrow 10.13\%$                             |
| 1.3B (dense)<br>1.3B (4 experts)  | $9.10 \\ 8.75 \downarrow 3.85\%$                                | $\begin{array}{c} 221.33 \\ 221.33 + 0.36 \uparrow 0.16\% \end{array}$ | $\begin{array}{c} 2.81 \\ 2.81 + 0.01 \uparrow 0.36\% \end{array}$ |
| 1.3B (dense)<br>1.3B (16 experts) | $8.48 \\ 7.42 \downarrow 12.53\%$                               | $885.32 \\ 885.32 + 4.87 \uparrow 0.55\%$                              | $\begin{array}{c} 2.81 \\ 2.81 + 0.04 \uparrow 1.42\% \end{array}$ |
| 1.3B (dense)<br>1.3B (32 experts) | $\begin{array}{c} 8.20 \\ 6.76 \downarrow\ 17.56\% \end{array}$ | $1770.65 \\ 1770.65 + 18.94 \uparrow 1.07\%$                           | $2.81 \\ 2.81 + 0.08 \uparrow 2.85\%$                              |

Table 3: **Performance gain versus computational overhead.** The table compares performance improvements and computational costs for models with 335M (trained with 4, 8, 16, and 32 experts) and 1.3B parameters (trained with 4, 16, and 32) experts. Training cost is measured in  $10^{19}$  FLOPs, inference cost is measured in  $10^{12}$  FLOPs. The dense baselines and corresponding mixture models were trained on the same data volume.

Regarding Computational Overhead As demonstrated in Table 3, the 1.3B parameter model with 32 experts achieves significant performance improvements almost cost-free, gaining nearly 18% in perplexity compared to the dense baseline while incurring only 1% additional cost during training and less than 3% during inference. Furthermore, a 335M parameter mixture of 32 experts reaches the same performance level as the 1.3B parameter model, but with a comparable training budget and three times lower inference cost. Despite the 335M parameter model incurring a higher percentage cost during inference, this computational overhead is not the definitive minimum necessary for effective routing: as detailed in § 3.4, our results indicate that the routing overhead can be significantly reduced by utilizing smaller routers and shorter prefix sizes. This suggests that further reductions in computational costs, without compromising routing quality, are feasible and present a valuable direction for future research.

### <span id="page-19-0"></span>A.4 COMMUNICATION OVERHEAD

In this section, we provide an estimate of the communication overhead involved in our model training compared to a dense baseline trained with standard distributed data parallelism. We quantify both the volume of data transmitted and the frequency of communication required during training. For simplicity of computation, we assume that all collective communications are bandwidth-optimal, meaning that the total number of bytes transferred per node is approximately 2K, where K is the size of the message (Thakur et al., 2005; Patarasuk & Yuan, 2009).

**Mixture of LLMs** Every router performs all gather operation to share the loss on the dataset chunk approximately every  $T=45 \mathrm{M}$  training tokens, as outlined in 3.2. This frequency was primarily determined by implementation specifics related to data storage.

We can define the number of tokens processed per step by a router as  $N_{\text{tokens\_per\_step}}$ , which can be calculated as:

$$N_{\text{tokens\_per\_step}} \leq S \times B_r$$

This leads to the minimum number of steps between communications,  $N_{\text{steps, per, com}}$ , given by:

$$N_{\text{steps\_per\_com}} \ge \frac{T}{S \times B_r}$$

The maximum number of communication events,  $N_{\text{comm}}$ , during the router's training can then be estimated as:

 $N_{\text{comm}} \leq \frac{N_{\text{steps\_router}}}{N_{\text{steps\_per\_comm}}} = \frac{N_{\text{steps\_router}} \times S \times B_r}{T}$ 

As shown in App. Table 2 we train routers for 128,000 steps with batch size 32. We use context size 1024. Therefore,  $N_{\rm comm}\approx 94 < 1\times 10^2$ .

The total amount of data sent and received by each node (router) can be calculated as the total number of sequences multiplied by the number of bytes per sequence. We consider that the loss is transferred in float16, which requires 2 bytes per element. The total number of sequences can be estimated as  $\frac{T \times N}{S}$ .

Each router needs to send and receive the loss values for each sequence to and from other routers, resulting in a factor of 2 for the data sent and received. Therefore, the data per router can be approximated as:

$$\mbox{Data per router} \approx 2 \times 2 \times \frac{T \times E}{S}$$

Given  $T=45\times 10^6$  – training tokens per router between communication steps,  $E\leq 32$  – number of experts and  $S=1{,}024$  – sequence length, we can estimated the data transferred per node as:

$$\text{Data per router} \leq 2 \times 2 \times \frac{T \times E}{S} = 4 \times \frac{45 \times 10^6 \times 32}{1,024} = 5.625 \text{MB}$$

Once routers are fully trained, they communicate only when necessary, such as when experts require new data for training.

For instance, consider that the message size is K bytes, equivalent to loss array of K/2 sequences in float16 precision. The frequency of communication after routers are fully trained is determined by the batch size of the expert B and the number of experts E. Specifically, the number of steps each expert makes between communications can be estimated as:

$$N_{\text{steps\_per\_comm}} \approx \frac{K}{2 \times B \times E}$$

Thus, communication can be scheduled approximately every  $\frac{K}{4 \times B \times E}$  expert's training steps. Assuming  $K = 2^{31} - 1$  bytes, B = 128 as used in the majority of our experiments, and  $E \in \{4, 8, 16, 32\}$  (see Appendix 2), the communication interval becomes:

<span id="page-20-0"></span>
$$\frac{2^{31} - 1}{2 \times 128 \times E} \approx \frac{2^{23}}{E} \le 2^{18} \quad \forall E \in \{4, 8, 16, 32\}$$
 (17)

Comparison with Distributed Training Let us assume that we train a model with a batch size of B on a single host, where W represents the model size in parameters. For simplicity, we consider only data parallelism and assume that the model weights can fit on one GPU. To compare communication costs, we examine the additional expense incurred when increasing throughput by a factor of E, i.e., utilizing E hosts instead of one.

In regular distributed training, scaling to E hosts necessitates copying all weights to each host and performing gradient synchronization at every training step. Assuming each parameter is represented by 32 bits (as optimizers typically operate in float32), the amount of data transferred to and from each node during gradient synchronization can be calculated using the bandwidth-optimal algorithm:

Data per node = 
$$2 \times W \times 4$$

Here,  $W \times 4$  represent the size of the model gradients in bytes.

If W = 1.3 × 10<sup>9</sup> parameters, then the amount of data transferred per node per training step is:

Data per node = 
$$2 \times 1.3 \times 10^9 \times 4 = 10.4$$
GB

Therefore, in regular distributed training, each node must send and receive gigabytes of data at every step. In contrast, our approach offers flexible communication strategies. We can transfer large volumes of data only a few times during training (see Equation [17\)](#page-20-0), or optimise for more frequent communications with smaller data sizes. This flexibility allows us to adapt based on network latency and other infrastructure constraints.

## <span id="page-21-0"></span>B DOWNSTREAM EVALUATION

<span id="page-21-1"></span>For the evaluation, we utilized the *lm-eval-harness* software [\(Gao et al.,](#page-11-14) [2024\)](#page-11-14). For the computation of perplexity, we compute the perplexity using the format: "Question: {question}. Answer: {answer}."

| Task                            | 32 experts, 2T | dense, 2T |
|---------------------------------|----------------|-----------|
| mmlu abstract algebra           | 0.1400         | 0.2300    |
| mmlu astronomy                  | 0.3553         | 0.2829    |
| mmlu business ethics            | 0.4900         | 0.5300    |
| mmlu college computer science   | 0.2500         | 0.2900    |
| mmlu college mathematics        | 0.1500         | 0.1400    |
| mmlu computer security          | 0.3500         | 0.3800    |
| mmlu elementary mathematics     | 0.2672         | 0.2487    |
| mmlu formal logic               | 0.3016         | 0.2937    |
| mmlu global facts               | 0.3100         | 0.2800    |
| mmlu high school geography      | 0.3586         | 0.3586    |
| mmlu high school macroeconomics | 0.3128         | 0.2872    |
| mmlu high school microeconomics | 0.3193         | 0.3193    |
| mmlu high school statistics     | 0.2917         | 0.2593    |
| mmlu human aging                | 0.4081         | 0.3857    |
| mmlu human sexuality            | 0.3969         | 0.3893    |
| mmlu international law          | 0.2314         | 0.2314    |
| mmlu logical fallacies          | 0.3067         | 0.2945    |
| mmlu machine learning           | 0.2589         | 0.2679    |
| mmlu medical genetics           | 0.3900         | 0.3500    |
| mmlu moral disputes             | 0.2803         | 0.2890    |
| mmlu moral scenarios            | 0.2380         | 0.2380    |
| mmlu nutrition                  | 0.3039         | 0.2516    |
| mmlu prehistory                 | 0.4167         | 0.3858    |
| mmlu professional accounting    | 0.2872         | 0.2518    |
| mmlu professional medicine      | 0.3566         | 0.2904    |
| mmlu professional psychology    | 0.3203         | 0.3186    |
| mmlu world religions            | 0.5731         | 0.4503    |
| mmlu anatomy                    | 0.4667         | 0.3926    |
| mmlu clinical knowledge         | 0.3434         | 0.2755    |
| mmlu college biology            | 0.4028         | 0.3819    |
| mmlu college chemistry          | 0.2800         | 0.2600    |
| mmlu college medicine           | 0.2832         | 0.2659    |
| mmlu college physics            | 0.1863         | 0.1961    |
| mmlu conceptual physics         | 0.3787         | 0.3787    |

Table 4: Downstream Evaluation Results (Part 1). This table presents a comparison in accuracy on downstream tasks between a 1.3B-parameter, 32-expert model and a 1.3B-parameter dense model. Both models were trained using the 2T tokens and nearly the same training FLOPs, with only a 1% difference. The inference cost of the expert model increased by less than 3% compared to the dense baseline FLOPs.

| Task                                     | 32 experts, 2T | dense, 2T |
|------------------------------------------|----------------|-----------|
| mmlu econometrics                        | 0.2719         | 0.2018    |
| mmlu electrical engineering              | 0.2552         | 0.2621    |
| mmlu high school biology                 | 0.3290         | 0.3097    |
| mmlu high school chemistry               | 0.2167         | 0.1970    |
| mmlu high school computer science        | 0.2400         | 0.2400    |
| mmlu high school european history        | 0.2848         | 0.3212    |
| mmlu high school government and politics | 0.4093         | 0.3834    |
| mmlu high school mathematics             | 0.1519         | 0.1444    |
| mmlu high school physics                 | 0.2980         | 0.2715    |
| mmlu high school psychology              | 0.4642         | 0.4330    |
| mmlu high school us history              | 0.3382         | 0.3137    |
| mmlu high school world history           | 0.2827         | 0.2996    |
| mmlu jurisprudence                       | 0.2407         | 0.2222    |
| mmlu management                          | 0.3786         | 0.3981    |
| mmlu marketing                           | 0.4915         | 0.4915    |
| mmlu miscellaneous                       | 0.5121         | 0.4994    |
| mmlu philosophy                          | 0.2958         | 0.3119    |
| mmlu professional law                    | 0.2634         | 0.2536    |
| mmlu public relations                    | 0.3545         | 0.4273    |
| mmlu security studies                    | 0.3224         | 0.3020    |
| mmlu sociology                           | 0.2786         | 0.2786    |
| mmlu us foreign policy                   | 0.3300         | 0.3400    |
| arc challenge                            | 0.3200         | 0.2892    |
| arc easy                                 | 0.6780         | 0.6549    |
| hellaswag                                | 0.4912         | 0.4663    |
| sciq                                     | 0.8950         | 0.9130    |

Table 5: Downstream Evaluation Results (Part 2)

### <span id="page-22-0"></span>C ADDITIONAL RESULTS

Effect of Prefix Length on Routing Performance We also conducted experiments by partitioning the data using a shorter prefix length of M = 32 tokens during training, whereas in our main results (see § [3\)](#page-4-1), we used a prefix length of M = 256 tokens.

We observed that training with a smaller prefix length resulted in better expert performance when routing during inference was done using a small number of tokens (see Figure [6\)](#page-22-1). This suggests that using a shorter prefix during training enhances the router's ability to make effective decisions with limited context.

We did not take any action to make sure that the router models could extrapolate to longer sequences than seen in training (e.g. interpolating positional encodings, using ALiBi [\(Press et al.,](#page-12-15) [2022\)](#page-12-15), etc.). As expected, we observe a performance degradation when making routing decisions for sequences longer than the ones used for training.

<span id="page-22-1"></span>![](_page_22_Figure_7.jpeg)

Figure 6: Test perplexity as a function of routing prefix length during inference for 1.3B parameter models with 16 experts. The orange curve represents a prefix length of 32, while the blue curve corresponds to a prefix length of 256.