# 3 Results: Building a generalist model

We validate the BTS approach through experiments with a seed language models of 2.7B parameters. We describe model [\(Section 3.1\)](#page-4-1), data [\(Section 3.2\)](#page-4-2), baseline [\(Section 3.3\)](#page-5-0), and evaluation [\(Section 3.4\)](#page-6-0) details and discuss experimental results in [Section 3.5.](#page-7-0)

## <span id="page-4-1"></span>3.1 Model details

<span id="page-4-3"></span>Seed model We pretrain a 2.7B parameter language model, following the same text recipe used in Llama 3 [\(Dubey et al.,](#page-14-0) [2024\)](#page-14-0). See [Table 1](#page-4-3) for architecture details. We employ a learning rate schedule that warms up from 0 to 4e-4 over 2000 steps, then undergoes a cosine decay to 1% of the peak learning rate. The seed model is trained for 2.2M steps on 15T tokens.

| Layers                | 20                  |
|-----------------------|---------------------|
| Model Dimension       | 3072                |
| FFN Dimension         | 12288               |
| Attention Heads       | 24                  |
| Key/Value Heads       | 1                   |
| Activation Function   | SwiGLU              |
| Vocabulary Size       | 128,000             |
| Positional Embeddings | RoPE (θ = 500, 000) |
|                       |                     |

Table 1 Architecture details for the 2.7B parameter seed model and expert models.

Expert models We create three copies of the seed model, each of which is continually trained for 96k training steps over a 200B token specialized data mixture to produce expert models for code, mathematics, and multilingual tasks. During the continued pretraining phase, we use a batch size of 2M tokens and a learning rate of 5e-6, followed immediately by a cosine decay schedule that reduces the learning rate to 1% of its initial value. This learning rate is derived by annealing from the final learning rate used at the end of seed model pretraining, adjusted to account for the reduced batch size in this continued pretraining phase. We adopted this learning rate strategy as it yielded the most stable learning during the continued pretraining phase

<span id="page-4-2"></span>BTS model We use four stitch layers to combine the seed model together with the three expert models. The four stitch layers are inserted after every five layers in the seed and expert models. We refer to the resulting model as the BTS model. As described in [Section 2,](#page-2-0) the four stitch layers alternate between a Merge-into-Expert layer and Experts-into-Hub stitch layer. Upon initialization, the BTS model is further trained for 15B tokens over 7000 steps using a batch size of 2M tokens. The optimization objective is to minimize the next-token prediction loss from the hub model's output. The learning rate schedule warms up from 0 to 5e-6 over 2000 steps, then undergoes a cosine decay to 1% of the peak learning rate. Note that during the BTS training phase, only the stitch layers are updated while all the parameters of the seed model and the expert models are frozen.

<span id="page-5-1"></span>

|                  | Training Params | Total Params | Active Params |
|------------------|-----------------|--------------|---------------|
| Expert upcycling |                 |              |               |
| BTX Sample       | 7.2B            | 7.2B         | 2.9B          |
| BTX Soft         | 7.2B            | 7.2B         | 7.2B          |
| BAM              | 8.4B            | 8.4B         | 8.4B          |
| Expert merging   |                 |              |               |
| Model Soup       | N/A             | 2.7B         | 2.7B          |
| BTM              | N/A             | 10.8B        | 10.8B         |
| Expert Routing   | 15k             | 10.8B        | 2.7B          |
| BAM Adapters     | 1.5B            | 9.9B         | 9.9B          |
| BTS              | 264M            | 11B          | 11B           |

Table 2 Training, total, and active parameter count for BTS and baselines. We use "expert upcycling" to describe MoE upcycling methods where the seed and experts themselves do not remain intact during the MoE training phase. These methods require significantly more training parameters, and thus are less modular, less flexible, and less interpretable. We use "expert merging" to describe methods, such as BTS, where the seed and expert models remain frozen during the merging phase. Expert merging methods require minimal number of training parameters, making them more modular and interpretable.

## 3.2 Data details

Seed model We adopt the same text pretraining mixture as Llama 3 [\(Dubey et al.,](#page-14-0) [2024\)](#page-14-0).

Expert models In the continued pretraining phase, each dense expert is trained on a specialized data mixture for 200B tokens:

- Code: We adopt a recipe similar to that of CodeLlama [\(Rozière et al.,](#page-15-6) [2023\)](#page-15-6) with > 85% code tokens, utilizing the code data subset of the seed model mixture.
- Math: We continue pretraining on the OpenWebMath dataset [\(Paster et al.,](#page-15-7) [2023\)](#page-15-7).
- Multilingual: We utilize a mixture of 90% non-English data and 10% English data, with each subset pulled from the seed model mixture, following the multilingual expert recipe described in [Dubey et al.](#page-14-0) [\(2024\)](#page-14-0).

BTS model The data mixture for the BTS training phrase consists of 15% expert domain tokens for each of the code, math, and multilingual domains. The remaining 55% of the mixture consists of the pretraining data utilized for the seed model outside of these domains.

## <span id="page-5-0"></span>3.3 Baselines

In addition to the seed and expert models, we also compare BTS with expert upcycling and expert merging baselines. We use expert upcycling to describe methods where the seed and expert models are used to initialize an MoE model, which is further trained. The entire MoE is updated during training and as such the experts and seed model themselves do not remain intact. This approach loses the flexibility and interpretability inherent in a more modular approach, and any model change requires updating a large number of parameters. On the other hand, we use expert merging to describe methods, such as BTS, in which the seed and expert parameters remain frozen during the merging phase.

#### Expert upcycling baselines:

- BTX [\(Sukhbaatar et al.,](#page-15-3) [2024\)](#page-15-3): We upcycle the seed model and three expert models into an MoE. Our baselines include two BTX variants, where the FFN experts employ one of two routing strategies: 1) sample top-1 routing [\(Sukhbaatar et al.,](#page-15-3) [2024\)](#page-15-3), where we use a Gumbel-Softmax [\(Jang et al.,](#page-14-4) [2016\)](#page-14-4) for the routing function, and 2) soft-routing, where all four experts are activated at all times. We use the same experimental setup as BTS runs, including training data and the learning rate schedule. See [Section 2.1](#page-2-1) for details on the MoE architecture.
- BAM [\(Zhang et al.,](#page-16-3) [2024\)](#page-16-3): We upcycle the seed model and the three expert models into an MoE with both attention experts and FFN experts. See [Section 2.1](#page-2-1) for a description of the attention experts architecture. We employ soft-routing for both sets of experts, ensuring that, like BTS, all FFN and attention parameters of the seed and expert models are activated during training and inference. We use the same experiment setup as BTS runs.

#### Expert merging baselines:

- Model soup [\(Wortsman et al.,](#page-15-8) [2022\)](#page-15-8): We uniformly average the weights of the seed and expert models. Unlike other baselines, no further training is required upon initialization.
- BTM [\(Li et al.,](#page-15-0) [2022\)](#page-15-0): We ensemble the output logits of the seed and expert models. The ensemble weights are estimated using Bayes' rule with a uniform prior [\(Li et al.,](#page-15-0) [2022;](#page-15-0) [Gururangan et al.,](#page-14-2) [2023\)](#page-14-2). Like the model soup baseline, no further training is required upon initialization.
- Expert routing: We train a linear router ∈ R dim×<sup>n</sup> that routes to either the seed model or one of the expert models. The router's training objective is a classification cross-entropy loss where the target is the model with the smallest next-token prediction loss for the input. Given a prompt, the router decides on the model and routes all subsequent tokens to the same model. During training, the routing decision is made based on the average embedding of the first t tokens in the input, where t is randomly sampled between 32 and 256. During inference, the routing decision is made based on the average embedding of the entire prompt. We train the linear router with a constant learning rate of 5e-4 and batch size of 1M. The model is trained for 1B tokens only, as we did not see an improvement in downstream metrics or training loss with further training.
- BAM with adapters [\(Zhang et al.,](#page-16-3) [2024\)](#page-16-3): We train an expert-intact variant of BAM with soft-routing, which we refer to as BAM with adapters. In this variant, each attention expert and each FFN expert's output undergo a linear adapter layer Wproj<sup>i</sup> ∈ R dim×dim. Formally, we replace [Equation 1](#page-2-3) and [Equation 2](#page-2-4) by the following:

$$y_{\text{MoE}} = \sum_{i \in \mathcal{T}} p_i(x) W_{\text{ffn proj}_i} (\text{FFN}_i(x))$$

$$y_{\text{MoA}} = \sum_{i \in \mathcal{M}} q_i(x) W_{\text{attn proj}_i} (\text{Attention}_i(x)).$$
(5)

Only the router and adapters are updated during training, while all other parameters remain frozen. We use the same experiment setup as BTS runs.

We show a comparison of the number of training, active, and total parameters in [Table 2.](#page-5-1) Note that BTS has the most total parameters of all variants, but only a small fraction of the training parameters of the expert upcycling variants.

## <span id="page-6-0"></span>3.4 Evaluation

We assess model performance with zero-shot and few-shot downstream tasks relevant to the expert domains.

- General Knowledge and Reasoning: To assess general knowledge and reasoning capabilities, we report MMLU (5-shot; [Hendrycks et al.,](#page-14-5) [2021a\)](#page-14-5) and Big-Bench Hard (3-shot; [Suzgun et al.,](#page-15-9) [2022\)](#page-15-9). In tables, we denote Big-Bench Hard as BBH.
- Code: For code generation capabilities, we evaluate on MBPP (3-shot; [Austin et al.,](#page-14-6) [2021\)](#page-14-6) and HumanEval (0-shot; [Chen et al.,](#page-14-7) [2021\)](#page-14-7) benchmarks. We denote HumanEval as HE in the results table for brevity.

<span id="page-7-1"></span>

|                   |       | General |       | Code  | Multilingual |           | Math  |           |       |
|-------------------|-------|---------|-------|-------|--------------|-----------|-------|-----------|-------|
|                   | MMLU  | BBH     | MBPP  | HE    | Flores(S)    | Flores(T) | GSM8K | MATH      | Avg.  |
| 2.7B Dense models |       |         |       |       |              |           |       |           |       |
| Seed Model        | 28.4  | 35.6    | 27.0  | 20.7  | 29.5         | 35.7      | 10.5  | 4.82      | 24.0  |
| Code Expert       | 30.3  | 35.2    | 32.0  | ∗25.0 | 29.0         | 35.5      | 11.4  | 4.40      | 25.4  |
| Multiling. Expert | 26.6  | 34.7    | 26.2  | 18.3  | ∗31.9        | ∗37.1     | 10.8  | 4.16      | 23.7  |
| Math Expert       | ∗36.3 | ∗37.2   | 26.2  | 16.5  | 23.6         | 32.7      | ∗20.5 | 10.1      | 25.4  |
| Expert upcycling  |       |         |       |       |              |           |       |           |       |
| BTX Sample        | 30.4  | 36.6    | 30.0  | 21.3  | 30.5         | 36.0      | 13.9  | 6.58      | 25.7  |
| BTX Soft          | 34.7  | 36.8    | 29.6  | 23.2  | 31.0         | 36.0      | 19.2  | 9.10      | 27.4  |
| BAM               | 35.2  | 37.1    | 29.8  | 22.6  | 31.0         | 36.1      | 20.3  | 10.1      | 27.8  |
| Expert merging    |       |         |       |       |              |           |       |           |       |
| Model Soup        | 30.7  | 37.0    | 29.6  | 22.6  | 29.5         | 36.2      | 13.6  | 6.46      | 25.7  |
| BTM               | 30.6  | 37.0    | 31.8  | 23.8  | 31.8         | 37.0      | 12.7  | 10.1      | 26.9  |
| Expert Routing    | 28.4  | 35.6    | 27.0  | 23.8  | 30.8         | 37.0      | 10.5  | 5.04      | 24.8  |
| BAM Adapters      | 34.0  | 37.0    | 28.8  | 22.6  | 31.0         | 36.1      | 18.8  | 10.0      | 27.3  |
| BTS               | 35.8  | 36.9    | ∗32.2 | 22.0  | 30.9         | 36.2      | 20.2  | ∗<br>10.6 | ∗28.1 |

Table 3 Performance of BTS against expert merging and upcycling methods, seed and expert models measured on popular benchmarks across several capabilities. Bolded numbers indicate the best performance among dense models or merged models, while an asterisk (<sup>∗</sup> ) denotes the best performance across all models. See [Section 3.4](#page-6-0) for benchmark details. Although dense expert models sometimes achieve the best results in their specialized domains, they often significantly under-perform in other domains. Among all merged models, BTS achieves the best average performance. Notably, BTS not only emerges as the most well-rounded generalist expert but also outperforms the corresponding domain-specific experts on MATH and MBPP tasks.

- Multilingual: For measuring multilingual capabilities, we use machine translation sub-tasks in Flores (1-shot; [Goyal et al.,](#page-14-8) [2022\)](#page-14-8). Specifically, we evaluate on seven languages: Dutch, Spanish, Portuguese, Vietnamese, Indonesian, Hindi, and French. We display the sub-tasks evaluations into two categories, 1) those with English as the source translation language (S), and 2) those with English as the target translation language (T).
- Math: For mathematical reasoning, we report the performance on GSM8K (8-shot; [Cobbe et al.,](#page-14-9) [2021\)](#page-14-9) and MATH (4-shot; [Hendrycks et al.,](#page-14-10) [2021b\)](#page-14-10).

## <span id="page-7-0"></span>3.5 Results

Results on general knowledge, code, multilingual, and math benchmarks for the seed model, expert models, and all expert merging and expert upcycling baselines are reported in [Table 3.](#page-7-1) We make the following observations:

- Expertmodels highlight datamix tradeoffs: While the dense expert models typically achieve the best results in their respective target domains, they often significantly underperform in other domains, highlighting that improving performance in one domain may come at the cost of regressing in others. For example, the Math expert outperforms all models in GSM8K, but lags behind the seed model substantially in coding tasks.
- Learned connections are important for expressive merging: Methods like BAM with adapters and BTS outperform expert merging methods without learned connections between experts, such Model Soup, BTM, and Expert Routing. This demonstrates the importance of adding learned, intermediate connections between experts.
- BTS achieves the best generalist performance: Among all model variants seed, expert, expert merging, and experts upcycling – BTS achieves the best average performance across tasks. Notably, BTS achieves

similar or better performance to the expert upcycling baselines at only a fraction of the training parameters.

• BTS can outperform individual experts in their specialized tasks: BTS emerges not only as the most well-rounded generalist model, but is also the only model which achieves better performance than any individual expert in some tasks. BTS outperforms the Code expert in MBPP and the Math expert in the MATH task.

