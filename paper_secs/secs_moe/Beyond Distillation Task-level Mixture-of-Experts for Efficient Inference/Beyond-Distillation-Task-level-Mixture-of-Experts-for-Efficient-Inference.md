# Beyond Distillation: Task-level Mixture-of-Experts for Efficient Inference

## Sneha Kudugunta, Yanping Huang, Ankur Bapna, Maxim Krikun, Dmitry Lepikhin, Thang Luong, Orhan Firat

Google Research

{snehark,huangyp,ankurbpn,krikun,lepikhin, thangluong,orhanf}@google.com

## Abstract

Sparse Mixture-of-Experts (MoE) has been a successful approach for scaling multilingual translation models to billions of parameters without a proportional increase in training computation. However, MoE models are prohibitively large and practitioners often resort to methods such as distillation for serving. In this work, we investigate routing strategies at different granularity (token, sentence, task) in MoE models to bypass distillation. Experiments on WMT and a web-scale dataset suggest that task-level routing (*task-MoE*) enables us to extract smaller, ready-to-deploy sub-networks from large sparse models.

On WMT, our task-MoE with 32 experts (533M parameters) outperforms the best performing token-level MoE model (*token-MoE*) by +1.0 BLEU on average across 30 language pairs. The peak inference throughput is also improved by a factor of 1.9x when we route by tasks instead of tokens. While distilling a token-MoE to a smaller dense model preserves only 32% of the BLEU gains, our sub-network task-MoE, by design, preserves all the gains with the same inference cost as the distilled student model. Finally, when scaling up to 200 language pairs, our 128-expert task-MoE (13B parameters) performs competitively with a token-level counterpart, while improving the peak inference throughput by a factor of 2.6x.

#### 1 Introduction

Scaling up neural network models has recently received great attention, given the significant quality improvements on a variety of tasks including natural language understanding [\(Raffel et al.,](#page-10-0) [2019;](#page-10-0) [Brown et al.,](#page-9-0) [2020\)](#page-9-0) and multilingual machine translation [\(Huang et al.,](#page-9-1) [2019;](#page-9-1) [Lepikhin et al.,](#page-10-1) [2020\)](#page-10-1).

While training massive models on large amounts of data can almost guarantee improved quality, there are two factors affecting their practicality and applicability: (1) *training efficiency* and (2) *inference efficiency*. Large dense models are often

prohibitively compute-intensive to train, with some models requiring TFlops-days of compute [\(Brown](#page-9-0) [et al.,](#page-9-0) [2020\)](#page-9-0). A recent line of work has proposed sparsely-gated Mixture-of-Experts (MoE) layers as an efficient alternative to dense models [\(Shazeer](#page-10-2) [et al.,](#page-10-2) [2017;](#page-10-2) [Lepikhin et al.,](#page-10-1) [2020;](#page-10-1) [Riabinin and](#page-10-3) [Gusev,](#page-10-3) [2020\)](#page-10-3) in order to address *training efficiency* limitations. In a vanilla sparsely-gated MoE model each token of the input sequence activates a different subset of the experts, hence the computation cost per token becomes only proportional to the size of the activated sub-network. However, they fail to meet requirements on *inference efficiency*.

Consider a long sequence where each token of the sequence activates a disjoint subset of available experts. From a practical standpoint, the inference trace of the full sequence spans several experts independently for every token, resulting in an independent pathway for each token. Although this is a desired property - adding flexibility to the model and increasing its capacity - it becomes prohibitive for inference for the following reasons: the model parameters in these large models are beyond the memory limit of a single accelerator device, and require model parallelism to shard them across a cluster of devices during inference. For models with MoE Layers, the input token would be dynamically routed to different experts allocated to different devices. This further adds communication cost across devices to the overall serving cost. Moreover, due to the sequential nature of the autoregressive decoding [\(Kasai et al.,](#page-10-4) [2020;](#page-10-4) [Chen et al.,](#page-9-2) [2018\)](#page-9-2), the added communication cost from model parallel decoders gets multiplied by the number of decoding steps. To add to this, serving MoE models efficiently requires batching a large number of input tokens together, otherwise only a subset of the MoE network will be activated leading to severe device under-utilization.

In this work, we study the *inference efficiency* of sparsely gated MoE models while taking into account the characteristics of the intended application, Multilingual Neural Machine Translation (MNMT). MNMT is an inherently multi-task learning problem, aimed at building a single neural network for translating multiple language pairs simultaneously. In a MNMT model, the extent to which parameters are shared across languages determines the magnitude of positive transfer [\(Baldwin and Ford,](#page-9-3) [1988\)](#page-9-3) and conversely task interference due to the capacity bottleneck [\(Arivazhagan et al.,](#page-9-4) [2019\)](#page-9-4). In an ideal scenario, we would want to efficiently train a single large MNMT model maximizing transfer while expanding the capacity bottleneck; meanwhile, we would like to enjoy the benefits of sparsely activated sub-networks per-task at inference time, i.e. extracting out a sub-network to decode for a particular language pair to actualize *inference efficiency*.

An alternative way to enjoy high inference efficiency from a large model is knowledge distillation [\(Hinton et al.,](#page-9-5) [2015\)](#page-9-5). However, [\(Fedus et al.,](#page-9-6) [2021\)](#page-9-6) found that only a small fraction of quality gains from a large sparse model can be preserved in the student models. Instead;

- We propose routing algorithms for MoE models with affordable serving costs (Section [3\)](#page-1-0). While vanilla MoEs route each sub-word token in the input to its preferred experts, we explore alternative routing strategies that are trained to leverage global task level information to route all tokens corresponding to a particular task collectively to the same set of experts. We decode different tasks separately and only load the subset of experts associated with the corresponding task during inference.
- We report the advantages of our task-level routing method in translation quality and inference cost on a multilingual WMT task (Section [4\)](#page-3-0). With the comparable inference cost, the task-level routing achieved +3.6 BLEU gain over the multilingual model training from scratch, and +2.1 BLEU gain over the dense student model distilled from the large tokenlevel /position-wise MoE (token-MoE) model.
- The observed quality gains from our approach are comparable with the token-MoE models while achieving 1.9x peak throughput and 6.3% of the decoder size.
- We scaled up the token-MoE model on a large scale in-house dataset and saw similar quality

gains (+3.6 BLEU) against the dense baseline (Section [5.2\)](#page-6-0). Compared to the token-level routing approach, our method achieves comparable quality gain, with 2.6x higher peak throughput and 1.6% of the decoder size.

• Finally, we analyze the routing decisions made in MoE models and motivate our method (Section [5.4\)](#page-7-0).

## <span id="page-1-1"></span>2 Scaling Transformers with Mixture-of-Experts

The Transformer [\(Vaswani et al.,](#page-10-5) [2017\)](#page-10-5) architecture is a popular model used for neural machine translation and other natural language understanding/generation problems. In sequence-to-sequence problems, the model consists of an encoder and decoder, each of which contains multiple Transformer layers. For further details, we refer the reader to the original paper [\(Vaswani et al.,](#page-10-5) [2017\)](#page-10-5).

We use the Mixture-of-Experts Transformer models proposed by [\(Lepikhin et al.,](#page-10-1) [2020\)](#page-10-1), where the MoE layers for the Transformers consist of E feed-forward networks (FFN), such that (FFN<sup>1</sup> . . . FFNE).

$$FFN_e(x_s) = wo_e \cdot ReLU(wi_e \cdot x_s)$$
$$y_s = \sum_{e=1}^{E} \mathcal{G}_{s,e} \cdot FFN_e(x_s)$$

Here, x<sup>s</sup> is the input token at position s to the MoE layer and each FFN<sup>e</sup> is a two layer neural network using a ReLU activation function. wi<sup>e</sup> and wo<sup>e</sup> are the input and output projection weights of the e-th expert. Finally, Gs,E is vector computed by the gating network (also referred as router). For each expert, most values of this vector are zeros, one value being positive. We use this vector to route the token to a select few experts. The entries chosen from Gs,E determine how much the expert contributes to the final output ys. Note that, in this work we choose the top 2 weight experts for each example to be comparable with the prior work.

The gating network Gs,E must be considered carefully for efficiency purposes: (1) the utilization of experts must be balanced and (2) the function must be efficient to implement at scale. For a more thorough discussion of MoE transformers, we direct the reader to [\(Lepikhin et al.,](#page-10-1) [2020\)](#page-10-1).

#### <span id="page-1-0"></span>3 Methods

In this section we describe our candidate routing strategies in the context of MNMT and discuss

<span id="page-2-0"></span>![](_page_2_Picture_0.jpeg)

Figure 1: Tokens are routed to the same expert based on task or some other prior in (a) task-based MoE whereas different tokens are routed to different experts in (b) token-based MoE models.

their trade-offs from the perspective of the training and inference efficiency. Multilingual models learn joint representations across languages to the extent of the parameters being shared [\(Wu and](#page-11-0) [Dredze,](#page-11-0) [2019;](#page-11-0) [Tiedemann,](#page-10-6) [2018;](#page-10-6) [Tan et al.,](#page-10-7) [2019;](#page-10-7) [Zhang et al.,](#page-11-1) [2020;](#page-11-1) [Östling and Tiedemann,](#page-10-8) [2016;](#page-10-8) [Kudugunta et al.,](#page-10-9) [2019\)](#page-10-9). While being beneficial for transfer, extreme sharing of the parameters exacerbates interference. Allowing dedicated (unshared) parameters are known to be effective at mitigating interference [\(Zhang et al.,](#page-11-2) [2021;](#page-11-2) [Kong et al.,](#page-10-10) [2021\)](#page-10-10) and MoE variants are inherently learn such partitioning across languages/tasks. Therefore we study the routing algorithm GATE(xs) of MoEs to mitigate interference, while enabling transfer and effective at inference.

#### <span id="page-2-1"></span>3.1 Routing Strategies

Given the sequential nature of the multilingual machine translation task, the routing decisions can be made at three different granularities, from bottom up (i) token-level, (ii) sentence-level and (iii) task-level, as detailed below.

Token-level Routing: This is the baseline discussed in Section [2](#page-1-1) where each token is routed independently.

Sentence-level Routing: Each sequence (sentence), and all tokens that form the sequence, are routed to the same expert. We change the routing algorithm to select experts by sentence representation, calculated by taking the average token

representations in a given sentence.

Task-level Routing: We select experts by task boundaries as opposed to making input-level decisions. In the context of MNMT, these task boundaries can either be defined by the target language (French-to-English and German-to-English are the same task) or the language pair (French-to-English and German-to-English are different tasks). Sentence and task level routing are formulated as follows:

$$\mathcal{G}_{s,E} = \mathrm{GATE}(\frac{1}{S}\sum_{s=1}^{S}x_s)$$
 (Sentence-level),  $\mathcal{G}_{s,E} = \mathrm{GATE}(\mathrm{task\_id}_s)$  (Task-level).

We illustrate the difference in Figure [1,](#page-2-0) in tokenbased MoE models (Figure [1b\)](#page-2-0), tokens from each datapoint are routed to different experts, whereas in task-level MoE models (Figure [1a\)](#page-2-0), tokens may be routed to the same expert based on task.

### 3.2 Inference Implications of Routing Strategies

While the MoE models discussed in [\(Shazeer et al.,](#page-10-2) [2017;](#page-10-2) [Lepikhin et al.,](#page-10-1) [2020\)](#page-10-1) train quickly relative to the number of parameters in terms of the wallclock time, they are expensive to serve. Consider a MoE with 512 experts and 50B parameters [\(Lep](#page-10-1)[ikhin et al.,](#page-10-1) [2020\)](#page-10-1). When employing token-level routing, each token can be independently routed to a different set of experts during inference. Given

that the entire model is too large to load into memory on a single accelerator, the two potential solutions to utilize this model for inference are: (i) Loading experts dynamically from host to device depending on routing decisions, or (ii) Utilizing model-parallelism over multiple accelerators for serving. While the first solution incurs heavy host-device communication costs, the second introduces significantly inter-device communication overhead.

Other practical approaches to serve a large MoE include model quantization, pruning and knowledge distillation (Cheng et al., 2017). While the first two strategies haven't been explored in the context of conditional computation, distillation (Hinton et al., 2015; Kim and Rush, 2016) has been found to introduce undesirable artifacts into the student model (Freitag et al., 2019; Bogoychev and Sennrich, 2019) in the context of NMT. Moreover, some studies have found that distilling large sparse models preserves only a small fraction of the gains achieved by scaling. On the other hand, if we limit the number of experts available to every task in the model to a small fraction of the total available capacity, it is possible to extract task-specific models for serving, alleviating the need for complex serving strategies or compression. Since decoding time complexity for auto-regressive encoderdecoder models is dominated by the decoder (Kasai et al., 2020), we can also pursue a hybrid strategy where the encoder utilizes more expensive routing strategies while the decoder of the model utilizes simpler and efficient routing.

Summarizing the *effective* decoding cost of the MoE models utilizing different routing strategies:

- Token/Sentence level routing: The routing decisions are made dynamically. Assuming each token/sentence makes disjoint choices, the server needs to load all *E* experts.
- Task-level routing: Tokens corresponding to each input sentence are routed to the same experts statically. The server only needs to pre-load K experts (assuming top-K routing).

#### <span id="page-3-0"></span>4 Experiments on 30 Language Pairs

We compare routing strategies at multiple levels in both, the encoder and the decoder, by conducting extensive experiments on two benchmarks: the public WMT dataset with 30 language pairs (Section 4.1) and an in-house web-scale dataset with 200

language pairs (Section 5). We start with WMT setup.

#### <span id="page-3-1"></span>4.1 Experimental Setup

For our experiments, we use parallel training and evaluation data from the WMT corpus and adopt the setup used by (Siddhant et al., 2020) with 15 languages, to and from English. Full training data details may be found in Table 3 in the Appendix. The amount of data ranges from more than 60 million sentence pairs in en-cs translation direction (en-cs) to roughly 150k sentence pairs for en-gu.

We use a temperature based data sampling strategy to train our models, similar to the strategy used to train the multilingual models in (Arivazhagan et al., 2019): if  $p_L$  is the probability that a sentence in the corpus belongs to language pair L, we sample from a distribution where the probability of sampling from L is proportional to  $p_L^{\frac{1}{T}}$ . All the experiments in this paper are performed on a model trained with a sampling temperature T=5.

We use the 142M Transformer Base (Vaswani et al., 2017) architecture (or enhanced versions of it with MoE layers) for all of our experiments with WMT. Our models are optimized using Adafactor (Shazeer and Stern, 2018) with momentum factorization and a per-parameter norm clipping threshold of 1.0. We followed a learning rate of 3.0, with 40K warm-up steps for the schedule, which is decayed with the inverse square root of the number of training steps after warm-up. BLEU scores presented in this paper are calculated using Sacre-BLEU (Post, 2018) on the WMT test sets.

**Multilingual baseline:** We train a Transformer Base model on this dataset as our multilingual dense baseline. We share all parameters across language pairs, including the softmax layer and input/output word embeddings. We use a 64k token Sentence Piece vocabulary (Kudo and Richardson, 2018). The vocabulary is shared on both the encoder and decoder side. Each sentence pair has a  $<2\times\times$  token pre-pended to the source sentence to indicate the target language, following Johnson et al. (2017).

**Mixture of Experts Models:** For MoE models, we replace the feed forward network (FFN) of alternate layers of the Transformer with a set of identical FFN experts as depicted in Figure 1b. For brevity, we provide aggregate BLEU scores in Section 4.2. We provide the full individual BLEU

<span id="page-4-1"></span>

| System                         | Routing G     | ranularity    | Throughput          |         | В     | LEU   |      |      |
|--------------------------------|---------------|---------------|---------------------|---------|-------|-------|------|------|
| System                         | Encoder       | Decoder       | Peak tokens/s       | Average | xx2en | en2xx | High | Low  |
| Bilingual Baselines            | -             | -             | $2.3 \times 10^{5}$ | 21.0    | 21.8  | 18.9  | 28.2 | 11.8 |
| Multilingual Transformer-Base  | -             | -             | 2.3 × 10            | 20.0    | 23.7  | 17.5  | 23.3 | 15.9 |
| Static MoE – 32 experts        | -             | -             | $2.3 \times 10^{5}$ | 17.6    | 25.0  | 10.2  | 20.9 | 13.5 |
| Token-level MoE – 32 experts   | Token         | Token         | $1.3 \times 10^{5}$ | 22.6    | 24.9  | 20.4  | 27.5 | 16.3 |
| Sentence-level MoE – 32 expert | Sentence      | Sentence      | $1.3 \times 10^{5}$ | 19.9    | 24.1  | 16.8  | 22.6 | 16.1 |
|                                | Language Pair | Language Pair |                     | 21.4    | 25.2  | 16.9  | 23.4 | 17.3 |
|                                | Target        | Target        |                     | 22.9    | 25.6  | 20.2  | 27.2 | 17.3 |
| Task-level MoE – 32 experts    | Language Pair | Token         | $2.3 \times 10^{5}$ | 22.4    | 25.6  | 20.3  | 26.9 | 16.8 |
| rask-level MOE – 32 experts    | Target        | Token         | 2.3 × 10°           | 22.3    | 24.5  | 20.4  | 26.8 | 16.6 |
|                                | Token         | Language Pair |                     | 23.0    | 26.2  | 20.3  | 27.2 | 17.6 |
|                                | Token         | Target        |                     | 23.6    | 26.0  | 21.1  | 28.5 | 17.4 |

Table 1: **Routing strategies for Mixture-of-Experts (MoE) models** – We compare routing experts by either tokens, sentence representations, or tasks (using either language pairs or target languages). For task-level MoE, routing can also be different between encoder and decoder. For results, *Average* is the average results of all language pairs, whereas xx2en and en2xx are the averages of translations into and from English respectively. *High* indicates high-resource language pairs (> 1 million sentence pairs) while Low is for low-resource language pairs (< 1 million sentence pairs).

scores in the Appendix A.3, along with bilingual baselines. In addition, we provide the number of parameters for different components of our models in Appendix A.4.

